"""
Unified Inference Engine Hook Factory with Sub-100ms Auto-Detection Heuristics.

Supports:
- NVIDIA TensorRT-LLM (via Triton /v2/health/ready)
- Hugging Face TGI (/info)
- SGLang (/get_model_info)
- vLLM (/version)
- Mock Inference Simulator (/health with role/paused payload)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Type

import aiohttp

from daemon.integrations.base import (
    AbstractInferenceEngineHook,
    BackendType,
    EngineDetectionError,
)
from daemon.integrations.mock import MockInferenceEngineHook
from daemon.integrations.sglang import SGLangInferenceEngineHook
from daemon.integrations.tensorrt_llm import TensorRTLLMEngineHook
from daemon.integrations.tgi import TGIEngineHook
from daemon.integrations.vllm import VLLMInferenceEngineHook

logger = logging.getLogger("sgm.daemon.integrations.factory")

ENGINE_HOOK_REGISTRY: Dict[str, Type[AbstractInferenceEngineHook]] = {
    BackendType.VLLM.value: VLLMInferenceEngineHook,
    BackendType.SGLANG.value: SGLangInferenceEngineHook,
    BackendType.TENSORRT_LLM.value: TensorRTLLMEngineHook,
    BackendType.TGI.value: TGIEngineHook,
    BackendType.MOCK.value: MockInferenceEngineHook,
    # Common aliases
    "trtllm": TensorRTLLMEngineHook,
    "trt": TensorRTLLMEngineHook,
    "tensorrt": TensorRTLLMEngineHook,
    "text_generation_inference": TGIEngineHook,
    "srt": SGLangInferenceEngineHook,
    "sim": MockInferenceEngineHook,
    "simulator": MockInferenceEngineHook,
}


class EngineHookFactory:
    """
    Factory class responsible for discovering, instantiating, and configuring
    the appropriate engine hook for any supported inference runtime.
    """

    @classmethod
    async def create(
        cls,
        engine_type: Optional[str] = "auto",
        engine_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        timeout: float = 5.0,
        abort_timeout: float = 0.05,
        timeout_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> AbstractInferenceEngineHook:
        """
        Creates an engine hook instance, automatically detecting backend type if 'auto'.

        Args:
            engine_type: Backend identifier ('vllm', 'tensorrt_llm', 'tgi', 'sglang', 'mock', 'auto').
                         If a URL string is passed here, it is treated as engine_url with type 'auto'.
            engine_url: Base HTTP URL of the target engine runtime.
            session: Optional shared aiohttp.ClientSession.
            timeout: Default HTTP timeout.
            abort_timeout: Fast abort timeout budget (<= 50ms).
            timeout_ms: Millisecond override for abort_timeout.
            **kwargs: Extra parameters passed to the hook constructor.

        Returns:
            Configured AbstractInferenceEngineHook instance.
        """
        # Handle positional arguments in either order: (type, url), (url, type), or (url,)
        if engine_type and (engine_type.startswith("http://") or engine_type.startswith("https://")):
            actual_url = engine_type
            actual_type = engine_url if engine_url and not (engine_url.startswith("http://") or engine_url.startswith("https://")) else "auto"
            engine_url = actual_url
            engine_type = actual_type

        effective_url = engine_url or "http://127.0.0.1:8001"
        target_type = (engine_type or "auto").lower()

        # Check environment override
        env_override = os.environ.get("SGM_ENGINE_TYPE")
        if env_override and env_override.lower() not in ("", "auto", "none"):
            target_type = env_override.lower()

        if target_type == "auto":
            try:
                fallback_arg = kwargs.pop("fallback", None)
                detected_backend = await cls.auto_detect(effective_url, session=session, fallback=fallback_arg)
                target_type = detected_backend.value if hasattr(detected_backend, "value") else str(detected_backend)
                logger.info("Auto-detected engine backend for %s: %s", effective_url, target_type)
            except EngineDetectionError:
                if kwargs.get("strict_detection", False):
                    raise
                logger.warning("Auto-detection failed for %s, falling back to vLLM", effective_url)
                target_type = BackendType.VLLM.value
        elif target_type not in ENGINE_HOOK_REGISTRY:
            raise ValueError(f"Unknown or unsupported engine type: '{target_type}'. Supported: {list(ENGINE_HOOK_REGISTRY.keys())}")

        hook_cls = ENGINE_HOOK_REGISTRY.get(target_type, VLLMInferenceEngineHook)
        return hook_cls(
            base_url=effective_url,
            timeout=timeout,
            abort_timeout=abort_timeout,
            timeout_ms=timeout_ms,
            session=session,
            **kwargs,
        )

    @classmethod
    def create_sync(
        cls,
        engine_type: str,
        engine_url: str = "http://127.0.0.1:8001",
        timeout: float = 5.0,
        abort_timeout: float = 0.05,
        timeout_ms: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
        **kwargs: Any,
    ) -> AbstractInferenceEngineHook:
        """
        Synchronously instantiates a hook for a known engine_type without network auto-detection.
        """
        target_type = engine_type.lower() if engine_type else "vllm"
        if target_type not in ENGINE_HOOK_REGISTRY:
            raise ValueError(f"Unknown or unsupported engine type: '{target_type}'. Supported: {list(ENGINE_HOOK_REGISTRY.keys())}")
        hook_cls = ENGINE_HOOK_REGISTRY.get(target_type, VLLMInferenceEngineHook)
        return hook_cls(
            base_url=engine_url,
            timeout=timeout,
            abort_timeout=abort_timeout,
            timeout_ms=timeout_ms,
            session=session,
            **kwargs,
        )

    @classmethod
    async def auto_detect(
        cls,
        engine_url: str,
        session: Optional[aiohttp.ClientSession] = None,
        timeout_s: float = 0.1,
        retries: int = 1,
        fallback: Optional[BackendType] = None,
    ) -> BackendType:
        """
        Executes sub-100ms discovery probes against the target endpoint to identify backend.

        Probing sequence:
        1. /v2/health/ready -> tensorrt_llm
        2. /info            -> tgi
        3. /get_model_info  -> sglang
        4. /version         -> vllm
        5. /health          -> mock (if role/paused in JSON) or fallback vllm
        """
        base = engine_url.rstrip("/")
        owns_session = session is None
        http_session = session or aiohttp.ClientSession()

        probe_timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=min(0.05, timeout_s))

        try:
            for attempt in range(max(1, retries)):
                t_start = time.monotonic()

                # Probe 1: Triton / TensorRT-LLM (/v2/health/ready)
                try:
                    async with http_session.get(f"{base}/v2/health/ready", timeout=probe_timeout) as resp:
                        if resp.status == 200:
                            elapsed_ms = (time.monotonic() - t_start) * 1000.0
                            logger.debug("TRT-LLM detected at %s in %.1fms", base, elapsed_ms)
                            return BackendType.TENSORRT_LLM
                except Exception:
                    pass

                # Probe 2: Hugging Face TGI (/info)
                try:
                    async with http_session.get(f"{base}/info", timeout=probe_timeout) as resp:
                        if resp.status == 200:
                            elapsed_ms = (time.monotonic() - t_start) * 1000.0
                            logger.debug("TGI detected at %s in %.1fms", base, elapsed_ms)
                            return BackendType.TGI
                except Exception:
                    pass

                # Probe 3: SGLang (/get_model_info)
                try:
                    async with http_session.get(f"{base}/get_model_info", timeout=probe_timeout) as resp:
                        if resp.status == 200:
                            elapsed_ms = (time.monotonic() - t_start) * 1000.0
                            logger.debug("SGLang detected at %s in %.1fms", base, elapsed_ms)
                            return BackendType.SGLANG
                except Exception:
                    pass

                # Probe 4: vLLM (/version)
                try:
                    async with http_session.get(f"{base}/version", timeout=probe_timeout) as resp:
                        if resp.status == 200:
                            elapsed_ms = (time.monotonic() - t_start) * 1000.0
                            logger.debug("vLLM detected at %s in %.1fms", base, elapsed_ms)
                            return BackendType.VLLM
                except Exception:
                    pass

                # Probe 5: /health (Mock or vLLM fallback)
                try:
                    async with http_session.get(f"{base}/health", timeout=probe_timeout) as resp:
                        if resp.status == 200:
                            try:
                                data = await resp.json()
                                if isinstance(data, dict):
                                    if "role" in data or "paused" in data or "port" in data:
                                        elapsed_ms = (time.monotonic() - t_start) * 1000.0
                                        logger.debug("Mock engine detected at %s in %.1fms", base, elapsed_ms)
                                        return BackendType.MOCK
                            except Exception:
                                pass
                            # Generic /health returns 200 -> default to vllm
                            return BackendType.VLLM
                except Exception:
                    pass

                if attempt < retries - 1:
                    await asyncio.sleep(min(0.2 * (2 ** attempt), 2.0))

            if fallback is not None:
                logger.info("Auto-detection probes failed, returning specified fallback: %s", fallback)
                return fallback

            raise EngineDetectionError(
                f"Failed to auto-detect inference engine at {engine_url} after {retries} attempt(s)"
            )

        finally:
            if owns_session and not http_session.closed:
                await http_session.close()
