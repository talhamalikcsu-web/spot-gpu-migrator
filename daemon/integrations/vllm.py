"""
vLLM Inference Engine Integration Hook for Spot GPU Migrator (SGM).

Coordinates OpenAI-compatible streaming API (/v1/chat/completions), fast engine abort
signaling (/abort), and prefix-caching continuation payload construction for instant
RadixAttention / Automatic Prefix Caching (APC) hit on Standby GPUs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from daemon.integrations.base import (
    AbortResult,
    AbstractInferenceEngineHook,
    BackendType,
    EngineMetadata,
)
from daemon.models import InferenceSession, TokenChunk

logger = logging.getLogger("sgm.daemon.integrations.vllm")


class VLLMInferenceEngineHook(AbstractInferenceEngineHook):
    """
    Production integration hook for vLLM (v0.6.0+) and SGLang (v0.3.0+).

    Interfaces with:
    1. /abort: Aborts in-flight inference requests in <= 50ms on preemption.
    2. /v1/chat/completions & /v1/completions: Dispatches streaming requests with prefix caching.
    3. /pause: Pauses generation when supported by upstream proxies or engines.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 5.0,
        abort_timeout: float = 0.05,
        engine_url: Optional[str] = None,
        timeout_ms: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        effective_url = engine_url or base_url or "http://127.0.0.1:8001"
        effective_abort_timeout = (timeout_ms / 1000.0) if timeout_ms is not None else abort_timeout
        super().__init__(base_url=effective_url, timeout=timeout)
        self.abort_timeout = effective_abort_timeout
        self._session = session
        self._owns_session = session is None

    @property
    def engine_url(self) -> str:
        return self.base_url

    @engine_url.setter
    def engine_url(self, value: str) -> None:
        self.base_url = value.rstrip("/")

    @property
    def timeout_ms(self) -> float:
        return self.abort_timeout * 1000.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout, sock_connect=2.0)
            self._session = aiohttp.ClientSession(timeout=client_timeout)
        return self._session

    async def close(self) -> None:
        """Closes the underlying aiohttp client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def abort_request(
        self,
        request_id: str,
        timeout_ms: Optional[float] = None,
    ) -> AbortResult:
        """
        Sends an abort cancellation signal to the vLLM engine (/abort).

        SLA Target: Dispatched and completed in <= 50ms to free GPU execution cycles
        without unhandled exceptions.
        """
        start_time = time.monotonic()
        limit_s = (timeout_ms / 1000.0) if timeout_ms is not None else self.abort_timeout
        url = f"{self.base_url}/abort"
        payload = {"request_id": request_id}

        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=limit_s),
            ) as resp:
                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                success = resp.status in (200, 204, 404)
                if success:
                    logger.info("vLLM aborted request %s in %.2fms (HTTP %d)", request_id, elapsed_ms, resp.status)
                else:
                    logger.warning("vLLM /abort returned unexpected status HTTP %d in %.2fms", resp.status, elapsed_ms)
                return AbortResult(
                    success=success,
                    latency_ms=elapsed_ms,
                    status_code=resp.status,
                    request_id=request_id,
                    message=f"HTTP {resp.status}",
                )
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.warning("vLLM /abort timed out after %.2fms for req %s", elapsed_ms, request_id)
            return AbortResult(
                success=False,
                latency_ms=elapsed_ms,
                status_code=408,
                request_id=request_id,
                message=f"Timeout after {elapsed_ms:.1f}ms",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.debug("vLLM /abort exception for req %s after %.2fms: %s", request_id, elapsed_ms, exc)
            return AbortResult(
                success=False,
                latency_ms=elapsed_ms,
                status_code=500,
                request_id=request_id,
                message=str(exc),
            )

    async def pause_request(
        self,
        request_id: str,
        timeout_ms: Optional[float] = None,
    ) -> AbortResult:
        """
        Attempts to pause inference on the local engine. If /pause is unavailable,
        falls back to aborting the request to prevent duplicate token generation.
        """
        url = f"{self.base_url}/pause"
        payload = {"request_id": request_id}
        start_time = time.monotonic()
        limit_s = (timeout_ms / 1000.0) if timeout_ms is not None else min(0.2, self.abort_timeout)
        try:
            session = await self._get_session()
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=limit_s),
            ) as resp:
                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                if resp.status == 200:
                    logger.info("Local engine paused request %s in %.2fms", request_id, elapsed_ms)
                    return AbortResult(
                        success=True,
                        latency_ms=elapsed_ms,
                        status_code=200,
                        request_id=request_id,
                        message="HTTP 200",
                    )
        except Exception as exc:
            logger.debug("Engine /pause endpoint not available (%s), falling back to /abort", exc)

        return await self.abort_request(request_id, timeout_ms=timeout_ms)

    def format_resumption_payload(
        self,
        session: InferenceSession,
        cutoff_sequence_id: Optional[int] = None,
        include_messages: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Formats an OpenAI / vLLM continuation payload with the exact token prefix
        to trigger vLLM's RadixAttention / PagedAttention Automatic Prefix Caching (APC).
        """
        cutoff = cutoff_sequence_id
        if cutoff is None:
            cutoff = session.last_flushed_sequence_id

        if cutoff >= 0 and session.generated_text:
            num_tokens = min(len(session.generated_text), cutoff + 1)
            prefix_tokens = session.generated_text[:num_tokens]
        elif session.generated_text:
            prefix_tokens = list(session.generated_text)
            num_tokens = len(prefix_tokens)
        else:
            prefix_tokens = []
            num_tokens = 0

        generated_prefix_str = "".join(prefix_tokens)
        full_prefix_prompt = f"{session.prompt}{generated_prefix_str}"

        total_gen = num_tokens if num_tokens > 0 else session.total_tokens_generated
        original_max = session.sampling_params.max_tokens if session.sampling_params else 512
        remaining_max_tokens = max(1, original_max - total_gen)

        messages: List[Dict[str, Any]] = []
        if session.prompt and generated_prefix_str:
            messages = [
                {"role": "user", "content": session.prompt},
                {"role": "assistant", "content": generated_prefix_str},
            ]
        elif session.prompt:
            messages = [{"role": "user", "content": session.prompt}]
        else:
            messages = [{"role": "user", "content": full_prefix_prompt}]

        payload: Dict[str, Any] = {
            "model": session.model,
            "prompt": full_prefix_prompt,
            "stream": session.is_streaming,
            "max_tokens": remaining_max_tokens,
            "temperature": session.sampling_params.temperature if session.sampling_params else 0.7,
            "top_p": session.sampling_params.top_p if session.sampling_params else 0.95,
            "stop": list(session.sampling_params.stop) if session.sampling_params else ["<|eot_id|>", "</s>"],
            "stream_options": {"include_usage": True},
            "presence_penalty": session.sampling_params.presence_penalty if session.sampling_params else 0.0,
            "frequency_penalty": session.sampling_params.frequency_penalty if session.sampling_params else 0.0,
            "request_id": session.request_id,
            "start_seq_id": cutoff + 1 if cutoff is not None else 0,
            "is_resume": True,
        }

        if include_messages:
            payload["messages"] = messages
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False if generated_prefix_str else True

        if session.prompt_tokens or session.generated_tokens:
            prompt_tokens = list(session.prompt_tokens)
            gen_tokens = list(session.generated_tokens[:num_tokens])
            payload["prompt_token_ids"] = prompt_tokens + gen_tokens

        payload["extra_body"] = {
            "cutoff_sequence_id": cutoff,
            "prefix_caching": True,
            "radix_attention_enabled": True,
            "cached_prefix_tokens": num_tokens,
        }

        payload.update(kwargs)
        return payload

    def build_prefix_caching_payload(
        self,
        session: InferenceSession,
        format_type: str = "chat",
        cutoff_sequence_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Constructs a continuation payload formatted specifically for vLLM / SGLang
        Automatic Prefix Caching (APC) / RadixAttention.
        """
        return self.format_resumption_payload(
            session=session,
            cutoff_sequence_id=cutoff_sequence_id,
            include_messages=(format_type == "chat"),
            **kwargs,
        )

    build_resumption_payload = format_resumption_payload

    async def resume_request(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepares and initiates resumption on the standby inference engine.

        Returns metadata descriptor including payload and headers for proxy or daemon.
        """
        target_endpoint = (resume_endpoint or self.base_url).rstrip("/")
        payload = self.build_prefix_caching_payload(session, format_type="chat")
        headers = {
            "Content-Type": "application/json",
            "X-SGM-Request-ID": session.request_id,
            "X-SGM-Client-Conn-ID": session.client_connection_id,
            "X-SGM-Resumed": "true",
            "X-SGM-Cutoff-Sequence-ID": str(session.last_flushed_sequence_id),
        }

        logger.info(
            "Resuming session %s on endpoint %s (cutoff seq: %d, remaining tokens: %d)",
            session.request_id,
            target_endpoint,
            session.last_flushed_sequence_id,
            payload.get("max_tokens", 0),
        )

        return {
            "endpoint": target_endpoint,
            "url": f"{target_endpoint}/v1/chat/completions",
            "payload": payload,
            "headers": headers,
            "starting_sequence_id": session.last_flushed_sequence_id + 1,
            "request_id": session.request_id,
        }

    async def stream_resumed(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
    ) -> AsyncIterator[TokenChunk]:
        """
        Connects to the standby engine, transmits the prefix-caching payload,
        and yields parsed SSE TokenChunks with strictly monotonic sequence IDs.
        """
        target_endpoint = (resume_endpoint or self.base_url).rstrip("/")
        url = f"{target_endpoint}/v1/chat/completions"
        payload = self.build_prefix_caching_payload(session, format_type="chat")
        headers = {
            "Content-Type": "application/json",
            "X-SGM-Request-ID": session.request_id,
            "X-SGM-Resumed": "true",
        }

        http_session = await self._get_session()
        next_seq = session.last_flushed_sequence_id + 1

        try:
            async with http_session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    logger.error("Standby engine %s returned HTTP %d for resumed req %s", url, resp.status, session.request_id)
                    return

                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            token_str = ""
                            choices = data.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                                token_str = delta.get("content", "")
                            elif "token" in data:
                                token_str = data["token"]

                            # Monotonic sequence id
                            seq_id = int(data.get("seq", next_seq))
                            yield TokenChunk(
                                request_id=session.request_id,
                                sequence_id=seq_id,
                                token=token_str,
                                raw_bytes=line_bytes,
                            )
                            next_seq = seq_id + 1
                        except Exception as parse_err:
                            logger.debug("Failed parsing resumed SSE line: %s", parse_err)
        except Exception as exc:
            logger.error("Error streaming resumed tokens for req %s from %s: %s", session.request_id, url, exc)

    async def health_check(self, endpoint: Optional[str] = None) -> bool:
        """Probes the /health endpoint of the inference engine."""
        target_url = f"{(endpoint or self.base_url).rstrip('/')}/health"
        try:
            session = await self._get_session()
            async with session.get(target_url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_engine_metadata(self) -> EngineMetadata:
        """Returns metadata descriptor for vLLM."""
        return EngineMetadata(
            backend=BackendType.VLLM,
            base_url=self.base_url,
            version="0.6.0",
            supports_abort_endpoint=True,
            supports_prefix_caching=True,
            native_streaming_endpoint=f"{self.base_url}/v1/chat/completions",
            openai_compatible_endpoint=f"{self.base_url}/v1/chat/completions",
            kv_cache_block_size=16,
            detected_via="heuristic",
        )

