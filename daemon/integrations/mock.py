"""
Mock LLM Inference Engine Integration Hook for Spot GPU Migrator (SGM).

Used for deterministic automated testing, preemption fault-injection benchmarks,
and continuous integration verification without requiring physical GPUs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import aiohttp

from daemon.integrations.base import BackendType, EngineMetadata
from daemon.integrations.vllm import VLLMInferenceEngineHook

logger = logging.getLogger("sgm.daemon.integrations.mock")


class MockInferenceEngineHook(VLLMInferenceEngineHook):
    """
    Hook connecting to local or simulated MockLLMEngine instances.
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
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            abort_timeout=abort_timeout,
            engine_url=engine_url,
            timeout_ms=timeout_ms,
            session=session,
        )

    def get_engine_metadata(self) -> EngineMetadata:
        """Returns metadata descriptor for MockLLMEngine."""
        return EngineMetadata(
            backend=BackendType.MOCK,
            base_url=self.base_url,
            version="1.0.0",
            supports_abort_endpoint=True,
            supports_prefix_caching=True,
            native_streaming_endpoint=f"{self.base_url}/v1/chat/completions",
            openai_compatible_endpoint=f"{self.base_url}/v1/chat/completions",
            kv_cache_block_size=16,
            detected_via="heuristic",
        )
