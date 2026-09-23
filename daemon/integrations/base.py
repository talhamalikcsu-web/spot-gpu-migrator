"""
Abstract Inference Engine Hook Interface for Spot GPU Migrator (SGM).

Defines the contract for controlling and resuming active inference sessions across
local and remote inference engines (vLLM, SGLang, HuggingFace TGI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from daemon.models import InferenceSession, TokenChunk

logger = logging.getLogger("sgm.daemon.integrations.base")


class AbstractInferenceEngineHook(ABC):
    """
    Abstract interface for coordinating inference engine lifecycles during spot preemption.

    Subclasses implement engine-specific control protocols (e.g. vLLM HTTP /abort,
    SGLang RadixAttention cache binding, or in-process AsyncLLMEngine hooks).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abstractmethod
    async def pause_request(self, request_id: str) -> bool:
        """
        Pauses or freezes inference for an active request at a clean sequence boundary.

        Args:
            request_id: Unique identifier of the request to halt.

        Returns:
            True if the engine successfully paused generation, False otherwise.
        """
        pass

    @abstractmethod
    async def resume_request(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches or prepares request resumption on the target standby engine.

        Args:
            session: Snapshot of the in-flight inference session with generated tokens.
            resume_endpoint: Target engine base URL (defaults to self.base_url).

        Returns:
            Dictionary containing invocation response or dispatch status.
        """
        pass

    @abstractmethod
    async def abort_request(self, request_id: str) -> bool:
        """
        Aborts an active inference request to free GPU resources immediately.

        Must complete within <= 50ms SLA to prevent preemption lockups.

        Args:
            request_id: Unique identifier of the request to abort.

        Returns:
            True if cancellation was acknowledged by the engine, False otherwise.
        """
        pass

    @abstractmethod
    def build_prefix_caching_payload(
        self,
        session: InferenceSession,
        format_type: str = "chat",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Constructs an OpenAI-compatible request payload structured for instant
        prefix-caching lookup (vLLM Automatic Prefix Caching / SGLang RadixAttention).

        Ensures identical prompt tokens and previously generated prefix tokens
        are supplied so that standby GPU achieves an immediate KV-cache hit.

        Args:
            session: Active inference session snapshot.
            format_type: 'chat' for /v1/chat/completions or 'completions' for /v1/completions.
            **kwargs: Extra parameters to pass through to payload.

        Returns:
            Complete JSON-serializable request payload dictionary.
        """
        pass

    @abstractmethod
    async def stream_resumed(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
    ) -> AsyncIterator[TokenChunk]:
        """
        Streams resumed tokens from the standby inference engine, yielding TokenChunk
        instances with monotonic sequence numbering starting from cutoff + 1.

        Args:
            session: Active inference session snapshot.
            resume_endpoint: Optional override for the target inference engine URL.

        Yields:
            TokenChunk instances continuing the stream seamlessly.
        """
        pass

    @abstractmethod
    async def health_check(self, endpoint: Optional[str] = None) -> bool:
        """
        Checks whether the inference engine endpoint is healthy and ready for inference.

        Args:
            endpoint: Optional override URL to check (defaults to self.base_url).

        Returns:
            True if healthy (HTTP 200), False otherwise.
        """
        pass
