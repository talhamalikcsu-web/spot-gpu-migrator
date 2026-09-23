"""
Abstract Inference Engine Hook Interface for Spot GPU Migrator (SGM).

Defines the contract for controlling and resuming active inference sessions across
local and remote inference engines (vLLM, SGLang, NVIDIA TensorRT-LLM, HuggingFace TGI, Mock).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from daemon.models import InferenceSession, SGMError, TokenChunk

logger = logging.getLogger("sgm.daemon.integrations.base")


class BackendType(str, Enum):
    """Supported inference backend types."""
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt_llm"
    TGI = "tgi"
    MOCK = "mock"


class EngineDetectionError(SGMError):
    """Raised when inference engine auto-detection fails across all probes."""
    pass


class AbortResult(dict):
    """Result of an engine abort or pause request."""

    def __init__(
        self,
        success: bool,
        latency_ms: float,
        status_code: int = 200,
        request_id: str = "",
        message: str = "",
        backend: Optional[BackendType] = None,
    ) -> None:
        super().__init__(
            success=success,
            latency_ms=latency_ms,
            status_code=status_code,
            request_id=request_id,
            message=message,
            backend=backend.value if hasattr(backend, "value") else (str(backend) if backend else None),
        )
        self.success = success
        self.latency_ms = latency_ms
        self.status_code = status_code
        self.request_id = request_id
        self.message = message
        self.backend = backend

    def __bool__(self) -> bool:
        return self.success

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self.success == other
        return super().__eq__(other)


class ResumptionDescriptor(BaseModel):
    """Payload and connection metadata required to resume generation on Standby."""
    endpoint: str
    url: str
    backend: BackendType
    payload: Dict[str, Any]
    headers: Dict[str, str]
    starting_sequence_id: int
    request_id: str
    remaining_tokens: int


class EngineMetadata(BaseModel):
    """Discovery and capability descriptor for an inference engine instance."""
    backend: BackendType
    base_url: str
    version: str = "unknown"
    model_name: str = ""
    supports_abort_endpoint: bool = False
    supports_prefix_caching: bool = True
    native_streaming_endpoint: str
    openai_compatible_endpoint: str
    kv_cache_block_size: int = 16
    detected_via: str = "heuristic"


class AbstractInferenceEngineHook(ABC):
    """
    Abstract interface for coordinating inference engine lifecycles during spot preemption.

    Subclasses implement engine-specific control protocols:
    - vLLM: Dedicated HTTP POST /abort and Automatic Prefix Caching (APC)
    - SGLang: Dedicated HTTP POST /abort and RadixAttention tree hints
    - NVIDIA TensorRT-LLM: Triton decoupled /generate_stream, sub-50ms socket abort, KV-cache reuse
    - HuggingFace TGI: Native /generate_stream, OpenAI /compat, sub-50ms hyper socket closure
    - Mock: Fast offset replay engine for deterministic testing and simulation
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 5.0,
        abort_timeout: float = 0.05,
        timeout_ms: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.abort_timeout = (timeout_ms / 1000.0) if timeout_ms is not None else abort_timeout

    def __await__(self):
        """Allows hook instances to be awaited seamlessly in async factory workflows."""
        async def _identity():
            return self
        return _identity().__await__()

    async def close(self) -> None:
        """Closes any underlying network sessions or open sockets."""
        pass

    @abstractmethod
    async def pause_request(self, request_id: str, timeout_ms: Optional[float] = None) -> AbortResult:
        """
        Pauses or freezes inference for an active request at a clean sequence boundary.

        Args:
            request_id: Unique identifier of the request to halt.
            timeout_ms: Optional deadline in milliseconds.

        Returns:
            AbortResult containing operation status, latency, and response details.
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
    async def abort_request(self, request_id: str, timeout_ms: Optional[float] = None) -> AbortResult:
        """
        Aborts an active inference request to free GPU resources immediately.

        Must complete within <= 50ms SLA to prevent preemption lockups.

        Args:
            request_id: Unique identifier of the request to abort.
            timeout_ms: Optional deadline in milliseconds.

        Returns:
            AbortResult indicating cancellation acknowledgement and measured latency.
        """
        pass

    @abstractmethod
    def build_prefix_caching_payload(
        self,
        session: InferenceSession,
        format_type: str = "chat",
        cutoff_sequence_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Constructs an engine request payload structured for instant prefix-caching lookup.

        Ensures identical prompt tokens and previously generated prefix tokens
        are supplied so that standby GPU achieves an immediate KV-cache hit.

        Args:
            session: Active inference session snapshot.
            format_type: Endpoint or schema format indicator.
            cutoff_sequence_id: Optional explicit sequence cutoff identifier.
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

    @abstractmethod
    def get_engine_metadata(self) -> EngineMetadata:
        """
        Returns discovery and capability descriptor for the inference engine.
        """
        pass
