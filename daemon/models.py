"""
Data models, enums, and exception types for the Spot GPU Migrator (SGM).

Defines serialization contracts for preemption events, node lifecycle states,
in-flight inference sessions, SGM-P2P protocol payloads, and token streams.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeLifecycleState(str, Enum):
    """Finite State Machine states for spot node lifecycle."""
    HEALTHY = "HEALTHY"
    PREEMPTION_DETECTED = "PREEMPTION_DETECTED"
    BUFFERING_INGRESS = "BUFFERING_INGRESS"
    STATE_STREAMING = "STATE_STREAMING"
    HANDOVER_COMPLETE = "HANDOVER_COMPLETE"
    TERMINATED = "TERMINATED"


class MessageType(int, Enum):
    """SGM-P2P wire protocol message types."""
    HANDOVER_INIT = 0x01
    REQUEST_STATE = 0x02
    TOKEN_DELTA = 0x03
    KV_CACHE_DESC = 0x04
    HANDOVER_FIN = 0x05
    HANDOVER_ACK = 0x06
    HANDOVER_NACK = 0x07
    HEARTBEAT = 0x08


class FrameFlags(int, Enum):
    """Bitfield flags for SGM-P2P binary frames."""
    NONE = 0x0000
    IS_COMPRESSED = 0x0001
    IS_LAST_CHUNK = 0x0002


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class SGMError(Exception):
    """Base exception for all Spot GPU Migrator errors."""
    pass


class MetadataPollTimeoutError(SGMError):
    """Raised when cloud metadata endpoint exceeds response timeout."""
    pass


class IMDSv2TokenExpiredError(SGMError):
    """Raised when AWS IMDSv2 returns 401 Unauthorized."""
    pass


class P2PConnectionFailedError(SGMError):
    """Raised when connection to peer node on port 9002 fails."""
    pass


class CRC32VerificationError(SGMError):
    """Raised when a binary frame fails CRC32 integrity check."""
    pass


class TokenSequenceGapError(SGMError):
    """Raised when incoming token stream has a missing or discontinuous sequence ID."""
    pass


class ClientDisconnectedError(SGMError):
    """Raised when downstream client terminates HTTP/SSE connection."""
    pass


class MigrationHandoverError(SGMError):
    """Raised when node state transfer or upstream swap fails."""
    pass


# ---------------------------------------------------------------------------
# Preemption & Event Models
# ---------------------------------------------------------------------------

class PreemptionEvent(BaseModel):
    """Standardized event emitted by cloud metadata watchdogs."""
    provider: str = Field(..., description="Cloud provider: 'aws', 'gcp', or 'runpod'")
    action: str = Field(default="terminate", description="Action signaled by hypervisor")
    deadline: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of deadline")
    deadline_seconds: float = Field(default=30.0, description="Remaining grace period in seconds")
    timestamp_ns: int = Field(default_factory=time.time_ns, description="Detection timestamp in nanoseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific metadata")


# ---------------------------------------------------------------------------
# LLM & Inference State Models
# ---------------------------------------------------------------------------

class SamplingParams(BaseModel):
    """Sampling parameters matching OpenAI / vLLM specifications."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    stop: List[str] = Field(default_factory=lambda: ["<|eot_id|>", "</s>"])
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)


class KVCacheRef(BaseModel):
    """Descriptor for KV cache block references in memory or shared storage."""
    block_ids: List[int] = Field(default_factory=list)
    layer_stride: int = Field(default=0)
    num_layers: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TokenDelta(BaseModel):
    """Incremental batch of recently generated tokens."""
    request_id: str
    sequence_id: int
    token_id: int
    token_str: str
    timestamp_ns: int = Field(default_factory=time.time_ns)


class InferenceSession(BaseModel):
    """
    Operational snapshot of an in-flight LLM inference request.
    Conforms to SPEC-002 Message Type 0x02 (REQUEST_STATE).
    """
    request_id: str = Field(default_factory=lambda: f"req-{uuid.uuid4()}")
    client_connection_id: str = Field(default="")
    model: str = Field(default="meta-llama/Llama-3-8b-instruct")
    prompt: str = Field(default="", description="Original user prompt text")
    prompt_tokens: List[int] = Field(default_factory=list, description="Token IDs of original prompt")
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    generated_tokens: List[int] = Field(default_factory=list, description="Token IDs generated so far")
    generated_text: List[str] = Field(default_factory=list, description="Decoded token strings")
    last_flushed_sequence_id: int = Field(default=-1, description="Last sequence ID flushed to client")
    total_tokens_generated: int = Field(default=0)
    is_streaming: bool = Field(default=True)
    stream_cutoff_timestamp_ns: int = Field(default=0)
    kv_cache: Optional[KVCacheRef] = Field(default=None)


# Alias for RequestSession conforming to SPEC-003
RequestSession = InferenceSession


class TokenChunk(BaseModel):
    """Parsed Server-Sent Event (SSE) token chunk."""
    request_id: str
    sequence_id: int
    token: str
    token_id: int = Field(default=0)
    is_final: bool = Field(default=False)
    raw_bytes: bytes = Field(default=b"")


# ---------------------------------------------------------------------------
# SGM-P2P Handover Control Payloads
# ---------------------------------------------------------------------------

class HandoverInit(BaseModel):
    """Initiates P2P handover session (Message Type 0x01)."""
    protocol_version: str = Field(default="1.0")
    source_node_id: str
    target_node_id: str
    preemption_deadline_utc: str = Field(default="")
    active_request_count: int = Field(default=0)
    timestamp_ns: int = Field(default_factory=time.time_ns)


class HandoverAck(BaseModel):
    """Standby confirmation of readiness (Message Type 0x06)."""
    source_node_id: str
    accepted_request_ids: List[str] = Field(default_factory=list)
    rejected_request_ids: List[str] = Field(default_factory=list)
    standby_ready_timestamp_ns: int = Field(default_factory=time.time_ns)
    resume_port: int = Field(default=8002)


class HandoverNack(BaseModel):
    """Standby rejection or error notification (Message Type 0x07)."""
    source_node_id: str
    error_code: int = Field(default=1)
    error_message: str = Field(default="Unknown error")
    timestamp_ns: int = Field(default_factory=time.time_ns)


class PreemptionAlertPayload(BaseModel):
    """REST payload sent from Node Daemon to Ingress Proxy upon preemption."""
    node_id: str
    provider: str
    deadline_seconds: float
    cutoff_sequence_id: int = Field(default=-1)
    active_request_ids: List[str] = Field(default_factory=list)
    timestamp_ns: int = Field(default_factory=time.time_ns)


class HandoverReadyPayload(BaseModel):
    """REST payload sent from Standby Node Daemon to Ingress Proxy upon readiness."""
    request_id: str
    starting_sequence_id: int
    resume_host: str = Field(default="127.0.0.1")
    resume_port: int = Field(default=8002)
    timestamp_ns: int = Field(default_factory=time.time_ns)
