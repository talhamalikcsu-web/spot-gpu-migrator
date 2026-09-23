"""
Stream Multiplexer and Upstream Connection Manager.

Multiplexes downstream client HTTP/SSE connections across Active and Standby upstreams,
handles keep-alive heartbeats during handover buffering, and performs seamless mid-stream
socket swapping with deduplication.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable, Dict, Optional

from daemon.models import ClientDisconnectedError, TokenChunk, TokenSequenceGapError
from proxy.buffer import TokenDeduplicator

logger = logging.getLogger("sgm.proxy.upstream")


class AbstractStreamMultiplexer(ABC):
    """Multiplexes downstream client connection across Active and Standby upstreams."""

    @abstractmethod
    async def stream_with_handover(
        self,
        request_id: str,
        active_upstream: AsyncIterator[TokenChunk],
        standby_factory: Callable[[], Awaitable[AsyncIterator[TokenChunk]]],
    ) -> AsyncIterator[bytes]:
        """
        Yields raw SSE chunks to downstream client.
        Seamlessly transitions to standby_factory generator when preemption occurs.
        """
        pass


class StreamSessionState:
    """Internal state tracking for an active streaming request inside the proxy."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.preemption_signaled = asyncio.Event()
        self.handover_ready = asyncio.Event()
        self.cutoff_sequence_id: int = -1
        self.is_completed: bool = False
        self.error: Optional[Exception] = None


class StreamMultiplexer(AbstractStreamMultiplexer):
    """
    Concrete Stream Multiplexer implementing zero-downtime upstream failover.
    """

    def __init__(
        self,
        deduplicator: Optional[TokenDeduplicator] = None,
        keepalive_interval_s: float = 0.8,
    ) -> None:
        self.deduplicator = deduplicator or TokenDeduplicator()
        self.keepalive_interval_s = keepalive_interval_s
        self.active_sessions: Dict[str, StreamSessionState] = {}

    def get_or_create_session(self, request_id: str) -> StreamSessionState:
        if request_id not in self.active_sessions:
            self.active_sessions[request_id] = StreamSessionState(request_id)
        return self.active_sessions[request_id]

    def trigger_preemption(self, request_id: str, cutoff_seq: int) -> None:
        """Notifies the session that preemption has started on the active node."""
        session = self.active_sessions.get(request_id)
        if session:
            session.cutoff_sequence_id = cutoff_seq
            session.preemption_signaled.set()
            logger.info("Multiplexer [%s]: Preemption signaled at cutoff seq %d", request_id, cutoff_seq)

    def trigger_handover_ready(self, request_id: str) -> None:
        """Notifies the session that the standby node is ready to stream."""
        session = self.active_sessions.get(request_id)
        if session:
            session.handover_ready.set()
            logger.info("Multiplexer [%s]: Standby handover marked READY", request_id)

    def _format_sse_chunk(self, chunk: TokenChunk) -> bytes:
        """Formats a TokenChunk as an OpenAI-compatible Server-Sent Event (SSE)."""
        payload = {
            "id": f"chatcmpl-{chunk.request_id}",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.token},
                    "finish_reason": "stop" if chunk.is_final else None,
                }
            ],
            "seq": chunk.sequence_id,
        }
        return f"id: {chunk.sequence_id}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

    async def stream_with_handover(
        self,
        request_id: str,
        active_upstream: AsyncIterator[TokenChunk],
        standby_factory: Callable[[], Awaitable[AsyncIterator[TokenChunk]]],
    ) -> AsyncIterator[bytes]:
        """
        Streams SSE tokens to client. If preemption occurs:
        1. Consumes active stream up to cutoff or EOF.
        2. Enters buffering mode emitting ': keep-alive\\n\\n' heartbeats.
        3. Swaps upstream to Standby, deduplicates replayed tokens, and finishes stream.
        """
        session = self.get_or_create_session(request_id)
        switched_to_standby = False
        last_activity_time = time.monotonic()

        try:
            # ---------------------------------------------------------------
            # Phase 1: Stream from Active Node
            # ---------------------------------------------------------------
            async for chunk in active_upstream:
                filtered = self.deduplicator.filter_chunk(request_id, chunk)
                if filtered is not None:
                    yield self._format_sse_chunk(filtered)
                    last_activity_time = time.monotonic()

                # If preemption is signaled, check if we've reached cutoff
                if session.preemption_signaled.is_set():
                    if session.cutoff_sequence_id >= 0 and chunk.sequence_id >= session.cutoff_sequence_id:
                        logger.info(
                            "Multiplexer [%s]: Reached preemption cutoff seq %d. Swapping upstream.",
                            request_id,
                            session.cutoff_sequence_id,
                        )
                        break

            # If preemption was signaled OR the active node cut off early
            if session.preemption_signaled.is_set():
                switched_to_standby = True
                logger.info("Multiplexer [%s]: Active stream ended. Entering BUFFERING mode.", request_id)

                # -----------------------------------------------------------
                # Phase 2: Buffering & Heartbeats (: keep-alive\n\n)
                # -----------------------------------------------------------
                buffering_start = time.monotonic()
                max_buffering_s = 2.5  # SLA timeout ceiling
                while not session.handover_ready.is_set():
                    now = time.monotonic()
                    if now - buffering_start >= max_buffering_s:
                        logger.warning(
                            "Multiplexer [%s]: Handover readiness wait exceeded %.1fs, forcing standby connection",
                            request_id,
                            max_buffering_s,
                        )
                        session.handover_ready.set()
                        break

                    # Send keep-alive ping if idle duration exceeds threshold
                    if now - last_activity_time >= self.keepalive_interval_s:
                        logger.debug("Multiplexer [%s]: Emitting keep-alive comment ping", request_id)
                        yield b": keep-alive\n\n"
                        last_activity_time = now

                    try:
                        # Wait for standby readiness or timeout tick
                        await asyncio.wait_for(session.handover_ready.wait(), timeout=0.100)
                    except asyncio.TimeoutError:
                        pass

                # -----------------------------------------------------------
                # Phase 3: Stream from Standby Node (Warm Target)
                # -----------------------------------------------------------
                logger.info("Multiplexer [%s]: Connecting to Standby upstream...", request_id)
                standby_upstream = await standby_factory()

                async for chunk in standby_upstream:
                    filtered = self.deduplicator.filter_chunk(request_id, chunk)
                    if filtered is not None:
                        yield self._format_sse_chunk(filtered)
                        last_activity_time = time.monotonic()

            # End of stream signal
            yield b"data: [DONE]\n\n"
            session.is_completed = True

        except (ConnectionResetError, BrokenPipeError) as conn_err:
            logger.warning("Downstream client disconnected prematurely for request %s: %s", request_id, conn_err)
            raise ClientDisconnectedError(f"Client disconnected for request {request_id}") from conn_err
        except Exception as exc:
            logger.error("Error during multiplexed stream for %s: %s", request_id, exc)
            session.error = exc
            raise
        finally:
            self.active_sessions.pop(request_id, None)
            self.deduplicator.reset_request(request_id)
