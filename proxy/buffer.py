"""
Token Ring Buffer and Deduplicator for SGM Ingress Proxy.

Maintains an in-memory ring buffer per active connection and enforces strictly monotonic
token sequences, preventing duplicate or dropped tokens during node switchover.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Dict, Generic, List, Optional, TypeVar

from daemon.models import TokenChunk, TokenSequenceGapError

logger = logging.getLogger("sgm.proxy.buffer")

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-capacity circular buffer indexed by sequence ID.
    Thread/task-safe for sequential writes per stream.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._slots: List[Optional[T]] = [None] * capacity
        self._seq_ids: List[int] = [-1] * capacity

    def put(self, sequence_id: int, item: T) -> None:
        """Stores an item at the position corresponding to sequence_id."""
        index = sequence_id % self.capacity
        self._slots[index] = item
        self._seq_ids[index] = sequence_id

    def get(self, sequence_id: int) -> Optional[T]:
        """Retrieves an item by sequence_id if it has not been overwritten."""
        index = sequence_id % self.capacity
        if self._seq_ids[index] == sequence_id:
            return self._slots[index]
        return None

    def get_slice(self, start_seq: int, end_seq: int) -> List[T]:
        """
        Retrieves all available contiguous items in range [start_seq, end_seq].
        Returns only items that are currently valid in the ring buffer.
        """
        result: List[T] = []
        for seq in range(start_seq, end_seq + 1):
            item = self.get(seq)
            if item is not None:
                result.append(item)
            else:
                break
        return result

    def clear(self) -> None:
        """Clears all entries in the ring buffer."""
        self._slots = [None] * self.capacity
        self._seq_ids = [-1] * self.capacity


class AbstractTokenDeduplicator(ABC):
    """Tracks token sequence IDs and prevents duplicate delivery during switchover."""

    @abstractmethod
    def record_flushed(self, request_id: str, seq_id: int) -> None:
        """Record that token with seq_id has been successfully flushed to the downstream client."""
        pass

    @abstractmethod
    def filter_chunk(self, request_id: str, chunk: TokenChunk) -> Optional[TokenChunk]:
        """
        Evaluate incoming chunk from upstream.
        Returns chunk if seq_id == last_flushed + 1.
        Returns None if seq_id <= last_flushed (duplicate).
        Raises TokenSequenceGapError if seq_id > last_flushed + 1.
        """
        pass


class TokenDeduplicator(AbstractTokenDeduplicator):
    """
    Concrete token deduplicator implementing exactly-once token delivery semantics.
    """

    def __init__(self, buffer_capacity: int = 1024) -> None:
        self.buffer_capacity = buffer_capacity
        self._last_flushed: Dict[str, int] = {}
        self._buffers: Dict[str, RingBuffer[TokenChunk]] = {}

    def get_last_flushed(self, request_id: str) -> int:
        """Returns the monotonic sequence index of the last token emitted downstream."""
        return self._last_flushed.get(request_id, -1)

    def record_flushed(self, request_id: str, seq_id: int) -> None:
        """Records that sequence ID has been delivered to downstream client."""
        current = self._last_flushed.get(request_id, -1)
        if seq_id > current:
            self._last_flushed[request_id] = seq_id

    def filter_chunk(self, request_id: str, chunk: TokenChunk) -> Optional[TokenChunk]:
        """
        Filters incoming chunk against the stream's sequence history.
        - seq <= last_flushed: Discarded as duplicate.
        - seq == last_flushed + 1: Accepted, stored in ring buffer, flushed marker updated.
        - seq > last_flushed + 1: Omission gap detected -> raises TokenSequenceGapError.
        """
        last_flushed = self.get_last_flushed(request_id)

        # 1. Deduplication check: re-emitted tokens are dropped
        if chunk.sequence_id <= last_flushed:
            logger.debug(
                "Deduplicator [%s]: Discarding duplicate token seq %d (last flushed: %d)",
                request_id,
                chunk.sequence_id,
                last_flushed,
            )
            return None

        # 2. Gap detection: Standby skipped tokens
        expected_seq = last_flushed + 1
        if chunk.sequence_id != expected_seq:
            logger.error(
                "Deduplicator [%s]: Token gap detected! Expected seq %d, got %d",
                request_id,
                expected_seq,
                chunk.sequence_id,
            )
            raise TokenSequenceGapError(
                f"Stream gap detected for {request_id}: expected {expected_seq}, received {chunk.sequence_id}"
            )

        # 3. Valid contiguous token
        if request_id not in self._buffers:
            self._buffers[request_id] = RingBuffer[TokenChunk](capacity=self.buffer_capacity)

        self._buffers[request_id].put(chunk.sequence_id, chunk)
        self.record_flushed(request_id, chunk.sequence_id)
        return chunk

    def reset_request(self, request_id: str) -> None:
        """Cleans up internal state for a completed or aborted request."""
        self._last_flushed.pop(request_id, None)
        self._buffers.pop(request_id, None)
