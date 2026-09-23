"""
Unit tests for RingBuffer and TokenDeduplicator.
Conforms to SPEC-003 Suite: test_proxy_buffer.py.
"""

import pytest

from daemon.models import TokenChunk, TokenSequenceGapError
from proxy.buffer import RingBuffer, TokenDeduplicator


def test_ring_buffer_put_get():
    """Verifies basic circular buffer insert, retrieval, and slice queries."""
    rb = RingBuffer[str](capacity=4)
    rb.put(0, "token_0")
    rb.put(1, "token_1")
    rb.put(2, "token_2")

    assert rb.get(0) == "token_0"
    assert rb.get(1) == "token_1"
    assert rb.get(2) == "token_2"
    assert rb.get(3) is None

    # Slice
    sliced = rb.get_slice(0, 2)
    assert sliced == ["token_0", "token_1", "token_2"]


def test_ring_buffer_overwrite():
    """Verifies that circular overwrites invalidate old sequence IDs."""
    rb = RingBuffer[str](capacity=4)
    for i in range(6):
        rb.put(i, f"token_{i}")

    # Seq 0 and 1 have been overwritten (indices 0 and 1 now hold 4 and 5)
    assert rb.get(0) is None
    assert rb.get(1) is None
    assert rb.get(4) == "token_4"
    assert rb.get(5) == "token_5"


def test_deduplicator_monotonic_filtering():
    """Verifies sequential token filtering and state progression."""
    dedup = TokenDeduplicator()
    req_id = "req-test-1"

    chunk0 = TokenChunk(request_id=req_id, sequence_id=0, token="Hello")
    chunk1 = TokenChunk(request_id=req_id, sequence_id=1, token=" world")

    res0 = dedup.filter_chunk(req_id, chunk0)
    assert res0 is not None
    assert res0.sequence_id == 0
    assert dedup.get_last_flushed(req_id) == 0

    res1 = dedup.filter_chunk(req_id, chunk1)
    assert res1 is not None
    assert res1.sequence_id == 1
    assert dedup.get_last_flushed(req_id) == 1


def test_deduplicator_discards_duplicates():
    """Verifies that standby node replayed tokens (seq <= last_flushed) are discarded."""
    dedup = TokenDeduplicator()
    req_id = "req-test-2"

    for seq in range(3):
        chunk = TokenChunk(request_id=req_id, sequence_id=seq, token=f"tok_{seq}")
        assert dedup.filter_chunk(req_id, chunk) is not None

    assert dedup.get_last_flushed(req_id) == 2

    # Standby re-emits sequence 1 and 2
    dup1 = TokenChunk(request_id=req_id, sequence_id=1, token="tok_1")
    dup2 = TokenChunk(request_id=req_id, sequence_id=2, token="tok_2")

    assert dedup.filter_chunk(req_id, dup1) is None
    assert dedup.filter_chunk(req_id, dup2) is None
    assert dedup.get_last_flushed(req_id) == 2

    # Next valid token (seq 3) is accepted
    next_chunk = TokenChunk(request_id=req_id, sequence_id=3, token="tok_3")
    res = dedup.filter_chunk(req_id, next_chunk)
    assert res is not None
    assert res.sequence_id == 3
    assert dedup.get_last_flushed(req_id) == 3


def test_deduplicator_gap_detection():
    """Verifies that missing token sequence raises TokenSequenceGapError."""
    dedup = TokenDeduplicator()
    req_id = "req-test-3"

    chunk0 = TokenChunk(request_id=req_id, sequence_id=0, token="First")
    dedup.filter_chunk(req_id, chunk0)

    # Missing seq 1, jumps to seq 2
    chunk2 = TokenChunk(request_id=req_id, sequence_id=2, token="Third")
    with pytest.raises(TokenSequenceGapError, match="Stream gap detected"):
        dedup.filter_chunk(req_id, chunk2)
