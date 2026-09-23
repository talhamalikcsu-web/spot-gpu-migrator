"""
Milestone 3 Verification: Sliding Window Stop Sequence Filtering.

Conforms to SPEC-005:
- AC-M3-04: Multi-token stop sequences (e.g. <|eot_id|>) must never leak partial tokens (e.g. <|)
- FAIL-04: Partial Stop Token Leakage mitigation
- TEST-M3-04
"""

from __future__ import annotations

import pytest

from daemon.integrations.tgi import SlidingWindowStopFilter


def test_sliding_window_suppresses_partial_stop_sequence():
    """
    TEST-M3-04: Verifies that partial slices of stop sequences (e.g. '<|') are buffered
    and never emitted to downstream clients when generation stops at <|eot_id|>.
    """
    filter_ = SlidingWindowStopFilter(stop_sequences=["<|eot_id|>", "</s>", "ClientExit"])

    # Normal tokens
    out1 = filter_.feed("Hello")
    assert out1 == ["Hello"]

    out2 = filter_.feed(" world")
    assert out2 == [" world"]

    # Partial stop word fragments arriving piece-by-piece
    out3 = filter_.feed("<|")
    assert out3 == [], "Partial stop prefix '<|' must be buffered, not emitted!"

    out4 = filter_.feed("eot")
    assert out4 == [], "Partial stop prefix '<|eot' must be buffered, not emitted!"

    out5 = filter_.feed("_id")
    assert out5 == [], "Partial stop prefix '<|eot_id' must be buffered, not emitted!"

    # Completes the stop word
    out6 = filter_.feed("|>")
    assert out6 == [], "Complete stop word '<|eot_id|>' must be discarded, not emitted!"
    assert filter_.matched_stop is True

    # Generation conclusion flush
    final = filter_.flush_final()
    assert final == [], "Flushing after matched stop must yield empty list"


def test_sliding_window_flushes_false_match():
    """
    Verifies that if a buffered prefix turns out to be a false match (e.g. '<|hello'),
    the buffered text is safely flushed downstream.
    """
    filter_ = SlidingWindowStopFilter(stop_sequences=["<|eot_id|>"])

    out1 = filter_.feed("Tags: ")
    assert out1 == ["Tags: "]

    out2 = filter_.feed("<|")
    assert out2 == [], "Should buffer candidate '<|'"

    # False match: next token is 'custom_tag'
    out3 = filter_.feed("custom_tag")
    assert out3 == ["<|custom_tag"], "False match must flush the held '<|' with subsequent text"
    assert filter_.matched_stop is False


def test_sliding_window_flush_on_stream_eof():
    """
    Verifies flush_final behavior on abrupt preemption or stream end:
    - If buffer contains exact stop word: discards it.
    - If buffer contains partial stop prefix: discards it (does not leak to client).
    - If buffer contains non-stop text: flushes it.
    """
    # Case 1: Stream cuts off while holding partial stop word '<|'
    filter_cut = SlidingWindowStopFilter(stop_sequences=["<|eot_id|>"])
    filter_cut.feed("Text before cutoff ")
    filter_cut.feed("<|")
    flushed = filter_cut.flush_final()
    assert flushed == [], "Cutoff midway through stop sequence must not leak partial fragment '<|'"

    # Case 2: Stream concludes with complete stop word in text
    filter_complete = SlidingWindowStopFilter(stop_sequences=["</s>"])
    filter_complete.feed("Sentence ending</s>")
    assert filter_complete.matched_stop is True
    assert filter_complete.flush_final() == []
