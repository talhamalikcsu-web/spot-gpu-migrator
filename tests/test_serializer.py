"""
Unit tests for SGM-P2P frame serialization and CRC32 verification.
Conforms to SPEC-003 Suite: test_serializer.py.
"""

import asyncio
import pytest

from daemon.models import (
    CRC32VerificationError,
    FrameFlags,
    InferenceSession,
    MessageType,
    SamplingParams,
)
from daemon.protocol.framing import (
    MAGIC_BYTES,
    HEADER_SIZE,
    pack_frame,
    unpack_frame_header,
    unpack_frame,
)
from daemon.memory.session import StateSerializer


def test_pack_and_unpack_frame():
    """Verifies binary framing, 16-byte fixed header, and payload round-trip."""
    payload = b'{"request_id":"req-123","prompt_tokens":[1,2,3]}'
    frame = pack_frame(MessageType.REQUEST_STATE, payload, flags=FrameFlags.NONE)

    assert len(frame) == HEADER_SIZE + len(payload)
    assert frame[:4] == MAGIC_BYTES

    version, msg_type, flags, payload_len, checksum = unpack_frame_header(frame[:HEADER_SIZE])
    assert version == 1
    assert msg_type == MessageType.REQUEST_STATE
    assert flags == 0
    assert payload_len == len(payload)

    unpacked_type, unpacked_flags, unpacked_payload = unpack_frame(frame)
    assert unpacked_type == MessageType.REQUEST_STATE
    assert unpacked_flags == 0
    assert unpacked_payload == payload


def test_crc32_checksum_mismatch():
    """Verifies that bit-level payload corruption triggers CRC32VerificationError."""
    payload = b"Hello Spot GPU Migrator"
    frame = bytearray(pack_frame(MessageType.TOKEN_DELTA, payload))

    # Corrupt a byte in payload
    frame[-1] ^= 0xFF

    with pytest.raises(CRC32VerificationError):
        unpack_frame(bytes(frame))


def test_invalid_magic_bytes():
    """Verifies rejection of frames lacking SGM1 magic header."""
    payload = b"test payload"
    frame = bytearray(pack_frame(MessageType.HEARTBEAT, payload))
    frame[0] = ord(b"X")

    with pytest.raises(ValueError, match="Invalid magic bytes"):
        unpack_frame(bytes(frame))


@pytest.mark.asyncio
async def test_state_serializer_p2p_transfer():
    """Tests P2P state streaming between StateSerializer instances over loopback TCP."""
    serializer = StateSerializer(node_id="test-node")

    sessions = [
        InferenceSession(
            request_id=f"req-{i}",
            model="meta-llama/Llama-3-8b-instruct",
            prompt="Hello world",
            prompt_tokens=[10, 20, 30],
            sampling_params=SamplingParams(temperature=0.7),
            generated_tokens=[100, 200],
            last_flushed_sequence_id=1,
            total_tokens_generated=2,
        )
        for i in range(3)
    ]

    listen_port = 19123
    listen_task = asyncio.create_task(
        serializer.receive_handover("127.0.0.1", listen_port, timeout=5.0)
    )

    # Allow server to bind
    await asyncio.sleep(0.05)

    ack = await serializer.send_handover("127.0.0.1", listen_port, sessions, timeout=5.0)
    assert len(ack.accepted_request_ids) == 3
    assert "req-0" in ack.accepted_request_ids

    received = await listen_task
    assert len(received) == 3
    assert received[0].request_id == "req-0"
    assert received[0].generated_tokens == [100, 200]
