"""
SGM-P2P Wire Protocol Framing (v1.0).

Implements the binary frame layout specified in SPEC-002:
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Magic Bytes ('S', 'G', 'M', '1')             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version (1)  | Msg Type (1B) |          Flags (2B)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload Length (4B, uint32)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     CRC32 Checksum (4B, uint32)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload Data (N Bytes) ...                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

from __future__ import annotations

import asyncio
import logging
import struct
import zlib
from typing import Tuple, Union

from daemon.models import CRC32VerificationError, FrameFlags, MessageType

logger = logging.getLogger("sgm.protocol.framing")

MAGIC_BYTES: bytes = b"SGM1"
PROTOCOL_VERSION: int = 1
HEADER_FORMAT: str = "!4sBBHII"
HEADER_SIZE: int = 16
MAX_PAYLOAD_SIZE: int = 16 * 1024 * 1024  # 16 MB limit per spec


def pack_frame(
    msg_type: Union[int, MessageType],
    payload: bytes,
    flags: int = FrameFlags.NONE,
    version: int = PROTOCOL_VERSION,
    compress: bool = False,
) -> bytes:
    """
    Encodes a message into an SGM-P2P binary frame with 16-byte fixed header and CRC32 checksum.

    Args:
        msg_type: Message type ID (e.g. HANDOVER_INIT=0x01, REQUEST_STATE=0x02).
        payload: Uncompressed or compressed raw payload bytes.
        flags: 16-bit bitfield of flags.
        version: Protocol version byte (default 1).
        compress: If True, compresses the payload with zlib and sets IS_COMPRESSED flag.

    Returns:
        Complete serialized binary frame including header and payload.

    Raises:
        ValueError: If payload length exceeds 16MB or invalid arguments provided.
    """
    msg_type_val = int(msg_type)
    final_payload = payload
    active_flags = flags

    if compress and len(payload) > 64:
        compressed = zlib.compress(payload)
        if len(compressed) < len(payload):
            final_payload = compressed
            active_flags |= FrameFlags.IS_COMPRESSED

    payload_len = len(final_payload)
    if payload_len > MAX_PAYLOAD_SIZE:
        raise ValueError(f"Payload size {payload_len} exceeds maximum allowed size of {MAX_PAYLOAD_SIZE} bytes")

    # IEEE 802.3 CRC32 of payload data
    checksum = zlib.crc32(final_payload) & 0xFFFFFFFF

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_BYTES,
        version,
        msg_type_val,
        active_flags,
        payload_len,
        checksum,
    )
    return header + final_payload


def unpack_frame_header(header_bytes: bytes) -> Tuple[int, int, int, int, int]:
    """
    Parses a 16-byte binary frame header.

    Args:
        header_bytes: Exactly 16 bytes.

    Returns:
        Tuple of (version, msg_type, flags, payload_len, checksum).

    Raises:
        ValueError: If header size is incorrect or magic bytes do not match.
    """
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Expected {HEADER_SIZE} bytes for header, got {len(header_bytes)}")

    magic, version, msg_type, flags, payload_len, checksum = struct.unpack(
        HEADER_FORMAT, header_bytes
    )

    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC_BYTES!r}, got {magic!r}")

    if payload_len > MAX_PAYLOAD_SIZE:
        raise ValueError(f"Declared payload length {payload_len} exceeds limit of {MAX_PAYLOAD_SIZE} bytes")

    return version, msg_type, flags, payload_len, checksum


def unpack_frame(raw_bytes: bytes) -> Tuple[int, int, bytes]:
    """
    Validates CRC32 and magic bytes, unpacking a full frame into (msg_type, flags, payload).
    Decompresses payload automatically if IS_COMPRESSED flag is present.

    Args:
        raw_bytes: Binary buffer containing at least 16 bytes.

    Returns:
        Tuple of (msg_type, flags, payload_bytes).

    Raises:
        ValueError: If frame is truncated or header invalid.
        CRC32VerificationError: If payload CRC32 fails verification.
    """
    if len(raw_bytes) < HEADER_SIZE:
        raise ValueError(f"Frame buffer too short ({len(raw_bytes)} bytes; minimum {HEADER_SIZE})")

    version, msg_type, flags, payload_len, expected_crc = unpack_frame_header(raw_bytes[:HEADER_SIZE])

    total_len = HEADER_SIZE + payload_len
    if len(raw_bytes) < total_len:
        raise ValueError(f"Truncated frame: expected {total_len} bytes, got {len(raw_bytes)}")

    payload = raw_bytes[HEADER_SIZE:total_len]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise CRC32VerificationError(
            f"CRC32 mismatch on message type 0x{msg_type:02x}: "
            f"expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )

    if flags & FrameFlags.IS_COMPRESSED:
        payload = zlib.decompress(payload)

    return msg_type, flags, payload


async def read_frame_async(reader: asyncio.StreamReader) -> Tuple[int, int, bytes]:
    """
    Asynchronously reads an entire frame from an asyncio.StreamReader.

    Args:
        reader: Async stream reader connected to peer.

    Returns:
        Tuple of (msg_type, flags, payload).

    Raises:
        asyncio.IncompleteReadError: If socket closes prematurely.
        CRC32VerificationError: If payload checksum fails.
        ValueError: If header is malformed.
    """
    header_bytes = await reader.readexactly(HEADER_SIZE)
    version, msg_type, flags, payload_len, expected_crc = unpack_frame_header(header_bytes)

    if payload_len > 0:
        payload = await reader.readexactly(payload_len)
    else:
        payload = b""

    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise CRC32VerificationError(
            f"CRC32 mismatch on message type 0x{msg_type:02x}: "
            f"expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )

    if flags & FrameFlags.IS_COMPRESSED:
        payload = zlib.decompress(payload)

    return msg_type, flags, payload


async def write_frame_async(
    writer: asyncio.StreamWriter,
    msg_type: Union[int, MessageType],
    payload: bytes,
    flags: int = FrameFlags.NONE,
    compress: bool = False,
) -> None:
    """
    Asynchronously encodes and writes a frame to an asyncio.StreamWriter, followed by drain().

    Args:
        writer: Async stream writer connected to peer.
        msg_type: Message type ID.
        payload: Payload bytes.
        flags: Flags bitfield.
        compress: Whether to compress payload.
    """
    frame = pack_frame(msg_type, payload, flags=flags, compress=compress)
    writer.write(frame)
    await writer.drain()
