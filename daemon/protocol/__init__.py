"""
SGM-P2P Wire Protocol framing and packet management.
"""

from daemon.protocol.framing import (
    MAGIC_BYTES,
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    pack_frame,
    unpack_frame_header,
    unpack_frame,
    read_frame_async,
    write_frame_async,
)

__all__ = [
    "MAGIC_BYTES",
    "HEADER_SIZE",
    "MAX_PAYLOAD_SIZE",
    "pack_frame",
    "unpack_frame_header",
    "unpack_frame",
    "read_frame_async",
    "write_frame_async",
]
