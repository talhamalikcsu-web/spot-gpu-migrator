"""
Chaos injection utilities for simulating network faults, bit-flips, and hypervisor deadlines.
"""

from __future__ import annotations

import random
from typing import Optional


def corrupt_frame_bytes(frame: bytes, corruption_rate: float = 0.1) -> bytes:
    """
    Randomly corrupts bits in the payload of an SGM-P2P frame to test CRC32 error detection.
    """
    if len(frame) <= 16 or random.random() > corruption_rate:
        return frame

    # Mutate a byte in the payload (after 16-byte header)
    idx = random.randint(16, len(frame) - 1)
    corrupted = bytearray(frame)
    corrupted[idx] ^= 0xFF
    return bytes(corrupted)


class ChaosCondition:
    """Configurable chaos injection flags for network & node simulation."""

    def __init__(
        self,
        fail_p2p_connection: bool = False,
        payload_corruption_rate: float = 0.0,
        network_delay_ms: float = 0.0,
    ) -> None:
        self.fail_p2p_connection = fail_p2p_connection
        self.payload_corruption_rate = payload_corruption_rate
        self.network_delay_ms = network_delay_ms
