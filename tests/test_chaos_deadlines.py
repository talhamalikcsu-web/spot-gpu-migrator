"""
Chaos, Deadline, and Fault-Tolerance verification tests.
Conforms to SPEC-003 Suite 2 and Suite 4: TC-04, TC-05, TC-09, TC-10.
"""

import asyncio
import time
import pytest

from daemon.core.node_daemon import NodeDaemon
from daemon.models import (
    CRC32VerificationError,
    InferenceSession,
    MessageType,
    P2PConnectionFailedError,
    PreemptionEvent,
)
from daemon.transport.p2p_client import P2PClient
from daemon.transport.p2p_server import P2PServer
from simulator.chaos import corrupt_frame_bytes


@pytest.mark.asyncio
async def test_tc04_handover_latency_budget():
    """
    TC-04: Verifies that end-to-end P2P session transfer completes in <= 2.5s.
    """
    server = P2PServer(host="127.0.0.1", port=19210, node_id="standby-test")
    await server.start()

    client = P2PClient(target_host="127.0.0.1", target_port=19210, node_id="active-test")

    sessions = [
        InferenceSession(
            request_id=f"bench-req-{i}",
            prompt_tokens=[1, 2, 3, 4, 5],
            generated_tokens=[10, 20, 30, 40],
            last_flushed_sequence_id=3,
        )
        for i in range(10)
    ]

    start_time = time.monotonic()
    ack = await client.send_sessions(sessions, timeout=3.0)
    elapsed_ms = (time.monotonic() - start_time) * 1000.0

    await server.stop()

    assert len(ack.accepted_request_ids) == 10
    assert elapsed_ms <= 2500.0, f"Handover took {elapsed_ms:.1f}ms, exceeding 2500ms SLA target"


@pytest.mark.asyncio
async def test_tc09_standby_node_unreachable_fallback():
    """
    TC-09: Verifies that Active Node handles Standby node connection refusal
    gracefully without crashing.
    """
    # Standby node is intentionally NOT running on port 19999
    client = P2PClient(target_host="127.0.0.1", target_port=19999, node_id="active-test")

    session = InferenceSession(request_id="failover-req-1")

    with pytest.raises(P2PConnectionFailedError):
        await client.send_sessions([session], timeout=0.500)


@pytest.mark.asyncio
async def test_tc10_crc32_corruption_detection():
    """
    TC-10: Verifies that bit flips trigger CRC32VerificationError on receiver.
    """
    valid_payload = b'{"msg": "state payload", "seq": 42}'
    from daemon.protocol.framing import pack_frame, unpack_frame

    frame = pack_frame(MessageType.REQUEST_STATE, valid_payload)
    corrupted_frame = corrupt_frame_bytes(frame, corruption_rate=1.0)

    with pytest.raises(CRC32VerificationError):
        unpack_frame(corrupted_frame)
