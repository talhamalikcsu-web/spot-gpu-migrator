"""
End-to-End Live Integration Tests for Zero-Downtime Migration.
Conforms to SPEC-003 Suite 3: TC-06, TC-07, TC-08.
"""

import asyncio
import json
import logging
import pytest
import aiohttp

from daemon.core.node_daemon import NodeDaemon
from daemon.models import PreemptionEvent
from proxy.ingress import IngressProxy
from simulator.mock_engine import MockLLMEngine, DEFAULT_GENERATION_CORPUS

logger = logging.getLogger("sgm.test.integration")

GROUND_TRUTH_WORDS = [
    "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " and",
    " runs", " across", " the", " wide", " open", " fields", " towards", " the", " green", " mountains."
]
GROUND_TRUTH_TEXT = "".join(GROUND_TRUTH_WORDS)


@pytest.fixture
async def full_cluster():
    """
    Spins up:
    - Active Mock Engine (port 18101)
    - Standby Mock Engine (port 18102)
    - Standby Node Daemon (port 19102, P2P port 19202)
    - Active Node Daemon (port 19101, P2P client targeting 19202)
    - SGM Ingress Proxy (port 18000, routing to 18101 and 18102)
    """
    # 1. Mock Engines
    active_engine = MockLLMEngine(
        port=18101,
        inter_token_delay_ms=30.0,
        corpus=GROUND_TRUTH_WORDS,
        node_role="active",
    )
    standby_engine = MockLLMEngine(
        port=18102,
        inter_token_delay_ms=30.0,
        corpus=GROUND_TRUTH_WORDS,
        node_role="standby",
    )
    await active_engine.start()
    await standby_engine.start()

    # 2. Ingress Proxy
    proxy = IngressProxy(
        port=18000,
        active_upstream_url="http://127.0.0.1:18101",
        standby_upstream_url="http://127.0.0.1:18102",
    )
    await proxy.start()

    # 3. Standby Node Daemon
    standby_daemon = NodeDaemon(
        node_id="standby-01",
        role="standby",
        control_port=19102,
        p2p_port=19202,
        proxy_url="http://127.0.0.1:18000",
        engine_url="http://127.0.0.1:18102",
    )
    await standby_daemon.start()

    # 4. Active Node Daemon
    active_daemon = NodeDaemon(
        node_id="active-01",
        role="active",
        control_port=19101,
        standby_host="127.0.0.1",
        standby_p2p_port=19202,
        proxy_url="http://127.0.0.1:18000",
        engine_url="http://127.0.0.1:18101",
    )
    await active_daemon.start()

    yield {
        "active_engine": active_engine,
        "standby_engine": standby_engine,
        "proxy": proxy,
        "active_daemon": active_daemon,
        "standby_daemon": standby_daemon,
    }

    # Teardown
    await active_daemon.stop()
    await standby_daemon.stop()
    await proxy.stop()
    await active_engine.stop()
    await standby_engine.stop()


@pytest.mark.asyncio
async def test_tc06_and_tc07_exactly_once_migration(full_cluster):
    """
    TC-06 & TC-07: Downstream Socket Continuity and Exactly-Once Token Delivery.
    Injects preemption when token 5 (' jumps') has been emitted.
    Verifies that client receives 100% of ground truth text with 0 duplicate and 0 omitted tokens.
    """
    active_daemon: NodeDaemon = full_cluster["active_daemon"]
    proxy: IngressProxy = full_cluster["proxy"]

    received_tokens = []
    received_raw_lines = []
    received_sequences = []

    req_payload = {
        "prompt": "Test prompt",
        "stream": True,
        "max_tokens": len(GROUND_TRUTH_WORDS),
    }

    async with aiohttp.ClientSession() as client:
        async with client.post("http://127.0.0.1:18000/v1/chat/completions", json=req_payload) as resp:
            assert resp.status == 200

            preemption_injected = False

            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
                received_raw_lines.append(line)

                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    payload = json.loads(line[5:].strip())
                    delta = payload["choices"][0]["delta"]["content"]
                    seq = payload.get("seq")
                    received_tokens.append(delta)
                    received_sequences.append(seq)

                    # When token 5 is emitted, trigger preemption!
                    if seq == 5 and not preemption_injected:
                        preemption_injected = True
                        logger.info("Triggering preemption at sequence 5!")
                        event = PreemptionEvent(
                            provider="aws",
                            action="terminate",
                            deadline_seconds=30.0,
                        )
                        # Fire preemption on active daemon
                        asyncio.create_task(active_daemon.on_preemption(event))

    client_received_text = "".join(received_tokens)

    logger.info("Ground Truth Text: %r", GROUND_TRUTH_TEXT)
    logger.info("Client Output Text: %r", client_received_text)
    logger.info("Received Sequence IDs: %s", received_sequences)

    # 1. Zero Duplicate Tokens & Monotonic Continuity
    for i in range(len(received_sequences) - 1):
        assert received_sequences[i+1] == received_sequences[i] + 1, (
            f"Sequence discontinuity: {received_sequences[i]} -> {received_sequences[i+1]}"
        )

    # 2. Strict Exactly-Once Token Delivery (100% String Identity)
    assert client_received_text == GROUND_TRUTH_TEXT, (
        f"Mismatch! Expected {GROUND_TRUTH_TEXT!r}, got {client_received_text!r}"
    )

    # 3. Verify exactly matching token count
    assert len(received_tokens) == len(GROUND_TRUTH_WORDS)
