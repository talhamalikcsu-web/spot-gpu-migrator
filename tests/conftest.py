"""
Pytest configuration and shared fixtures for SGM test suite.
"""

import asyncio
import os
import sys
import pytest

# Ensure root repository directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from daemon.core.node_daemon import NodeDaemon
from proxy.ingress import IngressProxy
from simulator.mock_engine import MockLLMEngine

GROUND_TRUTH_WORDS = [
    "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " and",
    " runs", " across", " the", " wide", " open", " fields", " towards", " the", " green", " mountains."
]
GROUND_TRUTH_TEXT = "".join(GROUND_TRUTH_WORDS)


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
    await asyncio.sleep(0.05)

