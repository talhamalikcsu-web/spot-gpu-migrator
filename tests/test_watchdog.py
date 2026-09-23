"""
Unit & Integration tests for Preemption Watchdogs.
Conforms to SPEC-003 Suite 1: TC-01, TC-02, TC-03.
"""

import asyncio
import pytest

from daemon.models import PreemptionEvent
from daemon.watchdog.aws import AWSPreemptionWatchdog
from daemon.watchdog.gcp import GCPPreemptionWatchdog
from daemon.watchdog.runpod import RunPodPreemptionWatchdog
from simulator.cloud_metadata import CloudMetadataServer


@pytest.fixture
async def simulator_server():
    """Starts local mock cloud metadata server on port 18001 for testing."""
    sim = CloudMetadataServer(host="127.0.0.1", port=18001)
    await sim.start()
    yield sim
    await sim.stop()


@pytest.mark.asyncio
async def test_tc01_aws_imdsv2_detection(simulator_server):
    """TC-01: AWS IMDSv2 Polling & Token Management (<300ms detection)."""
    base_url = "http://127.0.0.1:18001"
    watchdog = AWSPreemptionWatchdog(base_url=base_url, poll_interval_ms=50, timeout_ms=50)

    event_received = asyncio.Event()
    captured_event = []

    async def on_preempt(event: PreemptionEvent):
        captured_event.append(event)
        event_received.set()

    watchdog.register_callback(on_preempt)
    await watchdog.start()

    try:
        # 1. Healthy state check
        healthy_check = await watchdog.check_once()
        assert healthy_check is None

        # 2. Inject preemption notice
        simulator_server.trigger_preemption(provider="aws", notice_seconds=120)

        # 3. Verify callback fires in <= 300ms
        await asyncio.wait_for(event_received.wait(), timeout=0.300)
        assert len(captured_event) == 1
        ev = captured_event[0]
        assert ev.provider == "aws"
        assert ev.action == "terminate"
        assert ev.deadline_seconds <= 120.0
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_tc02_gcp_metadata_detection(simulator_server):
    """TC-02: GCP Compute Engine Metadata Verification (<300ms detection)."""
    base_url = "http://127.0.0.1:18001"
    watchdog = GCPPreemptionWatchdog(base_url=base_url, poll_interval_ms=50, timeout_ms=50)

    event_received = asyncio.Event()
    captured_event = []

    async def on_preempt(event: PreemptionEvent):
        captured_event.append(event)
        event_received.set()

    watchdog.register_callback(on_preempt)
    await watchdog.start()

    try:
        # Healthy check
        assert (await watchdog.check_once()) is None

        # Inject preemption
        simulator_server.trigger_preemption(provider="gcp", notice_seconds=30)

        await asyncio.wait_for(event_received.wait(), timeout=0.300)
        assert len(captured_event) == 1
        ev = captured_event[0]
        assert ev.provider == "gcp"
        assert ev.deadline_seconds == 30.0
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_tc03_runpod_webhook_notice():
    """TC-03: RunPod Webhook Notification (<100ms execution)."""
    watchdog = RunPodPreemptionWatchdog(status_url="")

    captured_event = []
    async def on_preempt(event: PreemptionEvent):
        captured_event.append(event)

    watchdog.register_callback(on_preempt)

    payload = {
        "event": "POD_TERMINATION_NOTICE",
        "pod_id": "pod-gpu-test-123",
        "grace_period_seconds": 30,
        "timestamp": 1790184425,
    }

    event = await watchdog.handle_webhook_payload(payload)
    assert event.provider == "runpod"
    assert event.deadline_seconds == 30.0
    assert len(captured_event) == 1
