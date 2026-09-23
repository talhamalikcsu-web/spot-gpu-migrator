"""
Automated Acceptance Tests for Milestone 2 Operational CLI Commands.

Validates:
1. `cli.py provision --provider aws`: Script generation, IMDSv2 enforcement, systemd config.
2. `cli.py provision --provider runpod`: vLLM prefix-caching startup, preemption webhook, daemon core.
3. `cli.py provision --stdout` and `--output` options.
4. `cli.py monitor` offline error handling & recovery loops without unhandled exceptions.
5. `cli.py monitor` live telemetry polling, in-flight stream counters, and real-time dollar savings counter.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import pytest
from aiohttp import web

from cli import (
    generate_aws_provisioning_script,
    generate_runpod_provisioning_script,
    run_provision_command,
)
from ui.dashboard import ClusterMonitor, DashboardState, SGMDashboard


class TestProvisioningCLI:
    """Test suite for cli.py provision commands."""

    def test_aws_script_generation_defaults(self):
        """Validates that AWS script includes IMDSv2, docker, nvidia-ctk, and systemd service."""
        script = generate_aws_provisioning_script(
            role="active",
            proxy_url="http://10.0.1.10:8000",
            standby_host="10.0.1.200",
            control_port=9001,
            p2p_port=9002,
            engine_url="http://127.0.0.1:8001",
        )
        assert "#!/usr/bin/env bash" in script
        assert "IMDS_TOKEN=$(curl -sS -X PUT" in script
        assert "X-aws-ec2-metadata-token-ttl-seconds: 21600" in script
        assert "nvidia-driver-550-server" in script
        assert "nvidia-ctk runtime configure" in script
        assert "sgm-node-daemon" in script
        assert "SGM_ROLE=${SGM_NODE_ROLE:-active}" in script
        assert "SGM_STANDBY_HOST=${STANDBY_PEER_IP:-10.0.1.200}" in script
        assert "SGM_CONTROL_PORT=9001" in script
        assert "SGM_P2P_PORT=9002" in script
        assert "systemctl enable --now sgm-daemon.service" in script

    def test_aws_script_generation_custom_standby(self):
        """Validates AWS script with standby role and custom ports."""
        script = generate_aws_provisioning_script(
            role="standby",
            proxy_url="http://192.168.1.10:8000",
            standby_host="192.168.1.20",
            control_port=9003,
            p2p_port=9004,
            engine_url="http://127.0.0.1:8002",
        )
        assert "SGM_ROLE=${SGM_NODE_ROLE:-standby}" in script
        assert "SGM_PROXY_URL=${PROXY_INGRESS_URL:-http://192.168.1.10:8000}" in script
        assert "SGM_STANDBY_HOST=${STANDBY_PEER_IP:-192.168.1.20}" in script
        assert "SGM_CONTROL_PORT=9003" in script
        assert "SGM_P2P_PORT=9004" in script
        assert "SGM_ENGINE_URL=http://127.0.0.1:8002" in script

    def test_runpod_script_generation(self):
        """Validates that RunPod script includes vLLM prefix caching, signal trapping, and daemon."""
        script = generate_runpod_provisioning_script(
            role="active",
            proxy_url="http://10.0.1.10:8000",
            standby_host="10.0.1.200",
            control_port=9001,
            p2p_port=9002,
            engine_url="http://127.0.0.1:8001",
            model="meta-llama/Llama-3-70b-instruct",
        )
        assert "#!/usr/bin/env bash" in script
        assert "nvidia-smi" in script
        assert "--enable-prefix-caching" in script
        assert "--block-size 16" in script
        assert "--model \"${MODEL_NAME:-meta-llama/Llama-3-70b-instruct}\"" in script
        assert "trap cleanup SIGTERM SIGINT" in script
        assert "python -m daemon.core" in script
        assert "--cloud-provider runpod" in script

    def test_run_provision_command_writes_file(self):
        """Verifies that run_provision_command writes the file to the requested destination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "custom_aws.sh")
            run_provision_command(
                provider="aws",
                role="active",
                output_path=out_file,
                proxy_url="http://10.0.1.10:8000",
                stdout_only=False,
            )
            assert os.path.exists(out_file)
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "IMDS_TOKEN" in content
            assert "SGM_ROLE=${SGM_NODE_ROLE:-active}" in content

    def test_run_provision_command_stdout(self, capsys):
        """Verifies that run_provision_command writes raw script to stdout when stdout_only is True."""
        run_provision_command(
            provider="runpod",
            role="active",
            stdout_only=True,
        )
        captured = capsys.readouterr()
        assert "#!/usr/bin/env bash" in captured.out
        assert "SGM RunPod Spot Automation" in captured.out
        assert "Panel" not in captured.out  # No Rich formatting markers


class TestMonitorCLI:
    """Test suite for cli.py monitor command and ClusterMonitor."""

    @pytest.mark.asyncio
    async def test_monitor_offline_host_graceful_handling(self):
        """
        Confirms that polling an unreachable host does NOT crash or raise unhandled exceptions.
        Marks cluster as OFFLINE and pauses financial accumulator.
        """
        dashboard = SGMDashboard()
        monitor = ClusterMonitor(
            proxy_url="http://127.0.0.1:59998",
            active_daemon_url="http://127.0.0.1:59997",
            standby_daemon_url="http://127.0.0.1:59996",
            poll_interval=0.2,
            dashboard=dashboard,
        )

        res = await monitor.poll_once()
        assert res["proxy_online"] is False
        assert res["active_online"] is False
        assert res["standby_online"] is False
        assert dashboard.state.cluster_status == "OFFLINE"
        assert dashboard.state.financial.nodes_count == 0
        for node in dashboard.state.nodes.values():
            assert node.status == "OFFLINE"

    @pytest.mark.asyncio
    async def test_monitor_live_cluster_telemetry(self):
        """
        Spins up a mock SGM Ingress Proxy and Node Daemons, verifies that ClusterMonitor
        polls /status and /health, updates nodes, in-flight streams, and financial counter.
        """
        # Mock Ingress Proxy server
        proxy_app = web.Application()
        proxy_state = {
            "preemption_active": False,
            "in_flight_count": 3,
            "active_upstream": "http://127.0.0.1:18001",
            "standby_upstream": "http://127.0.0.1:18002",
        }

        async def handle_proxy_status(request):
            return web.json_response({
                "active_upstream": proxy_state["active_upstream"],
                "standby_upstream": proxy_state["standby_upstream"],
                "preemption_active": proxy_state["preemption_active"],
                "in_flight_count": proxy_state["in_flight_count"],
                "in_flight_request_ids": ["req-1", "req-2", "req-3"],
            })

        async def handle_proxy_health(request):
            return web.json_response({
                "status": "healthy",
                "preemption_active": proxy_state["preemption_active"],
                "in_flight_requests": proxy_state["in_flight_count"],
            })

        async def handle_proxy_active_requests(request):
            return web.json_response({
                "requests": [
                    {"request_id": "req-1", "prompt": "Hello", "last_flushed_sequence_id": 5},
                    {"request_id": "req-2", "prompt": "World", "last_flushed_sequence_id": 12},
                    {"request_id": "req-3", "prompt": "SGM", "last_flushed_sequence_id": 8},
                ]
            })

        proxy_app.router.add_get("/status", handle_proxy_status)
        proxy_app.router.add_get("/health", handle_proxy_health)
        proxy_app.router.add_get("/internal/active-requests", handle_proxy_active_requests)

        # Mock Active Node Daemon server
        daemon_app = web.Application()

        async def handle_daemon_status(request):
            return web.json_response({
                "node_id": "spot-node-live-1",
                "role": "active",
                "state": "HEALTHY",
                "active_sessions_count": 3,
                "sessions": ["req-1", "req-2", "req-3"],
            })

        async def handle_daemon_health(request):
            return web.json_response({
                "status": "healthy",
                "node_id": "spot-node-live-1",
                "role": "active",
                "state": "HEALTHY",
            })

        daemon_app.router.add_get("/status", handle_daemon_status)
        daemon_app.router.add_get("/health", handle_daemon_health)

        # Start servers on ephemeral dynamic test ports (port 0)
        proxy_runner = web.AppRunner(proxy_app)
        await proxy_runner.setup()
        proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
        await proxy_site.start()
        proxy_port = list(proxy_site._server.sockets)[0].getsockname()[1]

        daemon_runner = web.AppRunner(daemon_app)
        await daemon_runner.setup()
        daemon_site = web.TCPSite(daemon_runner, "127.0.0.1", 0)
        await daemon_site.start()
        daemon_port = list(daemon_site._server.sockets)[0].getsockname()[1]

        try:
            dashboard = SGMDashboard()
            monitor = ClusterMonitor(
                proxy_url=f"http://127.0.0.1:{proxy_port}",
                active_daemon_url=f"http://127.0.0.1:{daemon_port}",
                poll_interval=0.1,
                dashboard=dashboard,
            )

            # Cycle 1: Steady state
            res = await monitor.poll_once()
            assert res["proxy_online"] is True
            assert res["active_online"] is True
            assert res["in_flight"] == 3
            assert dashboard.state.cluster_status == "OPERATIONAL"
            assert "spot-node-live-1" in dashboard.state.nodes
            assert dashboard.state.nodes["spot-node-live-1"].active_streams == 3
            assert dashboard.state.financial.nodes_count >= 1

            # Realized dollar savings accumulator is running
            savings = dashboard.state.financial.current_realized_savings_usd()
            assert savings >= dashboard.state.financial.baseline_savings_usd

            # Cycle 2: Inject preemption alert in proxy state
            proxy_state["preemption_active"] = True
            res2 = await monitor.poll_once()
            assert res2["preemption"] is True
            assert dashboard.state.cluster_status == "PREEMPTING"
            assert dashboard.state.preemption.is_preempting is True

            # Cycle 3: Preemption complete / resolved
            proxy_state["preemption_active"] = False
            res3 = await monitor.poll_once()
            assert res3["preemption"] is False
            assert dashboard.state.preemption.zero_loss_verified is True
            assert dashboard.state.cluster_status == "RESUMED"

        finally:
            await proxy_runner.cleanup()
            await daemon_runner.cleanup()
