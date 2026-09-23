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
from daemon.core.node_daemon import NodeDaemon
from ui.dashboard import (
    ClusterMonitor,
    DashboardState,
    NodeTelemetry,
    SGMDashboard,
    format_engine_badge,
    normalize_engine_name,
)


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


class TestUniversalEngineDashboardAndCLI:
    """Test suite for universal inference engine integrations in CLI and UI dashboard."""

    def test_node_telemetry_engine_type(self):
        """Validates that NodeTelemetry includes engine_type and defaults to vLLM."""
        node = NodeTelemetry(
            node_id="test-node-1",
            role="ACTIVE",
            provider="AWS (p4de.24xlarge)",
            status="HEALTHY",
            control_port=9001,
            p2p_port=9002,
            engine_port=8001,
            memory_info="80 GB VRAM",
        )
        assert node.engine_type == "vLLM"

        # Explicit engine types
        node_trt = NodeTelemetry(
            node_id="test-node-2",
            role="ACTIVE",
            provider="AWS (p4de.24xlarge)",
            status="HEALTHY",
            control_port=9001,
            p2p_port=9002,
            engine_port=8001,
            memory_info="80 GB VRAM",
            engine_type="TensorRT-LLM",
        )
        assert node_trt.engine_type == "TensorRT-LLM"

        node_tgi = NodeTelemetry(
            node_id="test-node-3",
            role="STANDBY",
            provider="GCP (a2-highgpu-8g)",
            status="HEALTHY",
            control_port=9003,
            p2p_port=9002,
            engine_port=8002,
            memory_info="80 GB VRAM",
            engine_type="HuggingFace TGI",
        )
        assert node_tgi.engine_type == "HuggingFace TGI"

    def test_engine_badge_formatting(self):
        """Validates that format_engine_badge applies expected Rich markup."""
        assert "vLLM" in format_engine_badge("vllm")
        assert "TRT-LLM" in format_engine_badge("tensorrt_llm")
        assert "TRT-LLM" in format_engine_badge("trt")
        assert "HF-TGI" in format_engine_badge("tgi")
        assert "SGLang" in format_engine_badge("sglang")
        assert "Mock" in format_engine_badge("mock")
        assert "Auto-Detect" in format_engine_badge("auto")

    def test_normalize_engine_name(self):
        """Validates canonical naming for supported engine backends."""
        assert normalize_engine_name("vllm") == "vLLM"
        assert normalize_engine_name("tensorrt_llm") == "TensorRT-LLM"
        assert normalize_engine_name("tgi") == "HuggingFace TGI"
        assert normalize_engine_name("sglang") == "SGLang"
        assert normalize_engine_name("mock") == "Mock Engine"
        assert normalize_engine_name("auto") == "Auto-Detect"

    def test_dashboard_renders_engine_in_header_and_table(self):
        """Validates that SGMDashboard displays engine in header and nodes table."""
        dashboard = SGMDashboard()
        dashboard.state.set_engine_type("tensorrt_llm")

        # Verify header rendering
        header_panel = dashboard._render_header()
        header_text = str(header_panel.renderable)
        assert "TRT-LLM" in header_text or "Engine:" in header_text

        # Verify nodes table rendering
        table_panel = dashboard._render_nodes_table()
        table = table_panel.renderable
        col_names = [col.header for col in table.columns]
        assert "Engine" in col_names

    def test_cli_argument_parsers_support_engine_type(self):
        """Validates that cli.py subparsers accept --engine-type with valid choices."""
        import subprocess

        for cmd in ["demo", "status", "monitor"]:
            proc = subprocess.run(
                [sys.executable, "cli.py", cmd, "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
            assert "--engine-type" in proc.stdout
            assert "tensorrt_llm" in proc.stdout
            assert "tgi" in proc.stdout
            assert "sglang" in proc.stdout

    @pytest.mark.asyncio
    async def test_cluster_monitor_engine_type_propagation(self):
        """Validates that ClusterMonitor propagates engine_type to dashboard and nodes."""
        dashboard = SGMDashboard()
        monitor = ClusterMonitor(
            proxy_url="http://127.0.0.1:59990",
            engine_type="tgi",
            poll_interval=0.1,
            dashboard=dashboard,
        )
        assert monitor.dashboard.state.engine_type == "HuggingFace TGI"

    @pytest.mark.asyncio
    async def test_node_daemon_engine_type_api(self):
        """Validates that NodeDaemon accepts engine_type and exposes it on /health and /status."""
        daemon = NodeDaemon(
            node_id="test-engine-node",
            role="active",
            control_port=0,
            p2p_port=0,
            engine_type="tensorrt_llm",
        )
        app = daemon._setup_routes()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = list(site._server.sockets)[0].getsockname()[1]

        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["engine_type"] == "tensorrt_llm"

                async with session.get(f"http://127.0.0.1:{port}/status") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["engine_type"] == "tensorrt_llm"
        finally:
            await runner.cleanup()

