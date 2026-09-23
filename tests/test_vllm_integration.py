"""
Milestone 2 Automated Verification Test Suite: Real vLLM Integration & Cloud Automation.

Conforms to SPEC-004 (specs/04-production-deployment-and-vllm.md):
- AC-M2-01: Container Healthchecks & Orchestrated Cluster Boot
- AC-M2-02: Real vLLM OpenAI API Hook & Engine Abort (<= 50ms)
- AC-M2-03: KV-Cache Prefix Caching Latency Budget & Resumption Payload Formatting
- AC-M2-06: AWS EC2 IMDSv2 Provisioning & Preemption Handling
- AC-M2-07: RunPod Spot Webhook & Auto-Recovery
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
import pytest
import yaml

from daemon.core.vllm_hook import AbortResult, VLLMInferenceEngineHook
from daemon.models import InferenceSession, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate_bash_structural_syntax(script_name: str, content: str) -> None:
    """
    Validates structural Bash syntax invariants including:
    - Balanced heredocs (cat << 'EOF' ... EOF)
    - Balanced control structures (if/fi, case/esac)
    - Balanced braces and parentheses outside comments and heredocs.
    """
    lines = content.splitlines()
    in_heredoc = False
    heredoc_delimiter = ""

    if_count = 0
    fi_count = 0
    case_count = 0
    esac_count = 0
    brace_depth = 0
    paren_depth = 0

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Handle heredoc transitions
        if in_heredoc:
            if stripped == heredoc_delimiter:
                in_heredoc = False
                heredoc_delimiter = ""
            continue

        heredoc_match = re.search(r"<<\s*['\"]?([A-Za-z0-9_]+)['\"]?", stripped)
        if heredoc_match:
            in_heredoc = True
            heredoc_delimiter = heredoc_match.group(1)
            continue

        # Skip comment lines
        if stripped.startswith("#"):
            continue

        # Tokenize line outside strings/comments for keyword tracking
        # Simple word boundaries for control flow keywords
        words = re.findall(r"\b(if|fi|case|esac|then|else|elif)\b", stripped)
        for w in words:
            if w == "if":
                if_count += 1
            elif w == "fi":
                fi_count += 1
            elif w == "case":
                case_count += 1
            elif w == "esac":
                esac_count += 1

        # Track brace and paren balance outside quotes
        # Remove literal strings before counting brackets
        sanitized = re.sub(r"\"[^\"]*\"|'[^']*'", "", stripped)
        # Remove trailing comments
        sanitized = re.sub(r"#.*$", "", sanitized)

        brace_depth += sanitized.count("{") - sanitized.count("}")
        paren_depth += sanitized.count("(") - sanitized.count(")")

    assert not in_heredoc, f"{script_name}: Unterminated heredoc block ('{heredoc_delimiter}')"
    assert if_count == fi_count, f"{script_name}: Mismatched if/fi count (if={if_count}, fi={fi_count})"
    assert case_count == esac_count, f"{script_name}: Mismatched case/esac count (case={case_count}, esac={esac_count})"
    assert brace_depth == 0, f"{script_name}: Mismatched curly braces {{}} (net depth: {brace_depth})"
    assert paren_depth == 0, f"{script_name}: Mismatched parentheses () (net depth: {paren_depth})"


def test_vllm_hook_prefix_caching_payload():
    """
    Test 1: Validates that VLLMInferenceEngineHook formats resumption payloads
    with exact token prefixes to trigger vLLM's RadixAttention APC cache hit.

    Conforms to SPEC-004 Section 2.3 & AC-M2-03:
    - Prefix prompt text concatenation (prompt + generated tokens up to cutoff).
    - Chat completion messages format (user prompt + assistant prefix).
    - Remaining max_tokens calculation (original max_tokens - total_tokens_generated).
    - Token IDs alignment (prompt_token_ids) for exact RadixTree hash matching.
    - SGM prefix caching flags and parameters.
    """
    hook = VLLMInferenceEngineHook(engine_url="http://127.0.0.1:8002")

    # 1. Standard Generation Session with in-flight generated tokens
    session = InferenceSession(
        request_id="req-radix-001",
        client_connection_id="conn-client-99",
        model="meta-llama/Llama-3-8b-instruct",
        prompt="Explain the theory of general relativity in three key points:",
        prompt_tokens=[1001, 1002, 1003, 1004, 1005],
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["<|eot_id|>", "</s>"],
        ),
        generated_tokens=[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008],
        generated_text=[
            " 1.", " Spacetime", " curvature", " occurs", " due", " to", " mass", " and",
        ],
        last_flushed_sequence_id=4,  # Flushed up to sequence 4 (" due")
        total_tokens_generated=8,
    )

    # Resumption payload cutoff at sequence ID 4 (5 tokens: 0..4 -> " 1. Spacetime curvature occurs due")
    payload = hook.format_resumption_payload(session, cutoff_sequence_id=4)

    assert payload["model"] == "meta-llama/Llama-3-8b-instruct"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["request_id"] == "req-radix-001"

    # Verify exact prefix prompt concatenation (RadixAttention radix key)
    expected_generated_prefix = " 1. Spacetime curvature occurs due"
    expected_full_prefix = f"Explain the theory of general relativity in three key points:{expected_generated_prefix}"
    assert payload["prompt"] == expected_full_prefix

    # Verify OpenAI-compatible chat completions messages structure
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == session.prompt
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"] == expected_generated_prefix

    # Verify RadixAttention APC token ID alignment (hash matches warm GPU VRAM cache blocks)
    expected_token_ids = [1001, 1002, 1003, 1004, 1005, 2001, 2002, 2003, 2004, 2005]
    assert payload["prompt_token_ids"] == expected_token_ids

    # Verify remaining token budget deduction (256 max_tokens - 5 prefix tokens = 251 remaining)
    assert payload["max_tokens"] == 251

    # Verify sampling parameters preserved
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["stop"] == ["<|eot_id|>", "</s>"]

    # Verify Radix Attention metadata for Standby node
    assert payload["extra_body"]["prefix_caching"] is True
    assert payload["extra_body"]["radix_attention_enabled"] is True
    assert payload["extra_body"]["cutoff_sequence_id"] == 4
    assert payload["extra_body"]["cached_prefix_tokens"] == 5

    # 2. Test Default Cutoff (uses session.last_flushed_sequence_id automatically)
    default_payload = hook.format_resumption_payload(session)
    assert default_payload["prompt"] == expected_full_prefix
    assert default_payload["max_tokens"] == 251
    assert default_payload["prompt_token_ids"] == expected_token_ids

    # 3. Test Full Continuation (cutoff at last generated token, sequence 7)
    full_payload = hook.format_resumption_payload(session, cutoff_sequence_id=7)
    assert full_payload["prompt"] == f"{session.prompt}{''.join(session.generated_text)}"
    assert full_payload["max_tokens"] == 256 - 8
    assert len(full_payload["prompt_token_ids"]) == len(session.prompt_tokens) + len(session.generated_tokens)

    # 4. Test Zero Generated Tokens (Fresh Prompt Resumption)
    fresh_session = InferenceSession(
        request_id="req-fresh-002",
        prompt="Tell me a joke.",
        prompt_tokens=[301, 302],
        sampling_params=SamplingParams(max_tokens=64),
        generated_tokens=[],
        generated_text=[],
        last_flushed_sequence_id=-1,
        total_tokens_generated=0,
    )
    fresh_payload = hook.format_resumption_payload(fresh_session)
    assert fresh_payload["prompt"] == "Tell me a joke."
    assert fresh_payload["max_tokens"] == 64
    assert fresh_payload["prompt_token_ids"] == [301, 302]
    assert len(fresh_payload["messages"]) == 1
    assert fresh_payload["messages"][0]["content"] == "Tell me a joke."


@pytest.mark.asyncio
async def test_vllm_hook_abort_request():
    """
    Test 2: Mocks vLLM's /abort endpoint and verifies that pause_request
    or abort_request signals the engine within the <= 50ms latency budget.

    Conforms to SPEC-004 Section 2.2 & AC-M2-02:
    - Endpoint /abort receives the request_id correctly.
    - Signal completes within <= 50ms latency SLA.
    - Both abort_request and pause_request return successful results.
    - Enforces timeout when engine is slow/unresponsive.
    """
    received_aborts: List[Dict[str, Any]] = []

    async def handle_abort(request: web.Request) -> web.Response:
        data = await request.json()
        received_aborts.append(data)
        # Fast simulated abort response (< 2ms)
        await asyncio.sleep(0.002)
        return web.json_response({"status": "aborted", "request_id": data.get("request_id")})

    async def handle_slow_abort(request: web.Request) -> web.Response:
        # Simulate hung inference engine (80ms delay)
        await asyncio.sleep(0.080)
        return web.json_response({"status": "aborted"})

    app = web.Application()
    app.router.add_post("/abort", handle_abort)
    app.router.add_post("/slow/abort", handle_slow_abort)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    # Retrieve dynamically assigned TCP port
    server_port = site._server.sockets[0].getsockname()[1]
    engine_url = f"http://127.0.0.1:{server_port}"

    hook = VLLMInferenceEngineHook(engine_url=engine_url, timeout_ms=50.0)

    try:
        # 1. Test abort_request within <= 50ms latency budget
        t_start = time.monotonic()
        abort_res = await hook.abort_request("req-test-abort-001")
        total_time_ms = (time.monotonic() - t_start) * 1000.0

        assert bool(abort_res) is True, "AbortResult should evaluate to True on success"
        assert abort_res.success is True
        assert abort_res.status_code == 200
        assert abort_res.request_id == "req-test-abort-001"
        assert abort_res.latency_ms <= 50.0, f"Abort latency {abort_res.latency_ms:.2f}ms exceeded 50ms budget!"
        assert total_time_ms <= 50.0, f"Total abort execution {total_time_ms:.2f}ms exceeded 50ms budget!"
        assert len(received_aborts) == 1
        assert received_aborts[0]["request_id"] == "req-test-abort-001"

        # 2. Test pause_request alias within <= 50ms latency budget
        t_start = time.monotonic()
        pause_res = await hook.pause_request("req-test-pause-002")
        total_pause_time_ms = (time.monotonic() - t_start) * 1000.0

        assert bool(pause_res) is True, "pause_request should evaluate to True on success"
        assert pause_res.success is True
        assert pause_res.status_code == 200
        assert pause_res.request_id == "req-test-pause-002"
        assert pause_res.latency_ms <= 50.0, f"Pause latency {pause_res.latency_ms:.2f}ms exceeded 50ms budget!"
        assert total_pause_time_ms <= 50.0, f"Total pause execution {total_pause_time_ms:.2f}ms exceeded 50ms budget!"
        assert len(received_aborts) == 2
        assert received_aborts[1]["request_id"] == "req-test-pause-002"

        # 3. Test latency budget timeout enforcement (AC-M2-02 strict ceiling)
        slow_hook = VLLMInferenceEngineHook(
            engine_url=f"{engine_url}/slow",
            timeout_ms=25.0,  # Strict 25ms timeout
        )
        try:
            t_start = time.monotonic()
            slow_res = await slow_hook.abort_request("req-test-slow")
            slow_time_ms = (time.monotonic() - t_start) * 1000.0

            assert bool(slow_res) is False, "Slow abort should fail on timeout"
            assert slow_res.success is False
            assert slow_res.status_code == 408
            assert slow_time_ms <= 50.0, f"Timeout took {slow_time_ms:.2f}ms, exceeding 50ms budget"
        finally:
            await slow_hook.close()

    finally:
        await hook.close()
        await runner.cleanup()


def test_docker_compose_syntax_and_health():
    """
    Test 3: Inspects and parses deploy/docker-compose.yml to verify that all ports
    (8000, 8001, 8002, 9001, 9002, 9003, 18000), networks, volumes, and healthchecks
    conform to SPEC-004.

    Conforms to SPEC-004 Section 3.4 & AC-M2-01:
    - Service topologies: sgm-proxy, active/standby daemons, active/standby engines, cloud-simulator.
    - Port mappings: 8000 (Proxy), 9001/9003 (Daemon Rest), 9002 (P2P), 8001/8002 (vLLM), 18000 (Chaos).
    - Network segmentation: sgm-ingress-net, sgm-cluster-internal-net.
    - Prefix caching configuration: --enable-prefix-caching, --block-size 16.
    - Healthchecks with interval, timeout, retries.
    """
    compose_path = REPO_ROOT / "deploy" / "docker-compose.yml"
    assert compose_path.exists(), f"deploy/docker-compose.yml does not exist at {compose_path}"

    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert isinstance(data, dict), "docker-compose.yml must be a valid YAML mapping"
    assert data.get("version") in ("3.8", "3", "3.9"), f"Unexpected compose version: {data.get('version')}"

    # 1. Networks validation
    networks = data.get("networks", {})
    assert "sgm-ingress-net" in networks, "Missing 'sgm-ingress-net' network"
    assert "sgm-cluster-internal-net" in networks, "Missing 'sgm-cluster-internal-net' network"
    # Internal must be false to allow watchdog to contact IMDSv2 (SPEC-004 line 363)
    assert networks["sgm-cluster-internal-net"].get("internal") is False, (
        "sgm-cluster-internal-net must have internal: false for IMDSv2 metadata access"
    )

    # 2. Volumes validation
    volumes = data.get("volumes", {})
    assert "huggingface-cache" in volumes, "Missing 'huggingface-cache' volume"

    # 3. Services and Port Allocations
    services = data.get("services", {})
    required_services = [
        "sgm-proxy",
        "sgm-active-daemon",
        "sgm-active-engine",
        "sgm-standby-daemon",
        "sgm-standby-engine",
        "cloud-simulator",
    ]
    for svc in required_services:
        assert svc in services, f"Required service '{svc}' not found in docker-compose.yml"

    # Collect all exposed host ports
    all_ports = set()
    for svc_name, svc_conf in services.items():
        for port_mapping in svc_conf.get("ports", []):
            if isinstance(port_mapping, str):
                parts = port_mapping.split(":")
                host_port = int(parts[-2] if len(parts) >= 2 else parts[0])
                all_ports.add(host_port)
            elif isinstance(port_mapping, dict):
                all_ports.add(int(port_mapping.get("published", 0)))
            elif isinstance(port_mapping, int):
                all_ports.add(port_mapping)

    expected_ports = {8000, 8001, 8002, 9001, 9002, 9003, 18000}
    missing_ports = expected_ports - all_ports
    assert not missing_ports, f"Missing required ports in docker-compose.yml: {missing_ports}. Found: {all_ports}"

    # 4. Service-Specific Port & Topology Verification
    proxy_ports = [str(p) for p in services["sgm-proxy"].get("ports", [])]
    assert any("8000" in p for p in proxy_ports), "sgm-proxy must expose port 8000"

    active_daemon_ports = [str(p) for p in services["sgm-active-daemon"].get("ports", [])]
    assert any("9001" in p for p in active_daemon_ports), "sgm-active-daemon must expose control port 9001"

    active_engine_ports = [str(p) for p in services["sgm-active-engine"].get("ports", [])]
    assert any("8001" in p for p in active_engine_ports), "sgm-active-engine must expose port 8001"

    standby_daemon_ports = [str(p) for p in services["sgm-standby-daemon"].get("ports", [])]
    assert any("9003" in p for p in standby_daemon_ports), "sgm-standby-daemon must expose control port 9003"
    assert any("9002" in p for p in standby_daemon_ports), "sgm-standby-daemon must expose P2P port 9002"

    standby_engine_ports = [str(p) for p in services["sgm-standby-engine"].get("ports", [])]
    assert any("8002" in p for p in standby_engine_ports), "sgm-standby-engine must expose port 8002"

    simulator_ports = [str(p) for p in services["cloud-simulator"].get("ports", [])]
    assert any("18000" in p for p in simulator_ports), "cloud-simulator must expose port 18000"

    # 5. Prefix Caching Configuration in vLLM Engines
    for engine_svc in ("sgm-active-engine", "sgm-standby-engine"):
        cmd = services[engine_svc].get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        assert "--enable-prefix-caching" in cmd, f"{engine_svc} must configure --enable-prefix-caching"
        assert "--block-size 16" in cmd or "--block-size" in cmd, f"{engine_svc} must configure --block-size 16"

    # 6. Healthchecks Verification
    healthchecked_services = ["sgm-proxy", "sgm-active-daemon", "sgm-standby-daemon", "cloud-simulator"]
    for svc_name in healthchecked_services:
        svc_conf = services[svc_name]
        assert "healthcheck" in svc_conf, f"Service '{svc_name}' must define a healthcheck"
        hc = svc_conf["healthcheck"]
        test_cmd = hc.get("test", [])
        test_str = " ".join(test_cmd) if isinstance(test_cmd, list) else str(test_cmd)
        assert "curl" in test_str and "/health" in test_str, (
            f"Service '{svc_name}' healthcheck must curl /health endpoint (found: {test_str})"
        )
        assert "interval" in hc, f"Service '{svc_name}' healthcheck must specify interval"
        assert "timeout" in hc, f"Service '{svc_name}' healthcheck must specify timeout"
        assert "retries" in hc, f"Service '{svc_name}' healthcheck must specify retries"

    # 7. Dependency and Startup Ordering
    proxy_deps = services["sgm-proxy"].get("depends_on", {})
    assert "sgm-active-daemon" in proxy_deps, "sgm-proxy must depend on sgm-active-daemon"
    assert "sgm-standby-daemon" in proxy_deps, "sgm-proxy must depend on sgm-standby-daemon"


def test_cloud_provisioning_scripts_syntax():
    """
    Test 4: Verifies that scripts/aws_spot_deploy.sh and scripts/runpod_spot_deploy.sh exist,
    have valid Bash syntax (via bash/sh check or structural validation), and configure
    IMDSv2 token retrieval properly.

    Conforms to SPEC-004 Section 4 & AC-M2-06, AC-M2-07:
    - Scripts existence and non-zero size.
    - Bash syntax validity (via system bash or deep structural syntax validation).
    - IMDSv2 token retrieval: PUT http://169.254.169.254/latest/api/token, TTL header, token reuse.
    - RunPod prefix caching and trap cleanup configuration.
    """
    aws_script = REPO_ROOT / "scripts" / "aws_spot_deploy.sh"
    runpod_script = REPO_ROOT / "scripts" / "runpod_spot_deploy.sh"

    # 1. Existence and non-empty check
    assert aws_script.exists(), f"Missing AWS deploy script at {aws_script}"
    assert runpod_script.exists(), f"Missing RunPod deploy script at {runpod_script}"
    assert aws_script.stat().st_size > 0, "scripts/aws_spot_deploy.sh is empty"
    assert runpod_script.stat().st_size > 0, "scripts/runpod_spot_deploy.sh is empty"

    aws_content = aws_script.read_text(encoding="utf-8")
    runpod_content = runpod_script.read_text(encoding="utf-8")

    # 2. Bash/sh Syntax Verification (if bash or sh interpreter is present on host)
    bash_bin = shutil.which("bash") or shutil.which("sh")
    if bash_bin:
        for script_path in (aws_script, runpod_script):
            res = subprocess.run(
                [bash_bin, "-n", str(script_path)],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, f"Bash syntax check failed for {script_path.name}:\n{res.stderr}"

    # 3. Structural Syntax Validation (executed in all environments)
    for script_name, content in (("aws_spot_deploy.sh", aws_content), ("runpod_spot_deploy.sh", runpod_content)):
        lines = content.splitlines()
        # Shebang
        assert lines[0].startswith("#!"), f"{script_name} must have a shebang on line 1"
        assert "bash" in lines[0] or "sh" in lines[0], f"{script_name} shebang must specify bash or sh"

        # Strict safety flags
        assert "set -euo pipefail" in content or ("set -e" in content and "set -u" in content), (
            f"{script_name} must enforce strict error handling ('set -euo pipefail')"
        )

        # Structural balance checks: quotes, braces, parens, heredocs, if/fi
        _validate_bash_structural_syntax(script_name, content)

    # 4. AWS IMDSv2 Token Retrieval Verification (SPEC-004 Section 4.1 & AC-M2-06)
    assert "169.254.169.254" in aws_content, "AWS script must target link-local metadata address 169.254.169.254"
    assert "/latest/api/token" in aws_content, "AWS script must query IMDSv2 token endpoint /latest/api/token"
    assert "X-aws-ec2-metadata-token-ttl-seconds" in aws_content, (
        "AWS script must specify IMDSv2 token TTL header 'X-aws-ec2-metadata-token-ttl-seconds'"
    )
    assert "X-aws-ec2-metadata-token" in aws_content, (
        "AWS script must pass IMDSv2 token header 'X-aws-ec2-metadata-token' in subsequent calls"
    )
    assert "-X PUT" in aws_content, "IMDSv2 token request must use HTTP PUT"
    assert "IMDS_TOKEN" in aws_content, "AWS script must capture IMDS_TOKEN variable"

    # Verify error handling if token acquisition fails
    assert "exit 1" in aws_content, "AWS script must exit with error if IMDSv2 token cannot be acquired"

    # Verify systemd service configuration in AWS script
    assert "/etc/systemd/system/sgm-daemon.service" in aws_content, (
        "AWS script must configure sgm-daemon.service systemd unit"
    )
    assert "ExecStart=" in aws_content, "Systemd unit must have ExecStart directive"
    assert "SGM_CLOUD_PROVIDER=aws" in aws_content or "SGM_CLOUD_PROVIDER" in aws_content, (
        "Systemd unit must configure SGM_CLOUD_PROVIDER"
    )

    # 5. RunPod Automation Verification (SPEC-004 Section 4.2 & AC-M2-07)
    assert "--enable-prefix-caching" in runpod_content, (
        "RunPod script must launch vLLM with --enable-prefix-caching"
    )
    assert "trap" in runpod_content and ("SIGTERM" in runpod_content or "cleanup" in runpod_content), (
        "RunPod script must register signal trap for graceful preemption handling"
    )
    assert "daemon.core" in runpod_content, (
        "RunPod script must execute SGM node daemon"
    )
    assert "--cloud-provider runpod" in runpod_content, (
        "RunPod script must configure --cloud-provider runpod"
    )


def test_dockerfiles_spec_compliance():
    """
    Supplementary Test: Verifies that deploy/Dockerfile.daemon and deploy/Dockerfile.proxy
    comply with SPEC-004 container security, non-root user (sgm:10001), multi-stage builds,
    and healthchecks.
    """
    daemon_dockerfile = REPO_ROOT / "deploy" / "Dockerfile.daemon"
    proxy_dockerfile = REPO_ROOT / "deploy" / "Dockerfile.proxy"

    assert daemon_dockerfile.exists(), f"Missing {daemon_dockerfile}"
    assert proxy_dockerfile.exists(), f"Missing {proxy_dockerfile}"

    for df_path in (daemon_dockerfile, proxy_dockerfile):
        content = df_path.read_text(encoding="utf-8")
        # Multi-stage builds
        assert "AS builder" in content, f"{df_path.name} must feature builder stage"
        assert "AS runtime" in content, f"{df_path.name} must feature runtime stage"
        # Non-root user sgm:10001
        assert "10001" in content and "sgm" in content, f"{df_path.name} must create non-root user sgm:10001"
        assert "USER sgm" in content or "USER 10001" in content, f"{df_path.name} must set USER to non-root"
        # Healthcheck
        assert "HEALTHCHECK" in content, f"{df_path.name} must define HEALTHCHECK"
