"""
Verification Tests for Cloud Spot Deployment Scripts and Container Configurations.
SPEC-004 Acceptance Criteria: AC-M2-01, AC-M2-05, AC-M2-06, AC-M2-07.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_aws_spot_deploy_script_structure():
    """
    AC-M2-06 Verification:
    Validates scripts/aws_spot_deploy.sh production hardening:
    - set -euo pipefail
    - IMDSv2 token endpoint and header
    - Hop limit 2 configuration
    - NVIDIA driver 550+ and nvidia-ctk setup
    - Systemd unit definition with non-zero timeout and ExecStop
    """
    script_path = REPO_ROOT / "scripts" / "aws_spot_deploy.sh"
    assert script_path.exists(), "scripts/aws_spot_deploy.sh must exist"

    content = script_path.read_text(encoding="utf-8")

    # Safety checks
    assert "set -euo pipefail" in content
    assert "IFS=$'\\n\\t'" in content

    # IMDSv2 token acquisition
    assert "169.254.169.254/latest/api/token" in content
    assert "X-aws-ec2-metadata-token-ttl-seconds" in content
    assert "X-aws-ec2-metadata-token" in content

    # Hop limit enforcement
    assert "http-put-response-hop-limit 2" in content or "HopLimit" in content

    # NVIDIA setup
    assert "nvidia-driver-550" in content
    assert "nvidia-ctk runtime configure" in content

    # Systemd unit
    assert "Description=Spot GPU Migrator" in content
    assert "ExecStart=/usr/bin/docker run" in content
    assert "ExecStop=/usr/bin/docker stop -t 30" in content
    assert "SGM_CLOUD_PROVIDER=aws" in content


def test_runpod_spot_deploy_script_structure():
    """
    AC-M2-07 Verification:
    Validates scripts/runpod_spot_deploy.sh configuration:
    - --enable-prefix-caching and --block-size 16
    - Health polling loop before daemon startup
    - Signal trapping (SIGTERM/SIGINT) for emergency drain
    - Daemon invocation with --cloud-provider runpod
    """
    script_path = REPO_ROOT / "scripts" / "runpod_spot_deploy.sh"
    assert script_path.exists(), "scripts/runpod_spot_deploy.sh must exist"

    content = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in content
    assert "--enable-prefix-caching" in content
    assert "--block-size 16" in content
    assert "trap cleanup SIGTERM SIGINT" in content
    assert "curl -s \"http://127.0.0.1:${VLLM_PORT}/health\"" in content
    assert "--cloud-provider runpod" in content


def test_dockerfile_daemon_hardening():
    """
    AC-M2-01 & AC-M2-05 Verification:
    Validates deploy/Dockerfile.daemon:
    - Multi-stage build (AS builder, AS runtime)
    - Python 3.12 slim
    - Non-root user sgm:10001
    - Healthcheck probing port 9001
    - Entrypoint running daemon.core.node_daemon
    """
    dockerfile_path = REPO_ROOT / "deploy" / "Dockerfile.daemon"
    assert dockerfile_path.exists(), "deploy/Dockerfile.daemon must exist"

    content = dockerfile_path.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim-bookworm AS builder" in content
    assert "FROM python:3.12-slim-bookworm AS runtime" in content
    assert "groupadd -g 10001 sgm" in content
    assert "useradd -u 10001 -g sgm" in content
    assert "USER sgm:sgm" in content
    assert "HEALTHCHECK" in content
    assert "http://127.0.0.1:9001/health" in content
    assert 'ENTRYPOINT ["python", "-m", "daemon.core.node_daemon"]' in content


def test_dockerfile_proxy_hardening():
    """
    Validates deploy/Dockerfile.proxy:
    - Multi-stage build
    - Non-root user sgm:10001
    - Port 8000 exposed
    - Healthcheck probing port 8000
    - Entrypoint running proxy.ingress
    """
    dockerfile_path = REPO_ROOT / "deploy" / "Dockerfile.proxy"
    assert dockerfile_path.exists(), "deploy/Dockerfile.proxy must exist"

    content = dockerfile_path.read_text(encoding="utf-8")

    assert "AS builder" in content
    assert "AS runtime" in content
    assert "USER sgm:sgm" in content
    assert "EXPOSE 8000" in content
    assert "http://127.0.0.1:8000/health" in content
    assert 'ENTRYPOINT ["python", "-m", "proxy.ingress"]' in content


def test_docker_compose_topology():
    """
    AC-M2-01 Verification:
    Validates deploy/docker-compose.yml cluster topology:
    - Services: sgm-proxy, sgm-active-daemon, sgm-standby-daemon,
      sgm-active-engine, sgm-standby-engine, cloud-simulator
    - Ports: 8000, 9001, 9002, 9003, 8001, 8002, 18000
    - Bridge networks: sgm-ingress-net, sgm-cluster-internal-net
    - Volume mapping for huggingface-cache
    """
    compose_path = REPO_ROOT / "deploy" / "docker-compose.yml"
    assert compose_path.exists(), "deploy/docker-compose.yml must exist"

    content = compose_path.read_text(encoding="utf-8")

    required_services = [
        "sgm-proxy:",
        "sgm-active-daemon:",
        "sgm-active-engine:",
        "sgm-standby-daemon:",
        "sgm-standby-engine:",
        "cloud-simulator:",
    ]
    for svc in required_services:
        assert svc in content, f"Missing service {svc} in docker-compose.yml"

    required_ports = [
        '"8000:8000"',
        '"9001:9001"',
        '"9002:9002"',
        '"9003:9003"',
        '"8001:8001"',
        '"8002:8002"',
        '"18000:18000"',
    ]
    for port in required_ports:
        assert port in content, f"Missing port mapping {port} in docker-compose.yml"

    assert "sgm-ingress-net:" in content
    assert "sgm-cluster-internal-net:" in content
    assert "huggingface-cache:" in content
    assert "--enable-prefix-caching" in content
