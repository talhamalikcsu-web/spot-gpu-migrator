"""
Chaos Injection Utilities for Spot GPU Migrator (SGM).

Provides CLI and programmatic chaos scenarios:
1. AWS 2-Minute Preemption Notice (120s deadline)
2. GCP 30-Second Preemption Notice (30s deadline)
3. Sudden Network Partition / Latency Injections
4. Rapid Metadata Service Flapping & 500 Blips
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("sgm.simulator.chaos_injector")


async def inject_aws_2min_preemption(
    simulator_url: str = "http://127.0.0.1:18000",
    notice_seconds: float = 120.0,
) -> Dict[str, Any]:
    """
    Injects an AWS EC2 Spot instance 2-minute preemption warning into the simulator.
    """
    url = f"{simulator_url.rstrip('/')}/chaos/inject-preemption"
    payload = {"provider": "aws", "notice_seconds": notice_seconds}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            logger.info("Injected AWS 2-minute preemption notice (%.1fs): %s", notice_seconds, data)
            return data


async def inject_gcp_30s_preemption(
    simulator_url: str = "http://127.0.0.1:18000",
    notice_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Injects a GCP Compute Engine 30-second preemption notice into the simulator.
    """
    url = f"{simulator_url.rstrip('/')}/chaos/inject-preemption"
    payload = {"provider": "gcp", "notice_seconds": notice_seconds}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            logger.info("Injected GCP 30-second preemption notice (%.1fs): %s", notice_seconds, data)
            return data


async def inject_network_partition(
    simulator_url: str = "http://127.0.0.1:18000",
    delay_ms: float = 5000.0,
) -> Dict[str, Any]:
    """
    Simulates a sudden network partition by injecting high artificial delay.
    """
    url = f"{simulator_url.rstrip('/')}/chaos/network-delay"
    payload = {"delay_ms": delay_ms}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            logger.warning("Injected sudden network partition delay (%.1fms): %s", delay_ms, data)
            return data


async def reset_chaos(
    simulator_url: str = "http://127.0.0.1:18000",
) -> Dict[str, Any]:
    """
    Resets all chaos conditions and provider states back to healthy.
    """
    url = f"{simulator_url.rstrip('/')}/chaos/reset"
    async with aiohttp.ClientSession() as session:
        async with session.post(url) as resp:
            data = await resp.json()
            logger.info("Reset simulator chaos state to HEALTHY: %s", data)
            return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="⚡ SGM Chaos Injector Tool")
    subparsers = parser.add_subparsers(dest="scenario", help="Chaos scenario to inject")

    aws_p = subparsers.add_parser("aws-preempt", help="Trigger AWS 2-minute preemption")
    aws_p.add_argument("--url", default="http://127.0.0.1:18000")
    aws_p.add_argument("--notice", type=float, default=120.0)

    gcp_p = subparsers.add_parser("gcp-preempt", help="Trigger GCP 30-second preemption")
    gcp_p.add_argument("--url", default="http://127.0.0.1:18000")
    gcp_p.add_argument("--notice", type=float, default=30.0)

    part_p = subparsers.add_parser("partition", help="Inject sudden network partition")
    part_p.add_argument("--url", default="http://127.0.0.1:18000")
    part_p.add_argument("--delay-ms", type=float, default=5000.0)

    reset_p = subparsers.add_parser("reset", help="Reset chaos state to healthy")
    reset_p.add_argument("--url", default="http://127.0.0.1:18000")

    args = parser.parse_args()

    if args.scenario == "aws-preempt":
        asyncio.run(inject_aws_2min_preemption(args.url, args.notice))
    elif args.scenario == "gcp-preempt":
        asyncio.run(inject_gcp_30s_preemption(args.url, args.notice))
    elif args.scenario == "partition":
        asyncio.run(inject_network_partition(args.url, args.delay_ms))
    elif args.scenario == "reset":
        asyncio.run(reset_chaos(args.url))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
