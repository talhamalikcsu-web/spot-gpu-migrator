"""
RunPod Termination Notice Watchdog.

Supports dual-channel preemption detection for RunPod spot pods:
1. Status poller against local agent endpoint (GET /runpod/preempt) or file flag.
2. Direct webhook ingestion method (handle_webhook_payload) triggered via HTTP POST /webhook/runpod-terminate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp

from daemon.models import PreemptionEvent
from daemon.watchdog.base import AbstractPreemptionWatchdog

logger = logging.getLogger("sgm.watchdog.runpod")


class RunPodPreemptionWatchdog(AbstractPreemptionWatchdog):
    """
    Preemption watchdog for RunPod spot GPU instances.
    """

    def __init__(
        self,
        status_url: str = "http://127.0.0.1:18000/runpod/preempt",
        trigger_file_path: str = "/tmp/runpod_terminate_notice",
        poll_interval_ms: int = 250,
        timeout_ms: int = 100,
    ) -> None:
        super().__init__(poll_interval_ms=poll_interval_ms, timeout_ms=timeout_ms)
        self.status_url = status_url
        self.trigger_file_path = trigger_file_path
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(
                total=self.timeout_s,
                connect=self.timeout_s,
                sock_read=self.timeout_s,
            )
            self._session = aiohttp.ClientSession(timeout=client_timeout)
        return self._session

    async def stop(self) -> None:
        await super().stop()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def handle_webhook_payload(self, payload: Dict[str, Any]) -> PreemptionEvent:
        """
        Synchronously handles an inbound webhook notice from RunPod hypervisor.
        """
        logger.critical("RunPod termination notice received via webhook: %s", payload)
        event_name = payload.get("event", "POD_TERMINATION_NOTICE")
        pod_id = payload.get("pod_id", "unknown-pod")
        grace_period = float(payload.get("grace_period_seconds", 30))

        event = PreemptionEvent(
            provider="runpod",
            action="terminate",
            deadline_seconds=grace_period,
            timestamp_ns=time.time_ns(),
            metadata={"event": event_name, "pod_id": pod_id, "payload": payload},
        )
        self._preemption_detected = True
        await self.notify_callbacks(event)
        return event

    async def check_once(self) -> Optional[PreemptionEvent]:
        """
        Probes local status endpoint and checks trigger file.
        """
        # 1. Check local termination flag file
        if os.path.exists(self.trigger_file_path):
            logger.critical("RunPod trigger file found at %s", self.trigger_file_path)
            return PreemptionEvent(
                provider="runpod",
                action="terminate",
                deadline_seconds=30.0,
                timestamp_ns=time.time_ns(),
                metadata={"trigger": "file", "path": self.trigger_file_path},
            )

        # 2. Check status poller endpoint
        if self.status_url:
            session = await self._get_session()
            try:
                async with session.get(self.status_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("preempted") is True:
                            grace_period = float(data.get("grace_period_seconds", 30))
                            logger.critical("RunPod status endpoint signaled preemption: %s", data)
                            return PreemptionEvent(
                                provider="runpod",
                                action="terminate",
                                deadline_seconds=grace_period,
                                timestamp_ns=time.time_ns(),
                                metadata=data,
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

        return None
