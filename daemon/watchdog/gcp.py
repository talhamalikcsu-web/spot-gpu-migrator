"""
Google Cloud Platform (GCP) Compute Engine Preemption Watchdog.

Polls the GCP hypervisor metadata service:
1. GET /computeMetadata/v1/instance/preempted (Header: 'Metadata-Flavor: Google')
   - Returns "FALSE" (healthy) or "TRUE" (preempted with 30s notice).
2. GET /computeMetadata/v1/instance/maintenance-event
   - Returns "NONE" (healthy) or "TERMINATE_ON_HOST_MAINTENANCE".
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from daemon.models import PreemptionEvent
from daemon.watchdog.base import AbstractPreemptionWatchdog

logger = logging.getLogger("sgm.watchdog.gcp")


class GCPPreemptionWatchdog(AbstractPreemptionWatchdog):
    """
    Preemption watchdog for GCP Spot / Preemptible VM instances.
    """

    def __init__(
        self,
        base_url: str = "http://metadata.google.internal",
        poll_interval_ms: int = 250,
        timeout_ms: int = 100,
    ) -> None:
        super().__init__(poll_interval_ms=poll_interval_ms, timeout_ms=timeout_ms)
        self.base_url = base_url.rstrip("/")
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

    async def check_once(self) -> Optional[PreemptionEvent]:
        """
        Executes preemption checks against GCP metadata endpoints.
        """
        session = await self._get_session()
        headers = {"Metadata-Flavor": "Google"}
        preempted_url = f"{self.base_url}/computeMetadata/v1/instance/preempted"

        try:
            async with session.get(preempted_url, headers=headers) as resp:
                if resp.status == 200:
                    body = (await resp.text()).strip()
                    if body.upper() == "TRUE":
                        logger.critical("GCP metadata signaled preemption: preempted=TRUE")
                        return PreemptionEvent(
                            provider="gcp",
                            action="terminate",
                            deadline_seconds=30.0,
                            timestamp_ns=time.time_ns(),
                            metadata={"check": "preempted", "response": body},
                        )
                elif resp.status == 403:
                    logger.warning("GCP metadata rejected request (HTTP 403). Missing Metadata-Flavor header?")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            # Immediate retry after 50ms per spec
            logger.debug("GCP metadata probe failed (%s), retrying after 50ms...", exc)
            await asyncio.sleep(0.050)
            try:
                async with session.get(preempted_url, headers=headers) as resp:
                    if resp.status == 200:
                        body = (await resp.text()).strip()
                        if body.upper() == "TRUE":
                            return PreemptionEvent(
                                provider="gcp",
                                action="terminate",
                                deadline_seconds=30.0,
                                timestamp_ns=time.time_ns(),
                                metadata={"check": "preempted_retry", "response": body},
                            )
            except Exception as retry_exc:
                logger.debug("GCP metadata retry failed: %s", retry_exc)
                raise

        # Secondary check: maintenance-event
        maint_url = f"{self.base_url}/computeMetadata/v1/instance/maintenance-event"
        try:
            async with session.get(maint_url, headers=headers) as resp:
                if resp.status == 200:
                    body = (await resp.text()).strip()
                    if body.upper() == "TERMINATE_ON_HOST_MAINTENANCE":
                        logger.critical("GCP signaled TERMINATE_ON_HOST_MAINTENANCE")
                        return PreemptionEvent(
                            provider="gcp",
                            action="terminate",
                            deadline_seconds=30.0,
                            timestamp_ns=time.time_ns(),
                            metadata={"check": "maintenance-event", "response": body},
                        )
        except Exception:
            pass

        return None
