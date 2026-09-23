"""
AWS IMDSv2 Preemption Watchdog.

Polls the AWS EC2 instance metadata service (IMDSv2) for spot termination notices:
1. Acquires session token via PUT /latest/api/token (TTL: 21600s).
2. Probes GET /latest/meta-data/spot/instance-action.
3. Parses JSON notice (action='terminate', time=...) with 120s warning notice.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from typing import Optional

import aiohttp

from daemon.models import PreemptionEvent
from daemon.watchdog.base import AbstractPreemptionWatchdog

logger = logging.getLogger("sgm.watchdog.aws")


class AWSPreemptionWatchdog(AbstractPreemptionWatchdog):
    """
    Preemption watchdog for AWS EC2 Spot instances using IMDSv2.
    """

    def __init__(
        self,
        base_url: str = "http://169.254.169.254",
        poll_interval_ms: int = 250,
        timeout_ms: int = 100,
        token_ttl_seconds: int = 21600,
    ) -> None:
        super().__init__(poll_interval_ms=poll_interval_ms, timeout_ms=timeout_ms)
        self.base_url = base_url.rstrip("/")
        self.token_ttl_seconds = token_ttl_seconds
        self._cached_token: Optional[str] = None
        self._token_acquired_at: float = 0.0
        self._token_refresh_interval_s: float = 3 * 3600.0  # Refresh every 3 hours
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

    async def _refresh_token(self) -> str:
        """Acquires a new IMDSv2 session token via PUT /latest/api/token."""
        session = await self._get_session()
        url = f"{self.base_url}/latest/api/token"
        headers = {"X-aws-ec2-metadata-token-ttl-seconds": str(self.token_ttl_seconds)}

        async with session.put(url, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ValueError(f"Failed to fetch IMDSv2 token (HTTP {resp.status}): {body}")
            token = (await resp.text()).strip()
            self._cached_token = token
            self._token_acquired_at = time.monotonic()
            logger.debug("Refreshed AWS IMDSv2 session token")
            return token

    async def _get_valid_token(self) -> str:
        now = time.monotonic()
        if (
            self._cached_token is None
            or (now - self._token_acquired_at) >= self._token_refresh_interval_s
        ):
            return await self._refresh_token()
        return self._cached_token

    async def check_once(self) -> Optional[PreemptionEvent]:
        """
        Polls AWS IMDSv2 spot instance-action endpoint.
        Returns PreemptionEvent if 200 OK, None if 404.
        """
        token = await self._get_valid_token()
        session = await self._get_session()
        url = f"{self.base_url}/latest/meta-data/spot/instance-action"
        headers = {"X-aws-ec2-metadata-token": token}

        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    return None  # Healthy

                if resp.status == 401:
                    # Invalidate token and retry once
                    logger.warning("AWS IMDSv2 token expired (401). Refreshing token immediately.")
                    token = await self._refresh_token()
                    headers["X-aws-ec2-metadata-token"] = token
                    async with session.get(url, headers=headers) as retry_resp:
                        if retry_resp.status == 404:
                            return None
                        if retry_resp.status == 200:
                            data = await retry_resp.json()
                            return self._parse_preemption_payload(data)
                        return None

                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except Exception:
                        text = await resp.text()
                        data = json.loads(text)
                    return self._parse_preemption_payload(data)

                logger.debug("Unexpected HTTP status from IMDSv2: %d", resp.status)
                return None

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("IMDSv2 request error: %s", exc)
            raise

    def _parse_preemption_payload(self, data: dict) -> PreemptionEvent:
        action = data.get("action", "terminate")
        deadline_str = data.get("time")
        deadline_seconds = 120.0

        if deadline_str:
            try:
                # Format: 2026-09-23T12:47:30Z
                target_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                diff = (target_dt - now_dt).total_seconds()
                if diff > 0:
                    deadline_seconds = diff
            except Exception as parse_err:
                logger.debug("Failed parsing deadline timestamp '%s': %s", deadline_str, parse_err)

        return PreemptionEvent(
            provider="aws",
            action=action,
            deadline=deadline_str,
            deadline_seconds=deadline_seconds,
            timestamp_ns=time.time_ns(),
            metadata=data,
        )
