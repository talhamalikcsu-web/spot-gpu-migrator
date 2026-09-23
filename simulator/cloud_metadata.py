"""
Cloud Chaos & Metadata Simulator Server.

Runs on port 18000 and mocks hypervisor spot metadata services:
- AWS IMDSv2 (PUT /latest/api/token, GET /latest/meta-data/spot/instance-action)
- GCP Compute Engine (GET /computeMetadata/v1/instance/preempted, maintenance-event)
- RunPod Spot Status (GET /runpod/preempt)
- Chaos API (POST /chaos/trigger-preemption, POST /chaos/inject-preemption, POST /chaos/reset)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
import uuid
from typing import Dict, Optional

from aiohttp import web

logger = logging.getLogger("sgm.simulator.cloud_metadata")


class CloudMetadataServer:
    """
    Mock hypervisor metadata server simulating AWS, GCP, and RunPod spot terminations.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 18000) -> None:
        self.host = host
        self.port = port

        # AWS state
        self.aws_preempted: bool = False
        self.aws_notice_seconds: float = 120.0
        self.aws_tokens: Dict[str, float] = {}  # token -> expires_at_monotonic

        # GCP state
        self.gcp_preempted: bool = False
        self.gcp_notice_seconds: float = 30.0
        self.gcp_maintenance_event: str = "NONE"

        # RunPod state
        self.runpod_preempted: bool = False
        self.runpod_notice_seconds: float = 30.0

        # Chaos controls
        self.artificial_delay_ms: float = 0.0

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    def reset(self) -> None:
        """Resets all cloud provider states back to healthy."""
        self.aws_preempted = False
        self.aws_notice_seconds = 120.0
        self.aws_tokens.clear()
        self.gcp_preempted = False
        self.gcp_notice_seconds = 30.0
        self.gcp_maintenance_event = "NONE"
        self.runpod_preempted = False
        self.runpod_notice_seconds = 30.0
        self.artificial_delay_ms = 0.0
        logger.info("Simulator reset to HEALTHY state for all cloud providers.")

    def trigger_preemption(self, provider: str = "aws", notice_seconds: Optional[float] = None) -> None:
        """Programmatically triggers preemption notice for a cloud provider."""
        prov = provider.lower()
        if prov in ("aws", "all"):
            self.aws_preempted = True
            if notice_seconds is not None:
                self.aws_notice_seconds = float(notice_seconds)
            logger.critical("Simulator: AWS preemption triggered (notice: %.1fs)", self.aws_notice_seconds)

        if prov in ("gcp", "all"):
            self.gcp_preempted = True
            if notice_seconds is not None:
                self.gcp_notice_seconds = float(notice_seconds)
            logger.critical("Simulator: GCP preemption triggered (notice: %.1fs)", self.gcp_notice_seconds)

        if prov in ("runpod", "all"):
            self.runpod_preempted = True
            if notice_seconds is not None:
                self.runpod_notice_seconds = float(notice_seconds)
            logger.critical("Simulator: RunPod preemption triggered (notice: %.1fs)", self.runpod_notice_seconds)

    # -----------------------------------------------------------------------
    # Route Handlers
    # -----------------------------------------------------------------------

    async def _apply_delay(self) -> None:
        if self.artificial_delay_ms > 0:
            await asyncio.sleep(self.artificial_delay_ms / 1000.0)

    # --- AWS IMDSv2 ---

    async def handle_aws_token(self, request: web.Request) -> web.Response:
        """PUT /latest/api/token -> Returns mock IMDSv2 session token."""
        await self._apply_delay()
        ttl_header = request.headers.get("X-aws-ec2-metadata-token-ttl-seconds", "21600")
        try:
            ttl_s = float(ttl_header)
        except ValueError:
            ttl_s = 21600.0

        token = f"aws-token-{uuid.uuid4().hex[:16]}"
        self.aws_tokens[token] = time.monotonic() + ttl_s
        logger.debug("AWS IMDSv2 token issued: %s (ttl: %.0fs)", token, ttl_s)
        return web.Response(text=token, content_type="text/plain")

    async def handle_aws_spot_action(self, request: web.Request) -> web.Response:
        """GET /latest/meta-data/spot/instance-action -> 404 healthy or 200 termination JSON."""
        await self._apply_delay()
        token = request.headers.get("X-aws-ec2-metadata-token")
        if not token:
            return web.Response(text="Unauthorized: Missing IMDSv2 Token", status=401)

        expires_at = self.aws_tokens.get(token)
        if expires_at is None or time.monotonic() > expires_at:
            return web.Response(text="Unauthorized: Token Expired or Invalid", status=401)

        if not self.aws_preempted:
            return web.Response(text="Not Found", status=404)

        deadline_dt = datetime.now(timezone.utc) + timedelta(seconds=self.aws_notice_seconds)
        deadline_iso = deadline_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = {
            "action": "terminate",
            "time": deadline_iso,
        }
        return web.json_response(payload)

    # --- GCP Metadata ---

    async def handle_gcp_preempted(self, request: web.Request) -> web.Response:
        """GET /computeMetadata/v1/instance/preempted -> 'FALSE' or 'TRUE'."""
        await self._apply_delay()
        flavor = request.headers.get("Metadata-Flavor")
        if flavor != "Google":
            return web.Response(text="Missing required Metadata-Flavor: Google header", status=403)

        body = "TRUE" if self.gcp_preempted else "FALSE"
        return web.Response(text=body, content_type="text/plain")

    async def handle_gcp_maintenance(self, request: web.Request) -> web.Response:
        """GET /computeMetadata/v1/instance/maintenance-event -> 'NONE' or 'TERMINATE_ON_HOST_MAINTENANCE'."""
        await self._apply_delay()
        flavor = request.headers.get("Metadata-Flavor")
        if flavor != "Google":
            return web.Response(text="Missing required Metadata-Flavor: Google header", status=403)

        status_text = "TERMINATE_ON_HOST_MAINTENANCE" if self.gcp_preempted else self.gcp_maintenance_event
        return web.Response(text=status_text, content_type="text/plain")

    # --- RunPod ---

    async def handle_runpod_preempt(self, request: web.Request) -> web.Response:
        """GET /runpod/preempt -> Status polling endpoint for RunPod."""
        await self._apply_delay()
        if self.runpod_preempted:
            return web.json_response({
                "preempted": True,
                "grace_period_seconds": self.runpod_notice_seconds,
            })
        return web.json_response({"preempted": False})

    # --- Chaos & Control API ---

    async def handle_chaos_trigger(self, request: web.Request) -> web.Response:
        """POST /chaos/trigger-preemption or /chaos/inject-preemption."""
        try:
            data = await request.json() if request.can_read_body else {}
        except Exception:
            data = {}

        provider = data.get("provider", "aws")
        notice_seconds = data.get("notice_seconds")
        self.trigger_preemption(provider=provider, notice_seconds=notice_seconds)

        return web.json_response({
            "status": "injected",
            "provider": provider,
            "aws_preempted": self.aws_preempted,
            "gcp_preempted": self.gcp_preempted,
            "runpod_preempted": self.runpod_preempted,
        })

    async def handle_chaos_reset(self, request: web.Request) -> web.Response:
        """POST /chaos/reset."""
        self.reset()
        return web.json_response({"status": "reset", "state": "healthy"})

    async def handle_chaos_network_delay(self, request: web.Request) -> web.Response:
        """POST /chaos/network-delay {"delay_ms": 150}."""
        data = await request.json()
        self.artificial_delay_ms = float(data.get("delay_ms", 0.0))
        return web.json_response({"status": "delay_configured", "delay_ms": self.artificial_delay_ms})

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /status -> Current simulator configuration."""
        return web.json_response({
            "aws_preempted": self.aws_preempted,
            "aws_notice_seconds": self.aws_notice_seconds,
            "gcp_preempted": self.gcp_preempted,
            "gcp_notice_seconds": self.gcp_notice_seconds,
            "runpod_preempted": self.runpod_preempted,
            "runpod_notice_seconds": self.runpod_notice_seconds,
            "artificial_delay_ms": self.artificial_delay_ms,
        })

    # -----------------------------------------------------------------------
    # Server Lifecycle
    # -----------------------------------------------------------------------

    def create_app(self) -> web.Application:
        app = web.Application()
        # AWS
        app.router.add_put("/latest/api/token", self.handle_aws_token)
        app.router.add_get("/latest/meta-data/spot/instance-action", self.handle_aws_spot_action)

        # GCP
        app.router.add_get("/computeMetadata/v1/instance/preempted", self.handle_gcp_preempted)
        app.router.add_get("/computeMetadata/v1/instance/maintenance-event", self.handle_gcp_maintenance)

        # RunPod
        app.router.add_get("/runpod/preempt", self.handle_runpod_preempt)

        # Chaos Control
        app.router.add_post("/chaos/trigger-preemption", self.handle_chaos_trigger)
        app.router.add_post("/chaos/inject-preemption", self.handle_chaos_trigger)
        app.router.add_post("/chaos/reset", self.handle_chaos_reset)
        app.router.add_post("/chaos/network-delay", self.handle_chaos_network_delay)
        app.router.add_get("/status", self.handle_status)

        return app

    async def start(self) -> None:
        """Starts the simulator server."""
        self._app = self.create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("Cloud Chaos & Metadata Simulator running on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stops the simulator server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("Cloud Metadata Simulator stopped.")
