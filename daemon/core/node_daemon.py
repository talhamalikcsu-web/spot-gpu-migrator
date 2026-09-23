"""
Node Daemon (SGM Core Supervisor).

Orchestrates the Node Lifecycle Finite State Machine (FSM), coordinates watchdog
notifications, manages in-flight session snapshots, triggers P2P state streaming,
and signals the Ingress Proxy.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import signal
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

import aiohttp
from aiohttp import web

from daemon.models import (
    HandoverAck,
    HandoverReadyPayload,
    InferenceSession,
    NodeLifecycleState,
    PreemptionAlertPayload,
    PreemptionEvent,
    SGMError,
)
from daemon.integrations.base import AbstractInferenceEngineHook
from daemon.integrations.factory import EngineHookFactory
from daemon.integrations.vllm import VLLMInferenceEngineHook
from daemon.transport.p2p_client import P2PClient
from daemon.transport.p2p_server import P2PServer
from daemon.watchdog.base import AbstractPreemptionWatchdog

logger = logging.getLogger("sgm.daemon.node")


class NodeDaemon:
    """
    Host supervisor and cluster participant managing the spot node lifecycle.
    """

    def __init__(
        self,
        node_id: str = "spot-node-01",
        role: str = "active",  # "active" or "standby"
        control_port: int = 9001,
        p2p_port: int = 9002,
        standby_host: str = "127.0.0.1",
        standby_p2p_port: int = 9002,
        proxy_url: str = "http://127.0.0.1:8000",
        watchdog: Optional[AbstractPreemptionWatchdog] = None,
        engine_url: str = "http://127.0.0.1:8001",
        engine_hook: Optional[AbstractInferenceEngineHook] = None,
        engine_type: Optional[str] = "auto",
    ) -> None:
        self.node_id = node_id
        self.role = role.lower()
        self.control_port = control_port
        self.p2p_port = p2p_port
        self.standby_host = standby_host
        self.standby_p2p_port = standby_p2p_port
        self.proxy_url = proxy_url.rstrip("/")
        self.engine_url = engine_url.rstrip("/")
        self.watchdog = watchdog
        self.engine_type = (engine_type or "auto").lower()
        self._custom_engine_hook = engine_hook is not None
        if self._custom_engine_hook:
            self.engine_hook = engine_hook
        elif self.engine_type != "auto":
            self.engine_hook = EngineHookFactory.create_sync(self.engine_type, engine_url=self.engine_url)
        else:
            self.engine_hook = VLLMInferenceEngineHook(base_url=self.engine_url)


        # FSM State
        self.state: NodeLifecycleState = NodeLifecycleState.HEALTHY
        self._state_lock = asyncio.Lock()

        # Session registry
        self.active_sessions: Dict[str, InferenceSession] = {}

        # Components
        self.p2p_client = P2PClient(
            target_host=self.standby_host,
            target_port=self.standby_p2p_port,
            node_id=self.node_id,
        )
        self.p2p_server: Optional[P2PServer] = None
        if self.role == "standby":
            self.p2p_server = P2PServer(
                host="0.0.0.0",
                port=self.p2p_port,
                node_id=self.node_id,
            )
            self.p2p_server.register_session_callback(self._on_standby_sessions_received)

        # Control API server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Hooks
        self.on_state_change: Optional[Callable[[NodeLifecycleState], None]] = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def set_state(self, new_state: NodeLifecycleState) -> None:
        """Transitions FSM state safely under lock."""
        async with self._state_lock:
            old_state = self.state
            if old_state == new_state:
                return
            logger.info("Node %s state transition: %s -> %s", self.node_id, old_state.value, new_state.value)
            self.state = new_state
            if self.on_state_change:
                self.on_state_change(new_state)

    def register_session(self, session: InferenceSession) -> None:
        """Registers an in-flight inference session."""
        self.active_sessions[session.request_id] = session
        logger.debug("Registered session %s (total active: %d)", session.request_id, len(self.active_sessions))

    def update_session_token(
        self,
        request_id: str,
        token_id: int,
        token_str: str,
        sequence_id: int,
    ) -> None:
        """Updates token generation progress for an active session."""
        session = self.active_sessions.get(request_id)
        if session:
            session.generated_tokens.append(token_id)
            session.generated_text.append(token_str)
            session.total_tokens_generated += 1
            session.last_flushed_sequence_id = sequence_id

    def complete_session(self, request_id: str) -> None:
        """Removes a finished session from active tracking."""
        self.active_sessions.pop(request_id, None)

    # -----------------------------------------------------------------------
    # Preemption Handover Orchestration
    # -----------------------------------------------------------------------

    async def on_preemption(self, event: PreemptionEvent) -> None:
        """
        Async callback invoked by PreemptionWatchdog when hypervisor notice is detected.
        Executes zero-downtime handover sequence within latency SLA.
        """
        if self.state != NodeLifecycleState.HEALTHY:
            logger.warning("Preemption callback received while in state %s; ignoring duplicate.", self.state)
            return

        start_time = time.monotonic()
        logger.critical(
            "PREEMPTION EVENT DETECTED: provider=%s, deadline=%.1fs. Commencing migration sequence.",
            event.provider,
            event.deadline_seconds,
        )

        # Step 1: PREEMPTION_DETECTED
        await self.set_state(NodeLifecycleState.PREEMPTION_DETECTED)

        # Synchronize active sessions from proxy if empty
        if not self.active_sessions and self.proxy_url:
            try:
                http_sess = await self._get_http_session()
                async with http_sess.get(f"{self.proxy_url}/internal/active-requests", timeout=0.300) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for r in data.get("requests", []):
                            req_id = r["request_id"]
                            self.active_sessions[req_id] = InferenceSession(
                                request_id=req_id,
                                prompt=r.get("prompt", ""),
                                last_flushed_sequence_id=r.get("last_flushed_sequence_id", -1),
                                model=r.get("model", "mock-llm"),
                            )
                        logger.info("Synchronized %d active sessions from Ingress Proxy", len(self.active_sessions))
            except Exception as sync_err:
                logger.debug("Could not sync active requests from proxy: %s", sync_err)

        # Find maximum cutoff sequence across active sessions
        cutoff_seq = -1
        for s in self.active_sessions.values():
            if s.last_flushed_sequence_id > cutoff_seq:
                cutoff_seq = s.last_flushed_sequence_id

        # Step 2: Notify Ingress Proxy
        await self._notify_proxy_preemption(event, cutoff_seq)

        # Step 3: BUFFERING_INGRESS (pause engine at clean boundary)
        await self.set_state(NodeLifecycleState.BUFFERING_INGRESS)
        await self._pause_local_engine()

        # Step 4: STATE_STREAMING (P2P transmission to Standby node)
        await self.set_state(NodeLifecycleState.STATE_STREAMING)
        try:
            sessions_to_migrate = list(self.active_sessions.values())
            logger.info("Streaming %d active sessions to Standby node %s:%d", len(sessions_to_migrate), self.standby_host, self.standby_p2p_port)
            
            ack = await self.p2p_client.send_sessions(sessions_to_migrate, timeout=3.0)
            logger.info("Standby ACK received: accepted=%s", ack.accepted_request_ids)

            # Step 5: HANDOVER_COMPLETE
            await self.set_state(NodeLifecycleState.HANDOVER_COMPLETE)
            
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.info("Handover sequence finished in %.2f ms (Target: <= 2500ms)", elapsed_ms)

        except Exception as exc:
            logger.error("P2P state transfer failed: %s. Proxy fallback required.", exc)
            await self._notify_proxy_failure(exc)
            raise

        # Step 6: TERMINATED
        await self.set_state(NodeLifecycleState.TERMINATED)
        await self._teardown_resources()

    async def _notify_proxy_preemption(self, event: PreemptionEvent, cutoff_seq: int) -> None:
        """Broadcasts preemption alert to SGM Ingress Proxy."""
        session = await self._get_http_session()
        payload = PreemptionAlertPayload(
            node_id=self.node_id,
            provider=event.provider,
            deadline_seconds=event.deadline_seconds,
            cutoff_sequence_id=cutoff_seq,
            active_request_ids=list(self.active_sessions.keys()),
        )
        url = f"{self.proxy_url}/internal/preemption-alert"
        try:
            async with session.post(url, json=payload.model_dump() if hasattr(payload, "model_dump") else payload.dict(), timeout=0.300) as resp:
                if resp.status == 200:
                    logger.info("Proxy successfully acknowledged preemption alert")
                else:
                    logger.warning("Proxy returned non-200 for preemption alert: HTTP %d", resp.status)
        except Exception as exc:
            logger.error("Failed to notify proxy of preemption: %s", exc)

    async def _notify_proxy_failure(self, error: Exception) -> None:
        """Signals proxy of handover failure so it can trigger prompt replay fallback."""
        session = await self._get_http_session()
        url = f"{self.proxy_url}/internal/handover-failed"
        payload = {
            "node_id": self.node_id,
            "error": str(error),
            "request_ids": list(self.active_sessions.keys()),
        }
        try:
            async with session.post(url, json=payload, timeout=0.5) as resp:
                logger.warning("Notified proxy of handover failure: HTTP %d", resp.status)
        except Exception as exc:
            logger.error("Could not notify proxy of handover failure: %s", exc)

    async def _pause_local_engine(self) -> None:
        """Sends pause / abort command to local LLM inference engine."""
        logger.info("Freezing local inference engine for node %s...", self.node_id)
        if self.engine_hook:
            for req_id in list(self.active_sessions.keys()):
                try:
                    await self.engine_hook.abort_request(req_id)
                except Exception as exc:
                    logger.debug("Failed aborting req %s via engine hook: %s", req_id, exc)

        session = await self._get_http_session()
        url = f"{self.engine_url}/pause"
        try:
            async with session.post(url, timeout=0.200) as resp:
                logger.debug("Local engine pause response: HTTP %d", resp.status)
        except Exception as exc:
            logger.debug("No engine response to pause (may be mock): %s", exc)

    async def _on_standby_sessions_received(self, sessions: List[InferenceSession]) -> None:
        """Standby hook: called when P2PServer receives session states."""
        logger.info("Standby node received %d sessions. Warming inference and notifying proxy...", len(sessions))
        session = await self._get_http_session()

        for s in sessions:
            self.active_sessions[s.request_id] = s
            # Notify Ingress Proxy that Standby is ready for this request
            ready_payload = HandoverReadyPayload(
                request_id=s.request_id,
                starting_sequence_id=s.last_flushed_sequence_id + 1,
                resume_host="127.0.0.1",
                resume_port=8002,
            )
            url = f"{self.proxy_url}/internal/handover-ready"
            try:
                dump = ready_payload.model_dump() if hasattr(ready_payload, "model_dump") else ready_payload.dict()
                async with session.post(url, json=dump, timeout=0.500) as resp:
                    logger.info("Notified proxy of handover readiness for req %s (HTTP %d)", s.request_id, resp.status)
            except Exception as exc:
                logger.error("Failed to notify proxy of handover readiness: %s", exc)

    async def _teardown_resources(self) -> None:
        """Gracefully frees resources prior to spot node termination."""
        logger.info("Tearing down node resources...")
        if self.engine_hook and hasattr(self.engine_hook, "close"):
            try:
                await self.engine_hook.close()
            except Exception as exc:
                logger.debug("Error closing engine hook: %s", exc)
        if self.watchdog:
            await self.watchdog.stop()
        if self.p2p_server:
            await self.p2p_server.stop()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # -----------------------------------------------------------------------
    # Control REST API (Port 9001)
    # -----------------------------------------------------------------------

    def _setup_routes(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/status", self._handle_status)
        app.router.add_post("/sessions/register", self._handle_session_register)
        app.router.add_post("/webhook/runpod-terminate", self._handle_runpod_webhook)
        app.router.add_post("/control/preempt", self._handle_manual_preempt)
        return app

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy" if self.state != NodeLifecycleState.TERMINATED else "terminated",
            "node_id": self.node_id,
            "role": self.role,
            "state": self.state.value,
            "engine_type": self.engine_type,
        })

    async def _handle_status(self, request: web.Request) -> web.Response:
        return web.json_response({
            "node_id": self.node_id,
            "role": self.role,
            "state": self.state.value,
            "engine_type": self.engine_type,
            "active_sessions_count": len(self.active_sessions),
            "sessions": list(self.active_sessions.keys()),
        })

    async def _handle_session_register(self, request: web.Request) -> web.Response:
        data = await request.json()
        session = InferenceSession(**data)
        self.register_session(session)
        return web.json_response({"status": "registered", "request_id": session.request_id})

    async def _handle_runpod_webhook(self, request: web.Request) -> web.Response:
        data = await request.json()
        logger.info("Inbound RunPod termination webhook: %s", data)
        event = PreemptionEvent(
            provider="runpod",
            action="terminate",
            deadline_seconds=float(data.get("grace_period_seconds", 30)),
            metadata=data,
        )
        asyncio.create_task(self.on_preemption(event))
        return web.json_response({"status": "acknowledged"})

    async def _handle_manual_preempt(self, request: web.Request) -> web.Response:
        data = await request.json() if request.can_read_body else {}
        provider = data.get("provider", "manual")
        deadline = float(data.get("deadline_seconds", 30.0))
        event = PreemptionEvent(provider=provider, action="terminate", deadline_seconds=deadline)
        asyncio.create_task(self.on_preemption(event))
        return web.json_response({"status": "preemption_triggered", "deadline_seconds": deadline})

    async def start(self) -> None:
        """Starts the Node Daemon control server, P2P server (if Standby), and watchdog."""
        if not self._custom_engine_hook and self.engine_type == "auto":
            try:
                self.engine_hook = await EngineHookFactory.create("auto", engine_url=self.engine_url)
                logger.info("Node Daemon engine hook auto-configured: %s", type(self.engine_hook).__name__)
            except Exception as exc:
                logger.warning("Engine auto-detection failed (%s), defaulting to vLLM", exc)
                self.engine_hook = VLLMInferenceEngineHook(base_url=self.engine_url)

        self._app = self._setup_routes()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.control_port)
        await self._site.start()
        logger.info("Node Daemon Control API listening on port %d (role=%s)", self.control_port, self.role)

        if self.p2p_server:
            await self.p2p_server.start()

        if self.watchdog:
            self.watchdog.register_callback(self.on_preemption)
            await self.watchdog.start()

    async def stop(self) -> None:
        """Stops control API, watchdog, and transport servers."""
        await self._teardown_resources()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("Node Daemon %s stopped.", self.node_id)


def main() -> None:
    """CLI and container entrypoint for SGM Node Daemon."""
    parser = argparse.ArgumentParser(
        prog="sgm-node-daemon",
        description="Spot GPU Migrator (SGM) Node Daemon & Preemption Watchdog",
    )
    parser.add_argument("--node-id", default=os.environ.get("SGM_NODE_ID", f"spot-node-{uuid.uuid4().hex[:8]}"))
    parser.add_argument("--role", default=os.environ.get("SGM_ROLE", "active"), choices=["active", "standby"])
    parser.add_argument("--control-port", type=int, default=int(os.environ.get("SGM_CONTROL_PORT", "9001")))
    parser.add_argument("--p2p-port", type=int, default=int(os.environ.get("SGM_P2P_PORT", "9002")))
    parser.add_argument("--standby-host", default=os.environ.get("SGM_STANDBY_HOST", "127.0.0.1"))
    parser.add_argument("--standby-p2p-port", type=int, default=int(os.environ.get("SGM_STANDBY_P2P_PORT", "9002")))
    parser.add_argument("--proxy-url", default=os.environ.get("SGM_PROXY_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--engine-url", default=os.environ.get("SGM_ENGINE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--engine-type", default=os.environ.get("SGM_ENGINE_TYPE", "auto"), choices=["vllm", "tensorrt_llm", "tgi", "sglang", "mock", "auto"])

    parser.add_argument("--cloud-provider", default=os.environ.get("SGM_CLOUD_PROVIDER", "aws"), choices=["aws", "gcp", "runpod", "mock", "none"])
    parser.add_argument("--metadata-url", default=os.environ.get("SGM_METADATA_URL", None))
    parser.add_argument("--poll-interval-ms", type=int, default=int(os.environ.get("SGM_POLL_INTERVAL_MS", "250")))
    parser.add_argument("--timeout-ms", type=int, default=int(os.environ.get("SGM_TIMEOUT_MS", "100")))
    parser.add_argument("--log-level", default=os.environ.get("SGM_LOG_LEVEL", "INFO"))

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    # Instantiate watchdog if active role
    watchdog: Optional[AbstractPreemptionWatchdog] = None
    if args.role == "active" and args.cloud_provider != "none":
        if args.cloud_provider == "aws":
            from daemon.watchdog.aws import AWSPreemptionWatchdog
            watchdog = AWSPreemptionWatchdog(
                base_url=args.metadata_url or "http://169.254.169.254",
                poll_interval_ms=args.poll_interval_ms,
                timeout_ms=args.timeout_ms,
            )
        elif args.cloud_provider == "gcp":
            from daemon.watchdog.gcp import GCPPreemptionWatchdog
            watchdog = GCPPreemptionWatchdog(
                base_url=args.metadata_url or "http://metadata.google.internal",
                poll_interval_ms=args.poll_interval_ms,
                timeout_ms=args.timeout_ms,
            )
        elif args.cloud_provider == "runpod":
            from daemon.watchdog.runpod import RunPodPreemptionWatchdog
            watchdog = RunPodPreemptionWatchdog(
                status_url=args.metadata_url or "http://127.0.0.1:8080/pod/status",
                poll_interval_ms=args.poll_interval_ms,
                timeout_ms=args.timeout_ms,
            )

    daemon = NodeDaemon(
        node_id=args.node_id,
        role=args.role,
        control_port=args.control_port,
        p2p_port=args.p2p_port,
        standby_host=args.standby_host,
        standby_p2p_port=args.standby_p2p_port,
        proxy_url=args.proxy_url,
        engine_url=args.engine_url,
        watchdog=watchdog,
        engine_type=args.engine_type,
    )

    async def run_daemon() -> None:
        await daemon.start()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Termination signal received. Draining resources...")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except (NotImplementedError, AttributeError):
                signal.signal(sig, lambda *_: stop_event.set())

        try:
            await stop_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await daemon.stop()

    try:
        asyncio.run(run_daemon())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Node Daemon shutdown complete.")


if __name__ == "__main__":
    main()

