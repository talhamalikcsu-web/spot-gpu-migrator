"""
SGM Ingress Proxy Server.

Listens on Port 8000 for OpenAI-compatible LLM inference requests, streams Server-Sent
Events (SSE) downstream to clients, tracks monotonic token sequences, and orchestrates
zero-downtime mid-stream socket handover on spot preemption.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
from aiohttp import web

from daemon.models import (
    HandoverReadyPayload,
    PreemptionAlertPayload,
    TokenChunk,
)
from proxy.buffer import TokenDeduplicator
from proxy.upstream import StreamMultiplexer

logger = logging.getLogger("sgm.proxy.ingress")


class IngressProxy:
    """
    Public entrypoint for inference requests that dynamically routes traffic
    between Active and Standby spot nodes.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        active_upstream_url: str = "http://127.0.0.1:8001",
        standby_upstream_url: str = "http://127.0.0.1:8002",
    ) -> None:
        self.host = host
        self.port = port
        self.active_upstream_url = active_upstream_url.rstrip("/")
        self.standby_upstream_url = standby_upstream_url.rstrip("/")
        
        self.deduplicator = TokenDeduplicator()
        self.multiplexer = StreamMultiplexer(deduplicator=self.deduplicator)

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._client_session: Optional[aiohttp.ClientSession] = None

        # Route table state
        self.is_preemption_active: bool = False
        self.active_node_id: str = "node-active"
        self.standby_node_id: str = "node-standby"
        self.in_flight_requests: Dict[str, Dict[str, Any]] = {}

    async def _get_client_session(self) -> aiohttp.ClientSession:
        if self._client_session is None or self._client_session.closed:
            # High socket keep-alive, no read timeout for long streaming
            client_timeout = aiohttp.ClientTimeout(total=None, sock_connect=5.0)
            self._client_session = aiohttp.ClientSession(timeout=client_timeout)
        return self._client_session

    def create_app(self) -> web.Application:
        """Configures aiohttp web application routes."""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_post("/v1/completions", self.handle_completions)
        app.router.add_post("/internal/preemption-alert", self.handle_preemption_alert)
        app.router.add_post("/internal/handover-ready", self.handle_handover_ready)
        app.router.add_post("/internal/handover-failed", self.handle_handover_failed)
        app.router.add_get("/internal/active-requests", self.handle_get_active_requests)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/status", self.handle_status)
        return app

    # -----------------------------------------------------------------------
    # OpenAI Inference Streaming Handlers
    # -----------------------------------------------------------------------

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Handles POST /v1/chat/completions with stream=True or False."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        is_streaming = body.get("stream", True)
        request_id = str(body.get("request_id") or f"req-{uuid.uuid4()}")
        body["request_id"] = request_id

        # Record in-flight request
        self.in_flight_requests[request_id] = {
            "body": body,
            "start_time": time.time(),
        }

        if not is_streaming:
            return await self._handle_non_streaming(request, body)

        # Prepare SSE StreamResponse to downstream client
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        # Choose initial upstream based on preemption state
        initial_url = self.standby_upstream_url if self.is_preemption_active else self.active_upstream_url
        logger.info(
            "Proxy routing chat request %s to %s (preemption_active=%s)",
            request_id,
            initial_url,
            self.is_preemption_active,
        )

        client_http = await self._get_client_session()

        # Active node generator
        active_gen = self._sse_chunk_generator(
            url=f"{initial_url}/v1/chat/completions",
            body=body,
            request_id=request_id,
            client_session=client_http,
        )

        # Standby node generator factory (invoked on preemption)
        async def standby_factory() -> AsyncIterator[TokenChunk]:
            # Standby resumes from last flushed sequence
            last_flushed = self.deduplicator.get_last_flushed(request_id)
            standby_body = dict(body)
            standby_body["start_seq_id"] = last_flushed + 1
            standby_body["is_resume"] = True
            logger.info(
                "Standby generator starting for %s at sequence %d",
                request_id,
                last_flushed + 1,
            )
            return self._sse_chunk_generator(
                url=f"{self.standby_upstream_url}/v1/chat/completions",
                body=standby_body,
                request_id=request_id,
                client_session=client_http,
            )

        try:
            # Stream downstream with multiplexer handling handover
            async for sse_bytes in self.multiplexer.stream_with_handover(
                request_id=request_id,
                active_upstream=active_gen,
                standby_factory=standby_factory,
            ):
                await response.write(sse_bytes)
        except Exception as exc:
            logger.error("Error during downstream write for req %s: %s", request_id, exc)
        finally:
            self.in_flight_requests.pop(request_id, None)

        return response

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        """Alias for /v1/completions."""
        return await self.handle_chat_completions(request)

    async def _handle_non_streaming(self, request: web.Request, body: dict) -> web.Response:
        client_http = await self._get_client_session()
        target_url = self.standby_upstream_url if self.is_preemption_active else self.active_upstream_url
        try:
            async with client_http.post(f"{target_url}/v1/chat/completions", json=body) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)
        except Exception as exc:
            logger.error("Non-streaming forward error: %s", exc)
            return web.json_response({"error": str(exc)}, status=503)

    async def _sse_chunk_generator(
        self,
        url: str,
        body: dict,
        request_id: str,
        client_session: aiohttp.ClientSession,
    ) -> AsyncIterator[TokenChunk]:
        """
        Connects to an upstream engine, reads the SSE stream line-by-line,
        and yields parsed TokenChunk objects.
        """
        try:
            async with client_session.post(url, json=body) as resp:
                if resp.status != 200:
                    logger.error("Upstream %s returned HTTP %d for req %s", url, resp.status, request_id)
                    return

                current_seq = 0
                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue

                    if line.startswith("id:"):
                        try:
                            current_seq = int(line[3:].strip())
                        except ValueError:
                            pass
                        continue

                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            break

                        try:
                            data = json.loads(payload_str)
                            token_str = ""
                            # Check standard delta choices
                            choices = data.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                                token_str = delta.get("content", "")
                            elif "token" in data:
                                token_str = data["token"]

                            seq_id = int(data.get("seq", current_seq))

                            yield TokenChunk(
                                request_id=request_id,
                                sequence_id=seq_id,
                                token=token_str,
                                raw_bytes=line_bytes,
                            )
                            current_seq = seq_id + 1
                        except Exception as parse_err:
                            logger.debug("Failed parsing SSE data line '%s': %s", line, parse_err)

        except Exception as exc:
            logger.warning("Upstream SSE read closed for %s (%s): %s", request_id, url, exc)

    # -----------------------------------------------------------------------
    # Internal Handover Signaling Endpoints
    # -----------------------------------------------------------------------

    async def handle_preemption_alert(self, request: web.Request) -> web.Response:
        """
        Received from Active Node Daemon.
        Marks Active node as terminating and transitions in-flight streams to BUFFERING.
        """
        data = await request.json()
        logger.critical("PROXY RECEIVED PREEMPTION ALERT: %s", data)
        self.is_preemption_active = True
        cutoff_seq = int(data.get("cutoff_sequence_id", -1))

        # Signal preemption to multiplexer for all active requests
        for req_id in list(self.in_flight_requests.keys()):
            self.multiplexer.trigger_preemption(req_id, cutoff_seq)

        return web.json_response({
            "status": "buffering_activated",
            "active_streams": len(self.in_flight_requests),
        })

    async def handle_handover_ready(self, request: web.Request) -> web.Response:
        """
        Received from Standby Node Daemon.
        Signals that Standby node is warmed and ready to take over a specific request.
        """
        data = await request.json()
        request_id = data.get("request_id")
        resume_port = data.get("resume_port", 8002)
        logger.info("PROXY RECEIVED HANDOVER READY: req_id=%s, resume_port=%d", request_id, resume_port)

        if request_id:
            self.multiplexer.trigger_handover_ready(request_id)

        return web.json_response({"status": "handover_dispatched"})

    async def handle_handover_failed(self, request: web.Request) -> web.Response:
        """Received if P2P transfer between nodes failed."""
        data = await request.json()
        logger.error("PROXY RECEIVED HANDOVER FAILED NOTIFICATION: %s", data)
        # In fallback mode, fresh requests route to standby without state replay
        return web.json_response({"status": "fallback_registered"})

    async def handle_get_active_requests(self, request: web.Request) -> web.Response:
        """Returns in-flight requests for node daemon state synchronization."""
        reqs = []
        for req_id, info in self.in_flight_requests.items():
            last_seq = self.deduplicator.get_last_flushed(req_id)
            body = info.get("body", {}) if isinstance(info, dict) else {}
            prompt = body.get("prompt") or str(body.get("messages", ""))
            reqs.append({
                "request_id": req_id,
                "prompt": prompt,
                "last_flushed_sequence_id": last_seq,
                "model": body.get("model", "mock-llm"),
            })
        return web.json_response({"requests": reqs})

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "preemption_active": self.is_preemption_active,
            "in_flight_requests": len(self.in_flight_requests),
        })

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response({
            "active_upstream": self.active_upstream_url,
            "standby_upstream": self.standby_upstream_url,
            "preemption_active": self.is_preemption_active,
            "in_flight_count": len(self.in_flight_requests),
            "in_flight_request_ids": list(self.in_flight_requests.keys()),
        })

    async def start(self) -> None:
        """Starts the Ingress Proxy aiohttp server."""
        self._app = self.create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("SGM Ingress Proxy listening on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stops the Ingress Proxy and frees resources."""
        if self._client_session and not self._client_session.closed:
            await self._client_session.close()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("SGM Ingress Proxy stopped.")


def main() -> None:
    """CLI and container entrypoint for SGM Ingress Reverse Proxy."""
    parser = argparse.ArgumentParser(
        prog="sgm-proxy",
        description="Spot GPU Migrator (SGM) Ingress Reverse Proxy",
    )
    parser.add_argument("--host", default=os.environ.get("SGM_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SGM_PROXY_PORT", "8000")))
    parser.add_argument("--active-upstream", default=os.environ.get("SGM_ACTIVE_UPSTREAM_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--standby-upstream", default=os.environ.get("SGM_STANDBY_UPSTREAM_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--log-level", default=os.environ.get("SGM_LOG_LEVEL", "INFO"))

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    proxy = IngressProxy(
        host=args.host,
        port=args.port,
        active_upstream_url=args.active_upstream,
        standby_upstream_url=args.standby_upstream,
    )

    async def run_proxy() -> None:
        await proxy.start()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Termination signal received. Shutting down Ingress Proxy...")
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
            await proxy.stop()

    try:
        asyncio.run(run_proxy())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Proxy shutdown complete.")


if __name__ == "__main__":
    main()

