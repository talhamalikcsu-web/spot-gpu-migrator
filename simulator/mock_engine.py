"""
Mock LLM Streaming Engine.

Emulates vLLM / HuggingFace text-generation worker producing SSE token streams with
monotonic sequence IDs. Supports starting generation from a sequence offset, enabling
seamless state resumption on Standby nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from aiohttp import web

logger = logging.getLogger("sgm.simulator.mock_engine")

DEFAULT_GENERATION_CORPUS = [
    "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " and",
    " runs", " across", " the", " wide", " open", " fields", " towards", " the", " green", " mountains.",
    " The", " sun", " sets", " behind", " the", " horizon,", " casting", " golden", " light", " upon",
    " the", " tranquil", " valley", " below.", " Deep", " learning", " models", " require", " substantial", " compute,",
    " making", " spot", " GPU", " migration", " essential", " for", " cost-effective", " large", " scale", " AI."
]


class MockLLMEngine:
    """
    Simulated LLM inference worker listening on port 8001 (Active) or 8002 (Standby).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        inter_token_delay_ms: float = 25.0,
        corpus: Optional[List[str]] = None,
        node_role: str = "active",
    ) -> None:
        self.host = host
        self.port = port
        self.inter_token_delay_s = inter_token_delay_ms / 1000.0
        self.corpus = corpus or DEFAULT_GENERATION_CORPUS
        self.node_role = node_role

        self.is_paused: bool = False
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_post("/v1/completions", self.handle_chat_completions)
        app.router.add_post("/pause", self.handle_pause)
        app.router.add_post("/abort", self.handle_abort)
        app.router.add_post("/resume", self.handle_resume)
        app.router.add_get("/health", self.handle_health)
        return app

    async def handle_pause(self, request: web.Request) -> web.Response:
        self.is_paused = True
        logger.info("Engine on port %d [%s] PAUSED", self.port, self.node_role)
        return web.json_response({"status": "paused"})

    async def handle_abort(self, request: web.Request) -> web.Response:
        self.is_paused = True
        data = await request.json() if request.can_read_body else {}
        req_id = data.get("request_id", "unknown")
        logger.info("Engine on port %d [%s] ABORTED req=%s", self.port, self.node_role, req_id)
        return web.json_response({"status": "aborted", "request_id": req_id})

    async def handle_resume(self, request: web.Request) -> web.Response:
        self.is_paused = False
        logger.info("Engine on port %d [%s] RESUMED", self.port, self.node_role)
        return web.json_response({"status": "resumed"})

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "role": self.node_role,
            "port": self.port,
            "paused": self.is_paused,
        })

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """
        Generates SSE stream of tokens with monotonic sequence IDs.
        Supports start_seq_id parameter to resume generation seamlessly.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        is_streaming = body.get("stream", True)
        request_id = body.get("request_id", f"mock-{int(time.time() * 1000)}")
        start_seq = int(body.get("start_seq_id", 0))
        max_tokens = int(body.get("max_tokens", len(self.corpus)))

        logger.info(
            "Engine [%s:%d] received completion req=%s (start_seq=%d, stream=%s)",
            self.node_role,
            self.port,
            request_id,
            start_seq,
            is_streaming,
        )

        # Slice corpus starting from start_seq
        if start_seq < len(self.corpus):
            tokens_to_emit = self.corpus[start_seq : start_seq + max_tokens]
        else:
            tokens_to_emit = []

        if not is_streaming:
            full_text = "".join(tokens_to_emit)
            return web.json_response({
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": full_text}}],
            })

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        current_seq = start_seq
        for idx, token in enumerate(tokens_to_emit):
            if self.is_paused:
                logger.info("Engine [%s:%d] stream paused at token seq %d", self.node_role, self.port, current_seq)
                break

            is_last = idx == len(tokens_to_emit) - 1
            payload = {
                "id": f"chatcmpl-{request_id}",
                "choices": [
                    {
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": "stop" if is_last else None,
                    }
                ],
                "token": token,
                "seq": current_seq,
            }
            line = f"id: {current_seq}\ndata: {json.dumps(payload)}\n\n"
            await response.write(line.encode("utf-8"))

            current_seq += 1
            if self.inter_token_delay_s > 0:
                await asyncio.sleep(self.inter_token_delay_s)

        # Write [DONE] if not paused
        if not self.is_paused:
            await response.write(b"data: [DONE]\n\n")

        return response

    async def start(self) -> None:
        self._app = self.create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("Mock LLM Engine [%s] listening on http://%s:%d", self.node_role, self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        logger.info("Mock LLM Engine [%s] stopped.", self.node_role)
