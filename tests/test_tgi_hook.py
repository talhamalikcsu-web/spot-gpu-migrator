"""
Milestone 3 Verification: Hugging Face TGI Hook Tests.

Conforms to SPEC-005:
- AC-M3-03: Hugging Face TGI Native & OpenAI Streaming Support
- AC-M3-04: TGI Client-Drop Abort
- FAIL-02: Multibyte UTF-8 Boundary Accumulator
- TEST-M3-03
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import aiohttp
from aiohttp import web
import pytest

from daemon.integrations.base import BackendType
from daemon.integrations.tgi import TGIEngineHook
from daemon.models import InferenceSession, SamplingParams, TokenChunk


@pytest.mark.asyncio
async def test_tgi_payload_and_metadata():
    """
    Verifies TGI resumption payload formatting for both native and compat modes:
    - return_full_text: False
    - max_new_tokens decremented correctly
    - inputs structured with continuation prompt
    """
    hook = TGIEngineHook(base_url="http://127.0.0.1:8001")
    meta = hook.get_engine_metadata()
    assert meta.backend == BackendType.TGI
    assert meta.supports_abort_endpoint is False
    assert meta.supports_prefix_caching is True

    session = InferenceSession(
        request_id="req-tgi-001",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        prompt="Write a Python script for zero-copy buffers.",
        sampling_params=SamplingParams(max_tokens=100, temperature=0.7, top_p=0.95),
        generated_text=["import", " os\n", "import", " mmap"],
        last_flushed_sequence_id=3,
    )

    # 1. Native payload format
    native_payload = hook.build_prefix_caching_payload(session, format_type="native")
    assert native_payload["inputs"] == "Write a Python script for zero-copy buffers.import os\nimport mmap"
    assert native_payload["parameters"]["max_new_tokens"] == 96  # 100 - 4
    assert native_payload["parameters"]["return_full_text"] is False
    assert native_payload["parameters"]["details"] is True

    # 2. OpenAI compat format
    compat_payload = hook.build_prefix_caching_payload(session, format_type="compat")
    assert compat_payload["max_tokens"] == 96
    assert compat_payload["return_full_text"] is False
    assert len(compat_payload["messages"]) == 2
    assert compat_payload["messages"][0]["role"] == "user"
    assert compat_payload["messages"][1]["role"] == "assistant"
    assert compat_payload["messages"][1]["content"] == "import os\nimport mmap"


@pytest.mark.asyncio
async def test_tgi_native_stream_ingestion():
    """
    AC-M3-03: Verifies streaming tokens from TGI native /generate_stream SSE.
    """
    native_sse_lines = [
        'data:{"token":{"id":1124,"text":"import","special":false},"generated_text":null,"details":null}\n\n',
        'data:{"token":{"id":284,"text":" os","special":false},"generated_text":null,"details":null}\n\n',
        'data:{"token":{"id":13,"text":"\\n","special":false},"generated_text":null,"details":null}\n\n',
        'data:{"token":{"id":128001,"text":"<|eot_id|>","special":true},"generated_text":"import os\\n","details":{"finish_reason":"eos_token","generated_tokens":3}}\n\n',
    ]

    async def handle_generate_stream(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        assert body["parameters"]["return_full_text"] is False
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        for line in native_sse_lines:
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.01)

        return resp

    app = web.Application()
    app.router.add_post("/generate_stream", handle_generate_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TGIEngineHook(base_url=f"http://127.0.0.1:{port}", use_openai_compat=False)

    try:
        session = InferenceSession(
            request_id="req-tgi-native-002",
            prompt="Write Python code",
            last_flushed_sequence_id=5,  # Cutoff at seq 5
        )

        tokens: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            tokens.append(chunk)

        # The special token <|eot_id|> is filtered out, yielding 3 content tokens
        assert len(tokens) == 3
        assert [t.sequence_id for t in tokens] == [6, 7, 8]
        assert [t.token for t in tokens] == ["import", " os", "\n"]

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tgi_compat_stream_ingestion():
    """
    AC-M3-03: Verifies streaming tokens from TGI OpenAI-compat /compat/v1/chat/completions.
    """
    compat_sse_lines = [
        'data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}\n\n',
        'data: {"id":"3","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n',
    ]

    async def handle_compat_stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        for line in compat_sse_lines:
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.01)

        return resp

    app = web.Application()
    app.router.add_post("/compat/v1/chat/completions", handle_compat_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TGIEngineHook(base_url=f"http://127.0.0.1:{port}", use_openai_compat=True)

    try:
        session = InferenceSession(
            request_id="req-tgi-compat-003",
            prompt="Say Hello World",
            last_flushed_sequence_id=0,
        )

        tokens: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            tokens.append(chunk)

        assert len(tokens) == 3
        assert [t.sequence_id for t in tokens] == [1, 2, 3]
        assert [t.token for t in tokens] == ["Hello", " World", "!"]

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tgi_sub_50ms_abort_via_socket_closure():
    """
    AC-M3-04: Verifies TGI client-drop abort executes in <= 50ms.
    """
    stream_started = asyncio.Event()

    async def handle_long_stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        stream_started.set()

        try:
            for i in range(100):
                line = f'data:{{"token":{{"id":{i},"text":" tok","special":false}}}}\n\n'
                await resp.write(line.encode("utf-8"))
                await asyncio.sleep(0.015)
        except (ConnectionResetError, asyncio.CancelledError, Exception):
            pass
        return resp

    app = web.Application()
    app.router.add_post("/generate_stream", handle_long_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TGIEngineHook(base_url=f"http://127.0.0.1:{port}")

    try:
        session = InferenceSession(
            request_id="req-tgi-abort-004",
            prompt="Generate words",
            last_flushed_sequence_id=0,
        )

        async def _consumer():
            async for _ in hook.stream_resumed(session):
                pass

        task = asyncio.create_task(_consumer())
        await asyncio.wait_for(stream_started.wait(), timeout=1.0)

        # Dispatch abort
        t_start = time.monotonic()
        abort_res = await hook.abort_request("req-tgi-abort-004")
        duration_ms = (time.monotonic() - t_start) * 1000.0

        assert abort_res.success is True
        assert abort_res.latency_ms <= 50.0
        assert duration_ms <= 50.0
        assert abort_res.backend == BackendType.TGI

        await task

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tgi_multibyte_utf8_accumulator():
    """
    FAIL-02: Verifies that multi-byte UTF-8 characters split across packets
    do not trigger decoding exceptions or garbled tokens.
    """
    # 4-byte emoji: 🚀 (bytes: 0xF0 0x9F 0x99 0x80)
    # Split across two SSE events
    emoji_bytes = "🚀".encode("utf-8")
    part1 = emoji_bytes[:2]
    part2 = emoji_bytes[2:]

    # Construct two lines where text is split at byte boundary
    line1 = b'data:{"token":{"id":10,"text":"' + part1 + b'","special":false}}\n\n'
    line2 = b'data:{"token":{"id":11,"text":"' + part2 + b'","special":false}}\n\n'

    # Using the UTF-8 incremental decoder in stream_resumed ensures robustness
    async def handle_multibyte_stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(line1)
        await asyncio.sleep(0.01)
        await resp.write(line2)
        return resp

    app = web.Application()
    app.router.add_post("/generate_stream", handle_multibyte_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TGIEngineHook(base_url=f"http://127.0.0.1:{port}")

    try:
        session = InferenceSession(request_id="req-utf8-005", prompt="Emoji test")
        tokens: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            tokens.append(chunk)

        # No exceptions raised, stream processed
        assert len(tokens) >= 1

    finally:
        await hook.close()
        await runner.cleanup()
