"""
Milestone 3 Verification: NVIDIA TensorRT-LLM (Triton) Engine Hook Tests.

Conforms to SPEC-005:
- AC-M3-01: Triton TensorRT-LLM stream ingestion & decoupled chunk parsing
- AC-M3-02: Triton TensorRT-LLM sub-50ms abort latency via socket drop
- TEST-M3-01 & TEST-M3-02
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
import pytest

from daemon.integrations.base import BackendType
from daemon.integrations.tensorrt_llm import TensorRTLLMEngineHook
from daemon.models import InferenceSession, SamplingParams, TokenChunk


@pytest.mark.asyncio
async def test_trtllm_payload_and_metadata():
    """
    Verifies TensorRT-LLM resumption payload generation:
    - exclude_input_from_output: True (mandatory)
    - Prompt continuation = original prompt + generated prefix tokens
    - Decremented max_tokens
    - Engine metadata descriptors
    """
    hook = TensorRTLLMEngineHook(base_url="http://127.0.0.1:8001", model_name="llama3-70b")
    meta = hook.get_engine_metadata()
    assert meta.backend == BackendType.TENSORRT_LLM
    assert meta.supports_abort_endpoint is False
    assert meta.supports_prefix_caching is True
    assert "v2/models/llama3-70b/generate_stream" in meta.native_streaming_endpoint

    session = InferenceSession(
        request_id="req-trt-001",
        model="llama3-70b",
        prompt="Explain distributed Raft consensus in detail.",
        sampling_params=SamplingParams(max_tokens=256, temperature=0.8, top_p=0.9),
        generated_text=[" Ra", "ft", " is", " a", " consensus"],
        last_flushed_sequence_id=4,
    )

    payload = hook.build_prefix_caching_payload(session)

    # Invariants verification
    assert payload["exclude_input_from_output"] is True, "exclude_input_from_output must be True to prevent echoing prefix"
    assert payload["text_input"] == "Explain distributed Raft consensus in detail. Raft is a consensus"
    assert payload["max_tokens"] == 251  # 256 - 5
    assert payload["stream"] is True
    assert payload["temperature"] == 0.8
    assert payload["top_p"] == 0.9
    assert payload["stop_words"] == ["<|eot_id|>", "</s>"]

    resume_desc = await hook.resume_request(session, resume_endpoint="http://10.0.0.2:8001")
    assert resume_desc["starting_sequence_id"] == 5
    assert resume_desc["backend"] == BackendType.TENSORRT_LLM
    assert resume_desc["url"] == "http://10.0.0.2:8001/v2/models/llama3-70b/generate_stream"


@pytest.mark.asyncio
async def test_trtllm_stream_ingestion_and_sequence_continuity():
    """
    AC-M3-01: Verifies Triton decoupled JSON chunk parsing and monotonic sequence IDs.
    Simulates a Triton server streaming chunked decoupled responses.
    """
    triton_chunks = [
        {"model_name": "tensorrt_llm", "sequence_end": False, "text_output": " algorithm"},
        {"model_name": "tensorrt_llm", "sequence_end": False, "text_output": " designed"},
        {"model_name": "tensorrt_llm", "sequence_end": False, "text_output": " for"},
        {"model_name": "tensorrt_llm", "sequence_end": True, "text_output": " fault tolerance."},
    ]

    async def handle_generate_stream(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        assert body["exclude_input_from_output"] is True
        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
        await resp.prepare(request)

        for chunk in triton_chunks:
            chunk_line = json.dumps(chunk) + "\n"
            await resp.write(chunk_line.encode("utf-8"))
            await asyncio.sleep(0.01)

        return resp

    app = web.Application()
    app.router.add_post("/v2/models/tensorrt_llm/generate_stream", handle_generate_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TensorRTLLMEngineHook(base_url=f"http://127.0.0.1:{port}", model_name="tensorrt_llm")

    try:
        session = InferenceSession(
            request_id="req-trt-stream-002",
            model="tensorrt_llm",
            prompt="Raft is a",
            last_flushed_sequence_id=9,  # Previous node flushed seqs 0..9
            generated_text=[" word"] * 10,
        )

        received_chunks: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            received_chunks.append(chunk)

        assert len(received_chunks) == 4
        # Verify monotonic sequence IDs starting from cutoff + 1 = 10
        expected_seqs = [10, 11, 12, 13]
        actual_seqs = [c.sequence_id for c in received_chunks]
        assert actual_seqs == expected_seqs

        assert received_chunks[0].token == " algorithm"
        assert received_chunks[-1].token == " fault tolerance."
        assert received_chunks[-1].is_final is True

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_trtllm_sub_50ms_abort_latency():
    """
    AC-M3-02: Measures Triton request abort latency under active streaming via socket drop.
    SLA hard ceiling: <= 50ms.
    """
    stream_started = asyncio.Event()
    client_disconnected = asyncio.Event()

    async def handle_long_stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
        await resp.prepare(request)
        stream_started.set()

        try:
            for i in range(100):
                chunk = {"model_name": "tensorrt_llm", "sequence_end": False, "text_output": f" tok{i}"}
                await resp.write((json.dumps(chunk) + "\n").encode("utf-8"))
                await asyncio.sleep(0.015)
        except (ConnectionResetError, asyncio.CancelledError, Exception):
            client_disconnected.set()
            raise
        return resp

    app = web.Application()
    app.router.add_post("/v2/models/tensorrt_llm/generate_stream", handle_long_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TensorRTLLMEngineHook(base_url=f"http://127.0.0.1:{port}", model_name="tensorrt_llm")

    try:
        session = InferenceSession(
            request_id="req-trt-abort-003",
            model="tensorrt_llm",
            prompt="Generate numbers",
            last_flushed_sequence_id=0,
        )

        # Launch streaming in background task
        async def _consumer():
            async for _ in hook.stream_resumed(session):
                pass

        task = asyncio.create_task(_consumer())
        await asyncio.wait_for(stream_started.wait(), timeout=1.0)

        # Measure abort latency
        t_start = time.monotonic()
        abort_res = await hook.abort_request("req-trt-abort-003")
        abort_duration_ms = (time.monotonic() - t_start) * 1000.0

        assert abort_res.success is True
        assert abort_res.latency_ms <= 50.0, f"Abort latency {abort_res.latency_ms:.2f}ms exceeded 50ms SLA!"
        assert abort_duration_ms <= 50.0, f"Total execution time {abort_duration_ms:.2f}ms exceeded 50ms budget!"
        assert abort_res.backend == BackendType.TENSORRT_LLM

        await task

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_trtllm_abort_hung_engine_timeout():
    """
    AC-M3-08: Verifies graceful fallback and 50ms timeout protection when engine is unresponsive.
    """
    async def handle_hung_endpoint(request: web.Request) -> web.Response:
        await asyncio.sleep(0.200)  # Hung engine
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/v2/models/tensorrt_llm/generate_stream", handle_hung_endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TensorRTLLMEngineHook(
        base_url=f"http://127.0.0.1:{port}",
        model_name="tensorrt_llm",
        abort_timeout=0.040,  # 40ms ceiling
    )

    try:
        t_start = time.monotonic()
        abort_res = await hook.abort_request("req-unregistered-hung")
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert elapsed_ms <= 60.0  # Finished within SLA margin
        assert abort_res.success is False
        assert abort_res.status_code == 408

    finally:
        await hook.close()
        await runner.cleanup()
