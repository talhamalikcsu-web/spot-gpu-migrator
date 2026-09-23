"""
Milestone 3 Automated Verification Test Suite: Universal Inference Engine Hooks.

Conforms to SPEC-005 (specs/05-universal-engine-hooks.md):
- AC-M3-01: Triton TensorRT-LLM stream ingestion & decoupled chunk parsing
- AC-M3-02: Triton TensorRT-LLM sub-50ms abort latency budget
- AC-M3-03: Hugging Face TGI native (/generate_stream) and OpenAI (/compat/v1/chat/completions) streaming
- AC-M3-04: TGI partial stop sequence suppression (SlidingWindowStopFilter)
- AC-M3-05: Unified Engine Hook Factory auto-detection across all 5 engines (vLLM, TRT-LLM, TGI, SGLang, Mock)
- AC-M3-08: Graceful fallback and zombie engine timeout enforcement (<= 50ms)

Contains the four required test cases:
1. test_trtllm_stream_parsing_and_resumption
2. test_trtllm_abort_latency_budget
3. test_tgi_native_and_compat_streaming
4. test_engine_hook_factory_auto_detection
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from aiohttp import web
import pytest

from daemon.integrations.base import (
    AbortResult,
    BackendType,
    EngineDetectionError,
    EngineMetadata,
)
from daemon.integrations.factory import ENGINE_HOOK_REGISTRY, EngineHookFactory
from daemon.integrations.mock import MockInferenceEngineHook
from daemon.integrations.sglang import SGLangInferenceEngineHook
from daemon.integrations.tensorrt_llm import TensorRTLLMEngineHook
from daemon.integrations.tgi import SlidingWindowStopFilter, TGIEngineHook
from daemon.integrations.vllm import VLLMInferenceEngineHook
from daemon.models import InferenceSession, SamplingParams, TokenChunk


# ---------------------------------------------------------------------------
# Helper: Ephemeral HTTP Server Fixture / Runner
# ---------------------------------------------------------------------------

async def _start_ephemeral_server(app: web.Application) -> tuple[web.AppRunner, str]:
    """Starts an ephemeral aiohttp web server on an OS-assigned port."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Test 1: Triton TensorRT-LLM Stream Parsing & Resumption
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trtllm_stream_parsing_and_resumption():
    """
    Verifies TensorRT-LLM stream parsing and resumption formatting.

    Requirements:
    - Mocks Triton /v2/models/meta-llama/generate_stream emitting JSON chunks.
    - Verifies TensorRTLLMEngineHook parses output tokens and yields TokenChunk
      with correct monotonic sequence IDs.
    - Verifies resumption payload formatting:
      - exclude_input_from_output: True (mandatory per SPEC-005 Section 2.4)
      - decremented max_tokens (original max_tokens - prefix token count)
      - continuation prompt concatenation (prompt + generated prefix)
    """
    received_requests: List[Dict[str, Any]] = []

    # 1. Setup mock Triton Inference Server with decoupled streaming
    async def handle_generate_stream(request: web.Request) -> web.StreamResponse:
        data = await request.json()
        received_requests.append(data)

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "application/json"},
        )
        await resp.prepare(request)

        # Emulate Triton Decoupled Transaction Protocol emitting chunked JSON lines
        chunks = [
            {
                "model_name": "meta-llama",
                "model_version": "1",
                "sequence_start": True,
                "sequence_end": False,
                "sequence_id": 0,
                "text_output": " Ra",
            },
            {
                "model_name": "meta-llama",
                "model_version": "1",
                "sequence_start": False,
                "sequence_end": False,
                "sequence_id": 0,
                "text_output": "ft",
            },
            {
                "model_name": "meta-llama",
                "model_version": "1",
                "sequence_start": False,
                "sequence_end": True,
                "sequence_id": 0,
                "text_output": " consensus",
            },
        ]

        for chunk in chunks:
            line = json.dumps(chunk) + "\n"
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.005)

        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v2/models/meta-llama/generate_stream", handle_generate_stream)

    runner, server_url = await _start_ephemeral_server(app)

    try:
        hook = TensorRTLLMEngineHook(base_url=server_url, model_name="meta-llama")

        # 2. Verify Resumption Payload Formatting
        session = InferenceSession(
            request_id="req-trt-test-001",
            client_connection_id="conn-client-01",
            model="meta-llama",
            prompt="Explain consensus algorithms in distributed systems:",
            prompt_tokens=[101, 102, 103],
            sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=64,
                stop=["<|eot_id|>", "</s>"],
            ),
            generated_tokens=[201, 202, 203],
            generated_text=[" Distributed", " systems", " use"],
            last_flushed_sequence_id=2,  # 3 tokens generated (0, 1, 2)
            total_tokens_generated=3,
        )

        payload = hook.build_prefix_caching_payload(session, cutoff_sequence_id=2)

        # Invariant 1: exclude_input_from_output must be True
        assert payload["exclude_input_from_output"] is True, (
            "exclude_input_from_output MUST be True for TRT-LLM to avoid echoing prompt and prefix"
        )

        # Invariant 2: Decremented token budget
        expected_remaining_tokens = 64 - 3  # 61
        assert payload["max_tokens"] == expected_remaining_tokens, (
            f"Expected max_tokens={expected_remaining_tokens}, got {payload['max_tokens']}"
        )

        # Invariant 3: Continuation prompt concatenation
        expected_continuation = "Explain consensus algorithms in distributed systems: Distributed systems use"
        assert payload["text_input"] == expected_continuation

        # Invariant 4: Preserved sampling parameters
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.95
        assert payload["stop_words"] == ["<|eot_id|>", "</s>"]
        assert payload["stream"] is True

        # Test fresh prompt resumption (cutoff=-1, zero tokens generated)
        fresh_session = InferenceSession(
            request_id="req-fresh-trt-002",
            model="meta-llama",
            prompt="Hello world",
            sampling_params=SamplingParams(max_tokens=32),
            generated_tokens=[],
            generated_text=[],
            last_flushed_sequence_id=-1,
            total_tokens_generated=0,
        )
        fresh_payload = hook.build_prefix_caching_payload(fresh_session)
        assert fresh_payload["exclude_input_from_output"] is True
        assert fresh_payload["max_tokens"] == 32
        assert fresh_payload["text_input"] == "Hello world"

        # 3. Verify Stream Ingestion, Decoupled Chunk Parsing, and Monotonic Sequence Numbering
        emitted_chunks: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            emitted_chunks.append(chunk)

        assert len(emitted_chunks) == 3, f"Expected 3 token chunks, got {len(emitted_chunks)}"

        # Verify parsed tokens
        assert emitted_chunks[0].token == " Ra"
        assert emitted_chunks[1].token == "ft"
        assert emitted_chunks[2].token == " consensus"

        # Verify monotonic sequence IDs starting from cutoff + 1 (2 + 1 = 3)
        assert emitted_chunks[0].sequence_id == 3
        assert emitted_chunks[1].sequence_id == 4
        assert emitted_chunks[2].sequence_id == 5

        # Verify terminal flag on final chunk
        assert emitted_chunks[0].is_final is False
        assert emitted_chunks[1].is_final is False
        assert emitted_chunks[2].is_final is True

        # Verify request ID continuity
        for chunk in emitted_chunks:
            assert chunk.request_id == "req-trt-test-001"

        # Verify server received the proper resumption payload over the wire
        assert len(received_requests) == 1
        assert received_requests[0]["exclude_input_from_output"] is True
        assert received_requests[0]["max_tokens"] == 61
        assert received_requests[0]["text_input"] == expected_continuation

        # 4. Verify Engine Metadata
        meta: EngineMetadata = hook.get_engine_metadata()
        assert meta.backend == BackendType.TENSORRT_LLM
        assert meta.model_name == "meta-llama"
        assert meta.supports_abort_endpoint is False
        assert meta.supports_prefix_caching is True

    finally:
        await hook.close()
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Test 2: Triton TensorRT-LLM Abort Latency Budget (<= 50ms)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trtllm_abort_latency_budget():
    """
    Verifies that abort_request completes within the strict <= 50ms latency budget.

    Conforms to SPEC-005 Section 2.3 & AC-M3-02:
    - Primary Mechanism: Transport Socket Drop (TCP FIN/RST) when an active stream is open.
      Triton decoupled API detects socket closure in <= 2ms and prunes the request.
    - Fallback Mechanism: Truncation injection via POST /generate_stream when stream is not registered.
    - Measured latency SLA: T_abort <= 50.0ms (target budget <= 20ms).
    - Timeout protection: Gracefully aborts hung/unresponsive engines within budget.
    """
    active_stream_started = asyncio.Event()

    async def handle_slow_stream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/json"},
        )
        await resp.prepare(request)
        active_stream_started.set()

        try:
            # Emit tokens slowly to simulate in-flight batching execution
            for idx in range(100):
                chunk = {
                    "model_name": "meta-llama",
                    "sequence_id": idx,
                    "sequence_end": False,
                    "text_output": f" tok_{idx}",
                }
                await resp.write((json.dumps(chunk) + "\n").encode("utf-8"))
                await asyncio.sleep(0.040)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    async def handle_truncation_abort(request: web.Request) -> web.Response:
        await asyncio.sleep(0.002)  # Fast 2ms simulated response
        return web.json_response({"status": "truncated", "max_tokens": 0})

    async def handle_hung_endpoint(request: web.Request) -> web.Response:
        await asyncio.sleep(0.150)  # Hangs for 150ms (exceeding budget)
        return web.json_response({"status": "timeout"})

    app = web.Application()
    app.router.add_post("/v2/models/meta-llama/generate_stream", handle_slow_stream)
    app.router.add_post("/v2/models/fallback-model/generate_stream", handle_truncation_abort)
    app.router.add_post("/v2/models/hung-model/generate_stream", handle_hung_endpoint)

    runner, server_url = await _start_ephemeral_server(app)

    try:
        # --- Part A: Primary Abort via Active Socket Closure ---
        hook = TensorRTLLMEngineHook(
            base_url=server_url,
            model_name="meta-llama",
            abort_timeout=0.05,
        )

        session = InferenceSession(
            request_id="req-live-abort-001",
            model="meta-llama",
            prompt="Count up to one hundred slowly:",
            sampling_params=SamplingParams(max_tokens=100),
            last_flushed_sequence_id=0,
        )

        received_chunks: List[TokenChunk] = []

        async def stream_worker():
            async for c in hook.stream_resumed(session):
                received_chunks.append(c)

        stream_task = asyncio.create_task(stream_worker())

        # Wait until streaming begins
        await asyncio.wait_for(active_stream_started.wait(), timeout=2.0)
        await asyncio.sleep(0.015)  # Let at least one chunk arrive

        # Measure abort_request latency under active stream
        t_start = time.monotonic()
        abort_res: AbortResult = await hook.abort_request("req-live-abort-001")
        measured_wall_ms = (time.monotonic() - t_start) * 1000.0

        # Enforce SLA Invariants
        assert abort_res.success is True, f"Abort failed: {abort_res.message}"
        assert abort_res.latency_ms <= 50.0, (
            f"Abort latency exceeded 50ms SLA hard ceiling: {abort_res.latency_ms:.2f}ms"
        )
        assert measured_wall_ms <= 50.0, (
            f"Wall-clock abort time exceeded 50ms: {measured_wall_ms:.2f}ms"
        )
        assert abort_res.status_code == 200
        assert abort_res.backend == BackendType.TENSORRT_LLM

        # Verify background stream terminates promptly
        try:
            await asyncio.wait_for(stream_task, timeout=0.1)
        except asyncio.TimeoutError:
            stream_task.cancel()

        # --- Part B: Fallback Abort (Non-registered stream, wire request) ---
        fallback_hook = TensorRTLLMEngineHook(
            base_url=server_url,
            model_name="fallback-model",
            abort_timeout=0.05,
        )

        t_start = time.monotonic()
        fallback_res = await fallback_hook.abort_request("req-fallback-002")
        measured_fallback_ms = (time.monotonic() - t_start) * 1000.0

        assert fallback_res.success is True
        assert fallback_res.latency_ms <= 50.0, (
            f"Fallback abort exceeded 50ms: {fallback_res.latency_ms:.2f}ms"
        )
        assert measured_fallback_ms <= 50.0

        # --- Part C: Repeated Trials (P99 Latency Budget Validation) ---
        latencies: List[float] = []
        for trial_idx in range(25):
            t0 = time.monotonic()
            res = await fallback_hook.abort_request(f"req-trial-{trial_idx}")
            dur_ms = (time.monotonic() - t0) * 1000.0
            assert res.success is True
            assert res.latency_ms <= 50.0, f"Trial {trial_idx} exceeded 50ms: {res.latency_ms:.2f}ms"
            latencies.append(dur_ms)

        latencies.sort()
        p99_idx = int(len(latencies) * 0.99)
        p99_latency = latencies[min(p99_idx, len(latencies) - 1)]
        assert p99_latency <= 35.0, (
            f"P99 abort latency exceeded 35ms target: {p99_latency:.2f}ms (all: {latencies})"
        )

        # --- Part D: Slow/Hung Engine Timeout Protection ---
        hung_hook = TensorRTLLMEngineHook(
            base_url=server_url,
            model_name="hung-model",
            timeout_ms=25.0,  # 25ms timeout override
        )

        t_start = time.monotonic()
        hung_res = await hung_hook.abort_request("req-hung-003", timeout_ms=25.0)
        hung_wall_ms = (time.monotonic() - t_start) * 1000.0

        # Must fail gracefully with HTTP 408 within <= 50ms
        assert hung_res.success is False
        assert hung_res.status_code == 408
        assert hung_wall_ms <= 50.0, (
            f"Hung engine abort timeout exceeded 50ms SLA ceiling: {hung_wall_ms:.2f}ms"
        )

    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Test 3: Hugging Face TGI Native & Compat Streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tgi_native_and_compat_streaming():
    """
    Verifies Hugging Face TGI native and OpenAI-compatible streaming protocols.

    Requirements:
    - Mocks TGI /generate_stream and /compat/v1/chat/completions.
    - Verifies TGIEngineHook token extraction and stop-sequence suppression.
    - Verifies prefix continuation payload structure:
      - Native: return_full_text: False, max_new_tokens decremented, inputs continuation.
      - Compat: messages list with user prompt and assistant prefix.
    - Verifies SlidingWindowStopFilter suppresses partial stop fragments (<|eot_id|>).
    """
    native_requests: List[Dict[str, Any]] = []
    compat_requests: List[Dict[str, Any]] = []

    # 1. Setup mock TGI Server
    async def handle_native_generate_stream(request: web.Request) -> web.StreamResponse:
        data = await request.json()
        native_requests.append(data)

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)

        # Emulate TGI Native SSE stream:
        # Includes standard tokens, partial stop sequence split across chunks, and special token
        events = [
            {"token": {"id": 101, "text": "import", "special": False}},
            {"token": {"id": 102, "text": " os", "special": False}},
            {"token": {"id": 103, "text": "\n", "special": False}},
            # Stop sequence '<|eot_id|>' split into two chunks: '<|' and 'eot_id|>'
            {"token": {"id": 104, "text": "<|", "special": False}},
            {"token": {"id": 105, "text": "eot_id|>", "special": False}},
            # Final special token
            {
                "token": {"id": 128001, "text": "<|eot_id|>", "special": True},
                "generated_text": "import os\n",
                "details": {"finish_reason": "eos_token", "generated_tokens": 3},
            },
        ]

        for ev in events:
            line = f"data:{json.dumps(ev)}\n\n"
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.005)

        await resp.write_eof()
        return resp

    async def handle_compat_chat_completions(request: web.Request) -> web.StreamResponse:
        data = await request.json()
        compat_requests.append(data)

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)

        compat_events = [
            {"choices": [{"delta": {"content": "def"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " run():"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "\n    pass"}, "finish_reason": "stop"}]},
        ]

        for ev in compat_events:
            line = f"data: {json.dumps(ev)}\n\n"
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.005)

        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/generate_stream", handle_native_generate_stream)
    app.router.add_post("/compat/v1/chat/completions", handle_compat_chat_completions)

    runner, server_url = await _start_ephemeral_server(app)

    try:
        hook_native = TGIEngineHook(base_url=server_url, use_openai_compat=False)
        hook_compat = TGIEngineHook(base_url=server_url, use_openai_compat=True)

        # 2. Verify Native Prefix Continuation Payload Structure
        session = InferenceSession(
            request_id="req-tgi-001",
            prompt="Write a Python script:",
            sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=128,
                stop=["<|eot_id|>", "</s>"],
            ),
            generated_text=[" #", " Begin", "\n"],
            last_flushed_sequence_id=2,  # 3 tokens generated (0, 1, 2)
            total_tokens_generated=3,
        )

        native_payload = hook_native.build_prefix_caching_payload(session, cutoff_sequence_id=2)

        # Invariant: return_full_text must be False
        assert native_payload["parameters"]["return_full_text"] is False, (
            "TGI native return_full_text MUST be False to prevent echoing prompt"
        )
        # Invariant: max_new_tokens decremented
        assert native_payload["parameters"]["max_new_tokens"] == 128 - 3  # 125
        # Invariant: inputs concatenated with generated prefix
        assert native_payload["inputs"] == "Write a Python script: # Begin\n"
        assert native_payload["parameters"]["stop"] == ["<|eot_id|>", "</s>"]
        assert native_payload["parameters"]["temperature"] == 0.7

        # 3. Verify OpenAI-Compat Continuation Payload Structure
        compat_payload = hook_compat.build_prefix_caching_payload(session, cutoff_sequence_id=2)
        assert len(compat_payload["messages"]) == 2
        assert compat_payload["messages"][0]["role"] == "user"
        assert compat_payload["messages"][0]["content"] == "Write a Python script:"
        assert compat_payload["messages"][1]["role"] == "assistant"
        assert compat_payload["messages"][1]["content"] == " # Begin\n"
        assert compat_payload["max_tokens"] == 125

        # 4. Verify Native Streaming, Token Extraction & Stop Sequence Suppression
        native_chunks: List[TokenChunk] = []
        async for chunk in hook_native.stream_resumed(session):
            native_chunks.append(chunk)

        # Tokens that should have been emitted: "import", " os", "\n"
        # The partial sequence '<|' followed by 'eot_id|>' formed '<|eot_id|>' which MUST be suppressed!
        # The final '<|eot_id|>' with special: True MUST be filtered!
        emitted_texts = [c.token for c in native_chunks]
        assert emitted_texts == ["import", " os", "\n"], (
            f"Stop sequence was not suppressed properly! Emitted tokens: {emitted_texts}"
        )

        # Verify no stop-word fragment leaked
        for token_text in emitted_texts:
            assert "<|" not in token_text, f"Partial stop fragment leaked: {token_text}"
            assert "eot_id" not in token_text, f"Stop fragment leaked: {token_text}"

        # Verify monotonic sequence IDs starting from cutoff + 1 (2 + 1 = 3)
        seq_ids = [c.sequence_id for c in native_chunks]
        assert seq_ids == [3, 4, 5], f"Expected sequence IDs [3, 4, 5], got {seq_ids}"

        # 5. Verify OpenAI-Compat Streaming
        compat_chunks: List[TokenChunk] = []
        async for chunk in hook_compat.stream_resumed(session):
            compat_chunks.append(chunk)

        compat_texts = [c.token for c in compat_chunks]
        assert compat_texts == ["def", " run():", "\n    pass"], (
            f"Unexpected compat stream tokens: {compat_texts}"
        )
        assert [c.sequence_id for c in compat_chunks] == [3, 4, 5]

        # 6. Verify Unit Behavior of SlidingWindowStopFilter directly
        filter_under_test = SlidingWindowStopFilter(stop_sequences=["<|eot_id|>", "</s>"])

        # Feed non-stop text
        assert filter_under_test.feed("hello ") == ["hello "]

        # Feed partial stop sequence: '<|'
        assert filter_under_test.feed("<|") == [], "Partial stop word prefix must be buffered"

        # Complete stop sequence: 'eot_id|>'
        assert filter_under_test.feed("eot_id|>") == [], "Completed stop sequence must be suppressed"
        assert filter_under_test.matched_stop is True

        # Subsequent tokens after stop must be dropped
        assert filter_under_test.feed(" extra text") == []
        assert filter_under_test.flush_final() == []

        # Test false-match buffering and recovery: '<|' followed by 'variable|>'
        false_match_filter = SlidingWindowStopFilter(stop_sequences=["<|eot_id|>"])
        assert false_match_filter.feed("value = ") == ["value = "]
        assert false_match_filter.feed("<|") == []  # Candidate prefix buffered
        flushed_false_match = false_match_filter.feed("not_a_stop>")  # Rejected as false match
        assert flushed_false_match == ["<|not_a_stop>"]

    finally:
        await hook_native.close()
        await hook_compat.close()
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Test 4: Unified Engine Hook Factory Auto-Detection & Fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_hook_factory_auto_detection():
    """
    Verifies EngineHookFactory auto-detection against mocked endpoints for each engine type.

    Requirements:
    - Tests EngineHookFactory.auto_detect against mocked endpoints:
      1. Triton / TensorRT-LLM: GET /v2/health/ready -> tensorrt_llm
      2. Hugging Face TGI:      GET /info            -> tgi
      3. SGLang:                GET /get_model_info  -> sglang
      4. vLLM:                  GET /version         -> vllm
      5. Mock Simulator:        GET /health (role/paused payload) -> mock
    - Verifies fallback behavior:
      - Generic /health without role payload -> falls back to vLLM
      - Dead / 404 endpoint -> raises EngineDetectionError (or returns fallback when set)
    - Verifies explicit type creation:
      - Synchronous creation via create_sync for all types
      - Asynchronous creation with explicit engine_type
      - Error handling for invalid/unsupported engine types
    """
    # 1. Setup mock server exposing all engine-specific signatures on distinct routes
    # We will spin up independent ephemeral apps for each engine type to guarantee isolation.

    # --- Engine 1: Triton / TensorRT-LLM Server ---
    trt_app = web.Application()
    async def _handle_trt_ready(request: web.Request) -> web.Response:
        return web.Response(status=200, text="OK")
    trt_app.router.add_get("/v2/health/ready", _handle_trt_ready)
    trt_runner, trt_url = await _start_ephemeral_server(trt_app)

    # --- Engine 2: Hugging Face TGI Server ---
    tgi_app = web.Application()
    async def _handle_tgi_info(request: web.Request) -> web.Response:
        return web.json_response({
            "version": "2.2.0",
            "model_id": "meta-llama/Llama-3-8b",
            "paged_attention": True,
        })
    tgi_app.router.add_get("/info", _handle_tgi_info)
    tgi_runner, tgi_url = await _start_ephemeral_server(tgi_app)

    # --- Engine 3: SGLang Server ---
    sglang_app = web.Application()
    async def _handle_sglang_info(request: web.Request) -> web.Response:
        return web.json_response({"model_path": "/opt/models/llama", "radix_cache": True})
    sglang_app.router.add_get("/get_model_info", _handle_sglang_info)
    sglang_runner, sglang_url = await _start_ephemeral_server(sglang_app)

    # --- Engine 4: vLLM Server ---
    vllm_app = web.Application()
    async def _handle_vllm_version(request: web.Request) -> web.Response:
        return web.json_response({"version": "0.6.0"})
    vllm_app.router.add_get("/version", _handle_vllm_version)
    vllm_runner, vllm_url = await _start_ephemeral_server(vllm_app)

    # --- Engine 5: Mock LLM Simulator Server ---
    mock_app = web.Application()
    async def _handle_mock_health(request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "role": "active",
            "port": 8001,
            "paused": False,
        })
    mock_app.router.add_get("/health", _handle_mock_health)
    mock_runner, mock_url = await _start_ephemeral_server(mock_app)

    # --- Engine 6: Generic /health Server (Default vLLM Fallback) ---
    generic_app = web.Application()
    async def _handle_generic_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})
    generic_app.router.add_get("/health", _handle_generic_health)
    generic_runner, generic_url = await _start_ephemeral_server(generic_app)

    # --- Engine 7: Dead / 404 Server (All probes fail) ---
    dead_app = web.Application()
    dead_runner, dead_url = await _start_ephemeral_server(dead_app)

    try:
        # 2. Test auto_detect on each mocked engine endpoint
        # Triton / TensorRT-LLM
        detected_trt = await EngineHookFactory.auto_detect(trt_url)
        assert detected_trt == BackendType.TENSORRT_LLM
        hook_trt = await EngineHookFactory.create(engine_url=trt_url)
        assert isinstance(hook_trt, TensorRTLLMEngineHook)

        # Hugging Face TGI
        detected_tgi = await EngineHookFactory.auto_detect(tgi_url)
        assert detected_tgi == BackendType.TGI
        hook_tgi = await EngineHookFactory.create(engine_url=tgi_url)
        assert isinstance(hook_tgi, TGIEngineHook)

        # SGLang
        detected_sglang = await EngineHookFactory.auto_detect(sglang_url)
        assert detected_sglang == BackendType.SGLANG
        hook_sglang = await EngineHookFactory.create(engine_url=sglang_url)
        assert isinstance(hook_sglang, SGLangInferenceEngineHook)

        # vLLM
        detected_vllm = await EngineHookFactory.auto_detect(vllm_url)
        assert detected_vllm == BackendType.VLLM
        hook_vllm = await EngineHookFactory.create(engine_url=vllm_url)
        assert isinstance(hook_vllm, VLLMInferenceEngineHook)

        # Mock Simulator
        detected_mock = await EngineHookFactory.auto_detect(mock_url)
        assert detected_mock == BackendType.MOCK
        hook_mock = await EngineHookFactory.create(engine_url=mock_url)
        assert isinstance(hook_mock, MockInferenceEngineHook)

        # 3. Test Fallback Behavior
        # Case A: Generic /health returns 200 without simulator payload -> falls back to vLLM
        detected_generic = await EngineHookFactory.auto_detect(generic_url)
        assert detected_generic == BackendType.VLLM
        hook_generic = await EngineHookFactory.create(engine_url=generic_url)
        assert isinstance(hook_generic, VLLMInferenceEngineHook)

        # Case B: Dead endpoint (all probes 404) -> auto_detect raises EngineDetectionError
        with pytest.raises(EngineDetectionError):
            await EngineHookFactory.auto_detect(dead_url, retries=1, timeout_s=0.05)

        # Case C: auto_detect with explicit fallback parameter -> returns fallback
        detected_with_fallback = await EngineHookFactory.auto_detect(
            dead_url, retries=1, timeout_s=0.05, fallback=BackendType.VLLM
        )
        assert detected_with_fallback == BackendType.VLLM

        # Case D: EngineHookFactory.create with auto on dead endpoint -> defaults to vLLM fallback
        hook_dead_fallback = await EngineHookFactory.create(engine_url=dead_url)
        assert isinstance(hook_dead_fallback, VLLMInferenceEngineHook)

        # Case E: EngineHookFactory.create with strict_detection=True -> raises EngineDetectionError
        with pytest.raises(EngineDetectionError):
            await EngineHookFactory.create(engine_url=dead_url, strict_detection=True)

        # 4. Test Explicit Type Creation (Sync and Async)
        # Synchronous creation
        s_trt = EngineHookFactory.create_sync("tensorrt_llm", "http://127.0.0.1:9001")
        assert isinstance(s_trt, TensorRTLLMEngineHook)

        s_tgi = EngineHookFactory.create_sync("tgi", "http://127.0.0.1:9002")
        assert isinstance(s_tgi, TGIEngineHook)

        s_sglang = EngineHookFactory.create_sync("sglang", "http://127.0.0.1:9003")
        assert isinstance(s_sglang, SGLangInferenceEngineHook)

        s_vllm = EngineHookFactory.create_sync("vllm", "http://127.0.0.1:9004")
        assert isinstance(s_vllm, VLLMInferenceEngineHook)

        s_mock = EngineHookFactory.create_sync("mock", "http://127.0.0.1:9005")
        assert isinstance(s_mock, MockInferenceEngineHook)

        # Aliases test
        assert isinstance(EngineHookFactory.create_sync("trtllm"), TensorRTLLMEngineHook)
        assert isinstance(EngineHookFactory.create_sync("text_generation_inference"), TGIEngineHook)
        assert isinstance(EngineHookFactory.create_sync("simulator"), MockInferenceEngineHook)

        # Asynchronous explicit creation (bypasses auto-detection)
        a_trt = await EngineHookFactory.create("tensorrt_llm", engine_url="http://127.0.0.1:9001")
        assert isinstance(a_trt, TensorRTLLMEngineHook)

        a_tgi = await EngineHookFactory.create(engine_type="tgi", engine_url="http://127.0.0.1:9002")
        assert isinstance(a_tgi, TGIEngineHook)

        # Error handling: Unsupported engine type
        with pytest.raises(ValueError) as excinfo:
            EngineHookFactory.create_sync("unknown_engine_backend")
        assert "Unknown or unsupported engine type" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo_async:
            await EngineHookFactory.create(engine_type="unknown_engine_backend")
        assert "Unknown or unsupported engine type" in str(excinfo_async.value)

    finally:
        await trt_runner.cleanup()
        await tgi_runner.cleanup()
        await sglang_runner.cleanup()
        await vllm_runner.cleanup()
        await mock_runner.cleanup()
        await generic_runner.cleanup()
        await dead_runner.cleanup()
