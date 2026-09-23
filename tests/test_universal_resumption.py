"""
Milestone 3 Verification: Universal Resumption & Sequence Continuity Across Backends.

Conforms to SPEC-005:
- AC-M3-06: Cross-Engine Monotonic Resumption Continuity
- Section 5.2: Formal Exactly-Once Delivery Invariants
  1. Active Node Cutoff Invariant
  2. Standby Resumption Invariant
  3. Zero Duplicate Tokens Invariant
  4. Zero Lost Tokens Invariant
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from aiohttp import web
import pytest

from daemon.integrations.tensorrt_llm import TensorRTLLMEngineHook
from daemon.integrations.tgi import TGIEngineHook
from daemon.models import InferenceSession, SamplingParams, TokenChunk


@pytest.mark.asyncio
async def test_trtllm_monotonic_resumption_invariants():
    """
    Simulates preemption cutoff at sequence K = 7 for TensorRT-LLM.
    Standby engine resumes from K + 1 = 8 up to completion.
    Verifies strictly monotonic sequence numbering with 0 duplicates and 0 drops.
    """
    total_tokens = ["Alpha", " Bravo", " Charlie", " Delta", " Echo", " Foxtrot", " Golf", " Hotel", " India", " Juliet", " Kilo", " Lima"]

    cutoff_k = 7
    active_tokens = total_tokens[: cutoff_k + 1]  # 0..7 (8 tokens)
    standby_expected_tokens = total_tokens[cutoff_k + 1 :]  # 8..11 (4 tokens)

    # Standby Triton endpoint mock
    async def handle_standby_stream(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        assert body["exclude_input_from_output"] is True
        resp = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
        await resp.prepare(request)

        for i, tok in enumerate(standby_expected_tokens):
            is_last = (i == len(standby_expected_tokens) - 1)
            chunk = {
                "model_name": "tensorrt_llm",
                "sequence_end": is_last,
                "text_output": tok,
            }
            await resp.write((json.dumps(chunk) + "\n").encode("utf-8"))
            await asyncio.sleep(0.005)

        return resp

    app = web.Application()
    app.router.add_post("/v2/models/trt-model/generate_stream", handle_standby_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TensorRTLLMEngineHook(base_url=f"http://127.0.0.1:{port}", model_name="trt-model")

    try:
        session = InferenceSession(
            request_id="req-universal-trt-01",
            model="trt-model",
            prompt="Spelling alphabet: ",
            sampling_params=SamplingParams(max_tokens=len(total_tokens)),
            generated_text=active_tokens,
            last_flushed_sequence_id=cutoff_k,  # 7
        )

        resumed_chunks: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            resumed_chunks.append(chunk)

        # Standby resumption invariant: emitted sequence IDs must start from K + 1 = 8
        resumed_seqs = [c.sequence_id for c in resumed_chunks]
        assert resumed_seqs == [8, 9, 10, 11]

        resumed_text = [c.token for c in resumed_chunks]
        assert resumed_text == standby_expected_tokens

        # Zero duplicates invariant
        active_seqs = set(range(cutoff_k + 1))
        assert active_seqs.intersection(set(resumed_seqs)) == set(), "Active and Standby emitted sequence IDs must not overlap!"

        # Zero lost tokens invariant
        combined_text = "".join(active_tokens) + "".join(resumed_text)
        assert combined_text == "".join(total_tokens)

    finally:
        await hook.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_tgi_monotonic_resumption_invariants():
    """
    Simulates preemption cutoff at sequence K = 4 for Hugging Face TGI.
    Standby engine resumes from K + 1 = 5.
    Verifies exactly-once delivery and zero dropped tokens.
    """
    corpus = ["Zero", " One", " Two", " Three", " Four", " Five", " Six", " Seven", " Eight"]
    cutoff_k = 4
    active_tokens = corpus[: cutoff_k + 1]  # 0..4 (5 tokens)
    standby_tokens = corpus[cutoff_k + 1 :]  # 5..8 (4 tokens)

    async def handle_standby_tgi(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        assert body["parameters"]["return_full_text"] is False
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        for i, tok in enumerate(standby_tokens):
            is_last = (i == len(standby_tokens) - 1)
            line = f'data:{{"token":{{"id":{i},"text":"{tok}","special":false}},"generated_text":{"true" if is_last else "null"}}}\n\n'
            await resp.write(line.encode("utf-8"))
            await asyncio.sleep(0.005)

        return resp

    app = web.Application()
    app.router.add_post("/generate_stream", handle_standby_tgi)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    port = site._server.sockets[0].getsockname()[1]
    hook = TGIEngineHook(base_url=f"http://127.0.0.1:{port}", use_openai_compat=False)

    try:
        session = InferenceSession(
            request_id="req-universal-tgi-02",
            prompt="Numbers: ",
            sampling_params=SamplingParams(max_tokens=len(corpus)),
            generated_text=active_tokens,
            last_flushed_sequence_id=cutoff_k,  # 4
        )

        resumed_chunks: List[TokenChunk] = []
        async for chunk in hook.stream_resumed(session):
            resumed_chunks.append(chunk)

        resumed_seqs = [c.sequence_id for c in resumed_chunks]
        assert resumed_seqs == [5, 6, 7, 8]
        resumed_text = [c.token for c in resumed_chunks]
        assert resumed_text == standby_tokens

        # Continuity check
        assert "".join(active_tokens) + "".join(resumed_text) == "".join(corpus)

    finally:
        await hook.close()
        await runner.cleanup()
