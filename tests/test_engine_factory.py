"""
Milestone 3 Verification: Unified Engine Hook Factory & Auto-Detection Tests.

Conforms to SPEC-005:
- AC-M3-05: Unified Engine Hook Factory Auto-Detection
- TEST-M3-05: Exercise heuristics against synthetic Triton, TGI, SGLang, vLLM, and Mock endpoints.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from aiohttp import web
import pytest

from daemon.integrations.base import (
    BackendType,
    EngineDetectionError,
)
from daemon.integrations.factory import EngineHookFactory
from daemon.integrations.mock import MockInferenceEngineHook
from daemon.integrations.sglang import SGLangInferenceEngineHook
from daemon.integrations.tensorrt_llm import TensorRTLLMEngineHook
from daemon.integrations.tgi import TGIEngineHook
from daemon.integrations.vllm import VLLMInferenceEngineHook


async def _spin_up_synthetic_server(routes: Dict[str, Any]) -> tuple[web.AppRunner, str]:
    """Helper to spin up a lightweight aiohttp server with custom probe endpoints."""
    app = web.Application()
    for path, handler in routes.items():
        app.router.add_get(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_auto_detect_triton_tensorrt_llm():
    """
    AC-M3-05: Probe 1: GET /v2/health/ready -> TensorRTLLMEngineHook.
    """
    async def handle_ready(request: web.Request) -> web.Response:
        return web.json_response({"status": "ready"})

    runner, url = await _spin_up_synthetic_server({"/v2/health/ready": handle_ready})
    try:
        t_start = time.monotonic()
        hook = await EngineHookFactory.create(url)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert isinstance(hook, TensorRTLLMEngineHook)
        assert hook.get_engine_metadata().backend == BackendType.TENSORRT_LLM
        assert elapsed_ms <= 150.0, f"Auto-detection took {elapsed_ms:.1f}ms (budget <= 150ms)"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_auto_detect_tgi():
    """
    AC-M3-05: Probe 2: GET /info -> TGIEngineHook.
    """
    async def handle_info(request: web.Request) -> web.Response:
        return web.json_response({
            "version": "2.2.0",
            "model_id": "meta-llama/Llama-3-8B-Instruct",
            "paged_attention": True,
        })

    runner, url = await _spin_up_synthetic_server({"/info": handle_info})
    try:
        t_start = time.monotonic()
        hook = await EngineHookFactory.create(url)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert isinstance(hook, TGIEngineHook)
        assert hook.get_engine_metadata().backend == BackendType.TGI
        assert elapsed_ms <= 150.0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_auto_detect_sglang():
    """
    AC-M3-05: Probe 3: GET /get_model_info -> SGLangInferenceEngineHook.
    """
    async def handle_sglang_info(request: web.Request) -> web.Response:
        return web.json_response({
            "model_path": "meta-llama/Llama-3-8B-Instruct",
            "tokenizer_path": "meta-llama/Llama-3-8B-Instruct",
            "is_generation": True,
        })

    runner, url = await _spin_up_synthetic_server({"/get_model_info": handle_sglang_info})
    try:
        t_start = time.monotonic()
        hook = await EngineHookFactory.create(url)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert isinstance(hook, SGLangInferenceEngineHook)
        assert hook.get_engine_metadata().backend == BackendType.SGLANG
        assert elapsed_ms <= 150.0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_auto_detect_vllm():
    """
    AC-M3-05: Probe 4: GET /version -> VLLMInferenceEngineHook.
    """
    async def handle_vllm_version(request: web.Request) -> web.Response:
        return web.json_response({"version": "0.6.1.post1"})

    runner, url = await _spin_up_synthetic_server({"/version": handle_vllm_version})
    try:
        t_start = time.monotonic()
        hook = await EngineHookFactory.create(url)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert isinstance(hook, VLLMInferenceEngineHook)
        assert hook.get_engine_metadata().backend == BackendType.VLLM
        assert elapsed_ms <= 150.0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_auto_detect_mock_simulator():
    """
    AC-M3-05: Probe 5: GET /health with role/paused -> MockInferenceEngineHook.
    """
    async def handle_mock_health(request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "role": "active",
            "port": 8001,
            "paused": False,
        })

    runner, url = await _spin_up_synthetic_server({"/health": handle_mock_health})
    try:
        t_start = time.monotonic()
        hook = await EngineHookFactory.create(url)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        assert isinstance(hook, MockInferenceEngineHook)
        assert hook.get_engine_metadata().backend == BackendType.MOCK
        assert elapsed_ms <= 150.0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_explicit_creation_and_sync_helpers():
    """
    Verifies that explicit engine types bypass auto-detection network probes.
    """
    # 1. Sync instantiation
    hook_vllm = EngineHookFactory.create_sync("vllm", "http://127.0.0.1:8001")
    assert isinstance(hook_vllm, VLLMInferenceEngineHook)

    hook_trt = EngineHookFactory.create_sync("tensorrt_llm", "http://127.0.0.1:8002")
    assert isinstance(hook_trt, TensorRTLLMEngineHook)

    hook_tgi = EngineHookFactory.create_sync("tgi", "http://127.0.0.1:8003")
    assert isinstance(hook_tgi, TGIEngineHook)

    hook_sglang = EngineHookFactory.create_sync("sglang", "http://127.0.0.1:8004")
    assert isinstance(hook_sglang, SGLangInferenceEngineHook)

    hook_mock = EngineHookFactory.create_sync("mock", "http://127.0.0.1:8005")
    assert isinstance(hook_mock, MockInferenceEngineHook)

    # 2. Async creation with explicit type (no probe needed)
    hook_async = await EngineHookFactory.create("tensorrt_llm", "http://127.0.0.1:8006")
    assert isinstance(hook_async, TensorRTLLMEngineHook)


@pytest.mark.asyncio
async def test_auto_detect_error_on_dead_endpoint():
    """
    Verifies EngineDetectionError is raised when an endpoint is completely unresponsive.
    """
    # Non-routable / unused local port
    dead_url = "http://127.0.0.1:59999"
    with pytest.raises(EngineDetectionError):
        await EngineHookFactory.auto_detect(dead_url, timeout_s=0.03, retries=1)
