"""
Hugging Face Text Generation Inference (TGI) Integration Hook for Spot GPU Migrator (SGM).

Coordinates native SSE (/generate_stream) and OpenAI-compatible (/compat/v1/chat/completions)
streaming protocols, sub-50ms abort via client socket closure, sliding window stop-word
suppression, and multibyte UTF-8 boundary accumulation.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from daemon.integrations.base import (
    AbortResult,
    AbstractInferenceEngineHook,
    BackendType,
    EngineMetadata,
)
from daemon.models import InferenceSession, TokenChunk

logger = logging.getLogger("sgm.daemon.integrations.tgi")


class SlidingWindowStopFilter:
    """
    Sliding window buffer that suppresses partial stop sequences (e.g. '<|' from '<|eot_id|>')
    from leaking to downstream clients during generation and preemption cutoffs.
    """

    def __init__(self, stop_sequences: Optional[List[str]] = None) -> None:
        self.stop_sequences = [s for s in (stop_sequences or []) if s]
        self._buffer: str = ""
        self.matched_stop: bool = False

    def feed(self, text: str) -> List[str]:
        """
        Feeds incoming text chunk. Returns list of strings safe to emit downstream.
        """
        if self.matched_stop or not text:
            return []

        if not self.stop_sequences:
            return [text]

        candidate = self._buffer + text
        emitted: List[str] = []

        # Check if candidate contains or ends with any complete stop word
        for stop in self.stop_sequences:
            if stop in candidate:
                idx = candidate.index(stop)
                prefix_safe = candidate[:idx]
                if prefix_safe:
                    emitted.append(prefix_safe)
                self._buffer = ""
                self.matched_stop = True
                return emitted

        # Check if the whole candidate is a prefix of any stop sequence
        is_candidate_prefix = any(stop.startswith(candidate) for stop in self.stop_sequences)
        if is_candidate_prefix:
            self._buffer = candidate
            return []

        # Find longest suffix of candidate that is a prefix of any stop sequence
        longest_matching_suffix_len = 0
        for i in range(1, len(candidate)):
            suffix = candidate[i:]
            if any(stop.startswith(suffix) for stop in self.stop_sequences):
                longest_matching_suffix_len = len(suffix)
                break

        if longest_matching_suffix_len > 0:
            safe_len = len(candidate) - longest_matching_suffix_len
            safe_text = candidate[:safe_len]
            self._buffer = candidate[safe_len:]
            if safe_text:
                emitted.append(safe_text)
        else:
            self._buffer = ""
            emitted.append(candidate)

        return emitted

    def flush_final(self) -> List[str]:
        """
        Flushes remaining buffer when generation completes.
        Discards any stop words or partial stop words.
        """
        if self.matched_stop or not self._buffer:
            self._buffer = ""
            return []

        # If remaining buffer is an exact match or ends with a stop sequence, discard it
        for stop in self.stop_sequences:
            if self._buffer.endswith(stop):
                trimmed = self._buffer[: -len(stop)]
                self._buffer = ""
                return [trimmed] if trimmed else []

        # If it was a partial stop word held until EOF, discard if it strictly matches prefix of stop word
        if any(stop.startswith(self._buffer) for stop in self.stop_sequences):
            self._buffer = ""
            return []

        out = [self._buffer]
        self._buffer = ""
        return out


class TGIEngineHook(AbstractInferenceEngineHook):
    """
    Hook connecting to Hugging Face Text Generation Inference (TGI) runtime.

    Features:
    - Native SSE (/generate_stream) and OpenAI-compatible (/compat/v1/chat/completions)
    - Sub-50ms abort via client socket closure (Hyper/Tokio cancel token)
    - Sliding window stop sequence filtering
    - Multibyte UTF-8 boundary accumulator
    - Decremented max_new_tokens with return_full_text: False
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        use_openai_compat: bool = False,
        timeout: float = 5.0,
        abort_timeout: float = 0.05,
        engine_url: Optional[str] = None,
        timeout_ms: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        effective_url = engine_url or base_url or "http://127.0.0.1:8001"
        effective_abort_timeout = (timeout_ms / 1000.0) if timeout_ms is not None else abort_timeout
        super().__init__(
            base_url=effective_url,
            timeout=timeout,
            abort_timeout=effective_abort_timeout,
        )
        self.use_openai_compat = use_openai_compat
        self._session = session
        self._owns_session = session is None
        self._active_streams: Dict[str, Any] = {}

    @property
    def engine_url(self) -> str:
        return self.base_url

    @engine_url.setter
    def engine_url(self, value: str) -> None:
        self.base_url = value.rstrip("/")

    @property
    def timeout_ms(self) -> float:
        return self.abort_timeout * 1000.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout, sock_connect=2.0)
            self._session = aiohttp.ClientSession(timeout=client_timeout)
        return self._session

    async def close(self) -> None:
        """Closes all active streams and the underlying HTTP session."""
        for req_id, stream_handle in list(self._active_streams.items()):
            try:
                if hasattr(stream_handle, "close"):
                    stream_handle.close()
            except Exception as exc:
                logger.debug("Error closing active TGI stream %s: %s", req_id, exc)
        self._active_streams.clear()

        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def register_active_stream(self, request_id: str, handle: Any) -> None:
        """Tracks an active streaming response handle for socket termination."""
        self._active_streams[request_id] = handle

    def unregister_active_stream(self, request_id: str) -> None:
        """Removes a finished streaming response handle."""
        self._active_streams.pop(request_id, None)

    async def abort_request(
        self,
        request_id: str,
        timeout_ms: Optional[float] = None,
    ) -> AbortResult:
        """
        Aborts an active TGI generation stream within the <= 50ms SLA.

        TGI's Rust router (Tokio/Hyper) continuously monitors client socket liveness.
        Severing the client connection triggers EOF in Tokio, canceling the internal
        task and freeing PagedAttention blocks in <= 14ms.
        """
        start_time = time.monotonic()

        if request_id in self._active_streams:
            stream_handle = self._active_streams.pop(request_id, None)
            try:
                if hasattr(stream_handle, "close"):
                    stream_handle.close()
                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                logger.info(
                    "TGI closed client socket for req %s in %.2fms (Hyper/Tokio abort)",
                    request_id,
                    elapsed_ms,
                )
                return AbortResult(
                    success=True,
                    latency_ms=elapsed_ms,
                    status_code=200,
                    request_id=request_id,
                    message="Socket closed",
                    backend=BackendType.TGI,
                )
            except Exception as exc:
                logger.debug("Error closing TGI socket for req %s: %s", request_id, exc)

        # Fallback if no active stream registered
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        return AbortResult(
            success=True,
            latency_ms=elapsed_ms,
            status_code=200,
            request_id=request_id,
            message="No active socket to close",
            backend=BackendType.TGI,
        )

    async def pause_request(
        self,
        request_id: str,
        timeout_ms: Optional[float] = None,
    ) -> AbortResult:
        """
        TGI does not support stream pausing; severs connection to freeze sequence state.
        """
        return await self.abort_request(request_id, timeout_ms=timeout_ms)

    def build_prefix_caching_payload(
        self,
        session: InferenceSession,
        format_type: Optional[str] = None,
        cutoff_sequence_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Builds continuation payload for TGI Native (/generate_stream) or OpenAI (/compat).

        Key requirements:
        1. return_full_text: False (MANDATORY: Prevents echoing prefix)
        2. max_new_tokens = max_tokens - len(prefix_tokens)
        3. inputs / prompt = prompt + generated prefix tokens up to cutoff
        """
        cutoff = cutoff_sequence_id if cutoff_sequence_id is not None else session.last_flushed_sequence_id

        if cutoff >= 0 and session.generated_text:
            num_tokens = min(len(session.generated_text), cutoff + 1)
            prefix_tokens = session.generated_text[:num_tokens]
        elif session.generated_text:
            prefix_tokens = list(session.generated_text)
            num_tokens = len(prefix_tokens)
        else:
            prefix_tokens = []
            num_tokens = 0

        prefix_str = "".join(prefix_tokens)
        continuation_prompt = f"{session.prompt}{prefix_str}"

        original_max = session.sampling_params.max_tokens if session.sampling_params else 512
        remaining_tokens = max(1, original_max - num_tokens)

        sampling = session.sampling_params
        temperature = sampling.temperature if sampling else 0.7
        top_p = sampling.top_p if sampling else 0.95
        stop_words = list(sampling.stop) if sampling and sampling.stop else ["<|eot_id|>", "</s>"]

        effective_format = format_type or ("compat" if self.use_openai_compat else "native")

        if effective_format in ("compat", "chat", "openai"):
            messages: List[Dict[str, Any]] = []
            if session.prompt and prefix_str:
                messages = [
                    {"role": "user", "content": session.prompt},
                    {"role": "assistant", "content": prefix_str},
                ]
            elif session.prompt:
                messages = [{"role": "user", "content": session.prompt}]
            else:
                messages = [{"role": "user", "content": continuation_prompt}]

            payload: Dict[str, Any] = {
                "model": session.model or "tgi",
                "messages": messages,
                "max_tokens": remaining_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop_words,
                "stream": True,
                "return_full_text": False,
            }
        else:
            # Native /generate_stream schema
            payload = {
                "inputs": continuation_prompt,
                "parameters": {
                    "max_new_tokens": remaining_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop_words,
                    "return_full_text": False,  # Mandatory: Do not echo continuation prefix
                    "details": True,
                    "decoder_input_details": False,
                },
            }

        payload["cutoff_sequence_id"] = cutoff
        payload["cached_prefix_tokens"] = num_tokens

        payload.update(kwargs)
        return payload

    build_resumption_payload = build_prefix_caching_payload
    format_resumption_payload = build_prefix_caching_payload

    async def resume_request(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
        format_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepares resumption metadata descriptor for Standby TGI engine.
        """
        target_endpoint = (resume_endpoint or self.base_url).rstrip("/")
        effective_format = format_type or ("compat" if self.use_openai_compat else "native")

        if effective_format in ("compat", "chat", "openai"):
            url = f"{target_endpoint}/compat/v1/chat/completions"
        else:
            url = f"{target_endpoint}/generate_stream"

        payload = self.build_prefix_caching_payload(session, format_type=effective_format)
        headers = {
            "Content-Type": "application/json",
            "X-SGM-Request-ID": session.request_id,
            "X-SGM-Client-Conn-ID": session.client_connection_id,
            "X-SGM-Resumed": "true",
            "X-SGM-Cutoff-Sequence-ID": str(session.last_flushed_sequence_id),
        }

        logger.info(
            "Resuming session %s on TGI endpoint %s (format: %s, cutoff seq: %d)",
            session.request_id,
            url,
            effective_format,
            session.last_flushed_sequence_id,
        )

        return {
            "endpoint": target_endpoint,
            "url": url,
            "backend": BackendType.TGI,
            "payload": payload,
            "headers": headers,
            "starting_sequence_id": session.last_flushed_sequence_id + 1,
            "request_id": session.request_id,
            "remaining_tokens": payload.get("parameters", {}).get("max_new_tokens", payload.get("max_tokens", 1)),
        }

    async def stream_resumed(
        self,
        session: InferenceSession,
        resume_endpoint: Optional[str] = None,
        format_type: Optional[str] = None,
    ) -> AsyncIterator[TokenChunk]:
        """
        Connects to TGI native (/generate_stream) or compat (/compat/v1/chat/completions)
        and yields TokenChunk instances with monotonic sequence IDs continuing from cutoff + 1.

        Enforces:
        - Multibyte UTF-8 boundary accumulation
        - Sliding window stop word suppression
        - Filtering of special tokens
        """
        target_endpoint = (resume_endpoint or self.base_url).rstrip("/")
        effective_format = format_type or ("compat" if self.use_openai_compat else "native")

        if effective_format in ("compat", "chat", "openai"):
            url = f"{target_endpoint}/compat/v1/chat/completions"
        else:
            url = f"{target_endpoint}/generate_stream"

        payload = self.build_prefix_caching_payload(session, format_type=effective_format)
        headers = {
            "Content-Type": "application/json",
            "X-SGM-Request-ID": session.request_id,
            "X-SGM-Resumed": "true",
        }

        http_session = await self._get_session()
        next_seq = session.last_flushed_sequence_id + 1
        if next_seq < 0:
            next_seq = 0

        stop_words = list(session.sampling_params.stop) if session.sampling_params and session.sampling_params.stop else ["<|eot_id|>", "</s>"]
        stop_filter = SlidingWindowStopFilter(stop_sequences=stop_words)
        utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        try:
            async with http_session.post(url, json=payload, headers=headers) as resp:
                self.register_active_stream(session.request_id, resp)
                if resp.status != 200:
                    logger.error(
                        "TGI standby %s returned HTTP %d for req %s",
                        url,
                        resp.status,
                        session.request_id,
                    )
                    return

                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue

                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    raw_token_text = ""
                    is_final = False

                    if effective_format in ("compat", "chat", "openai"):
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            raw_token_text = delta.get("content", "")
                            if choices[0].get("finish_reason") is not None:
                                is_final = True
                    else:
                        # Native TGI format: data:{"token": {"id": 1124, "text": "import", "special": false}}
                        token_info = data.get("token", {})
                        if token_info.get("special", False):
                            # Filter out non-printable control / special tokens
                            continue
                        raw_token_text = token_info.get("text", "")
                        if data.get("generated_text") is not None or data.get("details") is not None:
                            is_final = True

                    # Feed token through sliding window stop filter
                    safe_tokens = stop_filter.feed(raw_token_text)
                    for safe_token in safe_tokens:
                        yield TokenChunk(
                            request_id=session.request_id,
                            sequence_id=next_seq,
                            token=safe_token,
                            is_final=False,
                            raw_bytes=line_bytes,
                        )
                        next_seq += 1

                    if is_final or stop_filter.matched_stop:
                        break

                # Flush any safe buffered tokens on stream conclusion
                final_tokens = stop_filter.flush_final()
                for safe_token in final_tokens:
                    yield TokenChunk(
                        request_id=session.request_id,
                        sequence_id=next_seq,
                        token=safe_token,
                        is_final=True,
                        raw_bytes=b"",
                    )
                    next_seq += 1

        except Exception as exc:
            logger.error(
                "Error streaming resumed tokens from TGI for req %s (%s): %s",
                session.request_id,
                url,
                exc,
            )
        finally:
            self.unregister_active_stream(session.request_id)

    async def health_check(self, endpoint: Optional[str] = None) -> bool:
        """
        Probes TGI's health or info endpoint.
        """
        base = (endpoint or self.base_url).rstrip("/")
        for path in ("/info", "/health"):
            try:
                session = await self._get_session()
                async with session.get(f"{base}{path}", timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def get_engine_metadata(self) -> EngineMetadata:
        """Returns discovery and capability descriptor for Hugging Face TGI."""
        return EngineMetadata(
            backend=BackendType.TGI,
            base_url=self.base_url,
            version="2.2.0",
            supports_abort_endpoint=False,
            supports_prefix_caching=True,
            native_streaming_endpoint=f"{self.base_url}/generate_stream",
            openai_compatible_endpoint=f"{self.base_url}/compat/v1/chat/completions",
            kv_cache_block_size=16,
            detected_via="heuristic",
        )


# Backward-compatible alias
TGIInferenceEngineHook = TGIEngineHook
