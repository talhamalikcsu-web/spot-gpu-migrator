"""
SGLang Inference Engine Integration Hook for Spot GPU Migrator (SGM).

Optimized for SGLang Runtime (SRT) and RadixAttention KV-cache tree reuse.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from daemon.integrations.vllm import VLLMInferenceEngineHook
from daemon.models import InferenceSession

logger = logging.getLogger("sgm.daemon.integrations.sglang")


class SGLangInferenceEngineHook(VLLMInferenceEngineHook):
    """
    Concrete integration hook for SGLang (v0.3.0+).

    Leverages SGLang's RadixAttention router for sub-15ms prompt prefix reuse.
    """

    def build_prefix_caching_payload(
        self,
        session: InferenceSession,
        format_type: str = "chat",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Builds continuation payload with SGLang-specific RadixAttention parameters.
        """
        payload = super().build_prefix_caching_payload(session, format_type=format_type, **kwargs)
        # SGLang RadixAttention specific hinting
        payload["skip_special_tokens"] = False
        payload["spaces_between_special_tokens"] = True
        return payload
