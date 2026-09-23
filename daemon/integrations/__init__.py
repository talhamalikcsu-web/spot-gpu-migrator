"""
Inference engine integration hooks for vLLM and SGLang.
"""

from daemon.integrations.base import AbstractInferenceEngineHook
from daemon.integrations.vllm import AbortResult, VLLMInferenceEngineHook
from daemon.integrations.sglang import SGLangInferenceEngineHook

__all__ = [
    "AbstractInferenceEngineHook",
    "VLLMInferenceEngineHook",
    "SGLangInferenceEngineHook",
    "AbortResult",
]
