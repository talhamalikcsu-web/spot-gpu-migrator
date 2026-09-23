"""
Convenience re-export of VLLMInferenceEngineHook.
"""

from daemon.core.vllm_hook import AbortResult, VLLMInferenceEngineHook

__all__ = ["VLLMInferenceEngineHook", "AbortResult"]
