"""
Core vLLM Hook re-export from daemon.integrations.vllm.
"""

from daemon.integrations.vllm import AbortResult, VLLMInferenceEngineHook

__all__ = ["AbortResult", "VLLMInferenceEngineHook"]
