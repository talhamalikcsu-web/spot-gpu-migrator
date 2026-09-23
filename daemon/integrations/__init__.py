"""
Universal Inference Engine Hooks and Unified Factory for Spot GPU Migrator (SGM).
Supports NVIDIA TensorRT-LLM, Hugging Face TGI, SGLang, vLLM, and Mock engines.
"""

from daemon.integrations.base import (
    AbortResult,
    AbstractInferenceEngineHook,
    BackendType,
    EngineDetectionError,
    EngineMetadata,
    ResumptionDescriptor,
)
from daemon.integrations.factory import ENGINE_HOOK_REGISTRY, EngineHookFactory
from daemon.integrations.mock import MockInferenceEngineHook
from daemon.integrations.sglang import SGLangInferenceEngineHook
from daemon.integrations.tensorrt_llm import (
    TensorRTLLMEngineHook,
    TensorRTLLMInferenceEngineHook,
)
from daemon.integrations.tgi import (
    SlidingWindowStopFilter,
    TGIEngineHook,
    TGIInferenceEngineHook,
)
from daemon.integrations.vllm import VLLMInferenceEngineHook

VLLMEngineHook = VLLMInferenceEngineHook
SGLangEngineHook = SGLangInferenceEngineHook
MockEngineHook = MockInferenceEngineHook

__all__ = [
    "AbstractInferenceEngineHook",
    "AbortResult",
    "BackendType",
    "EngineDetectionError",
    "EngineMetadata",
    "ResumptionDescriptor",
    "EngineHookFactory",
    "ENGINE_HOOK_REGISTRY",
    "VLLMInferenceEngineHook",
    "VLLMEngineHook",
    "SGLangInferenceEngineHook",
    "SGLangEngineHook",
    "TensorRTLLMEngineHook",
    "TensorRTLLMInferenceEngineHook",
    "TGIEngineHook",
    "TGIInferenceEngineHook",
    "MockInferenceEngineHook",
    "MockEngineHook",
    "SlidingWindowStopFilter",
]
