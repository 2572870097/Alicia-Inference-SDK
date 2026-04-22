"""Public SDK exports for inference policies, sessions, and runtime helpers."""

from ._bootstrap import bootstrap_import_environment

bootstrap_import_environment()

from . import policy as _policy_exports
from .api import InferenceAPI
from .core.config import DeviceConfig, PolicyLoadConfig, RuntimeConfig, SUPPORTED_MODEL_TYPES
from .core.types import Observation, PolicyMetadata, PolicyStatus
from .factory import create_policy, load_policy
from .runtime.base import (
    AsyncInferenceWorker,
    BaseInferenceEngine,
    ObservationQueue,
    SmoothingConfig,
    TimedAction,
    TimedObservation,
    TimestampedActionQueue,
)
from .runtime.monitoring import get_inference_monitor, set_inference_monitor
from .session import InferenceSession


def __getattr__(name: str):
    if name in {
        "ACTInferenceEngine",
        "SmolVLAInferenceEngine",
        "PI0InferenceEngine",
        "ACT_AVAILABLE",
        "SMOLVLA_AVAILABLE",
        "PI0_AVAILABLE",
    }:
        return getattr(_policy_exports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # User-facing API
    "InferenceAPI",
    # Core classes
    "BaseInferenceEngine",
    "ACTInferenceEngine",
    "SmolVLAInferenceEngine",
    "PI0InferenceEngine",
    "InferenceSession",
    "create_policy",
    "load_policy",
    # Config
    "DeviceConfig",
    "RuntimeConfig",
    "PolicyLoadConfig",
    "SUPPORTED_MODEL_TYPES",
    "SmoothingConfig",
    # Data structures
    "Observation",
    "PolicyMetadata",
    "PolicyStatus",
    "TimedAction",
    "TimedObservation",
    # Components
    "TimestampedActionQueue",
    "ObservationQueue",
    "AsyncInferenceWorker",
    # Availability flags
    "ACT_AVAILABLE",
    "SMOLVLA_AVAILABLE",
    "PI0_AVAILABLE",
    "get_inference_monitor",
    "set_inference_monitor",
]
