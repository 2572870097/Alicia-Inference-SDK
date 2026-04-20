"""
Inference Engine Package for ACT, SmolVLA and PI0 models.

Provides optimized inference with LeRobot-style async architecture:
- Timestamp-aligned action queue (skip expired actions)
- Latency-adaptive chunk threshold
- Observation queue maxsize=1 (always use latest frame)
- Aggregate functions for overlapping chunks
- Gripper velocity clamping
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# [CRITICAL] Import lerobot libraries BEFORE sparkmind to prevent import conflicts.
# SparkMind might contain its own vendored or conflicting versions of libraries.
# By importing the installed 'lerobot' first, we ensure it's cached in sys.modules.
try:
    import lerobot
except ImportError:
    pass
except Exception as exc:
    logger.warning("Failed to preload installed lerobot package: %s", exc)
else:
    try:
        import lerobot.policies.rtc
    except Exception as exc:
        logger.warning("Failed to preload lerobot RTC policies, continuing: %s", exc)

# Fall back to the repo-local SparkMind checkout when the package is not installed.
repo_root = Path(__file__).resolve().parents[2]
sparkmind_root = repo_root / "SparkMind"
if sparkmind_root.is_dir():
    sparkmind_root_str = str(sparkmind_root)
    if sparkmind_root_str not in sys.path:
        sys.path.insert(0, sparkmind_root_str)

from .core.config import DeviceConfig, PolicyLoadConfig, RuntimeConfig, SUPPORTED_MODEL_TYPES
from .api import InferenceAPI
from .core.exceptions import (
    CheckpointError,
    ConfigurationError,
    DependencyError,
    DeviceUnavailableError,
    InstructionNotSupportedError,
    InferenceError,
    InferenceTimeoutError,
    ModelLoadError,
    ResourceStateError,
    SDKError,
    ValidationError,
)
from .core.types import Observation, PolicyMetadata, PolicyStatus
from .runtime.base import (
    BaseInferenceEngine,
    SmoothingConfig,
    LatencyEstimator,
    TimedAction,
    TimedObservation,
    TimestampedActionQueue,
    ObservationQueue,
    GripperSmoother,
    AsyncInferenceWorker,
    AGGREGATE_FUNCTIONS,
    get_aggregate_function,
    TraceRecorder,
    TraceEvent,
)
from .runtime.monitoring import get_inference_monitor, set_inference_monitor

ACTInferenceEngine = None
SmolVLAInferenceEngine = None
PI0InferenceEngine = None
ACT_AVAILABLE = False
SMOLVLA_AVAILABLE = False
PI0_AVAILABLE = False
_ENGINE_EXPORTS_LOADED = False


def _load_engine_exports() -> None:
    global ACTInferenceEngine, SmolVLAInferenceEngine, PI0InferenceEngine
    global ACT_AVAILABLE, SMOLVLA_AVAILABLE, PI0_AVAILABLE, _ENGINE_EXPORTS_LOADED

    if _ENGINE_EXPORTS_LOADED:
        return

    from .policy.act import ACTInferenceEngine as _ACTInferenceEngine, ACT_AVAILABLE as _ACT_AVAILABLE
    from .policy.pi0 import PI0InferenceEngine as _PI0InferenceEngine, PI0_AVAILABLE as _PI0_AVAILABLE
    from .policy.smolvla import (
        SmolVLAInferenceEngine as _SmolVLAInferenceEngine,
        SMOLVLA_AVAILABLE as _SMOLVLA_AVAILABLE,
    )

    ACTInferenceEngine = _ACTInferenceEngine
    SmolVLAInferenceEngine = _SmolVLAInferenceEngine
    PI0InferenceEngine = _PI0InferenceEngine
    ACT_AVAILABLE = _ACT_AVAILABLE
    SMOLVLA_AVAILABLE = _SMOLVLA_AVAILABLE
    PI0_AVAILABLE = _PI0_AVAILABLE
    _ENGINE_EXPORTS_LOADED = True


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower() if isinstance(model_type, str) else ""
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValidationError(
            f"Unknown model type: {model_type}",
            details={"supported_model_types": list(SUPPORTED_MODEL_TYPES)},
        )
    return normalized


def _resolve_device_config(
    *,
    config: Optional[PolicyLoadConfig],
    device_config: Optional[DeviceConfig],
    device: str,
) -> DeviceConfig:
    if config is not None:
        return config.device
    if device_config is not None:
        device_config.validate()
        return device_config

    resolved = DeviceConfig(device=device)
    resolved.validate()
    return resolved


def _resolve_smoothing_config(
    *,
    config: Optional[PolicyLoadConfig],
    runtime_config: Optional[RuntimeConfig],
    smoothing_config: Optional[SmoothingConfig],
) -> SmoothingConfig:
    if config is not None:
        return config.to_smoothing_config()
    if runtime_config is not None:
        return runtime_config.to_smoothing_config()
    if smoothing_config is not None:
        return smoothing_config
    return RuntimeConfig(enable_async_inference=True, aggregate_fn_name="latest_only").to_smoothing_config()


def _resolve_model_type(*, config: Optional[PolicyLoadConfig], model_type: str) -> str:
    if config is not None:
        config.validate()
        return config.normalized_model_type
    return _normalize_model_type(model_type)


def _raise_load_error(error: str, *, model_type: str, checkpoint_dir: str) -> None:
    details = {"model_type": model_type, "checkpoint_dir": checkpoint_dir}
    if not error:
        raise ModelLoadError("Model loading failed without a detailed error message", details=details)

    lowered = error.lower()
    if "checkpoint" in lowered and ("不存在" in error or "缺少" in error or "不受支持" in error):
        raise CheckpointError(error, details=details)
    if "目录不存在" in error or "缺少必需文件" in error or "格式不受支持" in error:
        raise CheckpointError(error, details=details)
    if "依赖未安装" in error or "缺少 safetensors" in error:
        raise DependencyError(error, details=details)
    raise ModelLoadError(error, details=details)


def create_policy(
    model_type: str = "act",
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    *,
    config: Optional[PolicyLoadConfig] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    device_config: Optional[DeviceConfig] = None,
) -> BaseInferenceEngine:
    """
    Create an unloaded policy instance.

    Preferred v1 usage is `create_policy(config=PolicyLoadConfig(...))`.
    Legacy keyword arguments remain supported for compatibility.
    """
    resolved_model_type = _resolve_model_type(config=config, model_type=model_type)
    resolved_device = _resolve_device_config(
        config=config,
        device_config=device_config,
        device=device,
    )
    resolved_smoothing = _resolve_smoothing_config(
        config=config,
        runtime_config=runtime_config,
        smoothing_config=smoothing_config,
    )

    _load_engine_exports()

    if resolved_model_type == "act":
        return ACTInferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
        )
    if resolved_model_type == "smolvla":
        return SmolVLAInferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
        )
    if resolved_model_type == "pi0":
        return PI0InferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
        )
    raise ValidationError(
        f"Unknown model type: {resolved_model_type}",
        details={"supported_model_types": list(SUPPORTED_MODEL_TYPES)},
    )


def load_policy(
    checkpoint_dir: Optional[str] = None,
    model_type: str = "act",
    *,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    tokenizer_path: Optional[str] = None,
    instruction: Optional[str] = None,
    config: Optional[PolicyLoadConfig] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    device_config: Optional[DeviceConfig] = None,
) -> BaseInferenceEngine:
    """
    Create and load a policy in one step.

    Raises:
        SDKError: if validation, loading, or optional instruction setup fails.
    """
    if config is not None:
        config.validate()
        checkpoint_dir = config.checkpoint_dir
        tokenizer_path = config.tokenizer_path
        instruction = config.instruction

    if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
        raise ValidationError("`checkpoint_dir` must be provided")

    resolved_model_type = _resolve_model_type(config=config, model_type=model_type)
    policy = create_policy(
        model_type=resolved_model_type,
        device=device,
        smoothing_config=smoothing_config,
        config=config,
        runtime_config=runtime_config,
        device_config=device_config,
    )

    try:
        valid, error = policy.validate_checkpoint(checkpoint_dir)
        if not valid:
            raise CheckpointError(
                error,
                details={"model_type": resolved_model_type, "checkpoint_dir": checkpoint_dir},
            )

        if resolved_model_type == "pi0":
            success, error = policy.load(checkpoint_dir, tokenizer_path=tokenizer_path)
        else:
            success, error = policy.load(checkpoint_dir)

        if not success:
            _raise_load_error(error, model_type=resolved_model_type, checkpoint_dir=checkpoint_dir)

        if instruction is not None:
            set_instruction = getattr(policy, "set_instruction", None)
            if callable(set_instruction):
                if not set_instruction(instruction):
                    raise InferenceError(
                        "Failed to set policy instruction",
                        details={"model_type": resolved_model_type, "instruction": instruction},
                    )
            else:
                raise InstructionNotSupportedError(
                    f"{resolved_model_type} policy does not support language instructions",
                    details={"model_type": resolved_model_type, "instruction": instruction},
                )

        return policy
    except SDKError:
        try:
            policy.unload()
        except Exception:
            logger.debug("Failed to unload policy after SDKError", exc_info=True)
        raise
    except Exception as exc:
        try:
            policy.unload()
        except Exception:
            logger.debug("Failed to unload policy after unexpected error", exc_info=True)
        raise ModelLoadError(
            "Unexpected error while loading policy",
            details={
                "model_type": resolved_model_type,
                "checkpoint_dir": checkpoint_dir,
                "cause": repr(exc),
            },
        ) from exc


class InferenceSession:
    """
    Developer-facing SDK session for model loading, chunk inference, and shutdown.

    Typical lifecycle:
    1. `load(model_type=..., checkpoint_dir=...)`
    2. `infer(observation)` or `step(observation)`
    3. `close()`
    """

    def __init__(self):
        self._policy: Optional[BaseInferenceEngine] = None

    @property
    def is_loaded(self) -> bool:
        return bool(self._policy is not None and self._policy.is_loaded)

    @property
    def policy(self) -> BaseInferenceEngine:
        return self._require_policy()

    @classmethod
    def open(
        cls,
        checkpoint_dir: Optional[str] = None,
        model_type: str = "act",
        *,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        tokenizer_path: Optional[str] = None,
        instruction: Optional[str] = None,
        config: Optional[PolicyLoadConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        device_config: Optional[DeviceConfig] = None,
    ) -> "InferenceSession":
        session = cls()
        session.load(
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            device=device,
            smoothing_config=smoothing_config,
            tokenizer_path=tokenizer_path,
            instruction=instruction,
            config=config,
            runtime_config=runtime_config,
            device_config=device_config,
        )
        return session

    def _require_policy(self) -> BaseInferenceEngine:
        if self._policy is None or not self._policy.is_loaded:
            raise ResourceStateError(
                "No loaded policy is available. Call `InferenceSession.load(...)` first."
            )
        return self._policy

    def load(
        self,
        checkpoint_dir: Optional[str] = None,
        model_type: str = "act",
        *,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        tokenizer_path: Optional[str] = None,
        instruction: Optional[str] = None,
        config: Optional[PolicyLoadConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        device_config: Optional[DeviceConfig] = None,
    ) -> "InferenceSession":
        self.close()
        self._policy = load_policy(
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            device=device,
            smoothing_config=smoothing_config,
            tokenizer_path=tokenizer_path,
            instruction=instruction,
            config=config,
            runtime_config=runtime_config,
            device_config=device_config,
        )
        return self

    def infer(self, observation: Observation) -> np.ndarray:
        """Run one raw model inference and return the action chunk."""
        return self.predict_chunk(observation)

    def predict_chunk(self, observation: Observation) -> np.ndarray:
        """Return the raw action chunk for the provided observation."""
        return self._require_policy().predict_chunk(observation)

    def step(self, observation: Observation) -> np.ndarray:
        """Return the next action for control-loop execution."""
        return self._require_policy().step(observation)

    def get_metadata(self) -> PolicyMetadata:
        return self._require_policy().get_metadata()

    def get_status(self) -> PolicyStatus:
        return self._require_policy().get_status()

    def start_async_inference(self) -> None:
        self._require_policy().start_async_inference()

    def stop_async_inference(self) -> None:
        self._require_policy().stop_async_inference()

    def unload(self) -> None:
        policy = self._policy
        self._policy = None
        if policy is not None:
            policy.unload()

    def close(self) -> None:
        self.unload()

    def __enter__(self) -> "InferenceSession":
        self._require_policy()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def __getattr__(name: str):
    if name in {
        "ACTInferenceEngine",
        "SmolVLAInferenceEngine",
        "PI0InferenceEngine",
        "ACT_AVAILABLE",
        "SMOLVLA_AVAILABLE",
        "PI0_AVAILABLE",
    }:
        _load_engine_exports()
        return globals()[name]
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
    "GripperSmoother",
    "AsyncInferenceWorker",
    "LatencyEstimator",
    # Utilities
    "AGGREGATE_FUNCTIONS",
    "get_aggregate_function",
    # Availability flags
    "ACT_AVAILABLE",
    "SMOLVLA_AVAILABLE",
    "PI0_AVAILABLE",
    "TraceRecorder",
    "TraceEvent",
    "get_inference_monitor",
    "set_inference_monitor",
    # Exceptions
    "SDKError",
    "ConfigurationError",
    "ValidationError",
    "CheckpointError",
    "DependencyError",
    "DeviceUnavailableError",
    "ModelLoadError",
    "InstructionNotSupportedError",
    "ResourceStateError",
    "InferenceError",
    "InferenceTimeoutError",
]
