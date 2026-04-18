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

from .base import (
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
from .monitoring import get_inference_monitor, set_inference_monitor
from .types import Observation, PolicyMetadata, PolicyStatus

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

    from .act import ACTInferenceEngine as _ACTInferenceEngine, ACT_AVAILABLE as _ACT_AVAILABLE
    from .pi0 import PI0InferenceEngine as _PI0InferenceEngine, PI0_AVAILABLE as _PI0_AVAILABLE
    from .smolvla import (
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


def create_policy(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> BaseInferenceEngine:
    """
    Factory function to create inference engine by type.
    
    Args:
        model_type: "act", "smolvla" or "pi0"
        device: PyTorch device string
        smoothing_config: Optional smoothing configuration (defaults to async mode)
        strict_device: If True, fail instead of silently falling back to another device
        
    Returns:
        Inference engine instance
    """
    # Default to async mode if no config provided
    if smoothing_config is None:
        smoothing_config = SmoothingConfig(
            enable_async_inference=True,
            aggregate_fn_name="latest_only",
        )

    _load_engine_exports()

    if model_type.lower() == "act":
        return ACTInferenceEngine(device=device, smoothing_config=smoothing_config, strict_device=strict_device)
    if model_type.lower() == "smolvla":
        return SmolVLAInferenceEngine(device=device, smoothing_config=smoothing_config, strict_device=strict_device)
    if model_type.lower() == "pi0":
        return PI0InferenceEngine(device=device, smoothing_config=smoothing_config, strict_device=strict_device)
    raise ValueError(f"Unknown model type: {model_type}")


def create_inference_engine(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> BaseInferenceEngine:
    """Backward-compatible alias for legacy callers."""
    return create_policy(
        model_type=model_type,
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    )


def load_policy(
    checkpoint_dir: str,
    model_type: str = "act",
    *,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
    tokenizer_path: Optional[str] = None,
    instruction: Optional[str] = None,
) -> BaseInferenceEngine:
    """
    Create and load a policy in one step.

    Raises:
        RuntimeError: if loading or optional instruction setup fails.
    """
    policy = create_policy(
        model_type=model_type,
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    )

    if model_type.lower() == "pi0":
        success, error = policy.load(checkpoint_dir, tokenizer_path=tokenizer_path)
    else:
        success, error = policy.load(checkpoint_dir)

    if not success:
        raise RuntimeError(error)

    if instruction is not None:
        set_instruction = getattr(policy, "set_instruction", None)
        if callable(set_instruction):
            if not set_instruction(instruction):
                raise RuntimeError("Failed to set policy instruction")
        else:
            raise RuntimeError(f"{model_type} policy does not support language instructions")

    return policy


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
    # Core classes
    "BaseInferenceEngine",
    "ACTInferenceEngine",
    "SmolVLAInferenceEngine",
    "PI0InferenceEngine",
    "create_policy",
    "load_policy",
    "create_inference_engine",
    # Config
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
]
