from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..runtime.base import SmoothingConfig
from .exceptions import ConfigurationError, ValidationError


SUPPORTED_MODEL_TYPES = ("act", "pi0", "smolvla")


@dataclass(frozen=True)
class DeviceConfig:
    """Stable device-selection contract for SDK callers."""

    device: str = "cuda:0"

    def validate(self) -> None:
        normalized = self.device.strip() if isinstance(self.device, str) else ""
        if not normalized:
            raise ConfigurationError("`device` must be a non-empty string")
        if normalized == "cpu" or normalized == "cuda":
            return
        if normalized.startswith("cuda:") and normalized.split(":", 1)[1].isdigit():
            return
        raise ConfigurationError(
            "`device` must be one of: `cpu`, `cuda`, or `cuda:<index>`"
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Developer-facing runtime config for control-loop behavior."""

    control_fps: float = 30.0
    gripper_max_velocity: float = 200.0
    enable_gripper_clamping: bool = True
    enable_async_inference: bool = True
    chunk_size_threshold: float = 0.5
    latency_ema_alpha: float = 0.2
    latency_safety_margin: float = 1.5
    aggregate_fn_name: str = "latest_only"
    obs_queue_maxsize: int = 1
    fallback_mode: str = "repeat"

    def validate(self) -> None:
        if self.control_fps <= 0:
            raise ConfigurationError("`control_fps` must be > 0")
        if self.gripper_max_velocity < 0:
            raise ConfigurationError("`gripper_max_velocity` must be >= 0")
        if not 0 < self.chunk_size_threshold <= 1:
            raise ConfigurationError("`chunk_size_threshold` must be in (0, 1]")
        if not 0 < self.latency_ema_alpha <= 1:
            raise ConfigurationError("`latency_ema_alpha` must be in (0, 1]")
        if self.latency_safety_margin <= 0:
            raise ConfigurationError("`latency_safety_margin` must be > 0")
        if self.obs_queue_maxsize <= 0:
            raise ConfigurationError("`obs_queue_maxsize` must be > 0")
        if self.fallback_mode not in {"repeat", "hold"}:
            raise ConfigurationError("`fallback_mode` must be one of: repeat, hold")

    def to_smoothing_config(self) -> SmoothingConfig:
        self.validate()
        return SmoothingConfig(
            control_fps=self.control_fps,
            gripper_max_velocity=self.gripper_max_velocity,
            enable_gripper_clamping=self.enable_gripper_clamping,
            enable_async_inference=self.enable_async_inference,
            chunk_size_threshold=self.chunk_size_threshold,
            latency_ema_alpha=self.latency_ema_alpha,
            latency_safety_margin=self.latency_safety_margin,
            aggregate_fn_name=self.aggregate_fn_name,
            obs_queue_maxsize=self.obs_queue_maxsize,
            fallback_mode=self.fallback_mode,
        )

    @classmethod
    def from_smoothing_config(cls, config: SmoothingConfig) -> "RuntimeConfig":
        return cls(
            control_fps=config.control_fps,
            gripper_max_velocity=config.gripper_max_velocity,
            enable_gripper_clamping=config.enable_gripper_clamping,
            enable_async_inference=config.enable_async_inference,
            chunk_size_threshold=config.chunk_size_threshold,
            latency_ema_alpha=config.latency_ema_alpha,
            latency_safety_margin=config.latency_safety_margin,
            aggregate_fn_name=config.aggregate_fn_name,
            obs_queue_maxsize=config.obs_queue_maxsize,
            fallback_mode=config.fallback_mode,
        )


@dataclass(frozen=True)
class PolicyLoadConfig:
    """Stable developer-facing config for creating and loading a policy."""

    checkpoint_dir: str
    model_type: str = "act"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tokenizer_path: Optional[str] = None
    instruction: Optional[str] = None

    def validate(self) -> None:
        model_type = self.model_type.lower() if isinstance(self.model_type, str) else ""
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValidationError(
                f"Unsupported model_type `{self.model_type}`. "
                f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ConfigurationError("`checkpoint_dir` must be a non-empty string")
        self.device.validate()
        self.runtime.validate()

    @property
    def normalized_model_type(self) -> str:
        self.validate()
        return self.model_type.lower()

    def to_smoothing_config(self) -> SmoothingConfig:
        self.validate()
        return self.runtime.to_smoothing_config()
