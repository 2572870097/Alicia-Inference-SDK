from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..runtime.base import SmoothingConfig


SUPPORTED_MODEL_TYPES = ("act", "pi0", "smolvla")


@dataclass(frozen=True)
class DeviceConfig:
    """Stable device-selection contract for SDK callers."""

    device: str = "cuda:0"

    def validate(self) -> None:
        normalized = self.device.strip() if isinstance(self.device, str) else ""
        if not normalized:
            raise ValueError("`device` must be a non-empty string")
        if normalized == "cpu" or normalized == "cuda":
            return
        if normalized.startswith("cuda:") and normalized.split(":", 1)[1].isdigit():
            return
        raise ValueError("`device` must be one of: `cpu`, `cuda`, or `cuda:<index>`")


@dataclass(frozen=True)
class RuntimeConfig:
    """Developer-facing runtime config for control-loop behavior."""

    control_fps: float = 30.0
    enable_async_inference: bool = True
    chunk_size_threshold: float = 0.5
    temporal_ensemble_coeff: Optional[float] = None
    obs_queue_maxsize: int = 1
    fallback_mode: str = "repeat"

    def validate(self) -> None:
        if self.control_fps <= 0:
            raise ValueError("`control_fps` must be > 0")
        if not 0 < self.chunk_size_threshold <= 1:
            raise ValueError("`chunk_size_threshold` must be in (0, 1]")
        if self.temporal_ensemble_coeff is not None and not math.isfinite(self.temporal_ensemble_coeff):
            raise ValueError("`temporal_ensemble_coeff` must be a finite float when provided")
        if self.obs_queue_maxsize <= 0:
            raise ValueError("`obs_queue_maxsize` must be > 0")
        if self.fallback_mode not in {"repeat", "hold"}:
            raise ValueError("`fallback_mode` must be one of: repeat, hold")

    def to_smoothing_config(self) -> SmoothingConfig:
        self.validate()
        return SmoothingConfig(
            control_fps=self.control_fps,
            enable_async_inference=self.enable_async_inference,
            chunk_size_threshold=self.chunk_size_threshold,
            temporal_ensemble_coeff=self.temporal_ensemble_coeff,
            obs_queue_maxsize=self.obs_queue_maxsize,
            fallback_mode=self.fallback_mode,
        )

    @classmethod
    def from_smoothing_config(cls, config: SmoothingConfig) -> "RuntimeConfig":
        return cls(
            control_fps=config.control_fps,
            enable_async_inference=config.enable_async_inference,
            chunk_size_threshold=config.chunk_size_threshold,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
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
    robot_type: Optional[str] = None
    policy_robot_type: Optional[str] = None

    def validate(self) -> None:
        model_type = self.model_type.lower() if isinstance(self.model_type, str) else ""
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type `{self.model_type}`. "
                f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError("`checkpoint_dir` must be a non-empty string")
        if self.robot_type is not None and (
            not isinstance(self.robot_type, str) or not self.robot_type.strip()
        ):
            raise ValueError("`robot_type` must be a non-empty string when provided")
        if self.policy_robot_type is not None and (
            not isinstance(self.policy_robot_type, str) or not self.policy_robot_type.strip()
        ):
            raise ValueError("`policy_robot_type` must be a non-empty string when provided")
        self.device.validate()
        self.runtime.validate()

    @property
    def normalized_model_type(self) -> str:
        self.validate()
        return self.model_type.lower()

    def to_smoothing_config(self) -> SmoothingConfig:
        self.validate()
        return self.runtime.to_smoothing_config()
