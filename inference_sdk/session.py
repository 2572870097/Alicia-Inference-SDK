"""Developer-facing inference session wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .core.config import DeviceConfig, PolicyLoadConfig, RuntimeConfig
from .core.types import Observation, PolicyMetadata, PolicyStatus
from .factory import load_policy
from .runtime.base import BaseInferenceEngine, SmoothingConfig


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
        robot_type: Optional[str] = None,
        policy_robot_type: Optional[str] = None,
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
            robot_type=robot_type,
            policy_robot_type=policy_robot_type,
        )
        return session

    def _require_policy(self) -> BaseInferenceEngine:
        if self._policy is None or not self._policy.is_loaded:
            raise RuntimeError(
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
        robot_type: Optional[str] = None,
        policy_robot_type: Optional[str] = None,
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
            robot_type=robot_type,
            policy_robot_type=policy_robot_type,
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


__all__ = ["InferenceSession"]
