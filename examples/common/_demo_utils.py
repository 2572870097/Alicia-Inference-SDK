#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from inference_sdk import Observation, SmoothingConfig
from inference_sdk.base import BaseInferenceEngine


class DummyPolicy(BaseInferenceEngine):
    """
    Minimal runnable policy used by SDK examples.

    It echoes the state as the next action and synthesizes a short action chunk
    with small offsets so that `step()` and `predict_chunk()` can both be shown
    without any external model files.
    """

    def __init__(self):
        super().__init__(SmoothingConfig(control_fps=10.0, enable_async_inference=False))
        self.model_type = "dummy"
        self.required_cameras = ["head", "wrist"]
        self.state_dim = 7
        self.action_dim = 7
        self.chunk_size = 3
        self.n_action_steps = 3
        self.requested_device = "cpu"
        self.actual_device = "cpu"
        self.is_loaded = True
        self._instruction = ""
        self._init_components()
        self.reset()

    def load(self, checkpoint_dir: str):
        self.is_loaded = True
        return True, ""

    def _predict_chunk(self, images, state):
        action = state[: self.action_dim].astype(np.float32)
        return np.stack([action, action + 1.0, action + 2.0])

    def set_instruction(self, instruction: str) -> bool:
        self._instruction = instruction
        return True

    def get_instruction(self) -> str:
        return self._instruction

    def unload(self):
        self.is_loaded = False


def build_synthetic_observation(
    required_cameras: Iterable[str],
    state_dim: int,
    step_idx: int = 0,
    instruction: Optional[str] = None,
) -> Observation:
    """
    Build a synthetic observation that works for demos and synthetic control loops.
    """
    images: Dict[str, np.ndarray] = {}
    height, width = 480, 640

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    grid_x = np.tile(x, (height, 1))
    grid_y = np.tile(y.reshape(-1, 1), (1, width))

    for offset, camera_role in enumerate(required_cameras):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (grid_x + step_idx * 5 + offset * 17) % 255
        frame[..., 1] = (grid_y + offset * 33) % 255
        frame[..., 2] = ((grid_x // 2) + (grid_y // 2) + step_idx * 3) % 255
        images[camera_role] = frame

    state = np.zeros(state_dim, dtype=np.float32)
    if state_dim > 0:
        ramp = np.linspace(-0.3, 0.3, state_dim, dtype=np.float32)
        state[:] = ramp
    if state_dim >= 7:
        state[-1] = 500.0 + 200.0 * math.sin(step_idx / 5.0)

    return Observation(images=images, state=state, instruction=instruction)


def pretty_print(label: str, payload: Any) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    print(f"{label}:")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
