#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_sdk import (
    DeviceConfig,
    InferenceSession,
    PolicyLoadConfig,
    RuntimeConfig,
    SDKError,
)


def build_synthetic_observation(
    required_cameras: Iterable[str],
    state_dim: int,
    *,
    instruction: Optional[str] = None,
    step_idx: int = 0,
):
    from inference_sdk import Observation

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
        state[:] = np.linspace(-0.3, 0.3, state_dim, dtype=np.float32)
    if state_dim >= 7:
        state[-1] = 500.0 + 200.0 * math.sin(step_idx / 5.0)

    return Observation(images=images, state=state, instruction=instruction)


def pretty_print(label: str, payload: Any) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    print(f"{label}:")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def run_policy_example(
    *,
    model_type: str,
    checkpoint_dir: str,
    device: str = "cuda:0",
    instruction: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    control_fps: float = 20.0,
    sync: bool = False,
) -> int:
    config = PolicyLoadConfig(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        device=DeviceConfig(device=device),
        runtime=RuntimeConfig(
            control_fps=control_fps,
            enable_async_inference=not sync,
        ),
        tokenizer_path=tokenizer_path,
        instruction=instruction,
    )

    try:
        session = InferenceSession()
        session.load(config=config)

        try:
            metadata = session.get_metadata()
            pretty_print("Metadata", metadata)
            pretty_print("Initial Status", session.get_status())

            observation = build_synthetic_observation(
                required_cameras=metadata.required_cameras,
                state_dim=metadata.state_dim,
                instruction=instruction,
            )

            chunk = session.infer(observation)
            pretty_print("Predicted Chunk", chunk)

            started_async = False
            if config.runtime.enable_async_inference:
                session.start_async_inference()
                started_async = True

            try:
                action = session.step(observation)
                pretty_print("Selected Action", action)
                pretty_print("Final Status", session.get_status())
            finally:
                if started_async:
                    session.stop_async_inference()
        finally:
            session.close()

        return 0
    except SDKError as exc:
        pretty_print("SDKError", exc.to_dict())
        return 1


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
