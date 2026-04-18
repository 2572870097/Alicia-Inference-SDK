#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Optional

from inference_sdk import SmoothingConfig, create_policy, load_policy

from common._demo_utils import build_synthetic_observation, pretty_print


def run_load_and_inspect(
    model_type: str,
    checkpoint_dir: str,
    *,
    device: str = "cuda:0",
    strict_device: bool = False,
    tokenizer_path: Optional[str] = None,
    instruction: Optional[str] = None,
) -> None:
    policy = None
    try:
        policy = load_policy(
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            device=device,
            strict_device=strict_device,
            tokenizer_path=tokenizer_path,
            instruction=instruction,
        )
        pretty_print("Metadata", policy.get_metadata())
        pretty_print("Status", policy.get_status())
    finally:
        if policy is not None:
            policy.unload()


def run_synthetic_control_loop(
    model_type: str,
    checkpoint_dir: str,
    *,
    device: str = "cuda:0",
    steps: int = 5,
    control_fps: float = 20.0,
    sync: bool = False,
    instruction: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> None:
    config = SmoothingConfig(
        control_fps=control_fps,
        enable_async_inference=not sync,
        aggregate_fn_name="latest_only",
    )

    policy = None
    try:
        policy = load_policy(
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            device=device,
            smoothing_config=config,
            tokenizer_path=tokenizer_path,
            instruction=instruction,
        )
        policy.start_async_inference()

        metadata = policy.get_metadata()
        pretty_print("Metadata", metadata)

        for step_idx in range(steps):
            observation = build_synthetic_observation(
                required_cameras=metadata.required_cameras,
                state_dim=metadata.state_dim,
                step_idx=step_idx,
                instruction=instruction,
            )
            action = policy.step(observation)
            status = policy.get_status()
            print(
                f"step={step_idx:02d} "
                f"action[:3]={action[:3].tolist()} "
                f"queue_size={status.queue_size} "
                f"latency_ms={status.latency_estimate_ms:.2f}"
            )
            time.sleep(1.0 / max(control_fps, 1.0))
    finally:
        if policy is not None:
            policy.stop_async_inference()
            policy.unload()


def run_manual_policy_lifecycle(
    model_type: str,
    checkpoint_dir: str,
    *,
    device: str = "cuda:0",
    instruction: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    sync: bool = False,
) -> None:
    policy = create_policy(
        model_type=model_type,
        device=device,
        smoothing_config=SmoothingConfig(enable_async_inference=not sync, control_fps=20.0),
    )

    try:
        if model_type == "pi0":
            success, error = policy.load(checkpoint_dir, tokenizer_path=tokenizer_path)
        else:
            success, error = policy.load(checkpoint_dir)
        if not success:
            raise RuntimeError(error)

        if instruction is not None:
            set_instruction = getattr(policy, "set_instruction", None)
            if callable(set_instruction):
                if not set_instruction(instruction):
                    raise RuntimeError("Failed to set instruction")

        pretty_print("Metadata", policy.get_metadata())

        metadata = policy.get_metadata()
        observation = build_synthetic_observation(
            required_cameras=metadata.required_cameras,
            state_dim=metadata.state_dim,
            step_idx=0,
            instruction=instruction,
        )

        chunk = policy.predict_chunk(observation)
        pretty_print("Predicted Chunk", chunk)

        policy.start_async_inference()
        action = policy.step(observation)
        pretty_print("Single Step Action", action)
        pretty_print("Status", policy.get_status())
    finally:
        policy.stop_async_inference()
        policy.unload()
