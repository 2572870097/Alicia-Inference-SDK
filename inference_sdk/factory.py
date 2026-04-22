"""Policy construction and loading helpers."""

from __future__ import annotations

import logging
from typing import Optional

from . import policy as policy_exports
from .core.config import DeviceConfig, PolicyLoadConfig, RuntimeConfig, SUPPORTED_MODEL_TYPES
from .runtime.base import BaseInferenceEngine, SmoothingConfig

logger = logging.getLogger(__name__)


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower() if isinstance(model_type, str) else ""
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported model types: {list(SUPPORTED_MODEL_TYPES)}"
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
    return RuntimeConfig(enable_async_inference=True).to_smoothing_config()


def _resolve_model_type(*, config: Optional[PolicyLoadConfig], model_type: str) -> str:
    if config is not None:
        config.validate()
        return config.normalized_model_type
    return _normalize_model_type(model_type)


def _raise_load_error(error: str, *, model_type: str, checkpoint_dir: str) -> None:
    if not error:
        raise RuntimeError(
            "Model loading failed without a detailed error message "
            f"(model_type={model_type}, checkpoint_dir={checkpoint_dir})"
        )

    lowered = error.lower()
    if "checkpoint" in lowered and ("不存在" in error or "缺少" in error or "不受支持" in error):
        raise ValueError(error)
    if "目录不存在" in error or "缺少必需文件" in error or "格式不受支持" in error:
        raise ValueError(error)
    if "依赖未安装" in error or "缺少 safetensors" in error:
        raise ImportError(error)
    raise RuntimeError(error)


def create_policy(
    model_type: str = "act",
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    *,
    config: Optional[PolicyLoadConfig] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    device_config: Optional[DeviceConfig] = None,
    robot_type: Optional[str] = None,
    policy_robot_type: Optional[str] = None,
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
    resolved_robot_type = config.robot_type if config is not None else robot_type
    resolved_policy_robot_type = (
        config.policy_robot_type if config is not None else policy_robot_type
    )

    if resolved_model_type == "act":
        return policy_exports.ACTInferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
            robot_type=resolved_robot_type,
            policy_robot_type=resolved_policy_robot_type,
        )
    if resolved_model_type == "smolvla":
        return policy_exports.SmolVLAInferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
        )
    if resolved_model_type == "pi0":
        return policy_exports.PI0InferenceEngine(
            device=resolved_device.device,
            smoothing_config=resolved_smoothing,
        )
    raise ValueError(
        f"Unknown model type: {resolved_model_type}. "
        f"Supported model types: {list(SUPPORTED_MODEL_TYPES)}"
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
    robot_type: Optional[str] = None,
    policy_robot_type: Optional[str] = None,
) -> BaseInferenceEngine:
    """
    Create and load a policy in one step.

    Raises:
        ValueError, RuntimeError, ImportError: if validation, loading, or optional instruction setup fails.
    """
    if config is not None:
        config.validate()
        checkpoint_dir = config.checkpoint_dir
        tokenizer_path = config.tokenizer_path
        instruction = config.instruction
        robot_type = config.robot_type
        policy_robot_type = config.policy_robot_type

    if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
        raise ValueError("`checkpoint_dir` must be provided")

    resolved_model_type = _resolve_model_type(config=config, model_type=model_type)
    policy = create_policy(
        model_type=resolved_model_type,
        device=device,
        smoothing_config=smoothing_config,
        config=config,
        runtime_config=runtime_config,
        device_config=device_config,
        robot_type=robot_type,
        policy_robot_type=policy_robot_type,
    )

    try:
        valid, error = policy.validate_checkpoint(checkpoint_dir)
        if not valid:
            raise ValueError(error)

        if resolved_model_type == "pi0":
            success, error = policy.load(checkpoint_dir, tokenizer_path=tokenizer_path)
        else:
            success, error = policy.load(checkpoint_dir)

        if not success:
            _raise_load_error(
                error,
                model_type=resolved_model_type,
                checkpoint_dir=checkpoint_dir,
            )

        if instruction is not None:
            set_instruction = getattr(policy, "set_instruction", None)
            if callable(set_instruction):
                if not set_instruction(instruction):
                    raise RuntimeError(
                        f"Failed to set policy instruction: {instruction}"
                    )
            else:
                raise RuntimeError(
                    f"{resolved_model_type} policy does not support language instructions"
                )

        return policy
    except Exception:
        try:
            policy.unload()
        except Exception:
            logger.debug("Failed to unload policy after unexpected error", exc_info=True)
        raise


__all__ = ["create_policy", "load_policy"]
