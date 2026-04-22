"""
ACT (Action Chunking Transformer) inference engine.

This engine keeps the SDK's queue-aware runtime, but delegates ACT policy loading
and normalization logic to the SparkMind LeRobot-compat implementation:

- `ACTPolicy.from_pretrained(...)` loads the exported policy config + weights
- `PolicyProcessorPipeline.from_pretrained(...)` loads the exported processor pipelines
- the SDK only bridges `images + state` into the policy's expected batch format
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..runtime.base import BaseInferenceEngine, SmoothingConfig
from ..runtime.device import resolve_torch_device
from .robot_adapter import RobotAdapter, create_robot_adapter

logger = logging.getLogger(__name__)


PRETRAINED_CHECKPOINT_FILES = ("config.json", "model.safetensors")
PRETRAINED_SUBDIR_NAME = "pretrained_model"
PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"


def _resolve_act_checkpoint_dir(checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir)
    if path.name == PRETRAINED_SUBDIR_NAME:
        return path

    pretrained_dir = path / PRETRAINED_SUBDIR_NAME
    if pretrained_dir.is_dir():
        return pretrained_dir

    return path


def _extract_camera_role(image_feature_key: str) -> Tuple[str, str]:
    suffix = image_feature_key.removeprefix("observation.images.")
    role = suffix[4:] if suffix.startswith("cam_") else suffix
    return suffix, role

ACT_AVAILABLE = False
ACT_IMPORT_ERROR: Optional[Exception] = None
ACT_MISSING_DEPENDENCIES = tuple(
    dependency
    for dependency in ("torch", "torchvision", "einops", "safetensors", "huggingface_hub", "draccus")
    if importlib.util.find_spec(dependency) is None
)
try:
    if ACT_MISSING_DEPENDENCIES:
        raise ImportError(f"Missing ACT dependencies: {', '.join(ACT_MISSING_DEPENDENCIES)}")

    from sparkmind.lerobot_compat.policies.act.configuration_act import ACTConfig
    from sparkmind.lerobot_compat.policies.act.modeling_act import ACTPolicy
    from sparkmind.lerobot_compat.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from sparkmind.lerobot_compat.processor.pipeline import PolicyProcessorPipeline
    import sparkmind.lerobot_compat.processor.batch_processor  # noqa: F401
    import sparkmind.lerobot_compat.processor.device_processor  # noqa: F401
    import sparkmind.lerobot_compat.processor.normalize_processor  # noqa: F401
    import sparkmind.lerobot_compat.processor.rename_processor  # noqa: F401

    ACT_AVAILABLE = True
    logger.info("SparkMind ACT policy loaded successfully")
except Exception as e:
    ACT_IMPORT_ERROR = e
    logger.warning("SparkMind ACT policy not available: %s", e)


class ACTInferenceEngine(BaseInferenceEngine):
    """
    ACT inference engine with SDK runtime scheduling and SparkMind-compatible
    ACT policy / processor loading.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        robot_type: Optional[str] = None,
        policy_robot_type: Optional[str] = None,
    ):
        super().__init__(smoothing_config)
        self.model_type = "act"
        device_selection = resolve_torch_device(device)
        self.requested_device = device_selection.requested
        self.actual_device = device_selection.actual
        self.device_warning = device_selection.warning
        if self.device_warning:
            logger.warning(self.device_warning)
        self.device = torch.device(self.actual_device)
        self.robot_adapter: RobotAdapter = create_robot_adapter(
            robot_type,
            policy_robot_type=policy_robot_type,
        )

        self.policy: Optional[Any] = None
        self.config: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.postprocessor: Optional[Any] = None
        self.loaded_n_action_steps: int = 1

        # Camera role mapping: image_feature -> camera aliases accepted by the SDK.
        self._camera_key_to_role: Dict[str, str] = {}
        self._role_to_camera_key: Dict[str, str] = {}
        self._camera_alias_to_key: Dict[str, str] = {}

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Perform a minimal ACT checkpoint path validation."""
        raw_path = Path(checkpoint_dir)
        if not raw_path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"

        resolved_path = _resolve_act_checkpoint_dir(checkpoint_dir)
        if not resolved_path.is_dir():
            return False, f"Checkpoint路径不是目录: {resolved_path}"

        return True, ""

    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load ACT policy from an exported checkpoint directory."""
        if not ACT_AVAILABLE:
            if ACT_MISSING_DEPENDENCIES:
                return False, f"ACT模型依赖未安装: {', '.join(ACT_MISSING_DEPENDENCIES)}"
            if ACT_IMPORT_ERROR is not None:
                return False, f"ACT模型依赖初始化失败: {ACT_IMPORT_ERROR}"
            return False, "ACT模型依赖未安装 (SparkMind lerobot_compat)"

        valid, error = self.validate_checkpoint(checkpoint_dir)
        if not valid:
            return False, error

        checkpoint_path = _resolve_act_checkpoint_dir(checkpoint_dir)

        try:
            config = ACTConfig.from_pretrained(str(checkpoint_path))
            config.device = self.actual_device
            # The exported checkpoint already contains full weights. Do not trigger
            # torchvision backbone downloads while reconstructing the model.
            config.pretrained_backbone_weights = None

            policy = ACTPolicy.from_pretrained(str(checkpoint_path), config=config)
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                str(checkpoint_path),
                config_filename=PREPROCESSOR_CONFIG_FILENAME,
                overrides={
                    "device_processor": {
                        "device": self.actual_device,
                        "float_dtype": "float32",
                    }
                },
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                str(checkpoint_path),
                config_filename=POSTPROCESSOR_CONFIG_FILENAME,
                overrides={
                    "device_processor": {
                        "device": "cpu",
                        "float_dtype": "float32",
                    }
                },
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )

            self.policy = policy
            self.config = policy.config
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor

            image_feature_keys = list(self.config.image_features.keys())
            self.required_cameras = []
            self._camera_key_to_role = {}
            self._role_to_camera_key = {}
            self._camera_alias_to_key = {}

            for key in image_feature_keys:
                if not key.startswith("observation.images."):
                    continue
                suffix, role = _extract_camera_role(key)
                if role not in self.required_cameras:
                    self.required_cameras.append(role)
                self._camera_key_to_role[key] = role
                self._role_to_camera_key[role] = key
                self._camera_alias_to_key[role] = key
                self._camera_alias_to_key[suffix] = key
                self._camera_alias_to_key[key] = key

            robot_state_feature = self.config.robot_state_feature
            action_feature = self.config.action_feature
            if robot_state_feature is None:
                raise ValueError("ACT导出模型缺少 observation.state 输入定义")
            if action_feature is None:
                raise ValueError("ACT导出模型缺少 action 输出定义")

            self.state_dim = int(robot_state_feature.shape[0])
            self.action_dim = int(action_feature.shape[0])
            self.chunk_size = int(self.config.chunk_size)
            self.loaded_n_action_steps = int(self.config.n_action_steps)
            self.n_action_steps = self.loaded_n_action_steps

            if self.chunk_size > 1 and self.n_action_steps <= 1:
                logger.warning(
                    "ACT checkpoint reports n_action_steps=%s with chunk_size=%s; "
                    "overriding execution to consume the full chunk for real-robot control.",
                    self.loaded_n_action_steps,
                    self.chunk_size,
                )
                self.n_action_steps = self.chunk_size

            logger.info("ACT policy loaded from %s", checkpoint_dir)
            logger.info("Required cameras: %s", self.required_cameras)
            logger.info("State dim: %s, Action dim: %s", self.state_dim, self.action_dim)
            logger.info("Chunk size: %s, N action steps: %s", self.chunk_size, self.n_action_steps)
            logger.info(
                "Using ACT robot adapter: runtime=%s, policy=%s",
                self.robot_adapter.robot_type or "generic",
                self.robot_adapter.policy_robot_type or "generic",
            )

            self.is_loaded = True
            self._init_components()
            self.reset()
            return True, ""

        except Exception as e:
            logger.error("Failed to load ACT model: %s", e)
            import traceback

            traceback.print_exc()
            return False, f"模型加载失败: {str(e)}"

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
        if self.preprocessor is not None and hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if self.postprocessor is not None and hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()
        super().reset()

    def build_inference_frame(self, images: Dict[str, np.ndarray], state: np.ndarray) -> Dict[str, torch.Tensor]:
        """Build the raw ACT inference frame before SparkMind pre/post processing.

        This mirrors the role of LeRobot's `build_inference_frame(...)`: it adapts
        SDK-native robot observations into the feature-keyed tensor dict expected by
        the exported ACT processor pipeline.
        """
        if self.config is None:
            raise RuntimeError("ACT model config is not loaded")

        frame: Dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(self.robot_adapter.state_to_policy(state)),
        }

        for camera_alias, image_bgr in images.items():
            camera_key = self._camera_alias_to_key.get(camera_alias)
            if camera_key is None:
                continue
            frame[camera_key] = self.robot_adapter.image_to_policy_tensor(image_bgr)

        expected_image_features = set(self.config.image_features.keys())
        provided_image_features = {key for key in frame if key.startswith("observation.images.")}
        missing = sorted(expected_image_features - provided_image_features)
        if missing:
            raise ValueError(
                "Missing required ACT camera observations: "
                f"{missing} (required_cameras={list(self.required_cameras)})"
            )

        return frame

    @torch.no_grad()
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Internal method to predict a chunk of actions.

        Returns:
            Action chunk (n_action_steps, action_dim) in robot action space.
        """
        if self.policy is None or self.preprocessor is None or self.postprocessor is None:
            raise RuntimeError("ACT model is not loaded")

        inference_frame = self.build_inference_frame(images, state)
        processed_batch = self.preprocessor(inference_frame)
        normalized_chunk = self.policy.predict_action_chunk(processed_batch)
        action_chunk = self.postprocessor(normalized_chunk)

        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.detach().cpu().numpy()
        else:
            action_chunk = np.asarray(action_chunk, dtype=np.float32)

        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]
        elif action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]

        action_chunk = action_chunk[: self.n_action_steps, : self.action_dim]
        return self.robot_adapter.action_from_policy(action_chunk)

    def unload(self):
        """Unload model and free memory."""
        self.policy = None
        self.config = None
        self.preprocessor = None
        self.postprocessor = None
        self.is_loaded = False
        self.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ACT model unloaded")
