from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

ALICIA_D_ROBOT_TYPE = "alicia_d"
ALICIA_M_ROBOT_TYPE = "alicia_m"
SUPPORTED_ACT_ROBOT_TYPES = (ALICIA_D_ROBOT_TYPE, ALICIA_M_ROBOT_TYPE)

_ROBOT_TYPE_ALIASES = {
    "d": ALICIA_D_ROBOT_TYPE,
    "aliciad": ALICIA_D_ROBOT_TYPE,
    "aliciadsdk": ALICIA_D_ROBOT_TYPE,
    "alicia_d": ALICIA_D_ROBOT_TYPE,
    "alicia-d": ALICIA_D_ROBOT_TYPE,
    "alicia d": ALICIA_D_ROBOT_TYPE,
    "alicia_d_sdk": ALICIA_D_ROBOT_TYPE,
    "alicia-d-sdk": ALICIA_D_ROBOT_TYPE,
    "alicia d sdk": ALICIA_D_ROBOT_TYPE,
    "m": ALICIA_M_ROBOT_TYPE,
    "aliciam": ALICIA_M_ROBOT_TYPE,
    "aliciamsdk": ALICIA_M_ROBOT_TYPE,
    "alicia_m": ALICIA_M_ROBOT_TYPE,
    "alicia-m": ALICIA_M_ROBOT_TYPE,
    "alicia m": ALICIA_M_ROBOT_TYPE,
    "alicia_m_sdk": ALICIA_M_ROBOT_TYPE,
    "alicia-m-sdk": ALICIA_M_ROBOT_TYPE,
    "alicia m sdk": ALICIA_M_ROBOT_TYPE,
}

_ALICIA_D_LIMITS_DEG = np.asarray(
    [
        (-157.5, 157.5),
        (-100.2, 100.2),
        (-34.3, 126.0),
        (-159.8, 159.8),
        (-89.9, 89.9),
        (-179.9, 179.9),
    ],
    dtype=np.float32,
)
_ALICIA_M_LIMITS_DEG = np.asarray(
    [
        (-157.5, 157.5),
        (-179.9, 0.0),
        (-179.9, 0.0),
        (-89.9, 89.9),
        (-89.9, 89.9),
        (-157.5, 157.5),
    ],
    dtype=np.float32,
)
_REVERSED_JOINT_INDEXES = {3, 5}
_PROPORTIONAL_JOINT_INDEX = 2


def normalize_act_robot_type(robot_type: Optional[str], *, field_name: str) -> Optional[str]:
    """Normalize supported Alicia robot type aliases."""
    if robot_type is None:
        return None
    if not isinstance(robot_type, str):
        raise ValueError(f"`{field_name}` must be a string when provided")

    raw_value = robot_type.strip()
    if not raw_value:
        raise ValueError(f"`{field_name}` must be a non-empty string when provided")

    normalized = raw_value.lower()
    canonical = _ROBOT_TYPE_ALIASES.get(
        normalized,
        _ROBOT_TYPE_ALIASES.get(normalized.replace("_", "").replace("-", "").replace(" ", "")),
    )
    if canonical is None:
        raise ValueError(
            f"Unsupported `{field_name}` `{robot_type}`. "
            f"Supported robot types: {list(SUPPORTED_ACT_ROBOT_TYPES)}"
        )
    return canonical


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(value, upper)))


def _map_joint_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    return ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min) + dst_min


def _convert_alicia_d_joint_to_alicia_m_deg(value_deg: float, joint_index: int) -> float:
    if joint_index == _PROPORTIONAL_JOINT_INDEX:
        negated = -value_deg
        src_min = -float(_ALICIA_D_LIMITS_DEG[joint_index, 1])
        src_max = -float(_ALICIA_D_LIMITS_DEG[joint_index, 0])
        dst_min = float(_ALICIA_M_LIMITS_DEG[joint_index, 0])
        dst_max = float(_ALICIA_M_LIMITS_DEG[joint_index, 1])
        mapped = _map_joint_value(
            _clamp(negated, src_min, src_max),
            src_min,
            src_max,
            dst_min,
            dst_max,
        )
        return _clamp(mapped, min(dst_min, dst_max), max(dst_min, dst_max))

    dst_min = float(_ALICIA_M_LIMITS_DEG[joint_index, 0])
    dst_max = float(_ALICIA_M_LIMITS_DEG[joint_index, 1])
    dst_origin = (dst_min + dst_max) / 2.0
    direction = -1.0 if joint_index in _REVERSED_JOINT_INDEXES else 1.0
    mapped = direction * value_deg + dst_origin
    return _clamp(mapped, min(dst_min, dst_max), max(dst_min, dst_max))


def _convert_alicia_m_joint_to_alicia_d_deg(value_deg: float, joint_index: int) -> float:
    if joint_index == _PROPORTIONAL_JOINT_INDEX:
        src_min = float(_ALICIA_M_LIMITS_DEG[joint_index, 0])
        src_max = float(_ALICIA_M_LIMITS_DEG[joint_index, 1])
        dst_min = -float(_ALICIA_D_LIMITS_DEG[joint_index, 1])
        dst_max = -float(_ALICIA_D_LIMITS_DEG[joint_index, 0])
        negated = _map_joint_value(
            _clamp(value_deg, src_min, src_max),
            src_min,
            src_max,
            dst_min,
            dst_max,
        )
        mapped = -negated
        d_min = float(_ALICIA_D_LIMITS_DEG[joint_index, 0])
        d_max = float(_ALICIA_D_LIMITS_DEG[joint_index, 1])
        return _clamp(mapped, min(d_min, d_max), max(d_min, d_max))

    src_min = float(_ALICIA_M_LIMITS_DEG[joint_index, 0])
    src_max = float(_ALICIA_M_LIMITS_DEG[joint_index, 1])
    src_origin = (src_min + src_max) / 2.0
    direction = -1.0 if joint_index in _REVERSED_JOINT_INDEXES else 1.0
    mapped = direction * (value_deg - src_origin)
    d_min = float(_ALICIA_D_LIMITS_DEG[joint_index, 0])
    d_max = float(_ALICIA_D_LIMITS_DEG[joint_index, 1])
    return _clamp(mapped, min(d_min, d_max), max(d_min, d_max))


def _convert_joint_space_between_alicia_variants(
    joints: np.ndarray,
    *,
    src_robot_type: str,
    dst_robot_type: str,
) -> np.ndarray:
    joint_array = np.asarray(joints, dtype=np.float32)
    if joint_array.ndim == 0:
        raise ValueError("Robot joint values must be at least one-dimensional")
    if joint_array.shape[-1] != 6:
        raise ValueError(
            "Alicia joint-space conversion expects 6 joint values, "
            f"got shape {list(joint_array.shape)} "
            f"(src_robot_type={src_robot_type}, dst_robot_type={dst_robot_type})"
        )

    flat_joints_deg = np.rad2deg(joint_array).reshape(-1, 6)
    converted_deg = np.empty_like(flat_joints_deg)

    for row_index, joint_row in enumerate(flat_joints_deg):
        for joint_index, joint_value in enumerate(joint_row):
            if src_robot_type == ALICIA_D_ROBOT_TYPE and dst_robot_type == ALICIA_M_ROBOT_TYPE:
                converted_deg[row_index, joint_index] = _convert_alicia_d_joint_to_alicia_m_deg(
                    float(joint_value),
                    joint_index,
                )
            elif src_robot_type == ALICIA_M_ROBOT_TYPE and dst_robot_type == ALICIA_D_ROBOT_TYPE:
                converted_deg[row_index, joint_index] = _convert_alicia_m_joint_to_alicia_d_deg(
                    float(joint_value),
                    joint_index,
                )
            else:
                raise ValueError(
                    "Unsupported Alicia robot conversion: "
                    f"{src_robot_type} -> {dst_robot_type}"
                )

    converted = np.deg2rad(converted_deg).astype(np.float32, copy=False)
    return converted.reshape(joint_array.shape)


class RobotAdapter:
    """Convert raw SDK observations/actions into the policy's expected interface."""

    def __init__(self, robot_type: Optional[str] = None, policy_robot_type: Optional[str] = None):
        normalized_robot_type = normalize_act_robot_type(robot_type, field_name="robot_type")
        normalized_policy_robot_type = normalize_act_robot_type(
            policy_robot_type,
            field_name="policy_robot_type",
        )

        if normalized_robot_type is None and normalized_policy_robot_type is not None:
            normalized_robot_type = normalized_policy_robot_type
        if normalized_policy_robot_type is None:
            normalized_policy_robot_type = normalized_robot_type

        self.robot_type = normalized_robot_type
        self.policy_robot_type = normalized_policy_robot_type

    @property
    def name(self) -> str:
        return self.robot_type or "generic"

    def image_to_policy_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).contiguous()

        if image_tensor.dtype == torch.uint8:
            return image_tensor.to(dtype=torch.float32) / 255.0

        image_tensor = image_tensor.to(dtype=torch.float32)
        if image_tensor.numel() > 0 and image_tensor.max().item() > 1.0:
            image_tensor = image_tensor / 255.0
        return image_tensor

    def state_to_policy(self, state: np.ndarray) -> np.ndarray:
        scaled_state = np.asarray(state, dtype=np.float32).copy()
        joint_dim = min(6, scaled_state.shape[0])
        if joint_dim == 6:
            scaled_state[:joint_dim] = self._convert_joints_to_policy_space(scaled_state[:joint_dim])
        if scaled_state.shape[0] >= 7:
            scaled_state[-1] = scaled_state[-1] / 1000.0
        return scaled_state

    def action_from_policy(self, actions: np.ndarray) -> np.ndarray:
        scaled_actions = np.asarray(actions, dtype=np.float32).copy()
        joint_dim = min(6, scaled_actions.shape[-1])
        if joint_dim == 6:
            scaled_actions[..., :joint_dim] = self._convert_joints_from_policy_space(
                scaled_actions[..., :joint_dim]
            )
        if scaled_actions.shape[-1] >= 7:
            scaled_actions[..., -1] = scaled_actions[..., -1] * 1000.0
        return scaled_actions

    def _convert_joints_to_policy_space(self, joints: np.ndarray) -> np.ndarray:
        return self._convert_joint_space(joints, src_robot_type=self.robot_type, dst_robot_type=self.policy_robot_type)

    def _convert_joints_from_policy_space(self, joints: np.ndarray) -> np.ndarray:
        return self._convert_joint_space(joints, src_robot_type=self.policy_robot_type, dst_robot_type=self.robot_type)

    def _convert_joint_space(
        self,
        joints: np.ndarray,
        *,
        src_robot_type: Optional[str],
        dst_robot_type: Optional[str],
    ) -> np.ndarray:
        if src_robot_type is None or dst_robot_type is None or src_robot_type == dst_robot_type:
            return np.asarray(joints, dtype=np.float32).copy()
        return _convert_joint_space_between_alicia_variants(
            joints,
            src_robot_type=src_robot_type,
            dst_robot_type=dst_robot_type,
        )


class AliciaDAdapter(RobotAdapter):
    def __init__(self, policy_robot_type: Optional[str] = None):
        super().__init__(robot_type=ALICIA_D_ROBOT_TYPE, policy_robot_type=policy_robot_type)


class AliciaMAdapter(RobotAdapter):
    def __init__(self, policy_robot_type: Optional[str] = None):
        super().__init__(robot_type=ALICIA_M_ROBOT_TYPE, policy_robot_type=policy_robot_type)


def create_robot_adapter(
    robot_type: Optional[str] = None,
    *,
    policy_robot_type: Optional[str] = None,
) -> RobotAdapter:
    normalized_robot_type = normalize_act_robot_type(robot_type, field_name="robot_type")
    normalized_policy_robot_type = normalize_act_robot_type(
        policy_robot_type,
        field_name="policy_robot_type",
    )

    if normalized_robot_type is None and normalized_policy_robot_type is not None:
        normalized_robot_type = normalized_policy_robot_type

    if normalized_robot_type == ALICIA_D_ROBOT_TYPE:
        return AliciaDAdapter(policy_robot_type=normalized_policy_robot_type)
    if normalized_robot_type == ALICIA_M_ROBOT_TYPE:
        return AliciaMAdapter(policy_robot_type=normalized_policy_robot_type)
    return RobotAdapter(robot_type=normalized_robot_type, policy_robot_type=normalized_policy_robot_type)


__all__ = [
    "ALICIA_D_ROBOT_TYPE",
    "ALICIA_M_ROBOT_TYPE",
    "SUPPORTED_ACT_ROBOT_TYPES",
    "RobotAdapter",
    "AliciaDAdapter",
    "AliciaMAdapter",
    "create_robot_adapter",
    "normalize_act_robot_type",
]
