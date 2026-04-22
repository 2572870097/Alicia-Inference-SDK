#!/usr/bin/env python3
"""ACT 推理接入 Alicia-M-SDK 真机控制示例。

流程:
1. 加载 ACT 模型
2. 打开头部/腕部相机
3. 读取 Alicia-M 当前关节 + 夹爪状态
4. 用 `api.step(...)` 做单步推理
5. 将动作下发到 Alicia-M 真机

默认行为面向当前仓库自带的 ACT checkpoint:
- 运行时机械臂: Alicia-M
- 模型动作空间: Alicia-D
- 默认启用 temporal ensembling 做真机闭环重规划

示例命令:
    python examples/act_alicia_m_real_robot.py \
        --port /dev/ttyACM0 \
        --head-camera 0 \
        --wrist-camera 1 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "models" / "ACT_pick_and_place_v2"
SUPPORTED_CAMERA_ROLES = {"head", "wrist"}

# Alicia-M-SDK 当前示例里的默认 MIT 参数。
MIT_KP = [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]
MIT_KD = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
MIT_TORQUE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MIT_VEL_REF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

LOGGER = logging.getLogger("examples.act_alicia_m_real_robot")


def _prepend_checkout(path: Path) -> None:
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


for local_checkout in (
    REPO_ROOT / "SparkMind",
    REPO_ROOT / "Alicia-M-SDK",
    REPO_ROOT / "Alicia-D-SDK",
):
    _prepend_checkout(local_checkout)

from inference_sdk import InferenceAPI

try:
    import alicia_m_sdk
except ImportError as exc:
    raise SystemExit(
        "无法导入 `alicia_m_sdk`。请先安装 Alicia-M-SDK，"
        "或者直接在当前工作区使用本地 `Alicia-M-SDK/` checkout。"
    ) from exc


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACT 推理接入 Alicia-M-SDK 真机控制示例。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="ACT 导出 checkpoint 目录。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="推理设备，例如 cpu / cuda:0。",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="Alicia-M 串口；不传则走 SDK 自动发现。",
    )
    parser.add_argument(
        "--robot-version",
        type=str,
        default="v1_1",
        help="Alicia-M 硬件版本。",
    )
    parser.add_argument(
        "--mode",
        choices=["pv", "mit"],
        default="pv",
        help="Alicia-M 控制模式。",
    )
    parser.add_argument(
        "--mit-interpolation",
        action="store_true",
        help="MIT 模式下启用固件插值；默认走逐帧 PD 跟随。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=120.0,
        help="机械臂速度参数，传给 Alicia-M-SDK。",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=120.0,
        help="夹爪速度参数，传给 Alicia-M-SDK。",
    )
    parser.add_argument(
        "--control-fps",
        type=float,
        default=15.0,
        help="控制环频率。",
    )
    parser.add_argument(
        "--head-camera",
        type=str,
        default="0",
        help="头部相机源，可以是索引或设备路径。",
    )
    parser.add_argument(
        "--wrist-camera",
        type=str,
        default="1",
        help="腕部相机源，可以是索引或设备路径；与 head 相同则复用同一路图像。",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="相机采集宽度。",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="相机采集高度。",
    )
    parser.add_argument(
        "--policy-robot-type",
        choices=["alicia_d", "alicia_m"],
        default=None,
        help="ACT 模型动作空间对应的机械臂类型；不传时对内置模型默认按 Alicia-D 处理。",
    )
    parser.add_argument(
        "--disable-temporal-ensemble",
        action="store_true",
        help="关闭 ACT temporal ensembling，改为 chunk queue 执行。",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="temporal ensembling 系数；仅在未关闭时生效。",
    )
    parser.add_argument(
        "--enable-async",
        action="store_true",
        help="关闭 temporal ensembling 后，可启用异步 chunk queue 推理。",
    )
    parser.add_argument(
        "--startup-home",
        action="store_true",
        help="启动控制环前先回零位。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="最大控制步数；0 表示持续运行直到 Ctrl+C。",
    )
    parser.add_argument(
        "--max-joint-delta-deg",
        type=float,
        default=8.0,
        help="每个控制步允许的最大关节变化量（度）。",
    )
    parser.add_argument(
        "--max-gripper-delta",
        type=float,
        default=120.0,
        help="每个控制步允许的最大夹爪变化量。",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=30,
        help="每多少步打印一次状态；0 表示关闭周期打印。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印更详细的日志。",
    )
    return parser.parse_args()


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_camera_source(value: str) -> int | str:
    raw_value = value.strip()
    if raw_value.isdigit():
        return int(raw_value)
    return raw_value


def _open_camera(name: str, source: int | str, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开{name}相机: {source}")

    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture


def _read_frame(capture: cv2.VideoCapture, name: str, width: int, height: int) -> np.ndarray:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"读取{name}相机图像失败")
    if width > 0 and height > 0 and frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def _resolve_policy_robot_type(args: argparse.Namespace) -> str | None:
    if args.policy_robot_type is not None:
        return args.policy_robot_type

    try:
        checkpoint_path = Path(args.checkpoint).resolve()
        if checkpoint_path == DEFAULT_CHECKPOINT_DIR.resolve():
            return "alicia_d"
    except OSError:
        pass

    return None


def _extract_robot_state(robot: Any) -> np.ndarray:
    state = robot.get_robot_state("joint_gripper")
    if state is None or getattr(state, "angles", None) is None:
        raise RuntimeError("Alicia-M 当前还没有可用的关节状态")
    return np.asarray(list(state.angles) + [float(state.gripper)], dtype=np.float32)


def _clip_action_delta(
    action: np.ndarray,
    current_state: np.ndarray,
    *,
    max_joint_delta_rad: float,
    max_gripper_delta: float,
) -> np.ndarray:
    target = np.asarray(action, dtype=np.float32).copy()
    target[:6] = current_state[:6] + np.clip(
        target[:6] - current_state[:6],
        -max_joint_delta_rad,
        max_joint_delta_rad,
    )
    target[6] = float(
        np.clip(
            target[6],
            current_state[6] - max_gripper_delta,
            current_state[6] + max_gripper_delta,
        )
    )
    target[6] = float(np.clip(target[6], 0.0, 1000.0))
    return target


def _switch_mode_if_needed(robot: Any, target_mode: str) -> None:
    current_mode = robot.control_mode.value.lower()
    if current_mode == target_mode:
        return

    print(
        f"当前 Alicia-M 模式是 {current_mode.upper()}，"
        f"即将切换到 {target_mode.upper()}。切换期间机械臂会短暂失能。"
    )
    input("确认后按 Enter 继续...")
    robot.switch_mode(target_mode)


def _maybe_go_home(robot: Any, speed: float, enabled: bool) -> None:
    if not enabled:
        return
    print("即将执行回零位。请确认机械臂周围无碰撞风险。")
    input("确认后按 Enter 继续...")
    robot.go_home(speed=speed)


def _validate_metadata(required_cameras: list[str], action_dim: int, state_dim: int) -> None:
    unsupported = sorted(set(required_cameras) - SUPPORTED_CAMERA_ROLES)
    if unsupported:
        raise RuntimeError(
            "当前示例只支持 `head` / `wrist` 相机角色，"
            f"但模型要求: {required_cameras}"
        )
    if state_dim < 7:
        raise RuntimeError(f"ACT 模型 state_dim={state_dim}，不足以驱动 Alicia-M 的 6 关节 + 夹爪")
    if action_dim < 7:
        raise RuntimeError(f"ACT 模型 action_dim={action_dim}，不足以驱动 Alicia-M 的 6 关节 + 夹爪")


def _log_runtime_summary(
    *,
    args: argparse.Namespace,
    policy_robot_type: str | None,
    temporal_ensemble_coeff: float | None,
) -> None:
    LOGGER.info("Alicia-M 真机 ACT 示例启动")
    LOGGER.info("checkpoint=%s", args.checkpoint)
    LOGGER.info("device=%s, control_fps=%.2f", args.device, args.control_fps)
    LOGGER.info(
        "robot_type=alicia_m, policy_robot_type=%s",
        policy_robot_type or "generic",
    )
    LOGGER.info(
        "robot_mode=%s, mit_interpolation=%s, speed=%.1f, gripper_speed=%.1f",
        args.mode,
        args.mit_interpolation,
        args.speed,
        args.gripper_speed,
    )
    LOGGER.info(
        "temporal_ensemble=%s, async=%s",
        "disabled" if temporal_ensemble_coeff is None else f"enabled(coeff={temporal_ensemble_coeff})",
        args.enable_async and temporal_ensemble_coeff is None,
    )


def main() -> int:
    args = _parse_args()
    _configure_logging(args.debug)

    policy_robot_type = _resolve_policy_robot_type(args)
    temporal_ensemble_coeff = None
    if not args.disable_temporal_ensemble:
        temporal_ensemble_coeff = args.temporal_ensemble_coeff

    _log_runtime_summary(
        args=args,
        policy_robot_type=policy_robot_type,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )

    max_joint_delta_rad = float(np.deg2rad(args.max_joint_delta_deg))
    enable_async = bool(args.enable_async and temporal_ensemble_coeff is None)

    api = InferenceAPI()
    robot = None
    head_capture = None
    wrist_capture = None

    try:
        load_result = api.load_model(
            model_type="act",
            checkpoint_dir=args.checkpoint,
            device=args.device,
            control_fps=args.control_fps,
            enable_async=enable_async,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            robot_type="alicia_m",
            policy_robot_type=policy_robot_type,
        )
        LOGGER.info("模型加载成功: %s", load_result)

        metadata = api.get_metadata()
        _validate_metadata(
            required_cameras=metadata.required_cameras,
            action_dim=metadata.action_dim,
            state_dim=metadata.state_dim,
        )
        LOGGER.info(
            "模型元数据: required_cameras=%s, state_dim=%d, action_dim=%d, chunk_size=%d, n_action_steps=%d",
            metadata.required_cameras,
            metadata.state_dim,
            metadata.action_dim,
            metadata.chunk_size,
            metadata.n_action_steps,
        )

        robot = alicia_m_sdk.create_robot(
            port=args.port,
            version=args.robot_version,
            control_aim="follower",
        )
        LOGGER.info("Alicia-M 连接成功，当前模式=%s", robot.control_mode.value.upper())
        _switch_mode_if_needed(robot, args.mode)
        _maybe_go_home(robot, args.speed, args.startup_home)

        head_source = _parse_camera_source(args.head_camera)
        wrist_source = _parse_camera_source(args.wrist_camera)
        reuse_head_for_wrist = args.head_camera.strip() == args.wrist_camera.strip()

        head_capture = _open_camera(
            "头部",
            head_source,
            width=args.camera_width,
            height=args.camera_height,
        )
        if reuse_head_for_wrist:
            wrist_capture = head_capture
            LOGGER.info("腕部相机复用头部相机源: %s", args.head_camera)
        else:
            wrist_capture = _open_camera(
                "腕部",
                wrist_source,
                width=args.camera_width,
                height=args.camera_height,
            )

        print(
            "确认工作空间、相机画面和机械臂起始姿态都已准备好。"
            "按 Enter 开始 ACT 真机控制，Ctrl+C 停止。"
        )
        input()

        control_interval = 1.0 / args.control_fps
        step_index = 0

        while args.max_steps <= 0 or step_index < args.max_steps:
            loop_start = time.perf_counter()

            current_state = _extract_robot_state(robot)
            head_frame = _read_frame(
                head_capture,
                "头部",
                width=args.camera_width,
                height=args.camera_height,
            )
            wrist_frame = head_frame if reuse_head_for_wrist else _read_frame(
                wrist_capture,
                "腕部",
                width=args.camera_width,
                height=args.camera_height,
            )

            action = np.asarray(
                api.step(
                    images={"head": head_frame, "wrist": wrist_frame},
                    state=current_state,
                ),
                dtype=np.float32,
            ).reshape(-1)

            if action.shape[0] < 7:
                raise RuntimeError(f"ACT 输出动作维度不足: {action.shape}")
            if action.shape[0] > 7 and step_index == 0:
                LOGGER.warning("ACT 输出维度为 %s，仅使用前 7 维驱动 Alicia-M", action.shape[0])

            target_state = _clip_action_delta(
                action[:7],
                current_state=current_state[:7],
                max_joint_delta_rad=max_joint_delta_rad,
                max_gripper_delta=args.max_gripper_delta,
            )

            mit_kwargs = {}
            if args.mode == "mit":
                mit_kwargs = {
                    "kp": MIT_KP,
                    "kd": MIT_KD,
                    "torque": MIT_TORQUE,
                    "vel_ref": MIT_VEL_REF,
                    "use_interpolation": args.mit_interpolation,
                }

            robot.set_robot_state(
                target_joints=target_state[:6].tolist(),
                gripper_value=float(target_state[6]),
                joint_format="rad",
                speed=args.speed,
                gripper_speed=args.gripper_speed,
                wait_for_completion=False,
                **mit_kwargs,
            )

            step_index += 1
            if args.status_every > 0 and step_index % args.status_every == 0:
                status = api.get_status()
                LOGGER.info(
                    "step=%d queue=%d fallback=%d joints_deg=%s gripper=%.1f",
                    step_index,
                    status.queue_size,
                    status.fallback_count,
                    np.round(np.rad2deg(target_state[:6]), 1).tolist(),
                    float(target_state[6]),
                )

            elapsed = time.perf_counter() - loop_start
            sleep_time = control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif args.status_every > 0 and step_index % args.status_every == 0:
                LOGGER.warning("控制环超时 %.1f ms", -sleep_time * 1000.0)

    except KeyboardInterrupt:
        LOGGER.info("收到 Ctrl+C，准备退出真机控制")
    finally:
        if head_capture is not None:
            head_capture.release()
        if wrist_capture is not None and wrist_capture is not head_capture:
            wrist_capture.release()
        api.unload_model()
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
