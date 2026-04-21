import logging
from dataclasses import dataclass
from typing import Optional

import torch

from ..core.exceptions import DeviceUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    actual: str
    warning: str = ""


def _normalize_device_name(device: Optional[str]) -> str:
    if not isinstance(device, str):
        return "cuda:0"

    normalized = device.strip()
    return normalized or "cuda:0"


def _parse_cuda_index(device: str) -> Optional[int]:
    if device == "cuda":
        return 0
    if not device.startswith("cuda:"):
        return None

    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return None


def _cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception as exc:
        logger.warning("Failed to query CUDA device count: %s", exc)
        return 0


def _is_supported_device_name(device: str) -> bool:
    if device == "cpu" or device == "cuda":
        return True
    if device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
        return True
    return False


def resolve_torch_device(requested_device: Optional[str] = None) -> DeviceSelection:
    requested = _normalize_device_name(requested_device)

    if not _is_supported_device_name(requested):
        message = (
            f"请求使用不受支持的设备 `{requested}`。"
            "当前仅支持 `cpu`、`cuda` 或 `cuda:<index>`。"
        )
        raise DeviceUnavailableError(message)

    if requested == "cpu":
        return DeviceSelection(requested=requested, actual=requested)

    if requested.startswith("cuda"):
        device_count = _cuda_device_count()
        if torch.cuda.is_available():
            requested_index = _parse_cuda_index(requested)
            if requested_index is not None and device_count > 0 and requested_index >= device_count:
                message = (
                    f"请求使用 `{requested}`，但当前只检测到 {device_count} 张 CUDA 设备。"
                )
                raise DeviceUnavailableError(message)

            return DeviceSelection(requested=requested, actual=requested)

        message = (
            f"请求使用 `{requested}`，但当前环境 CUDA 不可用。"
            f" torch.cuda.is_available()={torch.cuda.is_available()}，"
            f" torch.cuda.device_count()={device_count}。"
        )
        raise DeviceUnavailableError(message)

    return DeviceSelection(requested=requested, actual=requested)
