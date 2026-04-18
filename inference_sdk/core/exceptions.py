from __future__ import annotations

from typing import Any, Dict, Optional


class SDKError(Exception):
    """Base exception for all developer-facing SDK failures."""

    default_code = "SDK_ERROR"
    default_recoverable = False

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: Optional[bool] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}
        self.recoverable = self.default_recoverable if recoverable is None else recoverable

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
            "recoverable": self.recoverable,
        }


class ConfigurationError(SDKError):
    default_code = "SDK_CONFIG_INVALID"
    default_recoverable = True


class ValidationError(SDKError):
    default_code = "SDK_VALIDATION_ERROR"
    default_recoverable = True


class CheckpointError(SDKError):
    default_code = "SDK_CHECKPOINT_INVALID"
    default_recoverable = True


class DependencyError(SDKError):
    default_code = "SDK_DEPENDENCY_MISSING"
    default_recoverable = True


class DeviceUnavailableError(SDKError):
    default_code = "SDK_DEVICE_UNAVAILABLE"
    default_recoverable = True


class ModelLoadError(SDKError):
    default_code = "SDK_MODEL_LOAD_FAILED"
    default_recoverable = True


class InstructionNotSupportedError(SDKError):
    default_code = "SDK_INSTRUCTION_UNSUPPORTED"
    default_recoverable = True


class ResourceStateError(SDKError):
    default_code = "SDK_RESOURCE_STATE_INVALID"
    default_recoverable = True


class InferenceError(SDKError):
    default_code = "SDK_INFERENCE_FAILED"
    default_recoverable = True


class InferenceTimeoutError(SDKError):
    default_code = "SDK_INFERENCE_TIMEOUT"
    default_recoverable = True
