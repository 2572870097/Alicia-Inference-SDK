from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    """Unified SDK input object for real-time or offline inference."""

    images: Dict[str, np.ndarray]
    state: np.ndarray
    instruction: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class PolicyMetadata:
    model_type: str
    required_cameras: List[str]
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int
    requested_device: Optional[str] = None
    actual_device: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyStatus:
    is_loaded: bool
    model_type: str
    queue_size: int
    latency_estimate_ms: float
    fallback_count: int
    required_cameras: List[str]
    requested_device: Optional[str] = None
    actual_device: Optional[str] = None
    device_warning: str = ""
    async_inference_enabled: bool = False
