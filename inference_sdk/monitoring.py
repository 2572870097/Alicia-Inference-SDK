import threading
from typing import Optional, Protocol


class InferenceMonitor(Protocol):
    def register_thread(
        self,
        name: str,
        expected_interval: float = 1.0,
        timeout_threshold: float = 5.0,
        alert_callback=None,
    ) -> None:
        ...

    def unregister_thread(self, name: str) -> None:
        ...

    def heartbeat(self, name: str) -> None:
        ...


class NoOpMonitor:
    """Default monitor used when the SDK is embedded without app-level monitoring."""

    def register_thread(
        self,
        name: str,
        expected_interval: float = 1.0,
        timeout_threshold: float = 5.0,
        alert_callback=None,
    ) -> None:
        return None

    def unregister_thread(self, name: str) -> None:
        return None

    def heartbeat(self, name: str) -> None:
        return None


_monitor_lock = threading.Lock()
_monitor: InferenceMonitor = NoOpMonitor()


def get_inference_monitor() -> InferenceMonitor:
    with _monitor_lock:
        return _monitor


def set_inference_monitor(monitor: Optional[InferenceMonitor]) -> None:
    global _monitor
    with _monitor_lock:
        _monitor = monitor or NoOpMonitor()
