"""Package import bootstrap helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False


def bootstrap_import_environment() -> None:
    """Prepare third-party import order and local fallback paths once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # Import lerobot before SparkMind so the installed package wins module resolution.
    try:
        import lerobot  # noqa: F401
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Failed to preload installed lerobot package: %s", exc)
    else:
        try:
            import lerobot.policies.rtc  # noqa: F401
        except Exception as exc:
            logger.warning("Failed to preload lerobot RTC policies, continuing: %s", exc)

    repo_root = Path(__file__).resolve().parents[1]
    for sparkmind_root in (repo_root / "SparkMind", repo_root.parent / "SparkMind"):
        if not sparkmind_root.is_dir():
            continue

        sparkmind_root_str = str(sparkmind_root)
        if sparkmind_root_str not in sys.path:
            sys.path.insert(0, sparkmind_root_str)

    _BOOTSTRAPPED = True


__all__ = ["bootstrap_import_environment"]
