#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common._real_policy_demo import run_load_and_inspect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an ACT checkpoint and inspect its metadata.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the ACT checkpoint directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--strict-device", action="store_true", help="Fail instead of falling back on device mismatch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_load_and_inspect(
        model_type="act",
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        strict_device=args.strict_device,
    )


if __name__ == "__main__":
    main()
