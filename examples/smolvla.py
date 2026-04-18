#!/usr/bin/env python3

from __future__ import annotations

import argparse

from _shared import run_policy_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA inference example.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the SmolVLA checkpoint directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--instruction", default="place the block into the tray", help="Instruction used for SmolVLA inference.")
    parser.add_argument("--control-fps", type=float, default=20.0, help="Control-loop FPS for the runtime config.")
    parser.add_argument("--sync", action="store_true", help="Disable async inference and run in sync mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_policy_example(
        model_type="smolvla",
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        instruction=args.instruction,
        control_fps=args.control_fps,
        sync=args.sync,
    )


if __name__ == "__main__":
    raise SystemExit(main())
