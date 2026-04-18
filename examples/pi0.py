#!/usr/bin/env python3

from __future__ import annotations

import argparse

from _shared import run_policy_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI0 inference example.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the PI0 checkpoint directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path.")
    parser.add_argument("--instruction", default="pick up the apple", help="Instruction used for PI0 inference.")
    parser.add_argument("--control-fps", type=float, default=20.0, help="Control-loop FPS for the runtime config.")
    parser.add_argument("--sync", action="store_true", help="Disable async inference and run in sync mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_policy_example(
        model_type="pi0",
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        instruction=args.instruction,
        tokenizer_path=args.tokenizer_path,
        control_fps=args.control_fps,
        sync=args.sync,
    )


if __name__ == "__main__":
    raise SystemExit(main())
