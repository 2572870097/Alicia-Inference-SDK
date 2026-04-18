#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common._real_policy_demo import run_synthetic_control_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a policy against synthetic observations.")
    parser.add_argument("--model-type", choices=["act", "smolvla", "pi0"], required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=5, help="Number of control steps to run.")
    parser.add_argument("--control-fps", type=float, default=20.0, help="Target control frequency.")
    parser.add_argument("--sync", action="store_true", help="Disable async inference and run synchronously.")
    parser.add_argument("--instruction", default=None, help="Instruction for language-conditioned models.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path for PI0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_synthetic_control_loop(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        steps=args.steps,
        control_fps=args.control_fps,
        sync=args.sync,
        instruction=args.instruction,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
