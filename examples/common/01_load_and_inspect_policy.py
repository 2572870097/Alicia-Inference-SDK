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
    parser = argparse.ArgumentParser(description="Load a policy checkpoint and inspect its metadata.")
    parser.add_argument("--model-type", choices=["act", "smolvla", "pi0"], required=True)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path for PI0.")
    parser.add_argument("--instruction", default=None, help="Optional instruction for PI0/SmolVLA.")
    parser.add_argument("--strict-device", action="store_true", help="Fail instead of falling back on device mismatch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_load_and_inspect(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        strict_device=args.strict_device,
        tokenizer_path=args.tokenizer_path,
        instruction=args.instruction,
    )


if __name__ == "__main__":
    main()
