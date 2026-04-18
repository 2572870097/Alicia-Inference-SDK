#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common._real_policy_demo import run_manual_policy_lifecycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the lower-level create/load/start/step/unload lifecycle.")
    parser.add_argument("--model-type", choices=["act", "smolvla", "pi0"], required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--sync", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_manual_policy_lifecycle(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        instruction=args.instruction,
        tokenizer_path=args.tokenizer_path,
        sync=args.sync,
    )


if __name__ == "__main__":
    main()
