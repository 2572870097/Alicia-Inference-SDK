#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from common._demo_utils import DummyPolicy, build_synthetic_observation, pretty_print


def main() -> None:
    policy = DummyPolicy()

    observation = build_synthetic_observation(
        required_cameras=policy.get_required_cameras(),
        state_dim=policy.get_state_dim(),
        step_idx=0,
        instruction="close the gripper",
    )

    action = policy.step(observation)
    chunk = policy.predict_chunk(observation)

    pretty_print("Metadata", policy.get_metadata())
    pretty_print("Status", policy.get_status())
    pretty_print("Selected Action", action)
    pretty_print("Predicted Chunk", chunk)


if __name__ == "__main__":
    main()
