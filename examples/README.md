# Examples

This directory contains runnable usage examples for `inference_sdk`, organized
by scope:

- `common/`: model-agnostic examples that keep `--model-type` configurable.
- `act/`: ACT-specific examples.
- `pi0/`: PI0-specific examples.
- `smolvla/`: SmolVLA-specific examples.

## Quickstart

No checkpoint required:

```bash
python examples/common/00_quickstart_dummy_policy.py
```

## Generic Examples

When you want one script that can target any model:

```bash
python examples/common/01_load_and_inspect_policy.py \
  --model-type act \
  --checkpoint-dir /path/to/checkpoint
```

```bash
python examples/common/02_synthetic_control_loop.py \
  --model-type smolvla \
  --checkpoint-dir /path/to/checkpoint \
  --instruction "place the block into the tray"
```

## Model-Specific Examples

### ACT

```bash
python examples/act/01_load_and_inspect_act.py --checkpoint-dir /path/to/act_checkpoint
```

### PI0

```bash
python examples/pi0/01_load_and_inspect_pi0.py \
  --checkpoint-dir /path/to/pi0_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --instruction "pick up the apple"
```

### SmolVLA

```bash
python examples/smolvla/01_load_and_inspect_smolvla.py \
  --checkpoint-dir /path/to/smolvla_checkpoint \
  --instruction "place the block into the tray"
```
