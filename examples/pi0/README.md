# PI0 Examples

PI0-focused examples with tokenizer/instruction parameters exposed explicitly.

```bash
python examples/pi0/01_load_and_inspect_pi0.py \
  --checkpoint-dir /path/to/pi0_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --instruction "pick up the apple"

python examples/pi0/02_synthetic_control_loop_pi0.py \
  --checkpoint-dir /path/to/pi0_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --instruction "pick up the apple"

python examples/pi0/03_manual_lifecycle_pi0.py \
  --checkpoint-dir /path/to/pi0_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --instruction "pick up the apple"
```
