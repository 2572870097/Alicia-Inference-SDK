# 示例

[English](README.md)

当前 `examples/` 目录只保留每种推理算法一个可直接运行的示例：

- `examples/act.py`
- `examples/pi0.py`
- `examples/smolvla.py`

这三个示例都展示同一套推荐流程：

- 构造 v1 `PolicyLoadConfig`
- 按 `model_type` 加载 `InferenceSession`
- 打印 metadata 和运行状态
- 执行一次 `infer(...)` 获取 action chunk
- 执行一次 `step(...)`
- 正确停止并关闭 session

## ACT

```bash
python examples/act.py --checkpoint-dir /path/to/act_checkpoint
```

## PI0

```bash
python examples/pi0.py \
  --checkpoint-dir /path/to/pi0_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --instruction "pick up the apple"
```

## SmolVLA

```bash
python examples/smolvla.py \
  --checkpoint-dir /path/to/smolvla_checkpoint \
  --instruction "place the block into the tray"
```
