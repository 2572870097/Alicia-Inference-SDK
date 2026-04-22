# InferenceAPI 使用指南

## 概述

`InferenceAPI` 是一个封装好的推理接口，提供简洁的方法调用方式，开发者可以直接实例化并使用。

## 运行前准备

在这个仓库里，`pip install -e .[act]`、`.[pi0]`、`.[smolvla]` 只会安装 SDK
这一层的依赖。真正的策略实现来自 `sparkmind.*`，因此当前虚拟环境里还需要安
装 SparkMind checkout，例如：

```bash
source .venv/bin/activate
pip install -e .[all]
pip install -e ../SparkMind
```

不同模型在运行时还需要额外资产：

- `ACT`：合法的 ACT 导出 checkpoint 目录。
- `PI0`：合法的 PI0 checkpoint，以及 tokenizer 资产。优先通过
  `tokenizer_path` 或 `PI0_TOKENIZER_PATH` 指向本地目录。
- `SmolVLA`：合法的 SmolVLA checkpoint，以及基础 VLM 模型资产。离线环境建
  议通过 `SMOLVLA_VLM_MODEL_PATH` 指向本地的
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。

### 这些资产去哪里下载

- `ACT checkpoint`：
  当前仓库已自带一个可直接测试的导出 checkpoint：
  [`model/ACT_pick_and_place_v2`](model/ACT_pick_and_place_v2)。
  该模型卡对应的 Hugging Face repo 是 `z18820636149/ACT_pick_and_place_v2`。
- `PI0 checkpoint`：
  SDK 没有写死唯一下载地址。请使用你自己的导出 PI0 checkpoint，或来自其他
  Hugging Face / 本地目录的兼容导出结果。
- `PI0 tokenizer`：
  默认远端资产名是 `google/paligemma2-3b-mix-224`。
- `SmolVLA checkpoint`：
  SDK 没有写死唯一下载地址。请使用你自己的导出 SmolVLA checkpoint，或来自
  其他 Hugging Face / 本地目录的兼容导出结果。
- `SmolVLA 基础 VLM`：
  默认远端资产名是 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。

推荐的本地下载命令：

```bash
source .venv/bin/activate

# PI0 tokenizer
hf download google/paligemma2-3b-mix-224 \
  --local-dir models/google/paligemma2-3b-mix-224

hf download google/paligemma2-3b-mix-224 \
    --local-dir models/google/paligemma2-3b-mix-224 \
    --include "tokenizer*" "*.model" "special_tokens_map.json" "added_tokens.json" "config.json"
    
export PI0_TOKENIZER_PATH=$PWD/models/google/paligemma2-3b-mix-224

# SmolVLA 基础 VLM
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export SMOLVLA_VLM_MODEL_PATH=$PWD/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

SDK 会优先在这些位置查找本地资产：

- `PI0 tokenizer`：`tokenizer_path`、`PI0_TOKENIZER_PATH`、
  `checkpoint_dir/tokenizer`、仓库或当前目录下的 `models/...`。
- `SmolVLA 基础 VLM`：`SMOLVLA_VLM_MODEL_PATH`、
  `checkpoint_dir/vlm_model`、仓库或当前目录下的 `models/...`。

## 快速开始

### 基础使用

```python
from inference_sdk import InferenceAPI
import numpy as np

# 1. 创建 API 实例
api = InferenceAPI()

# 2. 加载模型
api.load_model(
    model_type="act",
    checkpoint_dir="/path/to/checkpoint",
    device="cuda:0",
    temporal_ensemble_coeff=0.01
)

# 3. 准备数据
images = {
    "head": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
}
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# 4. 执行推理
action = api.infer(images=images, state=state)
print(f"动作: {action}")

# 5. 卸载模型
api.unload_model()
```

### 使用上下文管理器

```python
with InferenceAPI() as api:
    api.load_model(
        model_type="act",
        checkpoint_dir="/path/to/checkpoint"
    )
    
    action = api.infer(images=images, state=state)
    # 自动卸载
```

## API 方法

### 模型管理

#### `load_model()`

加载推理模型。

```python
result = api.load_model(
    model_type="act",              # 模型类型: act, pi0, smolvla
    checkpoint_dir="/path/to/checkpoint",  # 检查点目录
    device="cuda:0",               # 设备
    tokenizer_path=None,           # PI0 tokenizer 路径
    instruction=None,              # 语言指令 (PI0/SmolVLA)
    control_fps=20.0,              # 控制频率
    enable_async=True,             # 启用异步推理
    temporal_ensemble_coeff=None   # 通用时间集成系数，例如 0.01
)
```

说明：

- `tokenizer_path` 当前只对 `PI0` 生效。
- `SmolVLA` 会根据 checkpoint 配置解析基础 VLM 资产；离线运行时请设置
  `SMOLVLA_VLM_MODEL_PATH`。
- `temporal_ensemble_coeff` 对所有 chunk policy 都可用。设置后会在每个控制步都
  重新推理，并对重叠的未来动作做指数加权融合。该模式会自动关闭 async queue
  预取。

**返回值：**
```python
{
    "success": True,
    "model_type": "act",
    "required_cameras": ["head", "wrist"],
    "state_dim": 7,
    "action_dim": 7,
    "chunk_size": 100
}
```

#### `unload_model()`

卸载当前模型，释放资源。

```python
api.unload_model()
```

#### `is_loaded()`

检查模型是否已加载。

```python
if api.is_loaded():
    print("模型已加载")
```

### 推理方法

#### `infer()`

执行推理，返回完整的动作块。

```python
action = api.infer(
    images={"head": head_img, "wrist": wrist_img},
    state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    instruction="pick up the apple"  # 可选
)
```

**参数：**
- `images`: 图像字典，key 为相机名，value 为 BGR 格式的 numpy 数组
- `state`: 机器人状态向量（列表或 numpy 数组）
- `instruction`: 可选的语言指令（仅 PI0/SmolVLA）

**返回值：** numpy 数组，形状为 `(action_dim * chunk_size,)` 或 `(chunk_size, action_dim)`

#### `step()`

执行单步推理，用于控制环。

```python
action = api.step(
    images=images,
    state=state,
    instruction=None
)
```

返回单个时间步的动作向量。

### 状态查询

#### `get_metadata()`

获取模型元数据。

```python
metadata = api.get_metadata()
print(metadata.model_type)
print(metadata.required_cameras)
print(metadata.state_dim)
print(metadata.action_dim)
```

#### `get_status()`

获取模型运行状态。

```python
status = api.get_status()
print(status.is_loaded)
print(status.queue_size)
```

#### `get_model_info()`

获取模型信息（简化版）。

```python
info = api.get_model_info()
# {
#     "is_loaded": True,
#     "model_type": "act",
#     "required_cameras": ["head", "wrist"],
#     "state_dim": 7,
#     "action_dim": 7,
#     "chunk_size": 100
# }
```

#### `print_info()`

打印模型信息到控制台。

```python
api.print_info()
# ==================================================
# 模型类型: act
# 需要相机: ['head', 'wrist']
# 状态维度: 7
# 动作维度: 7
# 动作块大小: 100
# ==================================================
```

### 异步推理控制

#### `start_async_inference()`

启动异步推理。

```python
api.start_async_inference()
```

#### `stop_async_inference()`

停止异步推理。

```python
api.stop_async_inference()
```

## 使用场景

### 场景 1: 单次推理

```python
api = InferenceAPI()
api.load_model(model_type="act", checkpoint_dir="/path/to/checkpoint")

# 执行一次推理
action = api.infer(images=images, state=state)

api.unload_model()
```

### 场景 2: 控制环

```python
api = InferenceAPI(auto_start_async=True)
api.load_model(
    model_type="act",
    checkpoint_dir="/path/to/checkpoint",
    enable_async=True
)

# 控制环
while running:
    # 获取观测
    images = get_camera_images()
    state = get_robot_state()
    
    # 执行单步推理
    action = api.step(images=images, state=state)
    
    # 执行动作
    execute_action(action)

api.unload_model()
```

### 场景 3: 语言条件推理 (PI0/SmolVLA)

```python
api = InferenceAPI()
api.load_model(
    model_type="pi0",
    checkpoint_dir="/path/to/checkpoint",
    tokenizer_path="/path/to/tokenizer",
    instruction="pick up the red cube"
)

# 推理时可以覆盖指令
action = api.infer(
    images=images,
    state=state,
    instruction="place it on the table"
)

api.unload_model()
```

如果你要改成 `SmolVLA`，除了 checkpoint 之外，还需要额外准备基础 VLM 资产；
离线环境建议先设置：

```bash
export SMOLVLA_VLM_MODEL_PATH=/path/to/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

### 场景 4: 批量推理

```python
api = InferenceAPI()
api.load_model(model_type="act", checkpoint_dir="/path/to/checkpoint")

results = []
for images, state in dataset:
    action = api.infer(images=images, state=state)
    results.append(action)

api.unload_model()
```

## 错误处理

```python
from inference_sdk import InferenceAPI

api = InferenceAPI()

try:
    api.load_model(
        model_type="act",
        checkpoint_dir="/path/to/checkpoint"
    )
    action = api.infer(images=images, state=state)

except Exception as e:
    print(f"错误: {type(e).__name__} - {e}")
finally:
    api.unload_model()
```

## 注意事项

1. **图像格式**：图像必须是 BGR 格式（OpenCV 默认格式）
2. **状态维度**：确保状态向量维度与模型要求一致
3. **相机名称**：图像字典的 key 必须与模型要求的相机名称匹配
4. **资源管理**：使用完毕后记得调用 `unload_model()` 释放资源
5. **异步推理**：启用异步推理可提高吞吐量，适合控制环场景
6. **模型资产**：PI0 还需要 tokenizer，SmolVLA 还需要基础 VLM 模型；仅有 checkpoint 不足以运行

## 完整示例

可以直接参考这两个示例入口：

- [examples/api_usage.py](examples/api_usage.py)：基础 API 用法、离线 smoke test、控制环调用方式
- [examples/act_alicia_m_real_robot.py](examples/act_alicia_m_real_robot.py)：ACT 接 Alicia-M-SDK 真机，读取头部/腕部相机并在闭环里执行 6 关节 + 夹爪动作
