# Alicia Inference SDK

[English](README.md)

用于在 FastAPI 应用层之外加载和运行 `ACT`、`PI0`、`SmolVLA` 策略的可复用 Python SDK。

这个仓库是面向开发者的推理层，主要统一了以下内容：

- 策略如何创建和加载
- 观测输入对象长什么样
- 元数据和运行状态如何表达
- 异步控制环推理如何启动和停止
- SDK 如何向调用方暴露错误

公开 SDK 用法主要记录在当前 README 和
[`INFERENCE_API_GUIDE.md`](INFERENCE_API_GUIDE.md) 中。

## 安装

在当前目录下以 editable 模式安装：

```bash
pip install -e .
```

按模型安装可选依赖：

```bash
pip install -e .[act]
pip install -e .[pi0]
pip install -e .[smolvla]
pip install -e .[all]
```

这些 extras 只覆盖 SDK 自身依赖，不等于模型已经可以运行。实际的策略实现会导
入 `sparkmind.*`，所以当前虚拟环境里还需要让 SparkMind 在同一环境中可用。
一个常见的工作区结构是把 `SparkMind` 放在当前仓库同级目录，并一并安装：

```bash
source .venv/bin/activate
pip install -e .[all]
pip install -e ../SparkMind
```

如果存在同级的 `SparkMind/` checkout，SDK 也会在运行时尝试把它加入
`sys.path` 作为兜底路径。但这只解决“源码能被找到”，并不会替你安装
SparkMind 自己的 Python 依赖，所以这些依赖仍然必须装在同一个虚拟环境里。

基础依赖：

- `numpy`
- `opencv-python`
- `pyyaml`

模型可选依赖：

- `torch`
- `transformers`
- `omegaconf`
- `safetensors`

## 支持的模型

- `act`
- `pi0`
- `smolvla`

`PI0` 和 `SmolVLA` 支持语言指令，`ACT` 不支持。

## 运行前提

在调用 `load_model(...)` 之前，请先确认对应模型的运行资产已经就绪：

- `act`：只支持合法的 ACT 导出 checkpoint 目录。
- `pi0`：需要合法的 PI0 checkpoint，以及 tokenizer 资产。SDK 会依次检查
  `tokenizer_path`、`PI0_TOKENIZER_PATH`、常见的本地 `models/...` 目录，
  最后才会尝试从 Hugging Face 加载。
- `smolvla`：需要合法的 SmolVLA checkpoint，以及 checkpoint 配置里声明的
  基础 VLM 资产。离线运行时，建议通过 `SMOLVLA_VLM_MODEL_PATH` 指向本地模型。

如果只想做本地 smoke test，这个仓库当前自带一个导出的 ACT checkpoint：
[`model/ACT_pick_and_place_v2`](model/ACT_pick_and_place_v2)。

### 资产下载路径

- `ACT checkpoint`：
  当前仓库已经自带一个导出的 ACT checkpoint，路径是
  [`model/ACT_pick_and_place_v2`](model/ACT_pick_and_place_v2)。这个模型卡里对
  应的 Hugging Face repo 是 `z18820636149/ACT_pick_and_place_v2`。
- `PI0 checkpoint`：
  SDK 没有写死唯一下载仓库。你可以使用任何符合格式要求的导出 PI0 checkpoint
  目录，然后通过 `checkpoint_dir` 传入。
- `PI0 tokenizer`：
  代码里的默认远端资产名是 `google/paligemma2-3b-mix-224`。
- `SmolVLA checkpoint`：
  SDK 没有写死唯一下载仓库。你可以使用任何符合格式要求的导出 SmolVLA
  checkpoint 目录，然后通过 `checkpoint_dir` 传入。
- `SmolVLA 基础 VLM`：
  代码里的默认远端资产名是
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。

推荐的本地下载方式：

```bash
source .venv/bin/activate

# PI0 tokenizer
hf download google/paligemma2-3b-mix-224 \
  --local-dir models/google/paligemma2-3b-mix-224
export PI0_TOKENIZER_PATH=$PWD/models/google/paligemma2-3b-mix-224

# SmolVLA 基础 VLM
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export SMOLVLA_VLM_MODEL_PATH=$PWD/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

SDK 的本地查找顺序：

- `PI0 tokenizer`：`tokenizer_path` -> `PI0_TOKENIZER_PATH` ->
  `checkpoint_dir/tokenizer` -> 本地 `models/google/paligemma2-3b-mix-224`
  这类目录。
- `SmolVLA 基础 VLM`：`SMOLVLA_VLM_MODEL_PATH` -> `checkpoint_dir/vlm_model` ->
  本地 `models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 这类目录。

## 公共 API

推荐的 v1 入口：

- `InferenceSession`
- `load_policy(...)`
- `create_policy(...)`
- `PolicyLoadConfig`
- `RuntimeConfig`
- `DeviceConfig`
- `Observation`
- `PolicyMetadata`
- `PolicyStatus`
- 标准 Python 异常（`ValueError`、`RuntimeError`、`ImportError`）

已加载的 session 或策略实例提供以下核心运行方法：

- `load(..., model_type=...)`
- `infer(observation)`
- `step(observation)`
- `predict_chunk(observation)`
- `get_metadata()`
- `get_status()`
- `start_async_inference()`
- `stop_async_inference()`
- `reset()`
- `unload()` / `close()`

## 快速开始

### 1. 使用 Session 加载并返回 action chunk

这是推荐的 v1 用法。

如果只想走最短的本地验证路径，优先使用仓库自带的
`model/ACT_pick_and_place_v2`。

```python
from inference_sdk import (
    DeviceConfig,
    InferenceSession,
    Observation,
    PolicyLoadConfig,
    RuntimeConfig,
)

config = PolicyLoadConfig(
    checkpoint_dir="model/ACT_pick_and_place_v2",
    model_type="act",
    device=DeviceConfig(device="cpu"),
    runtime=RuntimeConfig(
        control_fps=20.0,
        enable_async_inference=False,
    ),
)

try:
    session = InferenceSession()
    try:
        session.load(config=config)

        metadata = session.get_metadata()
        print(metadata)

        observation = Observation(
            images={
                "head": head_bgr_frame,
                "wrist": wrist_bgr_frame,
            },
            state=robot_state,
        )

        action_chunk = session.infer(observation)
        print(action_chunk)
    finally:
        session.close()
except Exception as exc:
    print(exc)
```

如果要加载 `PI0`，请通过 `tokenizer_path` 或 `PI0_TOKENIZER_PATH` 提供
tokenizer 资产；如果要加载 `SmolVLA`，请确保基础 VLM 资产已准备好，或通过
`SMOLVLA_VLM_MODEL_PATH` 指向本地目录。

### 2. 直接按模型参数加载 Session

如果调用方不想先构造配置对象，也可以直接按参数加载：

```python
from inference_sdk import InferenceSession, Observation

session = InferenceSession()
try:
    session.load(
        checkpoint_dir="/path/to/checkpoint",
        model_type="act",
        device="cuda:0",
    )
    action_chunk = session.infer(observation)
finally:
    session.close()
```

### 3. 兼容底层策略加载方式

为了兼容旧代码和高级用法，仍然支持原来的底层策略加载方式：

```python
from inference_sdk import load_policy

policy = load_policy(
    checkpoint_dir="/path/to/checkpoint",
    model_type="act",
    device="cuda:0",
)

try:
    metadata = policy.get_metadata()
    print(metadata)
finally:
    policy.unload()
```

## 数据契约

### Observation

`Observation` 是统一的 SDK 输入对象：

```python
Observation(
    images={"head": np.ndarray, "wrist": np.ndarray},
    state=np.ndarray,
    instruction="optional language instruction",
    timestamp=None,
)
```

输入要求：

- `images` 必须是非空的 `dict[str, np.ndarray]`
- 每张图像必须是 `(H, W, 3)` 形状
- 图像默认采用 OpenCV 风格的 `BGR` 格式
- `state` 必须是 1 维 `np.ndarray`
- `instruction` 是可选字段，仅对语言条件模型生效

### PolicyMetadata

`policy.get_metadata()` 返回的静态信息包括：

- `model_type`
- `required_cameras`
- `state_dim`
- `action_dim`
- `chunk_size`
- `n_action_steps`
- `requested_device`
- `actual_device`
- `extras`

### PolicyStatus

`policy.get_status()` 返回的运行时信息包括：

- `is_loaded`
- `model_type`
- `queue_size`
- `fallback_count`
- `required_cameras`
- `requested_device`
- `actual_device`
- `device_warning`
- `async_inference_enabled`

## 生命周期语义

推荐的生命周期如下：

1. 创建并加载一个 `InferenceSession`，或直接加载一个策略实例。
2. 读取 metadata，确认需要哪些相机角色和状态维度。
3. 在需要原始块推理时调用 `infer()`，或按需启动异步推理。
4. 在控制环里逐步执行时调用 `step()`。
5. 如果启动过异步推理，则显式停止。
6. 调用 `close()` / `unload()`，或使用上下文管理器。

关键运行规则：

- `InferenceSession.load(...)` 会根据 `model_type` 加载指定模型。
- `InferenceSession.infer(...)` 会对单次观测返回原始 `action chunk`。
- `load_policy(...)` 返回已加载的策略实例。
- `create_policy(...)` 返回未加载的策略实例。
- 异步推理不会自动启动，需要显式调用 `start_async_inference()`。
- `step()` 走带动作队列的控制环逻辑。
- `predict_chunk()` 会绕过调度队列，直接返回原始块预测结果。
- `close()` 是 `unload()` 的别名。

## 错误模型

v1 SDK 使用标准 Python 异常，而不是自定义异常层级：

- `ValueError`：配置非法、输入校验失败、模型类型不支持、checkpoint 校验失败
- `RuntimeError`：策略未加载、运行时状态异常、设备不可用、推理阶段失败
- `ImportError`：缺少可选依赖

示例：

```python
from inference_sdk import load_policy

try:
    policy = load_policy(checkpoint_dir="/bad/path", model_type="act")
except Exception as exc:
    print(type(exc).__name__, exc)
```

## 运行时配置

使用 `RuntimeConfig` 配置控制环行为：

```python
from inference_sdk import RuntimeConfig

runtime = RuntimeConfig(
    control_fps=30.0,
    enable_async_inference=True,
    temporal_ensemble_coeff=None,
    fallback_mode="repeat",
)
```

关键字段：

- `control_fps`
- `enable_async_inference`
- `temporal_ensemble_coeff`
- `fallback_mode`
- `obs_queue_maxsize`

旧的 `SmoothingConfig` 仍然可用，但在 v1 中，推荐开发者优先使用
`RuntimeConfig`。

## 设备配置

使用 `DeviceConfig` 控制设备选择：

```python
from inference_sdk import DeviceConfig

device = DeviceConfig(device="cuda:0")
```

行为约定：

- 当前仅支持 `cpu`、`cuda` 和 `cuda:<index>`
- 不受支持的设备字符串会在配置校验阶段直接报错
- 如果请求的设备不可用，SDK 会抛出 `RuntimeError`

## 模型专项说明

### ACT

- 只支持导出后的 checkpoint 目录格式
- 不支持语言指令

### PI0

- 需要 tokenizer 资产
- 可以通过 `tokenizer_path` 显式传入 tokenizer 路径
- 也可以从本地模型目录或环境变量中解析 tokenizer

### SmolVLA

- 需要基础 SmolVLM 模型资产
- 可以从本地路径或环境变量覆盖解析基础 VLM 资产

## 示例

当前仓库提供两个可直接运行的示例入口：

- [`examples/api_usage.py`](examples/api_usage.py)
- [`examples/act_alicia_m_real_robot.py`](examples/act_alicia_m_real_robot.py)

这些脚本分别演示：

- `api_usage.py`：使用仓库内置 ACT checkpoint 做基础加载、控制环推理，以及 PI0/SmolVLA 的运行前准备提示
- `act_alicia_m_real_robot.py`：将 ACT 推理接到 Alicia-M-SDK 真机控制环，采集头部/腕部相机图像并下发 6 关节 + 夹爪动作

## 仓库结构

```text
inference_sdk/
  __init__.py
  core/
    __init__.py
    config.py
    types.py
  policy/
    __init__.py
    act.py
    pi0.py
    smolvla.py
  runtime/
    __init__.py
    base.py
    device.py
    monitoring.py
examples/
model/
INFERENCE_API_GUIDE.md
pyproject.toml
```

目录职责划分：

- `inference_sdk/__init__.py`：稳定的开发者公开导入入口
- `inference_sdk/core/`：共享配置对象、数据结构
- `inference_sdk/policy/`：各推理算法对应的策略实现
- `inference_sdk/runtime/`：生命周期、设备选择、监控等运行时能力

## 兼容性说明

- 新代码应直接从 `inference_sdk` 导入
- SDK 可以独立运行，默认带有 no-op monitor
- 宿主应用可以通过 `set_inference_monitor(...)` 注入自己的 monitor
- 这个包设计为可独立 editable install 的推理 SDK

## 相关说明

- [`INFERENCE_API_GUIDE.md`](INFERENCE_API_GUIDE.md)
- [`examples/api_usage.py`](examples/api_usage.py)
- [`examples/act_alicia_m_real_robot.py`](examples/act_alicia_m_real_robot.py)
