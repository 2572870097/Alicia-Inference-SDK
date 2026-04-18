# Alicia Inference SDK

[English](README.md)

用于在 FastAPI 应用层之外加载和运行 `ACT`、`PI0`、`SmolVLA` 策略的可复用 Python SDK。

这个仓库是面向开发者的推理层，主要统一了以下内容：

- 策略如何创建和加载
- 观测输入对象长什么样
- 元数据和运行状态如何表达
- 异步控制环推理如何启动和停止
- SDK 如何向调用方暴露错误

v1 SDK 契约说明见
[`docs/v1-sdk-design.zh-CN.md`](docs/v1-sdk-design.zh-CN.md)。

## 安装

在当前目录下以 editable 模式安装：

```bash
pip install -e .
```

或者从 monorepo 根目录安装：

```bash
pip install -e Alicia-Inference-SDK
```

按模型安装可选依赖：

```bash
pip install -e .[act]
pip install -e .[pi0]
pip install -e .[smolvla]
pip install -e .[all]
```

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
- `SDKError` 及其子类

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

```python
from inference_sdk import (
    DeviceConfig,
    InferenceSession,
    Observation,
    PolicyLoadConfig,
    RuntimeConfig,
    SDKError,
)

config = PolicyLoadConfig(
    checkpoint_dir="/path/to/checkpoint",
    model_type="pi0",
    device=DeviceConfig(device="cuda:0"),
    runtime=RuntimeConfig(
        control_fps=20.0,
        enable_async_inference=True,
    ),
    tokenizer_path="/path/to/tokenizer",
    instruction="pick up the apple",
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
            instruction="pick up the apple",
        )

        action_chunk = session.infer(observation)
        print(action_chunk)
    finally:
        session.close()
except SDKError as exc:
    print(exc.code, exc.message, exc.details)
```

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
- `latency_estimate_ms`
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

v1 SDK 使用结构化异常，而不是只返回字符串错误：

- `SDKError`
- `ConfigurationError`
- `ValidationError`
- `CheckpointError`
- `DependencyError`
- `DeviceUnavailableError`
- `ModelLoadError`
- `InstructionNotSupportedError`
- `ResourceStateError`
- `InferenceError`
- `InferenceTimeoutError`

每个 `SDKError` 都暴露：

- `code`
- `message`
- `details`
- `recoverable`

示例：

```python
from inference_sdk import SDKError, load_policy

try:
    policy = load_policy(checkpoint_dir="/bad/path", model_type="act")
except SDKError as exc:
    print(exc.to_dict())
```

## 运行时配置

使用 `RuntimeConfig` 配置控制环行为：

```python
from inference_sdk import RuntimeConfig

runtime = RuntimeConfig(
    control_fps=30.0,
    enable_async_inference=True,
    aggregate_fn_name="latest_only",
    fallback_mode="repeat",
)
```

关键字段：

- `control_fps`
- `enable_async_inference`
- `aggregate_fn_name`
- `fallback_mode`
- `gripper_max_velocity`
- `latency_ema_alpha`
- `latency_safety_margin`
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

- SDK 会严格使用你传入的设备字符串
- 如果请求的设备不可用，SDK 会直接抛出 `DeviceUnavailableError`

## 模型专项说明

### ACT

- 支持旧版 checkpoint 目录格式
- 支持导出后的 checkpoint 目录格式
- 不支持语言指令

### PI0

- 需要 tokenizer 资产
- 可以通过 `tokenizer_path` 显式传入 tokenizer 路径
- 也可以从本地模型目录或环境变量中解析 tokenizer

### SmolVLA

- 需要基础 SmolVLM 模型资产
- 可以从本地路径或环境变量覆盖解析基础 VLM 资产

## 示例

仓库当前只保留每个算法一个示例入口：

- [`examples/act.py`](examples/act.py)
- [`examples/pi0.py`](examples/pi0.py)
- [`examples/smolvla.py`](examples/smolvla.py)

可运行命令见 [`examples/README.zh-CN.md`](examples/README.zh-CN.md)。

## 仓库结构

```text
inference_sdk/
  __init__.py
  core/
    __init__.py
    config.py
    exceptions.py
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
docs/
```

目录职责划分：

- `inference_sdk/__init__.py`：稳定的开发者公开导入入口
- `inference_sdk/core/`：共享配置对象、异常类型、数据结构
- `inference_sdk/policy/`：各推理算法对应的策略实现
- `inference_sdk/runtime/`：生命周期、设备选择、监控等运行时能力

## 兼容性说明

- 新代码应直接从 `inference_sdk` 导入
- SDK 可以独立运行，默认带有 no-op monitor
- 宿主应用可以通过 `set_inference_monitor(...)` 注入自己的 monitor
- 这个包设计为可独立 editable install 的推理 SDK

## 相关设计文档

- [`docs/v1-sdk-design.zh-CN.md`](docs/v1-sdk-design.zh-CN.md)
