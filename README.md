# Alicia Inference SDK

[中文文档](README.zh-CN.md)

Reusable Python SDK for loading and running `ACT`, `PI0`, and `SmolVLA` policies
outside the FastAPI application layer.

This repository is the developer-facing inference layer. It standardizes:

- how policies are created and loaded
- what an observation looks like
- what metadata and runtime status look like
- how async control-loop inference is started and stopped
- how errors are surfaced to SDK callers

The v1 SDK contract is documented in
[`docs/v1-sdk-design.md`](docs/v1-sdk-design.md).

## Installation

Install in editable mode from this directory:

```bash
pip install -e .
```

Or from the monorepo root:

```bash
pip install -e Alicia-Inference-SDK
```

Model-specific extras:

```bash
pip install -e .[act]
pip install -e .[pi0]
pip install -e .[smolvla]
pip install -e .[all]
```

Base dependencies:

- `numpy`
- `opencv-python`
- `pyyaml`

Optional model dependencies:

- `torch`
- `transformers`
- `omegaconf`
- `safetensors`

## Supported Models

- `act`
- `pi0`
- `smolvla`

`PI0` and `SmolVLA` support language instructions. `ACT` does not.

## Public API

Preferred v1 entry points:

- `InferenceSession`
- `load_policy(...)`
- `create_policy(...)`
- `PolicyLoadConfig`
- `RuntimeConfig`
- `DeviceConfig`
- `Observation`
- `PolicyMetadata`
- `PolicyStatus`
- `SDKError` and subclasses

Core runtime methods on a loaded session or policy:

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

## Quickstart

### 1. Session-style loading and chunk inference

This is the recommended v1 usage pattern.

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

### 2. Session-style loading with explicit model parameters

If the caller wants to pass parameters directly instead of building a config
object first:

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

### 3. Low-level policy loading

The low-level policy API is still supported for compatibility and advanced use:

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

## Data Contract

### Observation

`Observation` is the unified SDK input object:

```python
Observation(
    images={"head": np.ndarray, "wrist": np.ndarray},
    state=np.ndarray,
    instruction="optional language instruction",
    timestamp=None,
)
```

Input expectations:

- `images` must be a non-empty `dict[str, np.ndarray]`
- each image must have shape `(H, W, 3)`
- images are expected in OpenCV-style `BGR` format
- `state` must be a 1D `np.ndarray`
- `instruction` is optional and only used by language-conditioned models

### PolicyMetadata

Static information returned by `policy.get_metadata()`:

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

Runtime information returned by `policy.get_status()`:

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

## Lifecycle Semantics

The recommended lifecycle is:

1. Create and load an `InferenceSession`, or load a policy directly.
2. Read metadata to discover required camera roles and state size.
3. Call `infer()` for raw action chunk inference, or optionally start async inference.
4. Call `step()` in a control loop when you need one action at a time.
5. Stop async inference if it was started.
6. Call `close()` / `unload()`, or use a context manager.

Important runtime rules:

- `InferenceSession.load(...)` loads one model selected by `model_type`.
- `InferenceSession.infer(...)` returns the raw action chunk for one observation.
- `load_policy(...)` returns a loaded policy instance.
- `create_policy(...)` returns an unloaded policy instance.
- Async inference is not started automatically; call `start_async_inference()` explicitly.
- `step()` uses queue-aware control-loop behavior.
- `predict_chunk()` bypasses queue scheduling and returns raw chunk predictions.
- `close()` is an alias for `unload()`.

## Error Model

The v1 SDK uses typed exceptions instead of only returning strings:

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

Every `SDKError` exposes:

- `code`
- `message`
- `details`
- `recoverable`

Example:

```python
from inference_sdk import SDKError, load_policy

try:
    policy = load_policy(checkpoint_dir="/bad/path", model_type="act")
except SDKError as exc:
    print(exc.to_dict())
```

## Runtime Configuration

Use `RuntimeConfig` to configure control-loop behavior:

```python
from inference_sdk import RuntimeConfig

runtime = RuntimeConfig(
    control_fps=30.0,
    enable_async_inference=True,
    aggregate_fn_name="latest_only",
    fallback_mode="repeat",
)
```

Key fields:

- `control_fps`
- `enable_async_inference`
- `aggregate_fn_name`
- `fallback_mode`
- `gripper_max_velocity`
- `latency_ema_alpha`
- `latency_safety_margin`
- `obs_queue_maxsize`

The legacy `SmoothingConfig` remains available, but `RuntimeConfig` is the
preferred developer-facing config object for v1.

## Device Configuration

Use `DeviceConfig` to control device selection:

```python
from inference_sdk import DeviceConfig

device = DeviceConfig(device="cuda:0")
```

Behavior:

- the SDK uses the exact requested device string
- if the requested device is unavailable, the SDK raises `DeviceUnavailableError`

## Model-Specific Notes

### ACT

- supports legacy checkpoint directories
- supports exported checkpoint directories
- does not support language instructions

### PI0

- requires tokenizer assets
- tokenizer can be provided via `tokenizer_path`
- tokenizer may also be resolved from local model directories or environment

### SmolVLA

- requires the base SmolVLM model assets
- can resolve VLM assets from a local path or environment override

## Examples

Repository examples:

- [`examples/act.py`](examples/act.py)
- [`examples/pi0.py`](examples/pi0.py)
- [`examples/smolvla.py`](examples/smolvla.py)

See [`examples/README.md`](examples/README.md) for runnable commands.

## Repository Layout

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

Package responsibilities:

- `inference_sdk/__init__.py`: stable developer-facing import surface
- `inference_sdk/core/`: shared config objects, exceptions, and data types
- `inference_sdk/policy/`: model-specific policy implementations
- `inference_sdk/runtime/`: lifecycle, device resolution, and monitoring utilities

## Compatibility Notes

- new code should import from `inference_sdk`
- the SDK can run standalone with its built-in no-op monitor
- host applications can inject a monitor via `set_inference_monitor(...)`
- the package is designed to be installed independently via editable install

## Related Design Doc

- [`docs/v1-sdk-design.md`](docs/v1-sdk-design.md)
