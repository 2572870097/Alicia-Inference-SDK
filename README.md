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

The public SDK surface is documented in this README and in
[`INFERENCE_API_GUIDE.md`](INFERENCE_API_GUIDE.md).

## Installation

Install in editable mode from this directory:

```bash
pip install -e .
```

Model-specific extras:

```bash
pip install -e .[act]
pip install -e .[pi0]
pip install -e .[smolvla]
pip install -e .[all]
```

Those extras are only the SDK-side dependencies. The actual policy
implementations import `sparkmind.*`, so the active virtual environment also
needs SparkMind available in the same environment. A common workspace layout is
to keep `SparkMind` next to this repository and install that checkout too:

```bash
source .venv/bin/activate
pip install -e .[all]
pip install -e ../SparkMind
```

If a sibling `SparkMind/` checkout exists, the SDK also tries to add it to
`sys.path` at runtime as a fallback. That fallback only helps Python find the
source tree. SparkMind's own Python dependencies still must be installed in the
same virtual environment.

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

## Runtime Prerequisites

Before calling `load_model(...)`, make sure the runtime assets match the model
type:

- `act`: requires a valid exported ACT checkpoint directory.
- `pi0`: requires a valid PI0 checkpoint plus tokenizer assets. The SDK checks
  `tokenizer_path`, then `PI0_TOKENIZER_PATH`, then common local `models/...`
  folders before falling back to Hugging Face.
- `smolvla`: requires a valid SmolVLA checkpoint plus the base VLM assets
  declared by the checkpoint config. For offline runs, set
  `SMOLVLA_VLM_MODEL_PATH` to a local copy of that model.

For a local smoke test, this repository currently includes one exported ACT
checkpoint under [`model/ACT_pick_and_place_v2`](model/ACT_pick_and_place_v2).

### Asset Download Paths

- `ACT checkpoint`:
  This repository already bundles one exported ACT checkpoint at
  [`model/ACT_pick_and_place_v2`](model/ACT_pick_and_place_v2). Its model card
  points to the Hugging Face repo `z18820636149/ACT_pick_and_place_v2`.
- `PI0 checkpoint`:
  the SDK does not hardcode a single download repo. Use any exported PI0
  checkpoint directory that matches the expected format, then pass it through
  `checkpoint_dir`.
- `PI0 tokenizer`:
  the default remote asset is `google/paligemma2-3b-mix-224`.
- `SmolVLA checkpoint`:
  the SDK does not hardcode a single download repo. Use any exported SmolVLA
  checkpoint directory that matches the expected format, then pass it through
  `checkpoint_dir`.
- `SmolVLA base VLM`:
  the default remote asset is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

Recommended local download commands:

```bash
source .venv/bin/activate

# PI0 tokenizer
hf download google/paligemma2-3b-mix-224 \
  --local-dir models/google/paligemma2-3b-mix-224
export PI0_TOKENIZER_PATH=$PWD/models/google/paligemma2-3b-mix-224

# SmolVLA base VLM
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export SMOLVLA_VLM_MODEL_PATH=$PWD/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Local path resolution used by the SDK:

- `PI0 tokenizer`: `tokenizer_path` -> `PI0_TOKENIZER_PATH` ->
  `checkpoint_dir/tokenizer` -> local `models/google/paligemma2-3b-mix-224`
  style directories.
- `SmolVLA base VLM`: `SMOLVLA_VLM_MODEL_PATH` -> `checkpoint_dir/vlm_model` ->
  local `models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct` style directories.

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
- standard Python exceptions (`ValueError`, `RuntimeError`, `ImportError`)

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

For the shortest local smoke test, use the bundled ACT checkpoint under
`model/ACT_pick_and_place_v2`.

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

For `PI0`, pass tokenizer assets through `tokenizer_path` or
`PI0_TOKENIZER_PATH`. For `SmolVLA`, make sure the base VLM assets are locally
available or reachable via `SMOLVLA_VLM_MODEL_PATH`.

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

The v1 SDK uses standard Python exceptions instead of a custom exception
hierarchy:

- `ValueError` for invalid config, bad inputs, unsupported model types, and checkpoint validation failures
- `RuntimeError` for unloaded policy state, unavailable runtime/device state, and inference-time failures
- `ImportError` for missing optional dependencies

Example:

```python
from inference_sdk import load_policy

try:
    policy = load_policy(checkpoint_dir="/bad/path", model_type="act")
except Exception as exc:
    print(type(exc).__name__, exc)
```

## Runtime Configuration

Use `RuntimeConfig` to configure control-loop behavior:

```python
from inference_sdk import RuntimeConfig

runtime = RuntimeConfig(
    control_fps=30.0,
    enable_async_inference=True,
    temporal_ensemble_coeff=None,
    fallback_mode="repeat",
)
```

Key fields:

- `control_fps`
- `enable_async_inference`
- `temporal_ensemble_coeff`
- `fallback_mode`
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

- supported values are `cpu`, `cuda`, and `cuda:<index>`
- unsupported device strings fail validation early
- if the requested device is unavailable, the SDK raises `RuntimeError`

## Model-Specific Notes

### ACT

- supports exported checkpoint directories only
- does not support language instructions

### PI0

- requires tokenizer assets
- tokenizer can be provided via `tokenizer_path`
- tokenizer may also be resolved from local model directories or environment

### SmolVLA

- requires the base SmolVLM model assets
- can resolve VLM assets from a local path or environment override

## Examples

The repository currently ships two runnable example entrypoints:

- [`examples/api_usage.py`](examples/api_usage.py)
- [`examples/act_alicia_m_real_robot.py`](examples/act_alicia_m_real_robot.py)

Those scripts demonstrate:

- `api_usage.py`: ACT loading with the bundled sample checkpoint, control-loop style inference, and PI0/SmolVLA runtime prerequisite hints
- `act_alicia_m_real_robot.py`: wiring ACT inference into an Alicia-M real-robot loop with head/wrist cameras and 6-joint plus gripper execution via Alicia-M-SDK

## Repository Layout

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

Package responsibilities:

- `inference_sdk/__init__.py`: stable developer-facing import surface
- `inference_sdk/core/`: shared config objects and data types
- `inference_sdk/policy/`: model-specific policy implementations
- `inference_sdk/runtime/`: lifecycle, device resolution, and monitoring utilities

## Compatibility Notes

- new code should import from `inference_sdk`
- the SDK can run standalone with its built-in no-op monitor
- host applications can inject a monitor via `set_inference_monitor(...)`
- the package is designed to be installed independently via editable install

## Related Notes

- [`INFERENCE_API_GUIDE.md`](INFERENCE_API_GUIDE.md)
- [`examples/api_usage.py`](examples/api_usage.py)
- [`examples/act_alicia_m_real_robot.py`](examples/act_alicia_m_real_robot.py)
