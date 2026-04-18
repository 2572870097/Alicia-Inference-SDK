# Alicia Inference SDK

Reusable Python SDK for loading and running ACT, PI0, and SmolVLA policies outside
the FastAPI application layer.

Install in editable mode from this directory:

```bash
pip install -e .
```

Or from the monorepo root:

```bash
pip install -e Alicia-Inference-SDK
```

## Public entry points

- `create_policy(...)`: create an unloaded policy instance.
- `load_policy(...)`: create and load a policy in one call.
- `Observation`: unified input object for real-time and offline inference.
- `PolicyMetadata` / `PolicyStatus`: stable metadata and runtime status objects.

## Examples

- `examples/common/`: generic examples that can target any supported model.
- `examples/act/`: ACT-specific examples.
- `examples/pi0/`: PI0-specific examples.
- `examples/smolvla/`: SmolVLA-specific examples.

See [examples/README.md](examples/README.md) for concrete commands.

## Integration notes

- The SDK can run standalone with its built-in no-op monitor.
- Application code can inject a monitor via `set_inference_monitor(...)`.
- The backend keeps a compatibility layer under `core.inference`, but new code
  should import from `inference_sdk`.
- The package is designed to live beside `Alicia-D-SDK/` in the repo root and be
  installed independently via editable install.
