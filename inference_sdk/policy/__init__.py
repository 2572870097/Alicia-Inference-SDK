_LOADED_MODELS: set[str] = set()


def _load_model(model_type: str) -> None:
    global ACTInferenceEngine, PI0InferenceEngine, SmolVLAInferenceEngine
    global ACT_AVAILABLE, PI0_AVAILABLE, SMOLVLA_AVAILABLE

    if model_type in _LOADED_MODELS:
        return

    if model_type == "act":
        from .act import ACTInferenceEngine as _ACTInferenceEngine, ACT_AVAILABLE as _ACT_AVAILABLE

        ACTInferenceEngine = _ACTInferenceEngine
        ACT_AVAILABLE = _ACT_AVAILABLE
    elif model_type == "pi0":
        from .pi0 import PI0InferenceEngine as _PI0InferenceEngine, PI0_AVAILABLE as _PI0_AVAILABLE

        PI0InferenceEngine = _PI0InferenceEngine
        PI0_AVAILABLE = _PI0_AVAILABLE
    elif model_type == "smolvla":
        from .smolvla import (
            SmolVLAInferenceEngine as _SmolVLAInferenceEngine,
            SMOLVLA_AVAILABLE as _SMOLVLA_AVAILABLE,
        )

        SmolVLAInferenceEngine = _SmolVLAInferenceEngine
        SMOLVLA_AVAILABLE = _SMOLVLA_AVAILABLE
    else:
        raise AttributeError(f"Unknown model type: {model_type}")

    _LOADED_MODELS.add(model_type)


def __getattr__(name: str):
    if name in {"ACTInferenceEngine", "ACT_AVAILABLE"}:
        _load_model("act")
        return globals()[name]
    if name in {"PI0InferenceEngine", "PI0_AVAILABLE"}:
        _load_model("pi0")
        return globals()[name]
    if name in {"SmolVLAInferenceEngine", "SMOLVLA_AVAILABLE"}:
        _load_model("smolvla")
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ACTInferenceEngine",
    "ACT_AVAILABLE",
    "PI0InferenceEngine",
    "PI0_AVAILABLE",
    "SmolVLAInferenceEngine",
    "SMOLVLA_AVAILABLE",
]
