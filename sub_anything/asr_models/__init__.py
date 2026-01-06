"""ASR model implementations (one file per model)."""

from __future__ import annotations

import importlib
import pkgutil

from .base import ASRModel


def _discover_asr_models() -> dict[str, type[ASRModel]]:
    models: dict[str, type[ASRModel]] = {}

    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if module_info.name in {"__init__", "base", "google_base"}:
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")

        for obj in module.__dict__.values():
            if not isinstance(obj, type):
                continue
            if obj is ASRModel or not issubclass(obj, ASRModel):
                continue
            model_id = getattr(obj, "MODEL_ID", "")
            if not model_id:
                continue
            models[model_id] = obj

    return models


ASR_MODELS: dict[str, type[ASRModel]] = _discover_asr_models()

_PREFERRED_ORDER = [
    "chirp3",
    "long",
    "whisperx",
    "replicate-fast-whisper",
    "replicate-whisper",
    "whisper",
]
AVAILABLE_ASR_MODELS: list[str] = [
    *[m for m in _PREFERRED_ORDER if m in ASR_MODELS],
    *sorted(set(ASR_MODELS.keys()) - set(_PREFERRED_ORDER)),
]
