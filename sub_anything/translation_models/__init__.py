"""Translation model implementations."""

from __future__ import annotations

import importlib
import pkgutil

from .base import TranslationModel


def _discover_translation_models() -> dict[str, type[TranslationModel]]:
    models: dict[str, type[TranslationModel]] = {}

    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if module_info.name in {"__init__", "base"}:
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")

        for obj in module.__dict__.values():
            if not isinstance(obj, type):
                continue
            if obj is TranslationModel or not issubclass(obj, TranslationModel):
                continue
            model_id = getattr(obj, "MODEL_ID", "")
            if not model_id:
                continue
            models[model_id] = obj

    return models


TRANSLATION_MODELS: dict[str, type[TranslationModel]] = _discover_translation_models()
AVAILABLE_TRANSLATION_MODELS: list[str] = sorted(TRANSLATION_MODELS.keys())
