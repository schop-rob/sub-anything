"""Compatibility shim for older imports.

Translation model implementations now live in `sub_anything/translation_models/`.
"""

from .translation_models.openai import OpenAITranslationModel as TranslatorService

__all__ = ["TranslatorService"]

