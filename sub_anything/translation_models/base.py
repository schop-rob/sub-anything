"""Base class for translation model implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from ..models import TranscriptSegment


class TranslationModel(ABC):
    """Abstract base class for translation backends."""

    MODEL_ID: str = ""
    DISPLAY_NAME: str = ""
    REQUIRED_ENV: tuple[str, ...] = ()

    @classmethod
    def required_env(cls) -> tuple[str, ...]:
        return tuple(cls.REQUIRED_ENV)

    @classmethod
    def check(cls) -> tuple[bool, str]:
        missing = [var for var in cls.required_env() if not os.environ.get(var)]
        if missing:
            return False, f"Missing environment variables: {', '.join(missing)}"
        return True, ""

    @abstractmethod
    def translate(
        self,
        segments: list[TranscriptSegment],
        target_language: str,
        *,
        verbose: bool = False,
    ) -> list[TranscriptSegment]:
        raise NotImplementedError

