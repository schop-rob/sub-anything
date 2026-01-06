"""Base class for ASR model implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment


class ASRModel(ABC):
    """Abstract base class for an ASR model backend."""

    MODEL_ID: str = ""
    DISPLAY_NAME: str = ""
    DESCRIPTION: str = ""
    COST_PER_MINUTE: float = 0.0

    REQUIRED_ENV: tuple[str, ...] = ()

    # Chunking defaults (used by the CLI to segment long audio for this model).
    # Models that struggle with long uploads should override these.
    DEFAULT_CHUNK_DURATION_SECONDS: int = 3600
    DEFAULT_CHUNK_OVERLAP_SECONDS: int = 10

    # Output capabilities. If a model can't provide real timestamps, override to ("txt",).
    SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = ("srt", "txt")

    # Speaker diarization capability. Note: some models may require extra tokens (e.g. HF_TOKEN) at runtime.
    CAN_DIARIZE: bool = False

    @classmethod
    def required_env(cls) -> tuple[str, ...]:
        return tuple(cls.REQUIRED_ENV)

    @classmethod
    def can_provide_timestamps(cls) -> bool:
        return "srt" in cls.SUPPORTED_OUTPUT_FORMATS

    @classmethod
    def can_provide_plain_text(cls) -> bool:
        return "txt" in cls.SUPPORTED_OUTPUT_FORMATS

    @classmethod
    def can_diarize(cls) -> bool:
        return bool(cls.CAN_DIARIZE)

    # Backwards-compatible aliases.
    @classmethod
    def supports_timestamps(cls) -> bool:
        return cls.can_provide_timestamps()

    @classmethod
    def supports_plain_text(cls) -> bool:
        return cls.can_provide_plain_text()

    @classmethod
    def default_chunk_duration_seconds(cls) -> int:
        return int(cls.DEFAULT_CHUNK_DURATION_SECONDS)

    @classmethod
    def default_chunk_overlap_seconds(cls) -> int:
        return int(cls.DEFAULT_CHUNK_OVERLAP_SECONDS)

    @classmethod
    def check(cls) -> tuple[bool, str]:
        """Return (ok, error_message)."""
        missing = [var for var in cls.required_env() if not os.environ.get(var)]
        if missing:
            return False, f"Missing environment variables: {', '.join(missing)}"
        return True, ""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        raise NotImplementedError
