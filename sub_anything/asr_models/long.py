"""Google Long (Speech-to-Text V2) ASR model."""

from __future__ import annotations

REQUIRED_ENV = ("GOOGLE_APPLICATION_CREDENTIALS",)

from .google_base import GoogleSpeechV2ASRBase


class LongASR(GoogleSpeechV2ASRBase):
    MODEL_ID = "long"
    DISPLAY_NAME = "Google Long"
    DESCRIPTION = "Guaranteed timestamps, slightly less accurate"
    COST_PER_MINUTE = 0.016
    REQUIRED_ENV = REQUIRED_ENV

    GOOGLE_MODEL = "long"
    DEFAULT_LOCATION = "us-central1"
    DEFAULT_CHUNK_DURATION_SECONDS = 3600
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = True

    @classmethod
    def check(cls) -> tuple[bool, str]:
        return super().check()
