"""Google Chirp 3 (Speech-to-Text V2) ASR model."""

from __future__ import annotations

REQUIRED_ENV = ("GOOGLE_APPLICATION_CREDENTIALS",)

from .google_base import GoogleSpeechV2ASRBase


class Chirp3ASR(GoogleSpeechV2ASRBase):
    MODEL_ID = "chirp3"
    DISPLAY_NAME = "Google Chirp 3"
    DESCRIPTION = "Best quality, 70+ languages, may have timestamp gaps"
    COST_PER_MINUTE = 0.016
    REQUIRED_ENV = REQUIRED_ENV

    GOOGLE_MODEL = "chirp_3"
    DEFAULT_LOCATION = "eu"
    # Chirp 3 can be slow on very long BatchRecognize jobs; keep chunks small.
    DEFAULT_CHUNK_DURATION_SECONDS = 120
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = True

    @classmethod
    def check(cls) -> tuple[bool, str]:
        return super().check()
