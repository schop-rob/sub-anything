"""sub-anything: Transcribe any audio/video file to SRT subtitles."""

from .models import TranscriptSegment, Config
from .providers import (
    ASRProvider,
    Chirp3ASR,
    LongASR,
    WhisperXASR,
    OpenAIWhisperASR,
    ReplicateFastWhisperASR,
    ReplicateWhisperASR,
)
from .translator import TranslatorService

__version__ = "1.0.0"
__all__ = [
    "TranscriptSegment",
    "Config",
    "ASRProvider",
    "Chirp3ASR",
    "LongASR",
    "WhisperXASR",
    "OpenAIWhisperASR",
    "ReplicateFastWhisperASR",
    "ReplicateWhisperASR",
    "TranslatorService",
]
