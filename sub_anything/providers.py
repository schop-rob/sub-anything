"""Compatibility shim for older imports.

ASR model implementations now live in `sub_anything/asr_models/` (one file per model).
"""

from .asr_models.base import ASRModel as ASRProvider
from .asr_models.chirp3 import Chirp3ASR
from .asr_models.long import LongASR
from .asr_models.whisper import OpenAIWhisperASR
from .asr_models.whisperx import WhisperXASR
from .asr_models.replicate_fast_whisper import ReplicateFastWhisperASR
from .asr_models.replicate_whisper import ReplicateWhisperASR

__all__ = [
    "ASRProvider",
    "Chirp3ASR",
    "LongASR",
    "WhisperXASR",
    "OpenAIWhisperASR",
    "ReplicateFastWhisperASR",
    "ReplicateWhisperASR",
]
