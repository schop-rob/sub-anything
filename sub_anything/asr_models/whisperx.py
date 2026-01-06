"""WhisperX (Replicate) ASR model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment
from .base import ASRModel

REQUIRED_ENV = ("REPLICATE_API_TOKEN",)


class WhisperXASR(ASRModel):
    MODEL_ID = "whisperx"
    DISPLAY_NAME = "WhisperX (Replicate)"
    DESCRIPTION = "Excellent timestamps, fast, good diarization"
    COST_PER_MINUTE = 0.006
    REQUIRED_ENV = REQUIRED_ENV
    DEFAULT_CHUNK_DURATION_SECONDS = 3600
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = True

    def __init__(self):
        import replicate

        self._replicate = replicate

    @classmethod
    def check(cls) -> tuple[bool, str]:
        ok, msg = super().check()
        if not ok:
            return ok, msg
        try:
            import replicate  # noqa: F401
        except ImportError:
            return False, "Replicate library not installed. Run: pip install replicate"
        return True, ""

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        if verbose:
            print("  Uploading to Replicate...")

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        with open(audio_path, "rb") as audio_file:
            input_params: dict = {
                "audio_file": audio_file,
                "align_output": True,
                "batch_size": 64,
            }

            if language and language != "auto":
                input_params["language"] = language

            if enable_diarization:
                if hf_token:
                    input_params["diarization"] = True
                    input_params["huggingface_access_token"] = hf_token
                else:
                    print("Warning: Diarization requested but no HuggingFace token provided.")
                    print("Set HF_TOKEN environment variable for speaker labels.")

            if verbose:
                print("  Running WhisperX transcription...")

            max_retries = 3
            output = None

            for attempt in range(max_retries):
                try:
                    output = self._replicate.run(
                        "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
                        input=input_params,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 5
                        if verbose:
                            print(f"  Error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        if output is None:
            return []

        return self._parse_output(output)

    @staticmethod
    def _parse_output(output: dict) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        detected_language = output.get("detected_language")

        for seg in output.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = (seg.get("text") or "").strip()
            speaker = seg.get("speaker")

            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start_time=start,
                    end_time=end,
                    text=text,
                    confidence=0.95,
                    speaker=speaker,
                    language=detected_language,
                )
            )

        return segments
