"""OpenAI Whisper (Replicate) ASR model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment
from .base import ASRModel

REQUIRED_ENV = ("REPLICATE_API_TOKEN",)


class ReplicateWhisperASR(ASRModel):
    """OpenAI Whisper hosted on Replicate (supports large-v3)."""

    MODEL_ID = "replicate-whisper"
    DISPLAY_NAME = "OpenAI Whisper (Replicate)"
    DESCRIPTION = "Whisper large-v3 via Replicate (good accuracy, segment timestamps)"
    COST_PER_MINUTE = 0.006
    REQUIRED_ENV = REQUIRED_ENV

    DEFAULT_CHUNK_DURATION_SECONDS = 600
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = False

    def __init__(self, *, whisper_model: str = "large-v3"):
        self.whisper_model = whisper_model

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

    @staticmethod
    def _normalize_language_hint(language: str) -> str:
        lower = language.strip().lower().replace("_", "-")
        if lower in {"cmn-hans-cn", "cmn-hant-tw", "yue-hant-hk"}:
            return "zh"
        if lower.startswith("zh"):
            return "zh"
        if "-" in lower:
            return lower.split("-", 1)[0]
        return lower

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        if enable_diarization:
            print("Warning: diarization is not supported by replicate-whisper; ignoring --diarize.")

        import replicate

        input_params: dict = {
            "model": self.whisper_model,
            "translate": False,
            "transcription": "plain text",
        }

        if language and language not in {"auto"}:
            input_params["language"] = self._normalize_language_hint(language)

        with open(audio_path, "rb") as audio_file:
            input_params["audio"] = audio_file

            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    if verbose:
                        print(f"  Running Replicate Whisper ({self.whisper_model})...")
                    output = replicate.run("openai/whisper", input=input_params)
                    return self._parse_output(output)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 5
                        if verbose:
                            print(f"  Replicate error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        raise last_error  # pragma: no cover

    @staticmethod
    def _parse_output(output) -> list[TranscriptSegment]:
        # Replicate returns a dict-like object with segments/transcription.
        if not output:
            return []

        if isinstance(output, str):
            text = output.strip()
            if not text:
                return []
            return [TranscriptSegment(start_time=0.0, end_time=1.0, text=text, confidence=0.95)]

        detected_language = None
        transcription = None
        segments = None

        if isinstance(output, dict):
            detected_language = output.get("detected_language")
            transcription = output.get("transcription") or output.get("text")
            segments = output.get("segments")
        else:
            detected_language = getattr(output, "detected_language", None)
            transcription = getattr(output, "transcription", None) or getattr(output, "text", None)
            segments = getattr(output, "segments", None)

        out: list[TranscriptSegment] = []

        if segments:
            for seg in segments:
                start = getattr(seg, "start", None)
                end = getattr(seg, "end", None)
                text = getattr(seg, "text", None)
                if isinstance(seg, dict):
                    start = seg.get("start", start)
                    end = seg.get("end", end)
                    text = seg.get("text", text)

                if start is None or end is None:
                    continue
                text = (text or "").strip()
                if not text:
                    continue

                out.append(
                    TranscriptSegment(
                        start_time=float(start),
                        end_time=float(end),
                        text=text,
                        confidence=0.95,
                        language=detected_language,
                    )
                )

        if out:
            return out

        text = (transcription or "").strip()
        if not text:
            return []
        return [TranscriptSegment(start_time=0.0, end_time=1.0, text=text, confidence=0.95, language=detected_language)]
