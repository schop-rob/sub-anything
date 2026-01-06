"""Incredibly-fast Whisper (Replicate) ASR model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment
from .base import ASRModel

REQUIRED_ENV = ("REPLICATE_API_TOKEN",)


class ReplicateFastWhisperASR(ASRModel):
    """vaibhavs10/incredibly-fast-whisper (whisper-large-v3, very fast)."""

    MODEL_ID = "replicate-fast-whisper"
    DISPLAY_NAME = "Incredibly Fast Whisper (Replicate)"
    DESCRIPTION = "Whisper large-v3 via Replicate, extremely fast (chunk timestamps)"
    COST_PER_MINUTE = 0.006
    REQUIRED_ENV = REQUIRED_ENV

    DEFAULT_CHUNK_DURATION_SECONDS = 600
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = True

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
        if lower.startswith("zh") or lower in {"cmn-hans-cn", "cmn-hant-tw", "yue-hant-hk"}:
            return "chinese"
        if lower.startswith("en"):
            return "english"
        if lower.startswith("ja"):
            return "japanese"
        if lower.startswith("ko"):
            return "korean"
        if lower.startswith("fr"):
            return "french"
        if lower.startswith("de"):
            return "german"
        if lower.startswith("es"):
            return "spanish"
        if lower.startswith("it"):
            return "italian"
        if lower.startswith("ru"):
            return "russian"
        if lower.startswith("pt"):
            return "portuguese"
        return ""

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        import replicate

        input_params: dict = {
            "task": "transcribe",
            "timestamp": "chunk",
        }

        lang = None
        if language and language not in {"auto"}:
            lang = self._normalize_language_hint(language)
        if lang:
            input_params["language"] = lang

        if enable_diarization:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                input_params["diarise_audio"] = True
                input_params["hf_token"] = hf_token
            else:
                print("Warning: diarization requested but HF_TOKEN is not set; continuing without diarization.")

        with open(audio_path, "rb") as audio_file:
            input_params["audio"] = audio_file

            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    if verbose:
                        print("  Running Replicate incredibly-fast-whisper...")
                    output = replicate.run("vaibhavs10/incredibly-fast-whisper", input=input_params)
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
        if not output:
            return []

        if isinstance(output, str):
            text = output.strip()
            if not text:
                return []
            return [TranscriptSegment(start_time=0.0, end_time=1.0, text=text, confidence=0.95)]

        chunks = None
        text = None

        if isinstance(output, dict):
            chunks = output.get("chunks") or output.get("segments")
            text = output.get("text") or output.get("transcription")
        else:
            chunks = getattr(output, "chunks", None) or getattr(output, "segments", None)
            text = getattr(output, "text", None) or getattr(output, "transcription", None)

        out: list[TranscriptSegment] = []
        if chunks:
            for chunk in chunks:
                chunk_text = getattr(chunk, "text", None)
                timestamp = getattr(chunk, "timestamp", None)
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", chunk_text)
                    timestamp = chunk.get("timestamp", timestamp) or chunk.get("timestamps")

                chunk_text = (chunk_text or "").strip()
                if not chunk_text:
                    continue

                start = None
                end = None
                if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                    start, end = timestamp[0], timestamp[1]

                if start is None or end is None:
                    continue

                out.append(
                    TranscriptSegment(
                        start_time=float(start),
                        end_time=float(end),
                        text=chunk_text,
                        confidence=0.95,
                    )
                )

        if out:
            return out

        text = (text or "").strip()
        if not text:
            return []
        return [TranscriptSegment(start_time=0.0, end_time=1.0, text=text, confidence=0.95)]
