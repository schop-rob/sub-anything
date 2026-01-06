"""OpenAI Whisper ASR model."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx

from ..models import TranscriptSegment
from .base import ASRModel

REQUIRED_ENV = ("OPENAI_API_KEY",)


class OpenAIWhisperASR(ASRModel):
    MODEL_ID = "whisper"
    DISPLAY_NAME = "OpenAI Whisper"
    DESCRIPTION = "Good quality, uses OpenAI Whisper API (chunked for size limits)"
    COST_PER_MINUTE = 0.006
    REQUIRED_ENV = REQUIRED_ENV
    DEFAULT_CHUNK_DURATION_SECONDS = 600
    DEFAULT_CHUNK_OVERLAP_SECONDS = 10
    CAN_DIARIZE = False

    def __init__(self, *, model: str = "whisper-1"):
        self.model = model
        self._client = None

    @classmethod
    def check(cls) -> tuple[bool, str]:
        ok, msg = super().check()
        if not ok:
            return ok, msg
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "OpenAI library not installed. Run: pip install openai"
        return True, ""

    def _get_client(self):
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            organization = os.environ.get("OPENAI_ORGANIZATION") or os.environ.get("OPENAI_ORG")
            project = os.environ.get("OPENAI_PROJECT") or os.environ.get("OPENAI_PROJECT_ID")

            http_client = httpx.Client(timeout=300.0)
            client_kwargs: dict = {"api_key": api_key, "http_client": http_client}
            if base_url:
                client_kwargs["base_url"] = base_url
            if organization:
                client_kwargs["organization"] = organization
            if project:
                client_kwargs["project"] = project

            try:
                self._client = openai.OpenAI(**client_kwargs)
            except TypeError:
                client_kwargs.pop("project", None)
                self._client = openai.OpenAI(**client_kwargs)

        return self._client

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
            print("Warning: diarization is not supported by the OpenAI Whisper backend; ignoring --diarize.")

        client = self._get_client()

        kwargs: dict = {
            "model": self.model,
            "response_format": "verbose_json",
        }

        if language and language not in {"auto"}:
            kwargs["language"] = self._normalize_language_hint(language)

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                with open(audio_path, "rb") as f:
                    resp = client.audio.transcriptions.create(file=f, **kwargs)
                return self._parse_transcription(resp)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = (2**attempt) * 5
                    if verbose:
                        print(f"  OpenAI Whisper error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    if verbose:
                        print(f"  OpenAI Whisper error: {e}")
                    raise

        raise last_error  # pragma: no cover

    @staticmethod
    def _parse_transcription(resp) -> list[TranscriptSegment]:
        segments_data = getattr(resp, "segments", None)
        language = getattr(resp, "language", None)

        if segments_data is None and isinstance(resp, dict):
            segments_data = resp.get("segments")
            language = resp.get("language")

        if not segments_data:
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None) or ""
            text = str(text).strip()
            if not text:
                return []
            return [
                TranscriptSegment(
                    start_time=0.0,
                    end_time=1.0,
                    text=text,
                    confidence=0.9,
                    language=language,
                )
            ]

        out: list[TranscriptSegment] = []
        for seg in segments_data:
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
                    language=language,
                )
            )

        return out
