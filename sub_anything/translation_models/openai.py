"""OpenAI translation backend."""

from __future__ import annotations

import os
import time

import httpx

from ..models import TranscriptSegment
from .base import TranslationModel

REQUIRED_ENV = ("OPENAI_API_KEY",)


class OpenAITranslationModel(TranslationModel):
    MODEL_ID = "openai"
    DISPLAY_NAME = "OpenAI"
    REQUIRED_ENV = REQUIRED_ENV

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
    ):
        self.model = model
        self.batch_size = batch_size
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

            http_client = httpx.Client(timeout=120.0)
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

    def translate(
        self,
        segments: list[TranscriptSegment],
        target_language: str,
        *,
        verbose: bool = False,
    ) -> list[TranscriptSegment]:
        client = self._get_client()
        translated: list[TranscriptSegment] = []

        for batch_start in range(0, len(segments), self.batch_size):
            batch = segments[batch_start : batch_start + self.batch_size]
            translated.extend(self._translate_batch(client, batch, target_language, batch_start, verbose))

        return self._fix_overlaps(translated)

    def _translate_batch(
        self,
        client,
        batch: list[TranscriptSegment],
        target_language: str,
        batch_start: int,
        verbose: bool,
    ) -> list[TranscriptSegment]:
        texts = [f"{j + 1}. {seg.text}" for j, seg in enumerate(batch)]
        combined = "\n".join(texts)

        prompt = f"""Translate the following numbered lines to {target_language}.
Keep the numbering format exactly. Only output the translations, one per line.
Do not add extra commentary. Do not repeat lines.
Maintain the same tone and style.

{combined}"""

        try:
            result = self._call_with_retry(client, prompt, verbose)
            if not result:
                return batch

            translated_lines = result.strip().split("\n")
            out: list[TranscriptSegment] = []

            for j, seg in enumerate(batch):
                trans_text = self._extract_translation(translated_lines, j, seg.text)

                if verbose:
                    self._check_repetition(trans_text, batch_start + j + 1)

                new_duration = self._calculate_duration(seg, trans_text)

                out.append(
                    TranscriptSegment(
                        start_time=seg.start_time,
                        end_time=seg.start_time + new_duration,
                        text=trans_text,
                        confidence=seg.confidence,
                        speaker=seg.speaker,
                        language=target_language,
                    )
                )

            return out
        except Exception as e:
            if verbose:
                print(f"  Translation error: {e}, keeping original")
            return batch

    def _call_with_retry(self, client, prompt: str, verbose: bool, max_retries: int = 3) -> str:
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                return self._call_openai(client, prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 5
                    if verbose:
                        print(f"  OpenAI error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        raise last_error  # pragma: no cover

    def _call_openai(self, client, prompt: str) -> str:
        # Prefer Responses API when available (newer OpenAI SDKs), fall back to Chat Completions.
        responses = getattr(client, "responses", None)
        if responses is not None and hasattr(responses, "create"):
            try:
                resp = responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=0.0,
                )
            except Exception as e:
                if self._is_unsupported_temperature_error(e):
                    resp = responses.create(
                        model=self.model,
                        input=prompt,
                    )
                else:
                    raise
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text

        try:
            return self._call_chat_completions(client, prompt, temperature=0.0)
        except Exception as e:
            if self._is_unsupported_temperature_error(e):
                return self._call_chat_completions(client, prompt, temperature=None)
            if self._is_non_chat_model_error(e):
                return self._call_completions(client, prompt)
            raise

    @staticmethod
    def _is_non_chat_model_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("not a chat model" in msg) or ("v1/chat/completions" in msg) or ("chat/completions" in msg)

    @staticmethod
    def _is_unsupported_temperature_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("temperature" in msg) and (
            ("unsupported_value" in msg)
            or ("unsupported value" in msg)
            or ("does not support 0.0" in msg)
            or ("only the default" in msg)
        )

    def _call_chat_completions(self, client, prompt: str, temperature: float | None) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def _call_completions(self, client, prompt: str) -> str:
        for use_temperature in (True, False):
            kwargs: dict = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 2048,
            }
            if use_temperature:
                kwargs["temperature"] = 0.0

            try:
                resp = client.completions.create(**kwargs)
                text = getattr(resp.choices[0], "text", None)
                if text is None and isinstance(resp, dict):
                    text = resp.get("choices", [{}])[0].get("text")
                return (text or "").strip()
            except Exception as e:
                if use_temperature and self._is_unsupported_temperature_error(e):
                    continue
                raise

        raise RuntimeError("Failed to call v1/completions")  # pragma: no cover

    @staticmethod
    def _extract_translation(translated_lines: list[str], index: int, fallback: str) -> str:
        idx = index + 1
        prefixes = (f"{idx}.", f"{idx}:", f"{idx})", f"{idx} -")

        for line in translated_lines:
            stripped = line.lstrip()
            match_prefix = next((p for p in prefixes if stripped.startswith(p)), None)
            if match_prefix:
                value = stripped[len(match_prefix) :].strip()
                return value if value else fallback

        return fallback

    @staticmethod
    def _check_repetition(text: str, segment_num: int):
        tokens = text.lower().split()
        if len(tokens) >= 40:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.25:
                print(
                    f"  Warning: translation for segment {segment_num} looks highly repetitive; "
                    "consider a different translate model or smaller batch size"
                )

    @staticmethod
    def _calculate_duration(seg: TranscriptSegment, trans_text: str) -> float:
        orig_len = len(seg.text)
        trans_len = len(trans_text)
        duration = seg.end_time - seg.start_time

        if orig_len > 0 and trans_len > orig_len:
            ratio = min(trans_len / orig_len, 1.5)
            return duration * ratio
        return duration

    @staticmethod
    def _fix_overlaps(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        fixed = list(segments)
        for i in range(1, len(fixed)):
            if fixed[i].start_time < fixed[i - 1].end_time:
                fixed[i - 1] = TranscriptSegment(
                    start_time=fixed[i - 1].start_time,
                    end_time=fixed[i].start_time - 0.01,
                    text=fixed[i - 1].text,
                    confidence=fixed[i - 1].confidence,
                    speaker=fixed[i - 1].speaker,
                    language=fixed[i - 1].language,
                )
        return fixed
