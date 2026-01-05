"""Translation service using OpenAI GPT."""

import os
import time
from typing import Optional

import httpx

from .models import TranscriptSegment


class TranslatorService:
    """Translate transcript segments using OpenAI GPT."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 20,
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = None

    @classmethod
    def check_requirements(cls) -> tuple[bool, str]:
        """Check if requirements are met."""
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY environment variable not set"
        try:
            import openai
        except ImportError:
            return False, "OpenAI library not installed. Run: pip install openai"
        return True, ""

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            organization = os.environ.get("OPENAI_ORGANIZATION") or os.environ.get("OPENAI_ORG")
            project = os.environ.get("OPENAI_PROJECT") or os.environ.get("OPENAI_PROJECT_ID")

            # Work around openai<->httpx incompatibilities
            http_client = httpx.Client(timeout=120.0)
            client_kwargs = {"api_key": api_key, "http_client": http_client}

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
        verbose: bool = False,
    ) -> list[TranscriptSegment]:
        """Translate segments to target language."""
        client = self._get_client()
        translated = []

        for batch_start in range(0, len(segments), self.batch_size):
            batch = segments[batch_start:batch_start + self.batch_size]
            batch_translated = self._translate_batch(client, batch, target_language, batch_start, verbose)
            translated.extend(batch_translated)

        # Fix overlapping timestamps
        translated = self._fix_overlaps(translated)

        return translated

    def _translate_batch(
        self,
        client,
        batch: list[TranscriptSegment],
        target_language: str,
        batch_start: int,
        verbose: bool,
    ) -> list[TranscriptSegment]:
        """Translate a batch of segments."""
        texts = [f"{j + 1}. {seg.text}" for j, seg in enumerate(batch)]
        combined = "\n".join(texts)

        prompt = f"""Translate the following numbered lines to {target_language}.
Keep the numbering format exactly. Only output the translations, one per line.
Maintain the same tone and style.

{combined}"""

        try:
            response = self._call_with_retry(client, prompt, verbose)
            if response is None:
                return batch

            result = response.choices[0].message.content.strip()
            translated_lines = result.split("\n")

            translated = []
            for j, seg in enumerate(batch):
                trans_text = self._extract_translation(translated_lines, j, seg.text)

                if verbose:
                    self._check_repetition(trans_text, batch_start + j + 1)

                # Reflow timing based on text length change
                new_duration = self._calculate_duration(seg, trans_text)

                translated.append(
                    TranscriptSegment(
                        start_time=seg.start_time,
                        end_time=seg.start_time + new_duration,
                        text=trans_text,
                        confidence=seg.confidence,
                        speaker=seg.speaker,
                        language=target_language,
                    )
                )

                if verbose and seg.language:
                    print(f"  [{seg.language}] -> [{target_language}]")

            return translated

        except Exception as e:
            if verbose:
                print(f"  Translation error: {e}, keeping original")
            return batch

    def _call_with_retry(self, client, prompt: str, verbose: bool, max_retries: int = 3):
        """Call OpenAI API with retry logic."""
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    if verbose:
                        print(f"  OpenAI error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        return None

    @staticmethod
    def _extract_translation(translated_lines: list[str], index: int, fallback: str) -> str:
        """Extract translation for a specific index from response lines."""
        idx = index + 1
        prefixes = (f"{idx}.", f"{idx}:", f"{idx})", f"{idx} -")

        for line in translated_lines:
            stripped = line.lstrip()
            match_prefix = next((p for p in prefixes if stripped.startswith(p)), None)
            if match_prefix:
                return stripped[len(match_prefix):].strip()

        return fallback

    @staticmethod
    def _check_repetition(text: str, segment_num: int):
        """Warn about highly repetitive translations."""
        tokens = text.lower().split()
        if len(tokens) >= 40:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.25:
                print(
                    f"  Warning: translation for segment {segment_num} looks highly repetitive; "
                    "consider using a different translate model or smaller batch size"
                )

    @staticmethod
    def _calculate_duration(seg: TranscriptSegment, trans_text: str) -> float:
        """Calculate new duration based on text length change."""
        orig_len = len(seg.text)
        trans_len = len(trans_text)
        duration = seg.end_time - seg.start_time

        if orig_len > 0 and trans_len > orig_len:
            ratio = min(trans_len / orig_len, 1.5)
            return duration * ratio
        return duration

    @staticmethod
    def _fix_overlaps(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Fix overlapping timestamps between segments."""
        fixed = list(segments)
        for i in range(1, len(fixed)):
            if fixed[i].start_time < fixed[i-1].end_time:
                fixed[i-1] = TranscriptSegment(
                    start_time=fixed[i-1].start_time,
                    end_time=fixed[i].start_time - 0.01,
                    text=fixed[i-1].text,
                    confidence=fixed[i-1].confidence,
                    speaker=fixed[i-1].speaker,
                    language=fixed[i-1].language
                )
        return fixed
