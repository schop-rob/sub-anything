"""Shared implementation for Google Speech-to-Text V2 models."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from ..models import TranscriptSegment
from .base import ASRModel


class GoogleSpeechV2ASRBase(ASRModel):
    """Google Cloud Speech-to-Text V2 (BatchRecognize) backend."""

    GOOGLE_MODEL: str = ""
    DEFAULT_LOCATION: str = "eu"
    COST_PER_MINUTE: float = 0.016

    def __init__(
        self,
        *,
        project_id: str,
        gcs_bucket: str,
        location: str,
    ):
        self.project_id = project_id
        self.gcs_bucket = gcs_bucket
        self.location = location

        # Import Google libraries lazily (so `-h` works without them).
        from google.cloud import storage
        from google.cloud.speech_v2 import SpeechClient
        from google.api_core.client_options import ClientOptions

        self._storage_client = storage.Client()
        self._speech_client = SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
        )

    @classmethod
    def check(cls) -> tuple[bool, str]:
        ok, msg = super().check()
        if not ok:
            return ok, msg
        try:
            from google.cloud import storage  # noqa: F401
            from google.cloud.speech_v2 import SpeechClient  # noqa: F401
        except ImportError:
            return (
                False,
                "Google Cloud libraries not installed. Run: pip install google-cloud-speech google-cloud-storage",
            )
        return True, ""

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        # Normalize language code
        language_codes = ["auto"] if language in (None, "auto") else [self._normalize_language_code(language)]

        # Calculate timeout based on audio duration
        if audio_duration:
            timeout = max(600, int(600 + audio_duration))
        else:
            timeout = 1800

        # Upload to GCS
        blob_name = f"sub-anything-temp/{uuid.uuid4().hex}/{audio_path.name}"
        if verbose:
            print(f"  Uploading {audio_path.name} to GCS...")

        gcs_uri = self._upload_to_gcs(audio_path, blob_name)

        try:
            return self._transcribe_gcs(gcs_uri, language_codes, enable_diarization, verbose, timeout)
        finally:
            if verbose:
                print("  Cleaning up GCS...")
            self._delete_from_gcs(blob_name)

    def _upload_to_gcs(self, local_path: Path, blob_name: str) -> str:
        """Upload file to GCS and return gs:// URI."""
        from google.api_core import exceptions as google_exceptions

        bucket = self._storage_client.bucket(self.gcs_bucket)
        if not bucket.exists():
            raise google_exceptions.NotFound(f"Bucket '{self.gcs_bucket}' does not exist.")

        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.gcs_bucket}/{blob_name}"

    def _delete_from_gcs(self, blob_name: str):
        """Delete file from GCS."""
        try:
            bucket = self._storage_client.bucket(self.gcs_bucket)
            blob = bucket.blob(blob_name)
            blob.delete()
        except Exception:
            pass

    def _transcribe_gcs(
        self,
        gcs_uri: str,
        language_codes: list[str],
        enable_diarization: bool,
        verbose: bool,
        timeout: int = 1800,
    ) -> list[TranscriptSegment]:
        """Transcribe from GCS URI."""
        from google.cloud.speech_v2.types import cloud_speech
        from google.api_core import exceptions as google_exceptions

        parent = f"projects/{self.project_id}/locations/{self.location}"
        recognizer_path = f"{parent}/recognizers/_"

        features = cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        if enable_diarization:
            features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=1,
                max_speaker_count=6,
            )

        language_attempts = [language_codes]
        if self.GOOGLE_MODEL != "chirp_3" and language_codes != ["en-US"]:
            language_attempts.append(["en-US"])

        for lang_codes in language_attempts:
            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=lang_codes,
                model=self.GOOGLE_MODEL,
                features=features,
            )

            request = cloud_speech.BatchRecognizeRequest(
                recognizer=recognizer_path,
                config=config,
                files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
                recognition_output_config=cloud_speech.RecognitionOutputConfig(
                    inline_response_config=cloud_speech.InlineOutputConfig()
                ),
            )

            if verbose:
                print(f"  Sending to Google ({self.GOOGLE_MODEL}) with language: {lang_codes}...")
                print(f"  Waiting for transcription (timeout: {timeout // 60}m {timeout % 60}s)...")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    operation = self._speech_client.batch_recognize(request=request)
                    result = self._wait_for_operation(operation, timeout=timeout, verbose=verbose)
                    return self._parse_result(result, verbose)
                except google_exceptions.ResourceExhausted:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 10
                        if verbose:
                            print(f"  Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                except google_exceptions.InvalidArgument as e:
                    error_msg = str(e).lower()
                    if "language" in error_msg and "not supported" in error_msg:
                        if verbose:
                            print(f"  Language {lang_codes} not supported, trying fallback...")
                        break
                    raise
                except Exception as e:
                    if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                        if verbose:
                            print(
                                "  Operation timed out. The transcription may still be processing on Google's servers."
                            )
                        raise
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        if verbose:
                            print(f"  Error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        return []

    def _wait_for_operation(self, operation, timeout: int = 600, verbose: bool = False):
        """Wait for operation with progress feedback."""
        import threading

        result_container = {"result": None, "error": None, "done": False}

        def wait_for_result():
            try:
                result_container["result"] = operation.result(timeout=timeout)
            except Exception as e:
                result_container["error"] = e
            finally:
                result_container["done"] = True

        wait_thread = threading.Thread(target=wait_for_result, daemon=True)
        wait_thread.start()

        start_time = time.time()
        last_status_time = start_time
        status_interval = 30

        while not result_container["done"]:
            elapsed = time.time() - start_time

            if verbose and (time.time() - last_status_time) >= status_interval:
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                print(f"  Still processing... ({mins}m {secs}s elapsed)")
                last_status_time = time.time()

            time.sleep(1)

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]

    def _parse_result(self, result, verbose: bool) -> list[TranscriptSegment]:
        """Parse Google Speech API result into segments."""
        segments: list[TranscriptSegment] = []
        last_result_end_time = 0.0

        for file_result in result.results.values():
            transcript = file_result.transcript

            for result_item in transcript.results:
                if not result_item.alternatives:
                    continue

                alt = result_item.alternatives[0]
                confidence = getattr(alt, "confidence", 0.9)
                language = getattr(result_item, "language_code", None)

                result_end_time = None
                if hasattr(result_item, "result_end_offset"):
                    result_end_time = result_item.result_end_offset.total_seconds()

                if alt.words:
                    new_segments, last_result_end_time = self._process_words(
                        alt.words,
                        confidence,
                        language,
                        result_end_time,
                        last_result_end_time,
                        verbose,
                    )
                    segments.extend(new_segments)
                else:
                    text = alt.transcript.strip()
                    if not text:
                        continue

                    start_time = last_result_end_time
                    end_time = result_end_time if result_end_time else start_time + max(1.0, len(text) / 15.0)

                    segments.append(
                        TranscriptSegment(
                            start_time=start_time,
                            end_time=end_time,
                            text=text,
                            confidence=confidence,
                            language=language,
                        )
                    )
                    last_result_end_time = max(last_result_end_time, end_time)

        return segments

    def _process_words(
        self,
        word_infos,
        confidence: float,
        language: Optional[str],
        result_end_time: Optional[float],
        last_result_end_time: float,
        verbose: bool,
    ) -> tuple[list[TranscriptSegment], float]:
        """Process word-level timestamps into segments."""
        words = []
        valid_timing_count = 0
        min_start = None
        max_end = None

        for word_info in word_infos:
            word = word_info.word
            start = word_info.start_offset.total_seconds()
            end = word_info.end_offset.total_seconds()
            speaker = getattr(word_info, "speaker_label", None) or getattr(word_info, "speaker_tag", None)

            if end > start:
                valid_timing_count += 1
                min_start = start if min_start is None else min(min_start, start)
                max_end = end if max_end is None else max(max_end, end)

            words.append({"word": word, "start": start, "end": end, "speaker": speaker})

        timing_span = (max_end - min_start) if (min_start is not None and max_end is not None) else 0.0
        unique_end_times = {round(w["end"], 3) for w in words}

        timing_collapsed = (
            len(words) >= 5
            and (
                valid_timing_count < max(2, int(len(words) * 0.3))
                or timing_span < 0.1
                or (len(unique_end_times) <= 2 and len(words) >= 20)
            )
        )

        if result_end_time is not None and max_end is not None and max_end < result_end_time - 1.0:
            timing_collapsed = True

        if timing_collapsed and result_end_time is not None:
            if verbose:
                print("  Warning: word timestamps missing/collapsed; interpolating timings")

            window_start = last_result_end_time
            window_end = max(result_end_time, window_start + 0.01)
            duration = window_end - window_start
            per_word = duration / max(len(words), 1)

            for idx, w in enumerate(words):
                w["start"] = window_start + (idx * per_word)
                w["end"] = window_start + ((idx + 1) * per_word)

            segments = self._group_words_into_segments(words, confidence, language)
            return segments, window_end

        segments = self._group_words_into_segments(words, confidence, language)
        new_last = last_result_end_time
        if max_end is not None:
            new_last = max(new_last, max_end)
        if result_end_time is not None:
            new_last = max(new_last, result_end_time)
        return segments, new_last

    def _group_words_into_segments(
        self,
        words: list[dict],
        confidence: float,
        language: Optional[str],
    ) -> list[TranscriptSegment]:
        """Group words into subtitle segments."""
        segments: list[TranscriptSegment] = []
        current_words = []
        current_start = None

        for word in words:
            if current_start is None:
                current_start = word["start"]

            current_words.append(word)

            if len(current_words) >= 10 or word["word"].rstrip().endswith((".", "!", "?", "。", "！", "？")):
                text = " ".join(w["word"] for w in current_words).strip()
                if text:
                    segments.append(
                        TranscriptSegment(
                            start_time=current_start,
                            end_time=current_words[-1]["end"],
                            text=text,
                            confidence=confidence,
                            speaker=current_words[0].get("speaker"),
                            language=language,
                        )
                    )
                current_words = []
                current_start = None

        if current_words:
            text = " ".join(w["word"] for w in current_words).strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start_time=current_start if current_start is not None else current_words[0]["start"],
                        end_time=current_words[-1]["end"],
                        text=text,
                        confidence=confidence,
                        speaker=current_words[0].get("speaker"),
                        language=language,
                    )
                )

        return segments

    @staticmethod
    def _normalize_language_code(language: str) -> str:
        """Normalize common language shorthands to Google Speech-to-Text language codes."""
        lang = language.strip()
        lower = lang.lower().replace("_", "-")

        if lower in {"zh", "zh-cn", "zh-hans", "zh-hans-cn", "cn"}:
            return "cmn-Hans-CN"
        if lower in {"zh-tw", "zh-hant", "zh-hant-tw", "tw"}:
            return "cmn-Hant-TW"
        if lower in {"zh-hk", "yue", "yue-hk"}:
            return "yue-Hant-HK"

        return lang

