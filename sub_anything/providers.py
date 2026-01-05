"""ASR (Automatic Speech Recognition) providers."""

import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import TranscriptSegment


class ASRProvider(ABC):
    """Abstract base class for ASR providers."""

    name: str = "base"
    cost_per_minute: float = 0.0

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        """Transcribe audio file and return segments."""
        pass

    @classmethod
    def check_requirements(cls) -> tuple[bool, str]:
        """Check if provider requirements are met. Returns (ok, error_message)."""
        return True, ""


class GoogleASRProvider(ASRProvider):
    """Google Cloud Speech-to-Text provider (Chirp 3 and Long models)."""

    name = "google"
    cost_per_minute = 0.016

    def __init__(
        self,
        project_id: str,
        gcs_bucket: str,
        model: str = "chirp_3",  # "chirp_3" or "long"
        location: str = "us",
    ):
        self.project_id = project_id
        self.gcs_bucket = gcs_bucket
        self.model = model
        self.location = location

        # Import Google libraries
        from google.cloud import storage
        from google.cloud.speech_v2 import SpeechClient
        from google.api_core.client_options import ClientOptions

        self._storage_client = storage.Client()
        self._speech_client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com",
            )
        )

    @classmethod
    def check_requirements(cls) -> tuple[bool, str]:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        try:
            from google.cloud import storage
            from google.cloud.speech_v2 import SpeechClient
        except ImportError:
            return False, "Google Cloud libraries not installed. Run: pip install google-cloud-speech google-cloud-storage"
        return True, ""

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

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        verbose: bool = False,
        audio_duration: Optional[float] = None,
    ) -> list[TranscriptSegment]:
        """Transcribe using Google Cloud Speech-to-Text V2 API."""
        from google.cloud.speech_v2.types import cloud_speech
        from google.api_core import exceptions as google_exceptions

        # Normalize language code
        language_codes = ["auto"] if language in (None, "auto") else [self._normalize_language_code(language)]

        # Calculate timeout based on audio duration
        # Base: 10 minutes, plus 1 minute per minute of audio (generous buffer)
        if audio_duration:
            timeout = max(600, int(600 + audio_duration))
        else:
            timeout = 1800  # 30 minutes default if duration unknown

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

        # Build features
        features = cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        if enable_diarization:
            features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=1,
                max_speaker_count=6,
            )

        # Language fallback chain for retries
        language_attempts = [language_codes]

        # For non-chirp models, add fallback options
        if self.model != "chirp_3":
            if language_codes != ["en-US"]:
                language_attempts.append(["en-US"])  # Common fallback

        for lang_codes in language_attempts:
            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=lang_codes,
                model=self.model,
                features=features,
            )

            request = cloud_speech.BatchRecognizeRequest(
                recognizer=recognizer_path,
                config=config,
                files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
                recognition_output_config=cloud_speech.RecognitionOutputConfig(
                    inline_response_config=cloud_speech.InlineOutputConfig()
                )
            )

            if verbose:
                print(f"  Sending to Google ({self.model}) with language: {lang_codes}...")
                print(f"  Waiting for transcription (timeout: {timeout // 60}m {timeout % 60}s)...")

            # Execute with retry logic
            max_retries = 3
            result = None

            for attempt in range(max_retries):
                try:
                    operation = self._speech_client.batch_recognize(request=request)

                    # Poll for status instead of blocking
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
                    # If language not supported, try next language in fallback chain
                    if "language" in error_msg and "not supported" in error_msg:
                        if verbose:
                            print(f"  Language {lang_codes} not supported, trying fallback...")
                        break  # Break inner loop to try next language
                    raise
                except Exception as e:
                    # Check if it's a timeout error
                    if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                        if verbose:
                            print("  Operation timed out. The transcription may still be processing on Google's servers.")
                        raise
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5
                        if verbose:
                            print(f"  Error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        # If we get here, all language attempts failed
        return []

    def _wait_for_operation(self, operation, timeout: int = 600, verbose: bool = False):
        """Wait for operation with progress feedback."""
        import threading

        result_container = {"result": None, "error": None, "done": False}

        def wait_for_result():
            try:
                # Use Google's built-in result() which handles polling correctly
                result_container["result"] = operation.result(timeout=timeout)
            except Exception as e:
                result_container["error"] = e
            finally:
                result_container["done"] = True

        # Start waiting in background thread
        wait_thread = threading.Thread(target=wait_for_result, daemon=True)
        wait_thread.start()

        # Print progress while waiting
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

        # Check result
        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]

    def _parse_result(self, result, verbose: bool) -> list[TranscriptSegment]:
        """Parse Google Speech API result into segments."""
        segments = []
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
                        alt.words, confidence, language, result_end_time, last_result_end_time, verbose
                    )
                    segments.extend(new_segments)
                else:
                    # Fallback: use segment-level timing
                    text = alt.transcript.strip()
                    if not text:
                        continue

                    start_time = last_result_end_time
                    end_time = result_end_time if result_end_time else start_time + max(1.0, len(text) / 15.0)

                    segments.append(TranscriptSegment(
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        confidence=confidence,
                        language=language
                    ))
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

            words.append({
                "word": word,
                "start": start,
                "end": end,
                "speaker": speaker,
            })

        # Check for collapsed timing
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

        if result_end_time is not None and max_end is not None:
            if max_end < result_end_time - 1.0:
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
        else:
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
        segments = []
        current_words = []
        current_start = None

        for word in words:
            if current_start is None:
                current_start = word["start"]

            current_words.append(word)

            # Create segment every ~10 words or on sentence-ending punctuation
            if len(current_words) >= 10 or word["word"].rstrip().endswith((".", "!", "?", "。", "！", "？")):
                text = " ".join(w["word"] for w in current_words)
                segments.append(
                    TranscriptSegment(
                        start_time=current_start,
                        end_time=current_words[-1]["end"],
                        text=text.strip(),
                        confidence=confidence,
                        speaker=current_words[0].get("speaker"),
                        language=language,
                    )
                )
                current_words = []
                current_start = None

        if current_words:
            text = " ".join(w["word"] for w in current_words)
            segments.append(
                TranscriptSegment(
                    start_time=current_start if current_start is not None else current_words[0]["start"],
                    end_time=current_words[-1]["end"],
                    text=text.strip(),
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


class WhisperXProvider(ASRProvider):
    """WhisperX via Replicate API provider."""

    name = "whisperx"
    cost_per_minute = 0.006

    def __init__(self):
        import replicate
        self._replicate = replicate

    @classmethod
    def check_requirements(cls) -> tuple[bool, str]:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            return False, "REPLICATE_API_TOKEN environment variable not set"
        try:
            import replicate
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
        """Transcribe using WhisperX via Replicate API."""
        if verbose:
            print("  Uploading to Replicate...")

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        with open(audio_path, "rb") as audio_file:
            input_params = {
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
                        wait_time = (2 ** attempt) * 5
                        if verbose:
                            print(f"  Error: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

        if output is None:
            return []

        return self._parse_output(output)

    def _parse_output(self, output: dict) -> list[TranscriptSegment]:
        """Parse WhisperX output into segments."""
        segments = []
        detected_language = output.get("detected_language")

        for seg in output.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker")

            if not text:
                continue

            segments.append(TranscriptSegment(
                start_time=start,
                end_time=end,
                text=text,
                confidence=0.95,
                speaker=speaker,
                language=detected_language
            ))

        return segments
