#!/usr/bin/env python3
"""
sub-anything: Transcribe any audio/video file to SRT subtitles.

Supports multiple backends:
  - chirp3: Google Chirp 3 (best quality, may have timestamp issues)
  - long: Google Long model (guaranteed timestamps, slightly less accurate)
  - whisperx: WhisperX via Replicate (excellent timestamps + diarization)

Usage:
    sub-anything video.mp4
    sub-anything audio.mp3 --model whisperx
    sub-anything interview.wav --diarize --model whisperx
    sub-anything lecture.mp4 --translate en --language zh
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
MAX_CHUNK_DURATION_SECONDS = 3600  # 1 hour chunks
CHUNK_OVERLAP_SECONDS = 10
MAX_LINE_LENGTH = 42

# Cost estimates (USD per minute, approximate)
COST_PER_MINUTE = {
    "chirp3": 0.016,
    "long": 0.016,
    "whisperx": 0.006,  # ~$0.0014/sec * 60 / 14 (avg run time ratio)
}

AVAILABLE_MODELS = ["chirp3", "long", "whisperx"]


@dataclass
class TranscriptSegment:
    """A single transcribed segment with timing."""
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None
    language: Optional[str] = None


@dataclass
class Config:
    """Persistent configuration."""
    gcs_bucket: str = ""
    project_id: str = ""

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in ["gcs_bucket", "project_id"]})
        return cls()

    def save(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump({"gcs_bucket": self.gcs_bucket, "project_id": self.project_id}, f, indent=2)

    def is_complete(self) -> bool:
        return bool(self.gcs_bucket and self.project_id)


def check_dependencies():
    """Check and import required dependencies based on usage."""
    missing = []

    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_audio(video_path: Path, output_path: Path, verbose: bool = False) -> Path:
    """Extract audio from video file as WAV (16kHz mono for best compatibility)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path)
    ]
    if not verbose:
        cmd.insert(1, "-loglevel")
        cmd.insert(2, "error")

    subprocess.run(cmd, check=True)
    return output_path


def chunk_audio(audio_path: Path, output_dir: Path, chunk_duration: int = MAX_CHUNK_DURATION_SECONDS,
                overlap: int = CHUNK_OVERLAP_SECONDS, verbose: bool = False) -> list[tuple[Path, float]]:
    """Split audio into chunks with overlap. Returns list of (chunk_path, start_offset_seconds)."""
    total_duration = get_audio_duration(audio_path)

    if total_duration <= chunk_duration:
        return [(audio_path, 0.0)]

    chunks = []
    start = 0.0
    chunk_idx = 0

    while start < total_duration:
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.wav"
        duration = min(chunk_duration, total_duration - start)

        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-t", str(duration),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(chunk_path)
        ]
        if not verbose:
            cmd.insert(1, "-loglevel")
            cmd.insert(2, "error")

        subprocess.run(cmd, check=True)
        chunks.append((chunk_path, start))

        start += chunk_duration - overlap
        chunk_idx += 1

    return chunks


# ============================================================================
# Google Cloud Speech-to-Text Backend
# ============================================================================

def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> str:
    """Upload file to GCS and return gs:// URI."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


def delete_from_gcs(bucket_name: str, blob_name: str):
    """Delete file from GCS."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        blob.delete()
    except Exception:
        pass


def transcribe_google(
    gcs_uri: str,
    project_id: str,
    model: str,  # "chirp_3" or "long"
    language_codes: list[str],
    enable_diarization: bool = False,
    verbose: bool = False
) -> list[TranscriptSegment]:
    """Transcribe using Google Cloud Speech-to-Text V2 API."""
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core import exceptions as google_exceptions

    client = SpeechClient()

    recognizer_id = f"sub-anything-{uuid.uuid4().hex[:8]}"
    parent = f"projects/{project_id}/locations/us-central1"
    recognizer_path = f"{parent}/recognizers/{recognizer_id}"

    # Build features - CRITICAL: enable_word_time_offsets for timestamps
    features = cloud_speech.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    if enable_diarization:
        features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
            min_speaker_count=1,
            max_speaker_count=6,
        )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDecodingConfig(),
        language_codes=language_codes,
        model=model,
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
        print(f"  Sending to Google ({model})...")

    # Retry logic with exponential backoff
    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            operation = client.batch_recognize(request=request)
            result = operation.result(timeout=600)
            break
        except google_exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 10
                if verbose:
                    print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                if verbose:
                    print(f"  Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    if result is None:
        return []

    # Parse results - extract word-level timestamps
    segments = []

    for file_result in result.results.values():
        transcript = file_result.transcript

        for result_item in transcript.results:
            if not result_item.alternatives:
                continue

            alt = result_item.alternatives[0]

            # Build segments from word-level timestamps when available
            if alt.words:
                # Group words into subtitle-sized segments (roughly 10 words or natural pauses)
                current_words = []
                current_start = None

                for word_info in alt.words:
                    word = word_info.word
                    start = word_info.start_offset.total_seconds()
                    end = word_info.end_offset.total_seconds()
                    speaker = getattr(word_info, 'speaker_label', None)

                    if current_start is None:
                        current_start = start

                    current_words.append({
                        'word': word,
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })

                    # Create segment every ~10 words or on sentence-ending punctuation
                    if len(current_words) >= 10 or word.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
                        text = ' '.join(w['word'] for w in current_words)
                        segments.append(TranscriptSegment(
                            start_time=current_start,
                            end_time=current_words[-1]['end'],
                            text=text.strip(),
                            confidence=getattr(alt, 'confidence', 0.9),
                            speaker=current_words[0].get('speaker'),
                            language=getattr(result_item, 'language_code', None)
                        ))
                        current_words = []
                        current_start = None

                # Don't forget remaining words
                if current_words:
                    text = ' '.join(w['word'] for w in current_words)
                    segments.append(TranscriptSegment(
                        start_time=current_start,
                        end_time=current_words[-1]['end'],
                        text=text.strip(),
                        confidence=getattr(alt, 'confidence', 0.9),
                        speaker=current_words[0].get('speaker'),
                        language=getattr(result_item, 'language_code', None)
                    ))
            else:
                # Fallback: use segment-level timing
                text = alt.transcript.strip()
                if not text:
                    continue

                start_time = 0.0
                end_time = 0.0

                if hasattr(result_item, 'result_end_offset'):
                    end_time = result_item.result_end_offset.total_seconds()

                segments.append(TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    confidence=getattr(alt, 'confidence', 0.9),
                    language=getattr(result_item, 'language_code', None)
                ))

    return segments


# ============================================================================
# WhisperX via Replicate Backend
# ============================================================================

def transcribe_whisperx(
    audio_path: Path,
    language: Optional[str] = None,
    enable_diarization: bool = False,
    hf_token: Optional[str] = None,
    verbose: bool = False
) -> list[TranscriptSegment]:
    """Transcribe using WhisperX via Replicate API."""
    import replicate

    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable not set")

    if verbose:
        print("  Uploading to Replicate...")

    # Read and encode audio file
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Create a data URI for the audio
    audio_uri = f"data:audio/wav;base64,{base64.b64encode(audio_data).decode()}"

    input_params = {
        "audio_file": audio_uri,
        "align_output": True,  # Critical for word-level timestamps
        "batch_size": 64,
    }

    if language and language != "auto":
        input_params["language"] = language

    if enable_diarization:
        input_params["diarization"] = True
        if hf_token:
            input_params["huggingface_access_token"] = hf_token
        else:
            # Check for HF token in env
            hf_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_env:
                input_params["huggingface_access_token"] = hf_env
            else:
                print("Warning: Diarization requested but no HuggingFace token provided.")
                print("Set HF_TOKEN environment variable for speaker labels.")
                input_params["diarization"] = False

    if verbose:
        print("  Running WhisperX transcription...")

    # Run the model
    max_retries = 3
    output = None

    for attempt in range(max_retries):
        try:
            output = replicate.run(
                "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
                input=input_params
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

    # Parse WhisperX output
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
            confidence=0.95,  # WhisperX doesn't provide per-segment confidence
            speaker=speaker,
            language=detected_language
        ))

    return segments


# ============================================================================
# Common Processing Functions
# ============================================================================

def merge_chunk_segments(
    all_chunks: list[tuple[list[TranscriptSegment], float]],
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS
) -> list[TranscriptSegment]:
    """Merge segments from multiple chunks, handling overlaps."""
    if len(all_chunks) == 1:
        return all_chunks[0][0]

    merged = []

    for i, (segments, offset) in enumerate(all_chunks):
        # Adjust timing for chunk offset
        adjusted = []
        for seg in segments:
            adjusted.append(TranscriptSegment(
                start_time=seg.start_time + offset,
                end_time=seg.end_time + offset,
                text=seg.text,
                confidence=seg.confidence,
                speaker=seg.speaker,
                language=seg.language
            ))

        if i == 0:
            next_offset = all_chunks[i + 1][1] if i + 1 < len(all_chunks) else float('inf')
            cutoff = next_offset + (overlap_seconds / 2)
            for seg in adjusted:
                if seg.start_time < cutoff:
                    merged.append(seg)
        elif i == len(all_chunks) - 1:
            prev_end = offset + (overlap_seconds / 2)
            for seg in adjusted:
                if seg.start_time >= prev_end:
                    merged.append(seg)
        else:
            prev_end = offset + (overlap_seconds / 2)
            next_offset = all_chunks[i + 1][1]
            cutoff = next_offset + (overlap_seconds / 2)
            for seg in adjusted:
                if prev_end <= seg.start_time < cutoff:
                    merged.append(seg)

    merged.sort(key=lambda s: s.start_time)

    # Deduplicate
    deduplicated = []
    for seg in merged:
        if not deduplicated:
            deduplicated.append(seg)
        else:
            last = deduplicated[-1]
            if abs(seg.start_time - last.start_time) > 0.5 or seg.text != last.text:
                deduplicated.append(seg)

    return deduplicated


def wrap_text(text: str, max_length: int = MAX_LINE_LENGTH) -> str:
    """Wrap text to max_length characters per line."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + (1 if current_line else 0) <= max_length:
            current_line.append(word)
            current_length += word_length + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(
    segments: list[TranscriptSegment],
    mark_uncertain: bool = True,
    confidence_threshold: float = 0.7
) -> str:
    """Convert segments to SRT format."""
    srt_lines = []

    for i, seg in enumerate(segments, 1):
        text = seg.text

        if mark_uncertain and seg.confidence < confidence_threshold:
            text = f"[?] {text}"

        if seg.speaker:
            text = f"[{seg.speaker}] {text}"

        text = wrap_text(text)

        srt_lines.append(str(i))
        srt_lines.append(f"{format_srt_timestamp(seg.start_time)} --> {format_srt_timestamp(seg.end_time)}")
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def translate_segments(
    segments: list[TranscriptSegment],
    target_language: str,
    verbose: bool = False
) -> list[TranscriptSegment]:
    """Translate segments using OpenAI GPT with timing reflow."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)
    translated = []
    batch_size = 20

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        texts = [f"{j}. {seg.text}" for j, seg in enumerate(batch)]
        combined = "\n".join(texts)

        prompt = f"""Translate the following numbered lines to {target_language}.
Keep the numbering format exactly. Only output the translations, one per line.
Maintain the same tone and style.

{combined}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            translated_lines = result.split("\n")

            for j, seg in enumerate(batch):
                trans_text = seg.text
                for line in translated_lines:
                    if line.startswith(f"{j}."):
                        trans_text = line[len(f"{j}."):].strip()
                        break

                # Reflow timing
                orig_len = len(seg.text)
                trans_len = len(trans_text)
                duration = seg.end_time - seg.start_time

                if orig_len > 0 and trans_len > orig_len:
                    ratio = min(trans_len / orig_len, 1.5)
                    new_duration = duration * ratio
                else:
                    new_duration = duration

                translated.append(TranscriptSegment(
                    start_time=seg.start_time,
                    end_time=seg.start_time + new_duration,
                    text=trans_text,
                    confidence=seg.confidence,
                    speaker=seg.speaker,
                    language=target_language
                ))

                if verbose and seg.language:
                    print(f"  [{seg.language}] -> [{target_language}]")

        except Exception as e:
            if verbose:
                print(f"  Translation error: {e}, keeping original")
            translated.extend(batch)

    # Fix overlapping timestamps
    for i in range(1, len(translated)):
        if translated[i].start_time < translated[i-1].end_time:
            translated[i-1] = TranscriptSegment(
                start_time=translated[i-1].start_time,
                end_time=translated[i].start_time - 0.01,
                text=translated[i-1].text,
                confidence=translated[i-1].confidence,
                speaker=translated[i-1].speaker,
                language=translated[i-1].language
            )

    return translated


def mux_subtitles(video_path: Path, srt_path: Path, output_path: Path, verbose: bool = False):
    """Add subtitle track to video file (soft subs)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(srt_path),
        "-c", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        str(output_path)
    ]
    if not verbose:
        cmd.insert(1, "-loglevel")
        cmd.insert(2, "error")

    subprocess.run(cmd, check=True)


def prompt_for_config() -> Config:
    """Interactive prompt for missing configuration."""
    config = Config.load()

    if not config.project_id:
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and Path(creds_path).exists():
            with open(creds_path) as f:
                creds = json.load(f)
                config.project_id = creds.get("project_id", "")

        if not config.project_id:
            config.project_id = input("Enter your Google Cloud Project ID: ").strip()

    if not config.gcs_bucket:
        print("\nA Google Cloud Storage bucket is required for temporary audio uploads.")
        print("The audio will be uploaded, transcribed, then deleted.")
        config.gcs_bucket = input("Enter your GCS bucket name: ").strip()

    config.save()
    print(f"Configuration saved to {CONFIG_FILE}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video to SRT subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  chirp3   - Google Chirp 3: Best quality, 70+ languages, may have timestamp gaps
  long     - Google Long: Guaranteed timestamps, slightly less accurate
  whisperx - WhisperX (Replicate): Excellent timestamps, fast, good diarization

Examples:
  sub-anything video.mp4                          # Default (chirp3)
  sub-anything audio.mp3 --model whisperx         # Use WhisperX
  sub-anything interview.wav --model whisperx --diarize  # With speakers
  sub-anything lecture.mp4 --translate es         # Translate to Spanish
  sub-anything movie.mkv --mux                    # Embed subtitles in video

Environment variables:
  GOOGLE_APPLICATION_CREDENTIALS  - Required for chirp3/long models
  REPLICATE_API_TOKEN             - Required for whisperx model
  OPENAI_API_KEY                  - Required for --translate
  HF_TOKEN                        - Required for --diarize with whisperx
        """
    )

    parser.add_argument("input", type=Path, help="Input audio or video file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="chirp3",
                        help="Transcription model (default: chirp3)")
    parser.add_argument("--translate", metavar="LANG",
                        help="Translate subtitles to language (e.g., 'en', 'es', 'zh')")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--language", default="auto",
                        help="Source language hint (default: auto-detect)")
    parser.add_argument("--mux", action="store_true",
                        help="Embed subtitles into video file (soft subs)")

    args = parser.parse_args()

    # Check basic dependencies
    check_dependencies()
    from tqdm import tqdm

    # Validate input file
    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    input_ext = args.input.suffix.lower()
    is_video = input_ext in SUPPORTED_VIDEO_EXTENSIONS
    is_audio = input_ext in SUPPORTED_AUDIO_EXTENSIONS

    if not is_video and not is_audio:
        print(f"Error: Unsupported file format: {input_ext}")
        print(f"Supported video: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}")
        print(f"Supported audio: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
        sys.exit(1)

    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg not found in PATH")
        print("Install ffmpeg:")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        print("  Windows: choco install ffmpeg")
        sys.exit(1)

    # Check model-specific requirements
    use_google = args.model in ["chirp3", "long"]
    use_replicate = args.model == "whisperx"

    if use_google:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            print("Set it to the path of your Google Cloud service account JSON key")
            sys.exit(1)

        # Check Google dependencies
        try:
            from google.cloud import storage
            from google.cloud.speech_v2 import SpeechClient
        except ImportError:
            print("Error: Google Cloud libraries not installed")
            print("Run: pip install google-cloud-speech google-cloud-storage")
            sys.exit(1)

    if use_replicate:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Error: REPLICATE_API_TOKEN environment variable not set")
            sys.exit(1)

        try:
            import replicate
        except ImportError:
            print("Error: Replicate library not installed")
            print("Run: pip install replicate")
            sys.exit(1)

    if args.translate and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for translation")
        sys.exit(1)

    # Load config for Google models
    config = None
    if use_google:
        config = Config.load()
        if not config.is_complete():
            config = prompt_for_config()

    # Language setup
    language_codes = ["auto"] if args.language == "auto" else [args.language]

    # Output paths
    output_srt = args.input.with_suffix(".srt")
    output_video = None
    if args.mux and is_video:
        output_video = args.input.with_stem(args.input.stem + "_subtitled")

    print(f"Processing: {args.input.name}")
    print(f"Model: {args.model}")

    total_audio_minutes = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Extract audio if video
        if is_video:
            print("Extracting audio...")
            audio_path = temp_path / "audio.wav"
            extract_audio(args.input, audio_path, verbose=args.verbose)
        else:
            # For audio files, convert to WAV for consistency
            if input_ext != ".wav":
                print("Converting audio format...")
                audio_path = temp_path / "audio.wav"
                extract_audio(args.input, audio_path, verbose=args.verbose)
            else:
                audio_path = args.input

        # Get duration
        duration = get_audio_duration(audio_path)
        total_audio_minutes = duration / 60

        if args.verbose:
            print(f"Audio duration: {duration:.1f}s ({total_audio_minutes:.1f} min)")

        # Step 2: Chunk if necessary (for long files)
        needs_chunking = duration > MAX_CHUNK_DURATION_SECONDS

        if needs_chunking:
            print(f"Splitting into chunks ({int(duration // MAX_CHUNK_DURATION_SECONDS) + 1} chunks)...")
            chunks = chunk_audio(audio_path, temp_path, verbose=args.verbose)
        else:
            chunks = [(audio_path, 0.0)]

        # Step 3: Transcribe each chunk
        all_chunk_results = []
        chunk_iter = tqdm(chunks, desc="Transcribing", disable=not sys.stdout.isatty())

        for chunk_path, offset in chunk_iter:
            if use_google:
                # Upload to GCS
                blob_name = f"sub-anything-temp/{uuid.uuid4().hex}/{chunk_path.name}"

                if args.verbose:
                    print(f"  Uploading {chunk_path.name} to GCS...")

                gcs_uri = upload_to_gcs(chunk_path, config.gcs_bucket, blob_name)

                try:
                    google_model = "chirp_3" if args.model == "chirp3" else "long"
                    segments = transcribe_google(
                        gcs_uri=gcs_uri,
                        project_id=config.project_id,
                        model=google_model,
                        language_codes=language_codes,
                        enable_diarization=args.diarize,
                        verbose=args.verbose
                    )
                finally:
                    if args.verbose:
                        print("  Cleaning up GCS...")
                    delete_from_gcs(config.gcs_bucket, blob_name)

            elif use_replicate:
                segments = transcribe_whisperx(
                    audio_path=chunk_path,
                    language=args.language if args.language != "auto" else None,
                    enable_diarization=args.diarize,
                    verbose=args.verbose
                )

            all_chunk_results.append((segments, offset))

        # Step 4: Merge chunks
        if len(all_chunk_results) > 1:
            print("Merging chunks...")
            segments = merge_chunk_segments(all_chunk_results, CHUNK_OVERLAP_SECONDS)
        else:
            segments = all_chunk_results[0][0] if all_chunk_results else []

        if not segments:
            print("Warning: No speech detected in audio")
            sys.exit(0)

        print(f"Found {len(segments)} subtitle segments")

        # Step 5: Translate if requested
        if args.translate:
            print(f"Translating to {args.translate}...")
            segments = translate_segments(segments, args.translate, verbose=args.verbose)

        # Step 6: Generate SRT
        srt_content = segments_to_srt(segments)

        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"Saved: {output_srt}")

        # Step 7: Mux if requested
        if args.mux and is_video:
            print("Embedding subtitles into video...")
            mux_subtitles(args.input, output_srt, output_video, verbose=args.verbose)
            print(f"Saved: {output_video}")

    # Cost estimate
    cost_per_min = COST_PER_MINUTE.get(args.model, 0.01)
    estimated_cost = total_audio_minutes * cost_per_min
    print(f"\nEstimated cost: ${estimated_cost:.3f} ({total_audio_minutes:.1f} min @ ${cost_per_min}/min)")


if __name__ == "__main__":
    main()
