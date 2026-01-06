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
import json
import os
import sys
import tempfile
from pathlib import Path

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.json"

AVAILABLE_MODELS = ["chirp3", "long", "whisperx"]

# Cost estimates (USD per minute, approximate)
COST_PER_MINUTE = {
    "chirp3": 0.016,
    "long": 0.016,
    "whisperx": 0.006,
}


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


def bucket_exists(bucket_name: str) -> bool:
    """Check if a GCS bucket exists."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.exists()


def create_bucket(bucket_name: str, location: str = "us-central1") -> bool:
    """Create a GCS bucket. Returns True if successful."""
    from google.cloud import storage
    client = storage.Client()
    try:
        bucket = client.create_bucket(bucket_name, location=location)
        print(f"Created bucket: {bucket.name}")
        return True
    except Exception as e:
        print(f"Failed to create bucket: {e}")
        return False


def prompt_for_config(config):
    """Interactive prompt for missing configuration."""
    from sub_anything.models import Config

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

    # Verify bucket exists or offer to create it
    print(f"Checking bucket '{config.gcs_bucket}'...")
    if not bucket_exists(config.gcs_bucket):
        print(f"Bucket '{config.gcs_bucket}' does not exist.")
        create = input("Create it now? [Y/n]: ").strip().lower()
        if create in ("", "y", "yes"):
            if not create_bucket(config.gcs_bucket):
                print("Could not create bucket. Please create it manually in Google Cloud Console.")
                print("https://console.cloud.google.com/storage/browser")
                sys.exit(1)
        else:
            print("Please create the bucket manually or run again with a different bucket name.")
            sys.exit(1)
    else:
        print(f"Bucket '{config.gcs_bucket}' verified.")

    config.save(CONFIG_FILE)
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
  sub-anything                                   # Interactive wizard (TTY only)
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
    parser.add_argument("--google-location", metavar="LOC",
                        help="Google Speech-to-Text location for chirp3/long (e.g., us, eu, asia-northeast1). "
                             "Defaults: chirp3=eu, long=us-central1. When used with chirp3, it is saved to config.json.")
    parser.add_argument("--translate", metavar="LANG",
                        help="Translate subtitles to language (e.g., 'en', 'es', 'zh')")
    parser.add_argument("--translate-model", default="gpt-4o-mini",
                        help="OpenAI model for --translate (default: gpt-4o-mini)")
    parser.add_argument("--translate-batch-size", type=int, default=20,
                        help="Subtitle segments per translation request (default: 20)")
    parser.add_argument("--save-original", action="store_true",
                        help="When using --translate, also save the original transcript as *.orig.srt/.orig.txt")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--language", default="auto",
                        help="Source language hint (default: auto-detect)")
    parser.add_argument("--mux", action="store_true",
                        help="Embed subtitles into video file (soft subs)")
    parser.add_argument("--no-timestamps", action="store_true",
                        help="Output plain text (.txt) instead of SRT subtitles (.srt)")

    # If invoked without args, run the interactive wizard instead of showing an argparse error.
    if len(sys.argv) == 1:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            parser.print_help()
            sys.exit(2)
        from sub_anything.wizard import run_wizard
        args = run_wizard(config_file=CONFIG_FILE, available_models=AVAILABLE_MODELS)
    else:
        args = parser.parse_args()

    # Check basic dependencies
    check_dependencies()

    # Now import our modules (after checking tqdm is available)
    from tqdm import tqdm
    from sub_anything.models import Config
    from sub_anything.providers import GoogleASRProvider, WhisperXProvider
    from sub_anything.translator import TranslatorService
    from sub_anything.utils import (
        SUPPORTED_VIDEO_EXTENSIONS,
        SUPPORTED_AUDIO_EXTENSIONS,
        MAX_CHUNK_DURATION_SECONDS,
        CHUNK_OVERLAP_SECONDS,
        check_ffmpeg,
        get_audio_duration,
        extract_audio,
        chunk_audio,
        merge_chunk_segments,
        sanitize_segments,
        segments_to_srt,
        segments_to_text,
        mux_subtitles,
    )

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
        ok, error = GoogleASRProvider.check_requirements()
        if not ok:
            print(f"Error: {error}")
            sys.exit(1)

    if use_replicate:
        ok, error = WhisperXProvider.check_requirements()
        if not ok:
            print(f"Error: {error}")
            sys.exit(1)

    if args.translate:
        ok, error = TranslatorService.check_requirements()
        if not ok:
            print(f"Error: {error}")
            sys.exit(1)

    if args.translate and (args.translate_batch_size < 1 or args.translate_batch_size > 100):
        print("Error: --translate-batch-size must be between 1 and 100")
        sys.exit(1)

    if args.mux and args.no_timestamps:
        print("Error: --mux requires SRT output (remove --no-timestamps)")
        sys.exit(1)

    # Load config for Google models
    config = None
    provider = None

    if use_google:
        config = Config.load(CONFIG_FILE)
        if not config.is_complete():
            config = prompt_for_config(config)

        # Determine Google location
        if args.model == "chirp3":
            google_location = args.google_location or config.chirp_location or "eu"
            if args.google_location and args.google_location != config.chirp_location:
                config.chirp_location = args.google_location
                config.save(CONFIG_FILE)
        else:
            google_location = args.google_location or "us-central1"

        google_model = "chirp_3" if args.model == "chirp3" else "long"
        provider = GoogleASRProvider(
            project_id=config.project_id,
            gcs_bucket=config.gcs_bucket,
            model=google_model,
            location=google_location,
        )
    elif use_replicate:
        provider = WhisperXProvider()
        google_location = None

    # Output paths
    output_ext = ".txt" if args.no_timestamps else ".srt"
    output_transcript = args.input.with_suffix(output_ext)
    output_video = None
    if args.mux and is_video:
        output_video = args.input.with_stem(args.input.stem + "_subtitled")

    print(f"Processing: {args.input.name}")
    print(f"Model: {args.model}")
    if use_google and google_location:
        print(f"Google location: {google_location}")

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

        language = args.language if args.language != "auto" else None

        for chunk_path, offset in chunk_iter:
            # Calculate chunk duration for timeout
            chunk_duration = get_audio_duration(chunk_path) if chunk_path != audio_path else duration

            segments = provider.transcribe(
                audio_path=chunk_path,
                language=language,
                enable_diarization=args.diarize,
                verbose=args.verbose,
                audio_duration=chunk_duration,
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

        # Basic timestamp sanity checks
        if duration > 60 and len(segments) > 20:
            zero_duration = sum(1 for s in segments if s.end_time <= s.start_time + 1e-3)
            rounded_ends = {round(s.end_time, 3) for s in segments}
            max_end = max(s.end_time for s in segments)

            if zero_duration > 0 or (len(rounded_ends) <= 3 and max_end < duration * 0.2):
                print("Warning: Detected collapsed/invalid timestamps in output.")
                print("  Try rerunning with `--model long` (guaranteed timestamps) or `--model whisperx`.")

        segments = sanitize_segments(segments)

        print(f"Found {len(segments)} subtitle segments")

        # Step 5: Translate if requested
        if args.translate:
            if args.save_original:
                orig_path = output_transcript.with_suffix(f".orig{output_transcript.suffix}")
                original_content = segments_to_text(segments) if args.no_timestamps else segments_to_srt(segments)
                with open(orig_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                print(f"Saved: {orig_path}")

            print(f"Translating to {args.translate}...")
            translator = TranslatorService(
                model=args.translate_model,
                batch_size=args.translate_batch_size,
            )
            segments = translator.translate(
                segments,
                args.translate,
                verbose=args.verbose,
            )
            segments = sanitize_segments(segments)

        # Step 6: Generate SRT
        transcript_content = segments_to_text(segments) if args.no_timestamps else segments_to_srt(segments)

        with open(output_transcript, "w", encoding="utf-8") as f:
            f.write(transcript_content)

        print(f"Saved: {output_transcript}")

        # Step 7: Mux if requested
        if args.mux and is_video:
            print("Embedding subtitles into video...")
            mux_subtitles(args.input, output_transcript, output_video, verbose=args.verbose)
            print(f"Saved: {output_video}")

    # Cost estimate
    cost_per_min = COST_PER_MINUTE.get(args.model, 0.01)
    estimated_cost = total_audio_minutes * cost_per_min
    print(f"\nEstimated cost: ${estimated_cost:.3f} ({total_audio_minutes:.1f} min @ ${cost_per_min}/min)")


if __name__ == "__main__":
    main()
