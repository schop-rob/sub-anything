"""Utility functions for audio processing and SRT formatting."""

import re
import shutil
import subprocess
from pathlib import Path

from .models import TranscriptSegment


# Constants
MAX_CHUNK_DURATION_SECONDS = 3600  # 1 hour chunks
CHUNK_OVERLAP_SECONDS = 10
MAX_LINE_LENGTH = 42

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


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


def chunk_audio(
    audio_path: Path,
    output_dir: Path,
    chunk_duration: int = MAX_CHUNK_DURATION_SECONDS,
    overlap: int = CHUNK_OVERLAP_SECONDS,
    verbose: bool = False
) -> list[tuple[Path, float]]:
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


def sanitize_segments(segments: list[TranscriptSegment], min_duration: float = 0.2) -> list[TranscriptSegment]:
    """Ensure segments have non-negative, non-zero timestamps."""
    sanitized = []
    for seg in segments:
        start = max(0.0, float(seg.start_time))
        end = float(seg.end_time)
        if end <= start:
            end = start + min_duration

        sanitized.append(
            TranscriptSegment(
                start_time=start,
                end_time=end,
                text=seg.text,
                confidence=seg.confidence,
                speaker=seg.speaker,
                language=seg.language,
            )
        )

    return sanitized


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


_SRT_TS_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$")


def parse_srt_timestamp(value: str) -> float:
    """Parse an SRT timestamp (HH:MM:SS,mmm) into seconds."""
    m = _SRT_TS_RE.match(value.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours = int(m.group("h"))
    minutes = int(m.group("m"))
    secs = int(m.group("s"))
    millis = int(m.group("ms"))
    return (hours * 3600) + (minutes * 60) + secs + (millis / 1000.0)


def original_transcript_candidates(output_transcript: Path) -> list[Path]:
    """Return possible *.orig transcript paths for a given output transcript path."""
    suffix = output_transcript.suffix.lower()
    candidates = [output_transcript.with_suffix(f".orig{suffix}")]
    if suffix == ".txt":
        candidates.append(output_transcript.with_suffix(".orig.srt"))
    return candidates


def find_existing_original_transcript(output_transcript: Path) -> Path | None:
    for candidate in original_transcript_candidates(output_transcript):
        if candidate.exists():
            return candidate
    return None


def load_segments_from_srt(path: Path) -> list[TranscriptSegment]:
    """Load segments from an SRT file (best-effort)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    segments: list[TranscriptSegment] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Optional numeric index line
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        parts = line.split("-->", 1)
        try:
            start = parse_srt_timestamp(parts[0].strip())
            end = parse_srt_timestamp(parts[1].strip())
        except ValueError:
            i += 1
            continue

        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        raw_text = " ".join(text_lines).strip()
        if not raw_text:
            continue

        confidence = 1.0
        speaker = None
        text = raw_text

        if text.startswith("[?]"):
            confidence = 0.0
            text = text[3:].strip()

        if text.startswith("[") and "]" in text:
            label = text[1:text.index("]")]
            if label and label != "?":
                speaker = label
                text = text[text.index("]") + 1 :].strip()

        segments.append(
            TranscriptSegment(
                start_time=float(start),
                end_time=float(end),
                text=text,
                confidence=confidence,
                speaker=speaker,
            )
        )

    return segments


def load_segments_from_text(path: Path) -> list[TranscriptSegment]:
    """Load segments from a plain text transcript (one segment per non-empty line)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    segments: list[TranscriptSegment] = []
    t = 0.0
    for raw in lines:
        if not raw:
            continue

        confidence = 1.0
        speaker = None
        text = raw

        if text.startswith("[?]"):
            confidence = 0.0
            text = text[3:].strip()

        if text.startswith("[") and "]" in text:
            label = text[1:text.index("]")]
            if label and label != "?":
                speaker = label
                text = text[text.index("]") + 1 :].strip()

        segments.append(
            TranscriptSegment(
                start_time=t,
                end_time=t + 1.0,
                text=text,
                confidence=confidence,
                speaker=speaker,
            )
        )
        t += 1.0

    return segments


def load_segments_from_transcript(path: Path) -> list[TranscriptSegment]:
    """Load segments from either .srt or .txt transcript."""
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return load_segments_from_srt(path)
    if suffix == ".txt":
        return load_segments_from_text(path)
    raise ValueError(f"Unsupported transcript format: {path}")


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


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    """Convert segments to plain text (no timestamps)."""
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)

    return "\n".join(lines)


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
