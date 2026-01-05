"""Utility functions for audio processing and SRT formatting."""

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
