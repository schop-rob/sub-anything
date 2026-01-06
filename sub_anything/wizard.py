"""Interactive TUI wizard for sub-anything."""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
import time
import os
from pathlib import Path

from .models import Config
from .utils import SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _normalize_path_input(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    try:
        parts = shlex.split(raw)
        if len(parts) == 1:
            return parts[0]
    except ValueError:
        pass
    return raw.strip("\"'")


def _ask(prompt: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{hint}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    choices_lower = {c.lower(): c for c in choices}
    default = choices_lower.get(default.lower(), default)

    while True:
        value = _ask(prompt, default=default).strip()
        if value.isdigit():
            idx = int(value)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        key = value.lower()
        if key in choices_lower:
            return choices_lower[key]
        print(f"Choose one of: {', '.join(choices)} (or 1-{len(choices)})")


def _pick_choice(prompt: str, choices: list[str], default: str) -> str:
    """Pick a choice with arrow keys (falls back to text input)."""
    if not _is_interactive():
        return _ask_choice(prompt, choices, default)

    try:
        default_idx = next(i for i, c in enumerate(choices) if c.lower() == default.lower())
    except StopIteration:
        default_idx = 0

    if os.name == "nt":
        idx = _pick_choice_windows(prompt, choices, default_idx)
    else:
        idx = _pick_choice_posix(prompt, choices, default_idx)

    if idx is None:
        return _ask_choice(prompt, choices, default)

    return choices[idx]


def _render_choice_menu(prompt: str, choices: list[str], selected: int) -> None:
    width = shutil.get_terminal_size((80, 20)).columns
    pointer = "❯ "
    try:
        (pointer).encode(sys.stdout.encoding or "utf-8")
    except Exception:
        pointer = "> "

    print(prompt)
    for i, choice in enumerate(choices):
        prefix = pointer if i == selected else "  "
        print((prefix + choice)[:width])
    print("Use Up/Down and Enter. (q = type manually)")


def _pick_choice_posix(prompt: str, choices: list[str], selected: int) -> int | None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            # Clear screen section (choices + hint)
            print("\x1b[2J\x1b[H", end="")  # clear + home
            _render_choice_menu(prompt, choices, selected)

            ch = sys.stdin.buffer.read(1)
            if not ch:
                return None
            if ch in (b"\r", b"\n"):
                print()
                return selected
            if ch in (b"q", b"Q"):
                print()
                return None
            if ch == b"\x1b":
                seq = sys.stdin.buffer.read(2)
                if seq == b"[A":  # up
                    selected = (selected - 1) % len(choices)
                elif seq == b"[B":  # down
                    selected = (selected + 1) % len(choices)
                else:
                    # Try to consume the rest of longer sequences, ignore.
                    try:
                        sys.stdin.buffer.read(8)
                    except Exception:
                        pass
            elif ch in (b"k", b"K"):
                selected = (selected - 1) % len(choices)
            elif ch in (b"j", b"J"):
                selected = (selected + 1) % len(choices)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _pick_choice_windows(prompt: str, choices: list[str], selected: int) -> int | None:
    try:
        import msvcrt
    except Exception:
        return None

    while True:
        print("\x1b[2J\x1b[H", end="")
        _render_choice_menu(prompt, choices, selected)

        key = msvcrt.getwch()
        if key in ("\r", "\n"):
            print()
            return selected
        if key in ("q", "Q"):
            print()
            return None
        if key == "\x00" or key == "\xe0":
            key2 = msvcrt.getwch()
            if key2 == "H":  # up
                selected = (selected - 1) % len(choices)
            elif key2 == "P":  # down
                selected = (selected + 1) % len(choices)
        elif key in ("k", "K"):
            selected = (selected - 1) % len(choices)
        elif key in ("j", "J"):
            selected = (selected + 1) % len(choices)


def _list_media_files(directory: Path) -> list[Path]:
    supported = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
    files: list[Path] = []
    try:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in supported:
                files.append(child)
    except Exception:
        return []
    return sorted(files, key=lambda p: p.name.lower())


def _ask_input_file() -> Path:
    cwd = Path.cwd()
    cached_files = _list_media_files(cwd)

    while True:
        if cached_files:
            print("\nSupported files in this folder:")
            for i, p in enumerate(cached_files[:30], 1):
                print(f"  {i:>2}. {p.name}")
            if len(cached_files) > 30:
                print(f"  ... ({len(cached_files) - 30} more)")
            raw = _ask("Pick a number or paste a file path")
        else:
            raw = _ask("Paste a file path")

        raw = _normalize_path_input(raw)
        if raw.isdigit() and cached_files:
            idx = int(raw)
            if 1 <= idx <= len(cached_files):
                return cached_files[idx - 1]

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (cwd / path).resolve()

        if path.exists():
            return path

        # Common case: accidental trailing space
        trimmed = Path(str(path).rstrip()).expanduser()
        if trimmed.exists():
            return trimmed

        print(f"File not found: {path}")


def _intro_animation():
    if not _is_interactive():
        return

    width = shutil.get_terminal_size((80, 20)).columns
    msg = "sub-anything wizard"
    sparkle = "✦"
    try:
        (sparkle).encode(sys.stdout.encoding or "utf-8")
    except Exception:
        sparkle = "*"

    for i in range(12):
        glyph = sparkle if i % 2 == 0 else " "
        line = f"{msg} {glyph}"
        print("\r" + line.ljust(width), end="", flush=True)
        time.sleep(0.15)

    print("\r" + f"{msg} {sparkle}".ljust(width))


def run_wizard(
    *,
    config_file: Path,
    available_models: list[str],
) -> argparse.Namespace:
    """Prompt for arguments and return an argparse.Namespace compatible with sub_anything.py."""
    if not _is_interactive():
        raise RuntimeError("Wizard requires an interactive TTY")

    _intro_animation()
    print("Press Enter to accept defaults. Ctrl+C to quit.\n")

    input_path = _ask_input_file()

    model = _pick_choice(
        "Select model",
        choices=available_models,
        default="chirp3",
    )

    no_timestamps = _ask_yes_no("Output plain text (no timestamps)?", default=False)

    is_video = input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    mux = False
    if is_video:
        mux = _ask_yes_no("Embed subtitles into the video (mux)?", default=False)
        if mux and no_timestamps:
            print("Mux requires timestamps; switching output to SRT.")
            no_timestamps = False

    diarize = _ask_yes_no("Enable speaker diarization?", default=False)

    language = _ask("Source language hint (auto, zh, en-US, cmn-Hans-CN, ...)", default="auto")

    google_location = None
    if model in {"chirp3", "long"}:
        config = Config.load(config_file)
        default_loc = config.chirp_location if model == "chirp3" else "us-central1"
        google_location = _ask("Google location (us, eu, asia-northeast1, ...)", default=default_loc)

    translate = None
    translate_model = "gpt-4o-mini"
    translate_batch_size = 20
    save_original = False

    if _ask_yes_no("Translate the transcript?", default=False):
        translate = _ask("Translate to language (e.g. en, es)", default="en")
        save_original = _ask_yes_no("Also save the original (untranslated) transcript?", default=True)
        translate_model = _ask("Translation model", default=translate_model)
        translate_batch_size_raw = _ask("Translation batch size (smaller can reduce weird repeats)", default=str(translate_batch_size))
        try:
            translate_batch_size = int(translate_batch_size_raw)
        except ValueError:
            translate_batch_size = 20

    verbose = _ask_yes_no("Verbose output?", default=False)

    cmd = [
        "sub-anything",
        str(input_path),
        "--model",
        model,
        "--language",
        language,
    ]
    if google_location and model in {"chirp3", "long"}:
        cmd += ["--google-location", google_location]
    if diarize:
        cmd.append("--diarize")
    if mux:
        cmd.append("--mux")
    if no_timestamps:
        cmd.append("--no-timestamps")
    if translate:
        cmd += ["--translate", translate, "--translate-model", translate_model, "--translate-batch-size", str(translate_batch_size)]
        if save_original:
            cmd.append("--save-original")
    if verbose:
        cmd.append("--verbose")

    print("\nPlan:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))

    if not _ask_yes_no("Run now?", default=True):
        raise SystemExit(0)

    return argparse.Namespace(
        input=Path(input_path),
        verbose=verbose,
        model=model,
        google_location=google_location,
        translate=translate,
        translate_model=translate_model,
        translate_batch_size=translate_batch_size,
        save_original=save_original,
        diarize=diarize,
        language=language,
        mux=mux,
        no_timestamps=no_timestamps,
    )
