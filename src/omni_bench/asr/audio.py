"""ffmpeg helpers shared by the STT strategies.

Strategies consume 16 kHz mono audio. Video inputs are demuxed to a temporary
WAV; audio inputs are passed through untouched when they already match.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from omni_bench.asr.schema import AUDIO_SUFFIXES

TARGET_SAMPLE_RATE = 16_000


class AudioExtractionError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise AudioExtractionError("ffmpeg not found on PATH; it is required for ASR preprocessing.")
    return binary


def has_audio_stream(media_path: str | Path) -> bool:
    """False when the container carries no audio track (or ffprobe is absent)."""
    probe = shutil.which("ffprobe")
    if not probe:
        return True
    command = [
        probe,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return True
    if completed.returncode != 0:
        return True
    try:
        return bool(json.loads(completed.stdout or "{}").get("streams"))
    except json.JSONDecodeError:
        return True


def media_duration_s(media_path: str | Path) -> float | None:
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    command = [
        probe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return float(completed.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def extract_wav(media_path: str | Path, output_path: str | Path) -> Path:
    """Demux ``media_path`` to 16 kHz mono PCM WAV at ``output_path``.

    ffmpeg writes to a unique temp file in the same directory, which is then
    renamed over ``output_path``; a reader never sees a half-written WAV, even
    when several callers extract the same file at once.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(output.parent), prefix=f".{output.stem}.", suffix=".tmp.wav")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _run_extract(media_path, tmp)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return output


def _run_extract(media_path: str | Path, output: Path) -> None:
    command = [
        ffmpeg_binary(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise AudioExtractionError(
            f"ffmpeg failed for {media_path}: {completed.stderr.strip()[:500]}"
        )


_path_locks: dict[str, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    with _path_locks_guard:
        return _path_locks.setdefault(str(path.resolve()), threading.Lock())


def ensure_wav(media_path: str | Path, output_path: str | Path) -> Path:
    """Cached WAV of ``media_path`` at ``output_path``; extracted at most once per
    path within the process (per-path lock). The file only appears complete,
    via the atomic rename in :func:`extract_wav`."""
    output = Path(output_path)
    if output.exists():
        return output
    with _path_lock(output):
        if output.exists():
            return output
        return extract_wav(media_path, output)


@contextlib.contextmanager
def audio_source(media_path: str | Path, *, tmp_dir: str | Path | None = None) -> Iterator[Path]:
    """Yield a path the STT strategy can decode.

    Plain audio files are yielded as-is; anything else is demuxed to a temporary
    WAV that is removed on exit.
    """
    source = Path(media_path)
    if source.suffix.lower() in AUDIO_SUFFIXES:
        yield source
        return
    with tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None, prefix="omni-asr-") as tmp:
        yield extract_wav(source, Path(tmp) / f"{source.stem}.wav")
