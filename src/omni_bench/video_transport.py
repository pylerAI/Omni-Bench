"""How a local ``video_path`` reaches the server.

``file``   — ``file://`` URL; needs ``--allowed-local-media-path`` on the server.
``base64`` — inline ``data:<mime>;base64,...`` URL; works against remote servers
             that cannot see our filesystem. Files above ``threshold_mb`` are
             first transcoded (≤ ``max_height``p, h264, no audio) into a shared
             disk cache so request bodies stay bounded.

The cache is keyed like the ASR cache (resolved path + size + mtime) plus a tag
of the transcode parameters, so changing either invalidates the entry.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omni_bench.asr.schema import MediaRef

VIDEO_TRANSPORTS = ("file", "base64")
DEFAULT_TRANSCODE_CACHE = "/gpfs/public/artifacts/ail/nemotron-compare/cache/transcode"

VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}
VIDEO_SUFFIXES = tuple(VIDEO_MIME)


def video_mime(path: str | Path) -> str:
    return VIDEO_MIME.get(Path(path).suffix.lower(), "video/mp4")


@dataclass(frozen=True, slots=True)
class TranscodeStep:
    max_height: int
    crf: int


@dataclass(frozen=True, slots=True)
class TranscodeSettings:
    threshold_mb: float = 200.0
    cache_dir: str = DEFAULT_TRANSCODE_CACHE
    preset: str = "veryfast"
    threads: int = 8
    # Tried in order until the output fits under ``threshold_mb``; the last
    # step's output is used even if it is still larger.
    ladder: tuple[TranscodeStep, ...] = field(
        default_factory=lambda: (
            TranscodeStep(720, 28),
            TranscodeStep(480, 30),
            TranscodeStep(360, 32),
        )
    )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TranscodeSettings":
        raw = dict(raw or {})
        ladder = raw.pop("ladder", None)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if ladder:
            known["ladder"] = tuple(
                TranscodeStep(int(step["max_height"]), int(step["crf"])) for step in ladder
            )
        return cls(**known)

    @property
    def threshold_bytes(self) -> int:
        return int(self.threshold_mb * 1024 * 1024)

    @property
    def tag(self) -> str:
        spec = {"preset": self.preset, "threshold_mb": self.threshold_mb,
                "ladder": [asdict(step) for step in self.ladder], "v": 1}
        return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]

    def to_dict(self) -> dict[str, Any]:
        return {"threshold_mb": self.threshold_mb, "cache_dir": self.cache_dir,
                "preset": self.preset, "threads": self.threads,
                "ladder": [asdict(step) for step in self.ladder], "tag": self.tag}


def needs_transcode(path: str | Path, settings: TranscodeSettings) -> bool:
    return Path(path).stat().st_size > settings.threshold_bytes


def transcode_cache_path(path: str | Path, settings: TranscodeSettings) -> Path:
    key = MediaRef.from_path(path).key
    return Path(settings.cache_dir).expanduser() / key[:2] / f"{key}__{settings.tag}.mp4"


_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _thread_lock(key: str) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _run_ffmpeg(src: Path, dst: Path, step: TranscodeStep, settings: TranscodeSettings) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-i", str(src),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        # Downscale only; -2 keeps the width even as h264 requires. fps untouched.
        "-vf", f"scale=-2:'min({step.max_height},ih)'",
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(step.crf),
        "-pix_fmt", "yuv420p", "-threads", str(settings.threads),
        "-movflags", "+faststart",
        "-f", "mp4", str(dst),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr.strip()[-500:]}")


def ensure_transcoded(path: str | Path, settings: TranscodeSettings) -> Path:
    """Cached transcode of ``path``; builds it on a miss.

    Safe under concurrent threads (per-key lock) and processes (``flock`` on a
    sidecar lock file). The output appears only via atomic rename.
    """
    src = Path(path).expanduser().resolve()
    target = transcode_cache_path(src, settings)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(str(target)):
        lock_path = target.with_suffix(".lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if target.exists():
                    return target
                tmp = target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
                started = time.perf_counter()
                try:
                    for step in settings.ladder:
                        _run_ffmpeg(src, tmp, step, settings)
                        if tmp.stat().st_size <= settings.threshold_bytes:
                            break
                    meta = {
                        "source": str(src),
                        "source_bytes": src.stat().st_size,
                        "output_bytes": tmp.stat().st_size,
                        "step": asdict(step),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                        "settings": settings.to_dict(),
                    }
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
                target.with_suffix(".json").write_text(json.dumps(meta, indent=2))
                return target
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def resolve_transport_file(path: str | Path, settings: TranscodeSettings) -> Path:
    """The file whose bytes go on the wire: the original or its transcode."""
    src = Path(path).expanduser()
    if needs_transcode(src, settings):
        return ensure_transcoded(src, settings)
    return src


def video_data_url(path: str | Path, settings: TranscodeSettings) -> str:
    sent = resolve_transport_file(path, settings)
    payload = base64.b64encode(sent.read_bytes()).decode("ascii")
    return f"data:{video_mime(sent)};base64,{payload}"
