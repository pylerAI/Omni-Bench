"""Unified I/O schema for every ASR command.

Commands accept a :class:`TranscriptionRequest` and return a
:class:`Transcription`, regardless of which STT strategy runs underneath. The
JSON produced by :meth:`Transcription.to_dict` is the on-disk cache format and
the contract that downstream code (prompt formatting, analysis scripts) reads.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "asr/1"

#: Extensions treated as media the ASR pipeline can consume.
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".ts")
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")
MEDIA_SUFFIXES = VIDEO_SUFFIXES + AUDIO_SUFFIXES


@dataclass(slots=True, frozen=True)
class MediaRef:
    """Stable identity for a media file.

    ``key`` is what the cache is addressed by. It is derived from the resolved
    path plus size and mtime so that regenerating an upstream artifact (for
    example a re-extracted ``.wav``) invalidates the transcript instead of
    silently serving a stale one.
    """

    path: str
    size_bytes: int
    mtime_ns: int
    key: str

    @classmethod
    def from_path(cls, path: str | Path) -> "MediaRef":
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        digest = hashlib.sha1(
            f"{resolved}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        return cls(
            path=str(resolved),
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            key=digest,
        )


@dataclass(slots=True)
class TranscriptionRequest:
    """Input schema shared by every command and strategy."""

    media_path: Path
    language: str | None = None
    #: Strategy-specific overrides merged over the strategy's own defaults.
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.media_path = Path(self.media_path).expanduser()

    def media_ref(self) -> MediaRef:
        return MediaRef.from_path(self.media_path)


@dataclass(slots=True)
class TranscriptionSegment:
    index: int
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TranscriptionSegment":
        return cls(
            index=int(raw["index"]),
            start_s=float(raw["start_s"]),
            end_s=float(raw["end_s"]),
            text=str(raw["text"]),
        )


@dataclass(slots=True)
class Transcription:
    """Output schema shared by every command and strategy."""

    media: MediaRef
    backend: str
    model: str
    segments: list[TranscriptionSegment]
    language: str | None = None
    language_probability: float | None = None
    duration_s: float | None = None
    elapsed_s: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "media": asdict(self.media),
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_s": self.duration_s,
            "elapsed_s": self.elapsed_s,
            "params": self.params,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Transcription":
        media = raw["media"]
        return cls(
            media=MediaRef(**media),
            backend=str(raw["backend"]),
            model=str(raw["model"]),
            segments=[TranscriptionSegment.from_dict(s) for s in raw.get("segments", [])],
            language=raw.get("language"),
            language_probability=raw.get("language_probability"),
            duration_s=raw.get("duration_s"),
            elapsed_s=raw.get("elapsed_s"),
            params=raw.get("params") or {},
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def empty(cls, media: MediaRef, *, backend: str, model: str, reason: str) -> "Transcription":
        """A valid, cacheable transcript for media that carries no speech."""
        return cls(
            media=media,
            backend=backend,
            model=model,
            segments=[],
            params={"empty_reason": reason},
        )


def is_media_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_SUFFIXES


def iter_media_files(root: str | Path) -> list[Path]:
    """All media files under ``root`` (or ``root`` itself if it is a file)."""
    root_path = Path(root).expanduser()
    if root_path.is_file():
        return [root_path] if is_media_file(root_path) else []
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path, followlinks=True):
        for filename in filenames:
            if is_media_file(filename):
                found.append(Path(dirpath) / filename)
    return sorted(found)
