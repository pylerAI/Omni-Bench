"""Disk cache for transcripts.

Keyed by :class:`~omni_bench.asr.schema.MediaRef` (path + size + mtime) under a
namespace derived from the strategy, so switching engine or Whisper model never
serves a transcript produced by a different one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from omni_bench.asr.schema import MediaRef, Transcription


class TranscriptCache:
    def __init__(self, root: str | Path, namespace: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.namespace = namespace

    @property
    def namespace_dir(self) -> Path:
        return self.root / self.namespace

    def path_for(self, media: MediaRef) -> Path:
        return self.namespace_dir / media.key[:2] / f"{media.key}.json"

    def get(self, media: MediaRef) -> Transcription | None:
        path = self.path_for(media)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return Transcription.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            return None

    def put(self, transcription: Transcription) -> Path:
        path = self.path_for(transcription.media)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so concurrent workers never observe a partial file.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(transcription.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def count(self) -> int:
        if not self.namespace_dir.exists():
            return 0
        return sum(1 for _ in self.namespace_dir.rglob("*.json"))
