"""ASR subsystem.

Layering:

* ``schema``     — request/result dataclasses shared by everything
* ``strategies`` — swappable STT engines (strategy pattern + registry)
* ``cache``      — transcript store keyed by media identity
* ``commands``   — command objects that wrap strategy + cache
* ``format``     — transcript -> prompt block

Callers construct an :class:`AsrSettings` from YAML and call
:func:`build_transcribe_command`; nothing outside this package touches a
strategy directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omni_bench.asr.cache import TranscriptCache
from omni_bench.asr.commands import (
    BatchReport,
    BatchTranscribeCommand,
    Command,
    TranscribeCommand,
    build_requests,
)
from omni_bench.asr.format import DEFAULT_EMPTY_TEXT, DEFAULT_HEADER, format_transcript
from omni_bench.asr.schema import (
    MediaRef,
    Transcription,
    TranscriptionRequest,
    TranscriptionSegment,
    iter_media_files,
)
from omni_bench.asr.strategies import (
    SttStrategy,
    StrategySpec,
    available_strategies,
    build_strategy,
    register_strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "asr"


def resolve_cache_dir(value: str | Path) -> Path:
    """Relative cache paths resolve against the repo root, not the CWD."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(slots=True)
class AsrSettings:
    """Everything needed to turn media into a prompt block."""

    strategy: StrategySpec = field(default_factory=StrategySpec)
    cache_dir: Path = DEFAULT_CACHE_DIR
    #: Fail instead of transcribing inline when the cache misses. Keeps eval
    #: runs deterministic once `scripts/prepare_asr.py` has been run.
    strict_cache: bool = False
    with_timestamps: bool = True
    max_chars: int | None = None
    #: Drop transcript segments starting past this timestamp — see format_transcript.
    max_end_s: float | None = None
    header: str = DEFAULT_HEADER
    empty_text: str = DEFAULT_EMPTY_TEXT

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AsrSettings":
        data = dict(raw or {})
        strategy = StrategySpec.from_dict(data.pop("strategy", None))
        cache_dir = resolve_cache_dir(data.pop("cache_dir", DEFAULT_CACHE_DIR))
        known = {
            f for f in cls.__dataclass_fields__ if f not in {"strategy", "cache_dir"}  # type: ignore[attr-defined]
        }
        return cls(
            strategy=strategy,
            cache_dir=cache_dir,
            **{k: v for k, v in data.items() if k in known},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "cache_dir": str(self.cache_dir),
            "strict_cache": self.strict_cache,
            "with_timestamps": self.with_timestamps,
            "max_chars": self.max_chars,
            "max_end_s": self.max_end_s,
            "header": self.header,
            "empty_text": self.empty_text,
        }


def build_cache(settings: AsrSettings, strategy: SttStrategy) -> TranscriptCache:
    return TranscriptCache(settings.cache_dir, strategy.cache_namespace())


def build_transcribe_command(
    settings: AsrSettings, *, write_cache: bool = True
) -> TranscribeCommand:
    strategy = build_strategy(settings.strategy)
    return TranscribeCommand(strategy, build_cache(settings, strategy), write_cache=write_cache)


__all__ = [
    "AsrSettings",
    "BatchReport",
    "BatchTranscribeCommand",
    "Command",
    "DEFAULT_CACHE_DIR",
    "MediaRef",
    "StrategySpec",
    "SttStrategy",
    "TranscribeCommand",
    "TranscriptCache",
    "Transcription",
    "TranscriptionRequest",
    "TranscriptionSegment",
    "available_strategies",
    "build_cache",
    "build_requests",
    "build_strategy",
    "build_transcribe_command",
    "resolve_cache_dir",
    "format_transcript",
    "iter_media_files",
    "register_strategy",
]
