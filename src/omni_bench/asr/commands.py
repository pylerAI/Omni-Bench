"""Transcription commands.

Each command takes the shared request schema and returns the shared result
schema. The strategy is injected, so swapping the STT engine never changes a
caller. Commands are the only place that knows about the cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, TypeVar

from omni_bench.asr.cache import TranscriptCache
from omni_bench.asr.schema import Transcription, TranscriptionRequest
from omni_bench.asr.strategies import SttStrategy

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Command(ABC, Generic[TIn, TOut]):
    @abstractmethod
    def execute(self, payload: TIn) -> TOut:
        raise NotImplementedError

    def __call__(self, payload: TIn) -> TOut:
        return self.execute(payload)


@dataclass(slots=True)
class BatchOutcome:
    """Per-item result of a batch run; ``error`` is set only on failure."""

    media_path: str
    status: str  # "cached" | "transcribed" | "failed"
    elapsed_s: float | None = None
    segments: int | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchReport:
    total: int = 0
    cached: int = 0
    transcribed: int = 0
    failed: int = 0
    outcomes: list[BatchOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "cached": self.cached,
            "transcribed": self.transcribed,
            "failed": self.failed,
            "failures": [
                {"media_path": o.media_path, "error": o.error}
                for o in self.outcomes
                if o.status == "failed"
            ],
        }


class TranscribeCommand(Command[TranscriptionRequest, Transcription]):
    """Cache-aware single-item transcription."""

    def __init__(
        self,
        strategy: SttStrategy,
        cache: TranscriptCache | None = None,
        *,
        read_cache: bool = True,
        write_cache: bool = True,
    ) -> None:
        self.strategy = strategy
        self.cache = cache
        self.read_cache = read_cache
        self.write_cache = write_cache

    def lookup(self, request: TranscriptionRequest) -> Transcription | None:
        if not (self.cache and self.read_cache):
            return None
        return self.cache.get(request.media_ref())

    def execute(self, payload: TranscriptionRequest) -> Transcription:
        cached = self.lookup(payload)
        if cached is not None:
            return cached
        transcription = self.strategy.transcribe(payload)
        if self.cache and self.write_cache:
            self.cache.put(transcription)
        return transcription


class BatchTranscribeCommand(Command[Iterable[TranscriptionRequest], BatchReport]):
    """Fan a request list across threads; CTranslate2 releases the GIL."""

    def __init__(
        self,
        transcribe: TranscribeCommand,
        *,
        workers: int = 4,
        on_progress: Callable[[BatchOutcome], None] | None = None,
    ) -> None:
        self.transcribe = transcribe
        self.workers = max(1, workers)
        self.on_progress = on_progress

    def _run_one(self, request: TranscriptionRequest) -> BatchOutcome:
        media_path = str(request.media_path)
        try:
            cached = self.transcribe.lookup(request)
            if cached is not None:
                return BatchOutcome(media_path, "cached", segments=len(cached.segments))
            result = self.transcribe.execute(request)
            return BatchOutcome(
                media_path,
                "transcribed",
                elapsed_s=result.elapsed_s,
                segments=len(result.segments),
            )
        except Exception as exc:  # one bad file must not stop the batch
            return BatchOutcome(media_path, "failed", error=f"{type(exc).__name__}: {exc}")

    def execute(self, payload: Iterable[TranscriptionRequest]) -> BatchReport:
        requests = list(payload)
        report = BatchReport(total=len(requests))
        if not requests:
            return report

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._run_one, request) for request in requests]
            for future in as_completed(futures):
                outcome = future.result()
                report.outcomes.append(outcome)
                setattr(report, outcome.status, getattr(report, outcome.status) + 1)
                if self.on_progress:
                    self.on_progress(outcome)
        return report


def build_requests(
    media_paths: Iterable[str | Path],
    *,
    language: str | None = None,
    options: dict[str, Any] | None = None,
) -> list[TranscriptionRequest]:
    return [
        TranscriptionRequest(media_path=Path(path), language=language, options=dict(options or {}))
        for path in media_paths
    ]
