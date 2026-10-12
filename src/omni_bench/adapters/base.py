from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig


class BenchmarkAdapter(ABC):
    name: str
    #: Frames strategies this benchmark can run with; the first is its protocol
    #: default. Adapters that only ever hand over the original video keep "server".
    frame_modes: tuple[str, ...] = ("server",)
    #: Record field holding the model's answer text (what the reasoning strategy rewrites).
    response_field: str = "response"
    #: Per-sample records the run appends to; ``omni-bench rescore`` reads it back.
    records_file: str = "records.jsonl"

    @abstractmethod
    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Fields the official parser derives from ``record[response_field]``."""
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        records: list[dict[str, Any]],
        *,
        benchmark: BenchmarkConfig,
        output_dir: Path,
        frames_mode: str,
    ) -> dict[str, Any]:
        """Summary + output files from finished records (no inference). Shared by
        ``run`` and ``omni-bench rescore``."""
        raise NotImplementedError


LIMIT_MODES = ("head", "spread")


def apply_limit(items: list[Any], limit: int | None, mode: str | None = None) -> list[Any]:
    """First ``limit`` items, or (``spread``) ``limit`` evenly spaced ones.

    ``spread`` gives a smoke subset that covers the whole dataset (e.g. every
    Video-MME duration bucket) instead of only its first, shortest clips; the
    choice is deterministic so a resumed run picks the same items.
    """
    if limit is None or limit >= len(items):
        return items
    mode = (mode or "head").lower()
    if mode == "head":
        return items[:limit]
    if mode == "spread":
        if limit <= 0:
            return []
        step = len(items) / limit
        return [items[int(i * step)] for i in range(limit)]
    raise ValueError(f"Unknown limit_mode '{mode}'. Known: {list(LIMIT_MODES)}")
