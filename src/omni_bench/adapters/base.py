from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig


class BenchmarkAdapter(ABC):
    """One benchmark: run inference, then parse and summarize the records.

    ``run`` does inference and must end by calling ``finalize``. Parsing and
    summarizing are separate abstract methods, not helpers inside ``run``, so
    stored records can be re-parsed and re-summarized through the exact code a
    run uses, without inference (a later ``omni-bench rescore`` builds on this).
    """

    name: str
    #: Frames strategies this benchmark can run with; the first is its protocol
    #: default. Adapters that only ever hand over the original video keep "server".
    frame_modes: tuple[str, ...] = ("server",)
    #: Record field holding the model's answer text.
    response_field: str = "response"
    #: Per-sample records the run appends to.
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
        """Summary + output files from finished records (no inference)."""
        raise NotImplementedError


def apply_limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    return items[:limit]
