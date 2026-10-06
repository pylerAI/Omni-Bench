from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig


class BenchmarkAdapter(ABC):
    name: str

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
