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


def apply_limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    return items[:limit]
