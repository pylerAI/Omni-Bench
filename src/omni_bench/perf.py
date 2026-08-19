"""Throughput and latency stats for a finished benchmark run.

Computed from the records the adapters already write, so every benchmark gets
the same numbers without each adapter growing its own timing code. Attached to
``summary.json`` under ``perf``.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; ``q`` in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _numbers(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and value is not None:
            out.append(float(value))
    return out


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "p50": round(median(values), 3),
        "p90": round(_percentile(values, 0.90) or 0.0, 3),
        "p99": round(_percentile(values, 0.99) or 0.0, 3),
        "max": round(max(values), 3),
        "sum": round(sum(values), 1),
    }


def read_records(records_path: str | Path) -> list[dict[str, Any]]:
    path = Path(records_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_perf(
    records_path: str | Path,
    *,
    wall_s: float,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Wall-clock throughput plus per-request latency and token distributions.

    ``wall_s`` covers everything the adapter did — frame decoding, transcript
    lookup, and the requests — so ``samples_per_s`` is the number that predicts
    how long a rerun takes. Per-request ``latency_s`` is inflated by queueing at
    the server, so it is only comparable within a run.
    """
    records = read_records(records_path)
    latency = _numbers(records, "latency_s")
    prompt_tokens = _numbers(records, "prompt_tokens")
    completion_tokens = _numbers(records, "completion_tokens")
    errors = sum(1 for record in records if record.get("error"))

    perf: dict[str, Any] = {
        "samples": len(records),
        "errors": errors,
        "wall_s": round(wall_s, 1),
        "wall_min": round(wall_s / 60, 2),
        "samples_per_s": round(len(records) / wall_s, 4) if wall_s > 0 else None,
        "s_per_sample": round(wall_s / len(records), 4) if records else None,
        "concurrency": concurrency,
        "latency_s": _stats(latency),
        "prompt_tokens": _stats(prompt_tokens),
        "completion_tokens": _stats(completion_tokens),
    }

    if wall_s > 0:
        if prompt_tokens:
            perf["prompt_tokens_per_s"] = round(sum(prompt_tokens) / wall_s, 1)
        if completion_tokens:
            perf["output_tokens_per_s"] = round(sum(completion_tokens) / wall_s, 1)
        if prompt_tokens and completion_tokens:
            perf["total_tokens_per_s"] = round(
                (sum(prompt_tokens) + sum(completion_tokens)) / wall_s, 1
            )
    return perf
