from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    if not src.exists():
        return []
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_existing_keys(path: str | Path, key: str) -> set[str]:
    return {str(row.get(key)) for row in read_jsonl_records(path) if row.get(key) is not None}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if value is not None and value.__class__.__module__.startswith("pandas"):
        return None
    return value


def accuracy(correct: int, total: int) -> float:
    return round(correct / total * 100.0, 4) if total else 0.0


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[index], 6)


def _rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def summarize_throughput(rows: Iterable[dict[str, Any]], wall_time_s: float | None = None) -> dict[str, Any]:
    """Latency/throughput stats from per-record ``latency_s`` and token counts.

    Rates use the measured wall-clock when given; under concurrency the summed
    per-record latency overstates elapsed time, so pass the executor wall-clock.
    """
    rows_list = list(rows)
    latencies = [float(r["latency_s"]) for r in rows_list if r.get("latency_s") is not None]
    summed = sum(latencies)
    elapsed = wall_time_s if wall_time_s is not None else summed

    def _sum(key: str) -> int:
        return sum(int(r[key]) for r in rows_list if r.get(key) is not None)

    completion_tokens = _sum("completion_tokens")
    total_tokens = _sum("total_tokens")
    return {
        "samples": len(rows_list),
        "total_wall_time_s": round(elapsed, 6),
        "summed_latency_s": round(summed, 6),
        "avg_latency_s": round(summed / len(latencies), 6) if latencies else 0.0,
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "samples_per_sec": _rate(len(rows_list), elapsed),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_sec": _rate(completion_tokens, elapsed),
        "total_tokens_per_sec": _rate(total_tokens, elapsed),
    }


def summarize_accuracy(rows: Iterable[dict[str, Any]], group_keys: Iterable[str]) -> dict[str, Any]:
    rows_list = list(rows)
    summary: dict[str, Any] = {
        "total": len(rows_list),
        "correct": sum(1 for row in rows_list if row.get("is_correct")),
    }
    summary["accuracy"] = accuracy(summary["correct"], summary["total"])

    for key in group_keys:
        grouped: dict[str, dict[str, int | float]] = {}
        for row in rows_list:
            value = str(row.get(key, "unknown"))
            item = grouped.setdefault(value, {"total": 0, "correct": 0})
            item["total"] = int(item["total"]) + 1
            item["correct"] = int(item["correct"]) + int(bool(row.get("is_correct")))
        for item in grouped.values():
            item["accuracy"] = accuracy(int(item["correct"]), int(item["total"]))
        summary[f"by_{key}"] = grouped
    return summary
