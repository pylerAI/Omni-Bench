from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


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


def load_resumable_records(
    path: str | Path, key_fn: Callable[[dict[str, Any]], str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Successful records from a previous run, and their keys.

    Records carrying ``error`` are not treated as done, so the next invocation
    retries them. They are moved out of ``records.jsonl`` into
    ``records.errors.jsonl`` (kept for diagnosis) so the jsonl never holds a
    stale failure next to the retry's result.
    """
    src = Path(path)
    rows = read_jsonl_records(src)
    ok = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    if failed:
        errors_path = src.with_name(src.stem + ".errors.jsonl")
        for row in failed:
            append_jsonl(errors_path, row)
        tmp = src.with_suffix(src.suffix + ".tmp")
        write_jsonl(tmp, ok)
        tmp.replace(src)
    return ok, {key_fn(row) for row in ok}


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
