"""Re-score stored runs without inference.

For each benchmark dir of a model run, re-applies a reasoning strategy to the
stored records and runs the adapter's own ``parse_record`` / ``finalize`` — the
same code path ``omni-bench run`` uses — writing into a separate output dir.
The source run is only read.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from omni_bench.adapters import ADAPTER_NAMES, get_adapter
from omni_bench.config import DEFAULT_BENCHMARK_CONFIG, BenchmarkConfig, ModelConfig, _load_benchmark
from omni_bench.inference import REASONING_STRATEGIES, resolve_inference
from omni_bench.inference.reasoning import resplit_record
from omni_bench.inference.settings import DEFAULT_REASONING, MODEL_AXES, read_inference_block
from omni_bench.io import ensure_dir, read_json, read_jsonl_records, write_json, write_jsonl


def _benchmark_from(config_used: dict[str, Any] | None, name: str, benchmark_config: Path | None) -> BenchmarkConfig:
    if config_used and config_used.get("benchmarks"):
        return BenchmarkConfig(**config_used["benchmarks"][0])
    path = benchmark_config or DEFAULT_BENCHMARK_CONFIG
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in raw.get("benchmarks", []):
        if item.get("name") == name:
            return _load_benchmark(item)
    raise ValueError(f"No config_used.json in the run and no '{name}' entry in {path}; pass --benchmark-config")


def _model_from(config_used: dict[str, Any] | None, name: str) -> ModelConfig:
    if config_used and config_used.get("model"):
        raw = dict(config_used["model"])
        raw.pop("vllm", None)
        return ModelConfig(**raw)
    return ModelConfig(name=name, weight_path="")


def rescore_benchmark(
    src: Path,
    dst: Path,
    *,
    reasoning: str | None,
    benchmark_config: Path | None,
) -> dict[str, Any]:
    config_used = read_json(src / "config_used.json") if (src / "config_used.json").exists() else None
    name = (config_used or {}).get("benchmarks", [{}])[0].get("name") or src.name
    adapter = get_adapter(name)
    benchmark = _benchmark_from(config_used, name, benchmark_config)
    model = _model_from(config_used, src.parent.name)

    recorded = (config_used or {}).get("inference") or {}
    strategy_name = (
        reasoning
        or recorded.get("reasoning")
        or read_inference_block(model.extra, allowed=MODEL_AXES, where=f"model '{model.name}'").get("reasoning")
        or DEFAULT_REASONING
    )
    strategy = REASONING_STRATEGIES.get(strategy_name)()
    frames_mode = recorded.get("frames") or resolve_inference(model, benchmark, adapter.frame_modes).frames

    records_path = src / adapter.records_file
    records = read_jsonl_records(records_path) if records_path.exists() else read_json(src / "records.json")
    for record in records:
        resplit_record(strategy, record, adapter.response_field)
        # Error rows were never parsed by the run (fixed failure values).
        if not record.get("error"):
            record.update(adapter.parse_record(record))

    ensure_dir(dst)
    write_jsonl(dst / adapter.records_file, records)
    summary = adapter.finalize(records, benchmark=benchmark, output_dir=dst, frames_mode=frames_mode)
    original = read_json(src / "summary.json") if (src / "summary.json").exists() else {}
    if "perf" in original:
        # Timing from the original run; rescoring does not re-measure it.
        summary["perf"] = original["perf"]
    write_json(dst / "summary.json", summary)
    write_json(dst / "rescore_config.json", {
        "source": str(src), "reasoning": strategy_name, "frames": frames_mode,
        "benchmark": dataclasses.asdict(benchmark),
    })
    return summary


def rescore_run(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    benchmarks: list[str] | None = None,
    reasoning: str | None = None,
    benchmark_config: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve() / run_dir.name
    if out_root == run_dir or out_root.is_relative_to(run_dir) or run_dir.is_relative_to(out_root):
        raise ValueError(f"--out-dir must not overlap the source run: {out_root} vs {run_dir}")
    if reasoning is not None:
        REASONING_STRATEGIES.get(reasoning)
    results: dict[str, dict[str, Any]] = {}
    for src in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if not (src / "summary.json").exists():
            continue
        cfg = read_json(src / "config_used.json") if (src / "config_used.json").exists() else {}
        name = (cfg.get("benchmarks") or [{}])[0].get("name") or src.name
        if name not in ADAPTER_NAMES or (benchmarks and name not in benchmarks):
            continue
        results[name] = rescore_benchmark(
            src, out_root / src.name, reasoning=reasoning,
            benchmark_config=Path(benchmark_config) if benchmark_config else None,
        )
    if not results:
        raise ValueError(f"No benchmark results found under {run_dir}")
    return results
