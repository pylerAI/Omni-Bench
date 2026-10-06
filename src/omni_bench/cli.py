from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from omni_bench.adapters import ADAPTER_NAMES, get_adapter
from omni_bench.asr_client import AsrCommandPool, build_chat_client, resolve_audio_mode
from omni_bench.config import BenchmarkConfig, ModelConfig, load_config
from omni_bench.io import ensure_dir, write_json
from omni_bench.perf import summarize_perf
from omni_bench.report import render_report
from omni_bench.serving import serve_model, vllm_command


def main() -> None:
    parser = argparse.ArgumentParser(prog="omni-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run configured benchmark evaluations.")
    run_parser.add_argument("--config", required=True, help="Path to YAML config.")
    run_parser.add_argument(
        "--benchmark-config",
        default=None,
        help="Path to benchmark YAML config. Defaults to configs/benchmarks/default.yaml.",
    )
    run_parser.add_argument("--model", action="append", help="Model name to run. Repeatable.")
    run_parser.add_argument("--benchmark", action="append", help="Benchmark name to run. Repeatable.")
    run_parser.add_argument("--serve", action="store_true", help="Start vLLM serve for each model before evaluation.")
    run_parser.add_argument("--limit", type=int, default=None,
                            help="Evaluate only the first N items of each benchmark (overrides config).")
    run_parser.add_argument("--result-dir", default=None,
                            help="Override global.result_dir (e.g. a separate smoke-test root).")

    serve_parser = subparsers.add_parser("serve", help="Print or run a vLLM serve command for a model.")
    serve_parser.add_argument("--config", required=True, help="Path to YAML config.")
    serve_parser.add_argument("--model", required=True, help="Model name in config.")
    serve_parser.add_argument("--print-only", action="store_true", help="Print the command without running it.")

    subparsers.add_parser("list-benchmarks", help="List supported benchmark adapters.")

    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "serve":
        serve(args)
    elif args.command == "list-benchmarks":
        for name in sorted(ADAPTER_NAMES):
            print(name)


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, benchmark_path=args.benchmark_config)
    if args.result_dir:
        cfg.result_dir = Path(args.result_dir).expanduser().resolve()
    models = _filter_by_name(cfg.models, args.model)
    benchmarks = [b for b in _filter_by_name(cfg.benchmarks, args.benchmark) if b.enabled]
    if args.limit is not None:
        for benchmark in benchmarks:
            benchmark.limit = args.limit
    run_summary: dict[str, dict[str, object]] = {}
    # Shared across benchmarks so an STT engine is loaded at most once per config.
    asr_pool = AsrCommandPool()

    for model in models:
        context = serve_model(model, cfg.result_dir / "logs") if args.serve else contextlib.nullcontext()
        with context:
            model_summary: dict[str, object] = {}
            for benchmark in benchmarks:
                adapter = get_adapter(benchmark.name)
                # audio_mode is resolved per benchmark: a benchmark whose official
                # protocol excludes audio stays audio-free even in asr_text mode.
                client = build_chat_client(
                    model,
                    default_timeout_s=cfg.request_timeout_s,
                    benchmark=benchmark,
                    pool=asr_pool,
                )
                print(
                    f"[{model.name}/{benchmark.name}] audio_mode="
                    f"{resolve_audio_mode(model, benchmark)}"
                )
                output_dir = ensure_dir(
                    cfg.result_dir / model.name / (benchmark.result_subdir or benchmark.name)
                )
                write_config_snapshot(output_dir, args=args, model=model, benchmarks=[benchmark],
                                      result_dir=cfg.result_dir, timeout_s=cfg.request_timeout_s)
                started = time.perf_counter()
                summary = adapter.run(
                    benchmark=benchmark,
                    model=model,
                    client=client,
                    output_dir=output_dir,
                )
                wall_s = time.perf_counter() - started
                # Timed here rather than in each adapter, so all five report the
                # same throughput and latency fields.
                if isinstance(summary, dict):
                    summary["perf"] = summarize_perf(
                        output_dir,
                        wall_s=wall_s,
                        concurrency=benchmark.extra.get("concurrency")
                        or benchmark.extra.get("max_workers"),
                    )
                    write_json(output_dir / "summary.json", summary)
                    perf = summary["perf"]
                    print(
                        f"[{model.name}/{benchmark.name}] {perf['samples']} samples in "
                        f"{perf['wall_min']}min · {perf['samples_per_s']}/s · "
                        f"latency p50 {(perf['latency_s'] or {}).get('p50')}s "
                        f"p90 {(perf['latency_s'] or {}).get('p90')}s · errors {perf['errors']}"
                    )
                model_summary[benchmark.name] = summary
            run_summary[model.name] = model_summary

    ensure_dir(cfg.result_dir)
    write_json(cfg.result_dir / "run_summary.json", run_summary)
    write_overall_reports(cfg.result_dir, run_summary)
    try:
        report_path = render_report(cfg.result_dir)
        print(f"HTML report: {report_path}")
    except Exception as exc:  # report generation must never fail the run
        print(f"Skipped HTML report: {type(exc).__name__}: {exc}")


def _git_state() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]

    def git(*cmd: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def write_config_snapshot(output_dir: Path, *, args: argparse.Namespace, model: ModelConfig,
                          benchmarks: list[BenchmarkConfig], result_dir: Path,
                          timeout_s: float) -> None:
    """``config_used.json`` in the benchmark dir and the run (model) dir.

    The run-level copy is overwritten by each benchmark invocation, so the
    benchmark-level one is the authoritative record for that benchmark.
    """
    snapshot = {
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config_path": str(Path(args.config).resolve()),
        "benchmark_config_path": str(Path(args.benchmark_config).resolve()) if args.benchmark_config else None,
        "result_dir": str(result_dir),
        "request_timeout_s": timeout_s,
        "git": _git_state(),
        "model": dataclasses.asdict(model),
        "benchmarks": [dataclasses.asdict(b) for b in benchmarks],
    }
    write_json(output_dir / "config_used.json", snapshot)
    write_json(output_dir.parent / "config_used.json", snapshot)


def serve(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    matches = _filter_by_name(cfg.models, [args.model])
    model = matches[0]
    command = vllm_command(model)
    print(" ".join(command))
    if args.print_only:
        return
    with serve_model(model, cfg.result_dir / "logs"):
        print(f"Serving {model.name} at {model.resolved_base_url}. Press Ctrl+C to stop.")
        try:
            while True:
                import time

                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def _filter_by_name(items: Iterable[ModelConfig] | Iterable[BenchmarkConfig], names: list[str] | None):
    items_list = list(items)
    if not names:
        return items_list
    selected = [item for item in items_list if item.name in names]
    missing = set(names) - {item.name for item in selected}
    if missing:
        raise ValueError(f"Unknown names in config: {sorted(missing)}")
    return selected


def write_overall_reports(result_dir, run_summary: dict[str, dict[str, object]]) -> None:
    report = build_overall_report(run_summary)
    write_json(result_dir / "overall_report.json", report)
    (result_dir / "overall_report.md").write_text(format_overall_markdown(report), encoding="utf-8")


def build_overall_report(run_summary: dict[str, dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    metric_rows = []
    for model_name, benchmark_results in run_summary.items():
        metric_row: dict[str, object] = {"model": model_name}
        for benchmark_name, summary in benchmark_results.items():
            if not isinstance(summary, dict):
                continue
            for metric_name, value in representative_metrics(benchmark_name, summary).items():
                metric_row[f"{benchmark_name}.{metric_name}"] = value
        metric_rows.append(metric_row)
    return {"metrics": metric_rows}


def representative_metrics(benchmark_name: str, summary: dict[str, object]) -> dict[str, object]:
    if benchmark_name in {"av_speakerbench", "worldsense", "omnivideobench"}:
        return {"accuracy": summary.get("accuracy")}
    if benchmark_name == "videomme":
        return {"official_accuracy": summary.get("accuracy")}
    if benchmark_name == "omnidcbench":
        metrics = summary.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        return {
            "f1": metrics.get("f1"),
            "miou": metrics.get("miou"),
            "soda_m": metrics.get("soda_m"),
        }
    return {"accuracy": summary.get("accuracy")}


def format_overall_markdown(report: dict[str, list[dict[str, object]]]) -> str:
    return (
        "## Benchmark Metrics\n\n"
        + format_markdown_table(report.get("metrics", []))
        + "\nThroughput is measured separately with `vllm bench throughput` "
        "(see `results/throughput.json` and the HTML report).\n"
    )


def format_markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "| model |\n| --- |\n"
    columns = ["model"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def format_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
