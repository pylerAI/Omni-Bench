from __future__ import annotations

import argparse
import contextlib
from typing import Iterable

from omni_bench.adapters import ADAPTER_NAMES, get_adapter
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig, load_config
from omni_bench.io import ensure_dir, write_json
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
    models = _filter_by_name(cfg.models, args.model)
    benchmarks = [b for b in _filter_by_name(cfg.benchmarks, args.benchmark) if b.enabled]
    run_summary: dict[str, dict[str, object]] = {}

    for model in models:
        context = serve_model(model, cfg.result_dir / "logs") if args.serve else contextlib.nullcontext()
        with context:
            client = VllmChatClient(model, default_timeout_s=cfg.request_timeout_s)
            model_summary: dict[str, object] = {}
            for benchmark in benchmarks:
                adapter = get_adapter(benchmark.name)
                output_dir = ensure_dir(
                    cfg.result_dir / model.name / (benchmark.result_subdir or benchmark.name)
                )
                model_summary[benchmark.name] = adapter.run(
                    benchmark=benchmark,
                    model=model,
                    client=client,
                    output_dir=output_dir,
                )
            run_summary[model.name] = model_summary

    ensure_dir(cfg.result_dir)
    write_json(cfg.result_dir / "run_summary.json", run_summary)
    write_overall_reports(cfg.result_dir, run_summary)
    try:
        report_path = render_report(cfg.result_dir)
        print(f"HTML report: {report_path}")
    except Exception as exc:  # report generation must never fail the run
        print(f"Skipped HTML report: {type(exc).__name__}: {exc}")


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
    throughput_rows = []
    for model_name, benchmark_results in run_summary.items():
        metric_row: dict[str, object] = {"model": model_name}
        throughput_row: dict[str, object] = {"model": model_name}
        for benchmark_name, summary in benchmark_results.items():
            if not isinstance(summary, dict):
                continue
            for metric_name, value in representative_metrics(benchmark_name, summary).items():
                metric_row[f"{benchmark_name}.{metric_name}"] = value
            if benchmark_name == "videomme":
                for metric_name, value in throughput_metrics(summary).items():
                    throughput_row[metric_name] = value
        metric_rows.append(metric_row)
        throughput_rows.append(throughput_row)
    return {"metrics": metric_rows, "throughput": throughput_rows}


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


def throughput_metrics(summary: dict[str, object]) -> dict[str, object]:
    throughput = summary.get("throughput")
    throughput = throughput if isinstance(throughput, dict) else {}
    return {
        "avg_latency_s": throughput.get("avg_latency_s"),
        "p50_latency_s": throughput.get("p50_latency_s"),
        "p95_latency_s": throughput.get("p95_latency_s"),
        "samples_per_sec": throughput.get("samples_per_sec"),
        "tokens_per_sec": throughput.get("total_tokens_per_sec"),
    }


def format_overall_markdown(report: dict[str, list[dict[str, object]]]) -> str:
    return (
        "## Benchmark Metrics\n\n"
        + format_markdown_table(report.get("metrics", []))
        + "\n## Throughput Metrics\n\n"
        + "현재 throughput metric은 Video-MME에서만 측정합니다.\n\n"
        + format_markdown_table(report.get("throughput", []))
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
