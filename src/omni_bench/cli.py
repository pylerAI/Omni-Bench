from __future__ import annotations

import argparse
import contextlib
from typing import Iterable

from omni_bench.adapters import ADAPTER_NAMES, get_adapter
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig, load_config
from omni_bench.io import ensure_dir, write_json
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


if __name__ == "__main__":
    main()
