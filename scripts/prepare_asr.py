#!/usr/bin/env python
"""Transcribe every benchmark's media ahead of an evaluation run.

The transcripts land in the shared ASR cache, so `omni-bench run` with
``audio_mode: asr_text`` never blocks on Whisper. One worker process is pinned
per GPU; each process runs a thread pool over its share of the files. The job is
resumable — cached files are skipped.

    uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml
    uv run python scripts/prepare_asr.py --benchmark worldsense --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_bench.asr import (  # noqa: E402
    AsrSettings,
    BatchReport,
    BatchTranscribeCommand,
    TranscribeCommand,
    build_cache,
    build_requests,
    build_strategy,
    iter_media_files,
)
from omni_bench.config import DEFAULT_BENCHMARK_CONFIG  # noqa: E402

#: Where each benchmark keeps the media the eval client will hand to the ASR
#: layer. Video-MME is absent on purpose: it sends sampled frames only, with no
#: audio, so it stays identical to the native-omni runs.
MEDIA_ROOT_KEYS: dict[str, tuple[str, ...]] = {
    "av_speakerbench": ("data_path",),
    "omnidcbench": ("video_dir",),
    "omnivideobench": ("video_dir", "preprocess_cache_dir"),
    "worldsense": ("video_dir", "preprocess_cache_dir"),
}

#: WorldSense and OmniVideoBench do not hand the client the original video: their
#: adapters demux audio to a preprocess cache first and pass that ``.wav``. The
#: transcript cache is keyed by path, so transcribing only the originals leaves
#: those lookups missing. When the config does not name the directory, fall back
#: to the adapter's own default so the enumeration finds it.
PREPROCESS_CACHE_DEFAULTS: dict[str, tuple[str, str]] = {
    "worldsense": ("data_path", "preprocess_cache"),
    "omnivideobench": ("video_dir", "preprocess_cache"),
}


@dataclass(slots=True)
class Shard:
    gpu: int
    media_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asr-config", type=Path, default=None, help="YAML with the `asr:` block.")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG,
        help="Benchmark YAML used to locate media roots.",
    )
    parser.add_argument("--benchmark", action="append", help="Benchmark name. Repeatable. Default: all with media.")
    parser.add_argument("--media-root", action="append", type=Path, help="Extra directory or file to transcribe.")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU indices. Default: all visible.")
    parser.add_argument(
        "--threads-per-gpu",
        type=int,
        default=8,
        help="Concurrent transcriptions per GPU. Also sets CTranslate2 num_workers "
        "unless the config pins it, since without that the calls serialize inside "
        "the model and the threads buy nothing.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of files (debugging).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the ASR cache directory.")
    parser.add_argument("--model", default=None, help="Override the STT model id/path.")
    parser.add_argument("--strategy", default=None, help="Override the STT strategy name.")
    parser.add_argument("--language", default=None, help="Force a language instead of auto-detecting.")
    parser.add_argument("--report", type=Path, default=None, help="Where to write the run report JSON.")
    parser.add_argument("--dry-run", action="store_true", help="List the work without transcribing.")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> AsrSettings:
    raw: dict[str, Any] = {}
    if args.asr_config:
        with args.asr_config.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        raw = loaded.get("asr", loaded)
    settings = AsrSettings.from_dict(raw)
    if args.cache_dir:
        settings.cache_dir = args.cache_dir.expanduser()
    if args.model:
        settings.strategy.model = args.model
    if args.strategy:
        settings.strategy.name = args.strategy
    if args.language:
        settings.strategy.language = args.language
    return settings


def benchmark_media_roots(benchmark_config: Path, names: list[str] | None) -> list[Path]:
    with benchmark_config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    roots: list[Path] = []
    for item in raw.get("benchmarks", []):
        name = item.get("name")
        if name not in MEDIA_ROOT_KEYS:
            continue
        if names and name not in names:
            continue
        seen: set[Path] = set()
        for key in MEDIA_ROOT_KEYS[name]:
            value = item.get(key)
            if value:
                path = Path(str(value)).expanduser()
                if path not in seen:
                    seen.add(path)
                    roots.append(path)

        base_key, subdir = PREPROCESS_CACHE_DEFAULTS.get(name, (None, None))
        if base_key and not item.get("preprocess_cache_dir"):
            base = item.get(base_key)
            if base:
                derived = Path(str(base)).expanduser() / subdir
                if derived not in seen:
                    roots.append(derived)
                    if not derived.exists():
                        print(
                            f"  note: {name} demuxes audio into {derived} at eval time; "
                            "it does not exist yet, so run the benchmark once (or "
                            "pre-materialise it) and re-run this script, otherwise "
                            "those transcript lookups will miss."
                        )
    return roots


def collect_media(roots: Iterable[Path]) -> list[str]:
    seen: dict[str, None] = {}
    for root in roots:
        if not root.exists():
            print(f"  skip (missing): {root}")
            continue
        found = iter_media_files(root)
        print(f"  {len(found):>7} files  {root}")
        for path in found:
            seen.setdefault(str(path), None)
    return list(seen)


def resolve_gpus(spec: str | None) -> list[int]:
    if spec:
        return [int(part) for part in spec.split(",") if part.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "-1":
        return list(range(len([p for p in visible.split(",") if p.strip()])))
    try:
        import subprocess

        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        count = sum(1 for line in listing.stdout.splitlines() if line.startswith("GPU "))
        return list(range(count)) or [0]
    except Exception:
        return [0]


def shard(media_paths: list[str], gpus: list[int]) -> list[Shard]:
    shards = [Shard(gpu=gpu, media_paths=[]) for gpu in gpus]
    for index, path in enumerate(media_paths):
        shards[index % len(shards)].media_paths.append(path)
    return [s for s in shards if s.media_paths]


def run_shard(payload: tuple[Shard, dict[str, Any], int]) -> dict[str, Any]:
    """Executed in a worker process: one strategy instance pinned to one GPU."""
    shard_spec, settings_raw, threads = payload
    settings = AsrSettings.from_dict(settings_raw)
    settings.strategy.device_index = shard_spec.gpu
    # CTranslate2 serializes concurrent transcribe() calls unless the model was
    # built with num_workers > 1, so the thread pool alone would not parallelize.
    settings.strategy.options.setdefault("num_workers", threads)

    strategy = build_strategy(settings.strategy)
    cache = build_cache(settings, strategy)
    transcribe = TranscribeCommand(strategy, cache)

    done = 0
    total = len(shard_spec.media_paths)

    def progress(outcome) -> None:
        nonlocal done
        done += 1
        if done % 25 == 0 or done == total:
            print(f"[gpu{shard_spec.gpu}] {done}/{total}", flush=True)

    batch = BatchTranscribeCommand(transcribe, workers=threads, on_progress=progress)
    report = batch.execute(
        build_requests(shard_spec.media_paths, language=settings.strategy.language)
    )
    return report.to_dict()


def merge_reports(reports: list[dict[str, Any]]) -> BatchReport:
    merged = BatchReport()
    for report in reports:
        merged.total += report["total"]
        merged.cached += report["cached"]
        merged.transcribed += report["transcribed"]
        merged.failed += report["failed"]
    return merged


def main() -> int:
    args = parse_args()
    settings = load_settings(args)

    print("Media roots:")
    roots = benchmark_media_roots(args.benchmark_config, args.benchmark)
    roots.extend(args.media_root or [])
    media_paths = collect_media(roots)
    if args.limit:
        media_paths = media_paths[: args.limit]

    gpus = resolve_gpus(args.gpus)
    shards = shard(media_paths, gpus)

    print(
        f"\n{len(media_paths)} media files · strategy={settings.strategy.name} "
        f"· model={settings.strategy.model} · gpus={gpus} · threads/gpu={args.threads_per_gpu}"
    )
    print(f"cache: {settings.cache_dir}")
    if args.dry_run or not media_paths:
        return 0

    settings_raw = settings.to_dict()
    payloads = [(s, settings_raw, args.threads_per_gpu) for s in shards]

    if len(payloads) == 1:
        reports = [run_shard(payloads[0])]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=len(payloads)) as pool:
            reports = pool.map(run_shard, payloads)

    merged = merge_reports(reports)
    failures = [failure for report in reports for failure in report.get("failures", [])]
    summary = {
        "config_used": settings.to_dict(),
        "benchmarks": args.benchmark or sorted(MEDIA_ROOT_KEYS),
        "gpus": gpus,
        "threads_per_gpu": args.threads_per_gpu,
        "totals": {
            "total": merged.total,
            "cached": merged.cached,
            "transcribed": merged.transcribed,
            "failed": merged.failed,
        },
        "failures": failures,
    }

    report_path = args.report or (settings.cache_dir / "prepare_asr_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"\ntotal={merged.total} cached={merged.cached} "
        f"transcribed={merged.transcribed} failed={merged.failed}"
    )
    print(f"report: {report_path}")
    return 1 if merged.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
