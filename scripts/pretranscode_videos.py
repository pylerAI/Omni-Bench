"""Pre-build the base64-transport transcode cache for large videos.

The client transcodes on a cache miss too, but doing it inline stalls a worker
for minutes per long video. Running this ahead of an evaluation makes every
lookup a hit. Settings come from a model config's ``transcode:`` block so the
cache keys match what the client will look up.

    uv run --no-sync python scripts/pretranscode_videos.py \
        --config configs/models/nemotron_3_5_super_vl_bf16_nothink.yaml \
        --video-dir /gpfs/public/datasets/Video-MME/data \
        --video-dir /gpfs/public/datasets/OmniVideoBench/videos

Resumable: entries already in the cache are skipped. A JSON report is written
next to the cache (or to ``--report``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from omni_bench.video_transport import (  # noqa: E402
    VIDEO_SUFFIXES,
    TranscodeSettings,
    ensure_transcoded,
    needs_transcode,
    transcode_cache_path,
)


def load_settings(config_path: str | None, model_name: str | None) -> TranscodeSettings:
    if not config_path:
        return TranscodeSettings()
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    models = raw.get("models") or []
    if model_name:
        models = [m for m in models if m.get("name") == model_name]
    if not models:
        raise SystemExit(f"No model found in {config_path}")
    return TranscodeSettings.from_dict(models[0].get("transcode"))


def collect(video_dirs: list[str], files: list[str]) -> list[Path]:
    found: list[Path] = [Path(f) for f in files]
    for root in video_dirs:
        for dirpath, _, names in os.walk(root):
            found.extend(Path(dirpath) / n for n in names if n.lower().endswith(VIDEO_SUFFIXES))
    return sorted(set(found))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Model YAML whose transcode: block defines the cache settings.")
    ap.add_argument("--model", help="Model name inside --config (default: first).")
    ap.add_argument("--video-dir", action="append", default=[], help="Directory to scan. Repeatable.")
    ap.add_argument("--file", action="append", default=[], help="Single video file. Repeatable.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Parallel ffmpeg processes (default: nproc // threads, capped at 24).")
    ap.add_argument("--threads", type=int, default=None, help="x264 threads per job (default: from config).")
    ap.add_argument("--report", default=None, help="Report JSON path.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = load_settings(args.config, args.model)
    if args.threads:
        settings = replace(settings, threads=args.threads)
    jobs = args.jobs or max(1, min(24, (os.cpu_count() or 8) // settings.threads))

    videos = collect(args.video_dir, args.file)
    large = [p for p in videos if needs_transcode(p, settings)]
    todo = [p for p in large if not transcode_cache_path(p, settings).exists()]
    print(f"videos {len(videos)} · > {settings.threshold_mb}MB {len(large)} · "
          f"cached {len(large) - len(todo)} · to transcode {len(todo)} · jobs {jobs} × {settings.threads} threads")
    print(f"cache: {settings.cache_dir} (tag {settings.tag})")
    if args.dry_run or not todo:
        return

    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(ensure_transcoded, p, settings): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            src = futures[fut]
            try:
                out = fut.result()
                row = {"source": str(src), "output": str(out),
                       "source_mb": round(src.stat().st_size / 2**20, 1),
                       "output_mb": round(out.stat().st_size / 2**20, 1), "ok": True}
            except Exception as exc:  # keep going; the client retries on miss
                row = {"source": str(src), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(row)
            print(f"[{i}/{len(todo)}] {'OK ' if row['ok'] else 'ERR'} {src.name} "
                  f"{row.get('source_mb', '')} -> {row.get('output_mb', row.get('error', ''))}", flush=True)

    report = {
        "settings": settings.to_dict(),
        "video_dirs": args.video_dir,
        "videos": len(videos),
        "above_threshold": len(large),
        "transcoded": sum(r["ok"] for r in results),
        "failed": [r for r in results if not r["ok"]],
        "max_output_mb": max((r["output_mb"] for r in results if r["ok"]), default=None),
        "wall_s": round(time.perf_counter() - started, 1),
        "items": results,
    }
    out = Path(args.report or Path(settings.cache_dir) / f"pretranscode_report_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"done in {report['wall_s']}s · failed {len(report['failed'])} → {out}")


if __name__ == "__main__":
    main()
