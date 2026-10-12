"""Remove truncated records (hit max_tokens) so a resumed run retries them.

Each adapter decides what is done from the keys in its records file
(videomme / omnivideobench / av_speakerbench = question_id, worldsense =
worldsense_key, omnidcbench = clip_path), so deleting entries from that file is
the only way to trigger a re-run. The original file is backed up to ~/trash.

Truncation rule (in order):
1. If the record has ``finish_reason``, only ``== "length"`` counts (exact signal).
2. Otherwise (older records), ``completion_tokens >= max_tokens``, where
   max_tokens comes from ``--max-tokens`` > ``<bench>/config_used.json`` > the
   old SPEC default.
3. The response opens ``<think>`` but never closes it with ``</think>``.
4. OmniDCBench: records whose ``prediction_json`` is null are removed too.

Usage:
    # new run directory (benchmark subdirectories are discovered)
    python scripts/strip_truncated.py all --run-dir /gpfs/.../omni-bench/nemotron-3.5-super-vl-bf16-think --dry-run
    # old Qwen3.8 paths (env OMNI_BENCH_THINK_RESULTS / OMNI_BENCH_VMME_RESULTS)
    python scripts/strip_truncated.py all
"""
import argparse, json, os, shutil, time
from pathlib import Path

RESULTS = Path(os.environ.get("OMNI_BENCH_THINK_RESULTS", "results/qwen3.8-27b-whisper-srvthink"))
VMME = Path(os.environ.get("OMNI_BENCH_VMME_RESULTS", "results/qwen3.8-27b-srv-think"))
RECORD_FILE = {
    "videomme": "records.jsonl",
    "worldsense": "records.jsonl",
    "omnivideobench": "records.jsonl",
    "av_speakerbench": "records.jsonl",
    "omnidcbench": "predictions.jsonl",
}
LEGACY_LIMIT = {"omnidcbench": 16384}
DEFAULT_LIMIT = 8192


def legacy_spec(name: str) -> tuple[Path, int]:
    root = VMME if name == "videomme" else RESULTS
    return root / name / RECORD_FILE[name], LEGACY_LIMIT.get(name, DEFAULT_LIMIT)


def configured_max_tokens(bench_dir: Path, name: str) -> int | None:
    snap = bench_dir / "config_used.json"
    if not snap.exists():
        return None
    for bench in json.loads(snap.read_text()).get("benchmarks", []):
        if bench.get("name") == name and bench.get("max_tokens"):
            return int(bench["max_tokens"])
    return None


def is_truncated(rec: dict, limit: int | None, bench: str) -> bool:
    if rec.get("error"):
        return False  # error rows are retried by the adapter on resume anyway
    finish = rec.get("finish_reason")
    if finish is not None:
        if finish == "length":
            return True
    elif limit is not None and (rec.get("completion_tokens") or 0) >= limit:
        return True
    if bench == "omnidcbench" and rec.get("prediction_json") is None:
        return True
    resp = rec.get("response") or ""
    if "<think>" in resp and "</think>" not in resp:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmarks", nargs="+", choices=sorted(RECORD_FILE) + ["all"])
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run directory (<run-dir>/<bench>/records.jsonl). Default: the old Qwen3.8 paths.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max_tokens for records without finish_reason (default: config_used.json).")
    ap.add_argument("--report", type=Path, default=None, help="Report JSON path.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = sorted(RECORD_FILE) if "all" in args.benchmarks else args.benchmarks

    report = {}
    for name in names:
        if args.run_dir:
            path = args.run_dir / name / RECORD_FILE[name]
            limit = args.max_tokens or configured_max_tokens(path.parent, name)
        else:
            path, limit = legacy_spec(name)
            limit = args.max_tokens or limit
        if not path.exists():
            print(f"{name:16} file missing, skipped ({path})")
            continue
        recs = [json.loads(l) for l in path.open() if l.strip()]
        keep = [r for r in recs if not is_truncated(r, limit, name)]
        cut = len(recs) - len(keep)
        report[name] = {"path": str(path), "total": len(recs), "truncated": cut,
                        "kept": len(keep), "limit": limit}
        print(f"{name:16} total {len(recs):5} · truncated {cut:4} ({cut*100/max(len(recs),1):.1f}%) · kept {len(keep):5}")
        if args.dry_run or cut == 0:
            continue
        trash = Path(os.path.expanduser("~/trash"))
        trash.mkdir(parents=True, exist_ok=True)
        tag = path.parent.parent.name
        backup = trash / f"{tag}_{name}_{path.stem}.{int(time.time())}{path.suffix}"
        shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"{'':16} -> backup {backup}")
    default_report = (args.run_dir / "truncation_report.json") if args.run_dir else Path(
        os.environ.get("OMNI_BENCH_TRUNCATION_REPORT", "truncation_report.json"))
    out = args.report or default_report
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
