"""잘린(max_tokens 도달) 레코드를 제거해 재개 시 다시 돌게 만든다.

각 어댑터는 레코드 파일의 키 집합으로 done 을 판단하므로(videomme/omnivideobench/
av_speakerbench=question_id, worldsense=worldsense_key, omnidcbench=clip_path),
파일에서 항목을 지우는 것이 유일한 재실행 트리거다. 원본은 ~/trash 로 백업한다.

잘림 판정 (순서대로):
1. 레코드에 ``finish_reason`` 이 있으면 ``== "length"`` 만 본다 (정확한 신호).
2. 없으면(구버전 레코드) ``completion_tokens >= max_tokens``. max_tokens 는
   ``--max-tokens`` > ``<bench>/config_used.json`` > 구 SPEC 기본값 순으로 정한다.
3. 응답 본문에 ``<think>`` 가 열렸는데 ``</think>`` 가 없으면 잘림.
4. OmniDCBench 는 ``prediction_json`` 이 null 인 건도 제거.

사용 예:
    # 새 런 디렉터리 (벤치 하위 디렉터리를 자동 탐색)
    python scripts/strip_truncated.py all --run-dir /gpfs/.../omni-bench/nemotron-3.5-super-vl-bf16-think --dry-run
    # 구 Qwen3.8 경로 (환경변수 OMNI_BENCH_THINK_RESULTS / OMNI_BENCH_VMME_RESULTS)
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
        return False  # 에러 건은 어댑터가 재개 시 알아서 재시도한다
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
                    help="런 디렉터리 (<run-dir>/<bench>/records.jsonl). 생략 시 구 Qwen3.8 경로.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="finish_reason 이 없는 레코드에 쓸 상한 (기본: config_used.json).")
    ap.add_argument("--report", type=Path, default=None, help="리포트 JSON 경로.")
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
            print(f"{name:16} 파일 없음 — 건너뜀 ({path})")
            continue
        recs = [json.loads(l) for l in path.open() if l.strip()]
        keep = [r for r in recs if not is_truncated(r, limit, name)]
        cut = len(recs) - len(keep)
        report[name] = {"path": str(path), "total": len(recs), "truncated": cut,
                        "kept": len(keep), "limit": limit}
        print(f"{name:16} 전체 {len(recs):5} · 잘림 {cut:4} ({cut*100/max(len(recs),1):.1f}%) · 유지 {len(keep):5}")
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
        print(f"{'':16} → 백업 {backup}")
    default_report = (args.run_dir / "truncation_report.json") if args.run_dir else Path(
        os.environ.get("OMNI_BENCH_TRUNCATION_REPORT", "truncation_report.json"))
    out = args.report or default_report
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
