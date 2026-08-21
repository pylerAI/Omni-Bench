"""잘린(max_tokens 도달) 레코드를 제거해 재개 시 다시 돌게 만든다.

각 어댑터는 레코드 파일의 키 집합으로 done 을 판단하므로(videomme/omnivideobench/
av_speakerbench=question_id, worldsense=worldsense_key, omnidcbench=clip_path),
파일에서 항목을 지우는 것이 유일한 재실행 트리거다. 원본은 ~/trash 로 백업한다.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

RESULTS = Path(os.environ.get("OMNI_BENCH_THINK_RESULTS", "results/qwen3.8-27b-whisper-srvthink"))
VMME = Path(os.environ.get("OMNI_BENCH_VMME_RESULTS", "results/qwen3.8-27b-srv-think"))
# 벤치 → (레코드 파일, 원래 max_tokens)
SPEC = {
    "videomme":        (VMME / "videomme" / "records.jsonl", 8192),
    "worldsense":      (RESULTS / "worldsense" / "records.jsonl", 8192),
    "omnivideobench":  (RESULTS / "omnivideobench" / "records.jsonl", 8192),
    "av_speakerbench": (RESULTS / "av_speakerbench" / "records.jsonl", 8192),
    "omnidcbench":     (RESULTS / "omnidcbench" / "predictions.jsonl", 16384),
}


def is_truncated(rec: dict, limit: int, bench: str) -> bool:
    if (rec.get("completion_tokens") or 0) >= limit:
        return True
    # 캡셔닝은 JSON 이 안 뽑히면 사실상 유실 — 잘림과 동일하게 취급한다.
    if bench == "omnidcbench" and rec.get("prediction_json") is None and not rec.get("error"):
        return True
    # thinking 런은 </think> 를 닫지 못하면 최종 답이 없다. completion_tokens 임계값만
    # 보면 어댑터가 다른 상한을 쓰는 경우(omnivideobench 의 1024)를 놓치므로 함께 본다.
    resp = rec.get("response") or ""
    if "<think>" in resp or "</think>" in resp:
        if "</think>" not in resp:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="+", choices=sorted(SPEC) + ["all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    names = sorted(SPEC) if "all" in args.benchmarks else args.benchmarks

    report = {}
    for name in names:
        path, limit = SPEC[name]
        if not path.exists():
            print(f"{name:16} 파일 없음 — 건너뜀 ({path})")
            continue
        recs = [json.loads(l) for l in path.open() if l.strip()]
        keep = [r for r in recs if not is_truncated(r, limit, name)]
        cut = len(recs) - len(keep)
        report[name] = {"total": len(recs), "truncated": cut, "kept": len(keep), "limit": limit}
        print(f"{name:16} 전체 {len(recs):5} · 잘림 {cut:4} ({cut*100/max(len(recs),1):.1f}%) · 유지 {len(keep):5}")
        if args.dry_run or cut == 0:
            continue
        backup = Path(os.path.expanduser("~/trash")) / f"{name}_srvthink_records.{int(time.time())}.jsonl"
        shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"{'':16} → 백업 {backup.name}")
    out = Path(os.environ.get("OMNI_BENCH_TRUNCATION_REPORT", "truncation_report.json"))
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
