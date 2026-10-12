"""Re-score multiple-choice records from saved responses.

The official extractor takes the first A-D character anywhere in the reply, so a
prose answer beginning "Based on..." is read as B. This applies a stricter rule
and reports both numbers side by side. Pure post-processing — no inference.
"""
import json, re, sys, collections
from pathlib import Path

LETTERS = "ABCD"
# Explicit answer markers, most specific first.
PATTERNS = [
    re.compile(r"(?:final answer|best answer|correct answer|answer)\s*(?:is)?\s*[:\-]?\s*\(?\*{0,2}([ABCD])\*{0,2}\)?", re.I),
    re.compile(r"\*\*\(?([ABCD])\)?\*\*"),
    re.compile(r"\(([ABCD])\)"),
    re.compile(r"(?:^|\n)\s*([ABCD])\s*[.):]"),
]
STANDALONE = re.compile(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])")


def robust_extract(response: str | None, options: list[str] | None = None) -> str:
    text = (response or "").strip()
    if not text:
        return ""
    if len(text) <= 3:                      # bare "B" / "B." / "(B"
        m = STANDALONE.search(text)
        if m:
            return m.group(1)
    for pat in PATTERNS:                    # explicit markers: take the LAST one
        hits = pat.findall(text)
        if hits:
            return hits[-1].upper()
    hits = STANDALONE.findall(text)         # else last standalone letter
    if hits:
        return hits[-1].upper()
    if options:                             # else match option text
        low = text.lower()
        for opt in options:
            body = re.sub(r"^\s*[ABCD]\s*[.):]?\s*", "", str(opt)).strip().lower()
            if body and body in low:
                return str(opt).strip()[0].upper()
    return ""


def main(path: Path, answer_key="answer", options_key=None):
    recs = json.load(open(path)) if path.suffix == ".json" else [json.loads(l) for l in open(path)]
    off_ok = rob_ok = 0
    changed = collections.Counter()
    by_len = {"short": [0, 0, 0], "long": [0, 0, 0]}
    for r in recs:
        gold = str(r.get(answer_key) or "").strip()
        official = str(r.get("parsed_answer") or "").strip()
        robust = robust_extract(r.get("response"), r.get(options_key) if options_key else None)
        o, b = official == gold, robust == gold
        off_ok += o; rob_ok += b
        if o != b:
            changed[("고침" if b else "망침")] += 1
        bucket = "long" if (r.get("completion_tokens") or 0) > 100 else "short"
        by_len[bucket][0] += 1; by_len[bucket][1] += o; by_len[bucket][2] += b
    n = len(recs)
    print(f"{path.parent.name}: n={n}")
    print(f"  official parser : {off_ok/n*100:.2f}%  ({off_ok})")
    print(f"  robust  parser  : {rob_ok/n*100:.2f}%  ({rob_ok})   차이 {(rob_ok-off_ok)/n*100:+.2f}")
    print(f"  변경: {dict(changed)}")
    for k, (cnt, o, b) in by_len.items():
        if cnt:
            print(f"  {k:5s} n={cnt:5d}  official {o/cnt*100:5.2f}%  robust {b/cnt*100:5.2f}%")


if __name__ == "__main__":
    main(Path(sys.argv[1]), options_key=sys.argv[2] if len(sys.argv) > 2 else None)
