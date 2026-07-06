#!/usr/bin/env python3
"""Standalone entry point for the HTML results report.

``omni-bench run`` generates the report automatically; use this to regenerate it
from existing ``results/`` without re-running any benchmark.

    uv run python scripts/build_report.py [--results DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omni_bench.report import render_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = render_report(Path(args.results), Path(args.out) if args.out else None)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
