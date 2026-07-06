#!/usr/bin/env python3
"""Render benchmark results into a self-contained HTML report.

Reads ``results/<model>/<benchmark>/summary.json`` and emits ``results/report.html``:
headline stat tiles (single model) or a leaderboard table (multiple models), an
overall cross-benchmark accuracy chart, and per-benchmark paper-style (booktabs)
tables — one per dimension the adapters aggregate — that fold open/closed. The
output is a single file with inline CSS/SVG; no external assets.

``omni-bench run`` calls :func:`render_report` automatically; ``scripts/build_report.py``
exposes it as a standalone command.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
from pathlib import Path
from typing import Any

# ---- presentation metadata -------------------------------------------------

BENCH_LABEL = {
    "av_speakerbench": "AV-SpeakerBench",
    "worldsense": "WorldSense",
    "videomme": "Video-MME",
    "omnivideobench": "OmniVideoBench",
    "omnidcbench": "OmniDCBench",
}
BENCH_ORDER = ["av_speakerbench", "worldsense", "videomme", "omnivideobench", "omnidcbench"]
# captioning benchmarks report generation metrics rather than accuracy.
CAPTION_BENCH = {"omnidcbench"}

DIM_LABEL = {
    "by_category": "Category", "by_sub_category": "Sub-category", "by_task_id": "Task",
    "by_domain": "Domain", "by_task_domain": "Task domain", "by_task_type": "Task type",
    "by_duration": "Duration", "by_video_type": "Video type",
    "by_question_type": "Question type", "by_audio_type": "Audio type",
}
ORDERED_DIMS = {"by_duration"}  # intrinsic order — do not sort by score
LEAD_DIM = {"av_speakerbench": "by_category", "worldsense": "by_task_domain",
            "omnivideobench": "by_question_type"}
METRIC_LABEL = {"f1": "F1", "miou": "mIoU", "soda_m": "SODA-m",
                "precision_mean": "Precision", "recall_mean": "Recall"}

# categorical series colours (light, dark) — from the data-viz reference palette,
# fixed order, never cycled. Used only when comparing multiple models.
SERIES = [("#2a78d6", "#3987e5"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
          ("#008300", "#008300"), ("#4a3aa7", "#9085e9"), ("#e34948", "#e66767"),
          ("#e87ba4", "#d55181"), ("#eb6834", "#d95926")]


def esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def pct(v: float) -> str:
    return f"{v:.2f}"


# ---- data loading ----------------------------------------------------------

def load_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """{model: {benchmark: summary_dict}} for every summary.json found."""
    out: dict[str, dict[str, dict]] = {}
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if model_dir.name == "logs":
            continue
        benches: dict[str, dict] = {}
        for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            summary = bench_dir / "summary.json"
            if summary.exists():
                try:
                    benches[bench_dir.name] = json.loads(summary.read_text())
                except json.JSONDecodeError:
                    continue
        if benches:
            out[model_dir.name] = benches
    return out


def dimensions(summary: dict) -> list[tuple[str, dict]]:
    """Unique ``by_*`` dimensions, de-duplicating identical breakdowns."""
    seen: list[str] = []
    dims: list[tuple[str, dict]] = []
    for key, val in summary.items():
        if not (key.startswith("by_") and isinstance(val, dict) and val):
            continue
        sig = json.dumps(val, sort_keys=True)
        if sig in seen:
            continue
        seen.append(sig)
        dims.append((key, val))
    return dims


def headline(bench: str, summary: dict) -> tuple[float | None, str]:
    """(value, unit) — accuracy % for MCQ benches, F1 (0-100) for captioning."""
    if bench in CAPTION_BENCH:
        f1 = (summary.get("metrics") or {}).get("f1")
        return (None, "F1") if f1 is None else (f1 * 100 if f1 <= 1 else f1, "F1")
    acc = summary.get("accuracy")
    return (acc, "%") if acc is not None else (None, "%")


# ---- rendering: charts & tables --------------------------------------------

def comparison_chart(labels: list[str], models: list[str],
                     matrix: list[list[float | None]]) -> str:
    """Grouped horizontal accuracy bars. rows = benchmarks, groups = models."""
    if not labels:
        return "<p class='pending'>No accuracy benchmarks to compare yet.</p>"
    multi = len(models) > 1
    bar_h = 20 if multi else 32
    inner, gap, top, left, right, bottom = 4, 18, 8, 168, 52, 26
    width = 760
    plot_w = width - left - right
    group_h = len(models) * bar_h + (len(models) - 1) * inner
    height = top + bottom + len(labels) * group_h + (len(labels) - 1) * gap

    legend = ""
    if multi:
        chips = "".join(
            f"<span class='lg'><span class='sw s{i % 8}'></span>{esc(m)}</span>"
            for i, m in enumerate(models))
        legend = f"<div class='legend'>{chips}</div>"

    p = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' "
         f"aria-label='Overall accuracy by benchmark'>"]
    for t in range(0, 101, 25):
        x = left + plot_w * t / 100
        p.append(f"<line class='grid' x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height - bottom}'/>")
        p.append(f"<text class='tick' x='{x:.1f}' y='{height - bottom + 16}' text-anchor='middle'>{t}</text>")
    for r, label in enumerate(labels):
        gy = top + r * (group_h + gap)
        p.append(f"<text class='cat' x='{left - 12}' y='{gy + group_h / 2:.1f}' "
                 f"text-anchor='end' dominant-baseline='central'>{esc(label)}</text>")
        for m_i, _model in enumerate(models):
            val = matrix[r][m_i]
            y = gy + m_i * (bar_h + inner)
            p.append(f"<rect class='track' x='{left}' y='{y}' width='{plot_w}' height='{bar_h}' rx='4'/>")
            if val is None:
                continue
            bw = plot_w * val / 100
            cls = f"fill s{m_i % 8}" if multi else "fill"
            p.append(f"<rect class='{cls}' x='{left}' y='{y}' width='{bw:.1f}' height='{bar_h}' rx='4'>"
                     f"<title>{esc(models[m_i])} · {esc(label)}: {pct(val)}%</title></rect>")
            p.append(f"<text class='val' x='{left + bw + 8:.1f}' y='{y + bar_h / 2:.1f}' "
                     f"dominant-baseline='central'>{pct(val)}</text>")
    p.append("</svg>")
    return legend + "".join(p)


def dim_table(dim_key: str, per_model: dict[str, dict], models: list[str]) -> str:
    """Booktabs table for one dimension. Single model → in-row bar; multi → model
    columns with the best value per row in bold."""
    label = DIM_LABEL.get(dim_key, dim_key.replace("by_", "").replace("_", " ").title())
    # union of categories, preserving first-seen order
    cats: list[str] = []
    for m in models:
        for c in (per_model.get(m) or {}):
            if c not in cats and isinstance(per_model[m][c], dict):
                cats.append(c)

    def acc_of(m: str, c: str) -> float | None:
        st = (per_model.get(m) or {}).get(c)
        return st.get("accuracy") if isinstance(st, dict) else None

    if dim_key not in ORDERED_DIMS:
        def keyfn(c: str) -> float:
            vals = [v for v in (acc_of(m, c) for m in models) if v is not None]
            return sum(vals) / len(vals) if vals else -1
        cats.sort(key=keyfn, reverse=True)

    multi = len(models) > 1
    if multi:
        head = "".join(f"<th class='num'>{esc(m)}</th>" for m in models)
        rows = []
        for c in cats:
            vals = {m: acc_of(m, c) for m in models}
            best = max((v for v in vals.values() if v is not None), default=None)
            tds = []
            for m in models:
                v = vals[m]
                cell = "—" if v is None else pct(v)
                strong = v is not None and best is not None and abs(v - best) < 1e-9
                tds.append(f"<td class='num{' best' if strong else ''}'>{cell}</td>")
            rows.append(f"<tr><td class='cat'>{esc(c)}</td>{''.join(tds)}</tr>")
        table = (f"<table class='paper'><thead><tr><th class='cat'>{esc(label)}</th>{head}</tr></thead>"
                 f"<tbody>{''.join(rows)}</tbody></table>")
    else:
        m = models[0]
        rows = []
        for c in cats:
            st = (per_model.get(m) or {}).get(c) or {}
            acc = st.get("accuracy", 0.0)
            rows.append(
                f"<tr><td class='cat'>{esc(c)}</td>"
                f"<td class='barcell'><span class='track'><span class='fill' style='width:{acc:.2f}%'></span></span></td>"
                f"<td class='num'>{pct(acc)}</td>"
                f"<td class='n'>{st.get('correct', 0):,}/{st.get('total', 0):,}</td></tr>")
        table = (f"<table class='paper'><thead><tr><th class='cat'>{esc(label)}</th><th></th>"
                 f"<th class='num'>Acc</th><th class='n'>n</th></tr></thead>"
                 f"<tbody>{''.join(rows)}</tbody></table>")
    return f"<figure class='panel'><figcaption class='panel-title'>{esc(label)}</figcaption>{table}</figure>"


def caption_panel(per_model: dict[str, dict], models: list[str]) -> str:
    order = ["f1", "miou", "soda_m", "precision_mean", "recall_mean"]
    present = [k for k in order if any(k in (per_model.get(m) or {}).get("metrics", {}) for m in models)]
    if not present:
        return ("<figure class='panel'><figcaption class='panel-title'>Caption metrics</figcaption>"
                "<p class='pending'>No finalized metrics yet — run still in progress.</p></figure>")
    head = "".join(f"<th class='num'>{esc(m)}</th>" for m in models)
    rows = []
    for k in present:
        tds = []
        for m in models:
            v = (per_model.get(m) or {}).get("metrics", {}).get(k)
            tds.append(f"<td class='num'>{'—' if v is None else pct(v * 100 if v <= 1 else v)}</td>")
        rows.append(f"<tr><td class='cat'>{esc(METRIC_LABEL.get(k, k))}</td>{''.join(tds)}</tr>")
    table = (f"<table class='paper compact'><thead><tr><th class='cat'>Metric</th>{head}</tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table>")
    return f"<figure class='panel'><figcaption class='panel-title'>Caption metrics</figcaption>{table}</figure>"


def stat_tiles(models: list[str], benches_present: list[str],
               data: dict[str, dict[str, dict]]) -> str:
    """Single-model: one tile per benchmark. Multi-model: leaderboard table."""
    if len(models) == 1:
        m = models[0]
        tiles = []
        for b in benches_present:
            s = data[m][b]
            val, unit = headline(b, s)
            big = "—" if val is None else pct(val)
            unit_html = f"<span class='unit'>{esc(unit)}</span>" if val is not None and unit == "%" else ""
            if b in CAPTION_BENCH:
                mm = s.get("metrics") or {}
                sub = (f"mIoU {pct((mm.get('miou') or 0) * 100)} · {mm.get('num_videos', '?')} clips"
                       if mm else "captioning · pending")
                name = "F1"
            else:
                sub = f"{s.get('correct', 0):,} / {s.get('total', 0):,} correct"
                name = "Accuracy"
            tiles.append(
                f"<div class='tile'><div class='tile-label'>{esc(BENCH_LABEL.get(b, b))}</div>"
                f"<div class='tile-value'>{big}{unit_html}</div>"
                f"<div class='tile-sub'><span class='metric-name'>{name}</span>{esc(sub)}</div></div>")
        return f"<section class='tiles'>{''.join(tiles)}</section>"

    # multi-model leaderboard
    head = "".join(f"<th class='num'>{esc(BENCH_LABEL.get(b, b))}</th>" for b in benches_present)
    body = []
    col_best = {b: max((headline(b, data[m][b])[0] for m in models
                        if b in data[m] and headline(b, data[m][b])[0] is not None), default=None)
                for b in benches_present}
    for m in models:
        tds = []
        for b in benches_present:
            val = headline(b, data[m][b])[0] if b in data[m] else None
            cell = "—" if val is None else pct(val)
            strong = val is not None and col_best[b] is not None and abs(val - col_best[b]) < 1e-9
            tds.append(f"<td class='num{' best' if strong else ''}'>{cell}</td>")
        body.append(f"<tr><td class='cat'>{esc(m)}</td>{''.join(tds)}</tr>")
    units = "".join(f"<th class='unit-row'>{'F1' if b in CAPTION_BENCH else 'Acc %'}</th>"
                    for b in benches_present)
    return (f"<section class='leaderboard'><div class='section-label'>Leaderboard</div>"
            f"<div class='chart-wrap'><table class='paper'><thead>"
            f"<tr><th class='cat'>Model</th>{head}</tr>"
            f"<tr><th></th>{units}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>")


# ---- page assembly ---------------------------------------------------------

def build_html(data: dict[str, dict[str, dict]], generated: str) -> str:
    models = list(data.keys())
    benches_present = ([b for b in BENCH_ORDER if any(b in data[m] for m in models)]
                       + [b for m in models for b in data[m] if b not in BENCH_ORDER])
    benches_present = list(dict.fromkeys(benches_present))

    # cross-benchmark accuracy comparison (accuracy benches only — single axis)
    acc_benches = [b for b in benches_present if b not in CAPTION_BENCH]
    labels = [BENCH_LABEL.get(b, b) for b in acc_benches]
    matrix = [[(data[m][b].get("accuracy") if b in data[m] else None) for m in models]
              for b in acc_benches]
    comparison = comparison_chart(labels, models, matrix)

    tiles = stat_tiles(models, benches_present, data)

    sections = []
    for b in benches_present:
        bmodels = [m for m in models if b in data[m]]
        per_model = {m: data[m][b] for m in bmodels}
        val, unit = headline(b, per_model[bmodels[0]]) if len(bmodels) == 1 else (None, "")
        if len(bmodels) == 1 and val is not None:
            hl = f"<span class='headline'>{pct(val)}{'%' if unit == '%' else ' ' + unit}</span><span class='sep'>·</span>"
        else:
            hl = ""
        kind = "captioning" if b in CAPTION_BENCH else "accuracy"
        total = per_model[bmodels[0]].get("total") or \
            (per_model[bmodels[0]].get("metrics") or {}).get("num_videos")
        meta = f"{hl}<span>{esc(kind)}</span>"
        if total:
            meta += f"<span class='sep'>·</span><span>{int(total):,} samples</span>"
        if len(bmodels) > 1:
            meta += f"<span class='sep'>·</span><span>{len(bmodels)} models</span>"

        if b in CAPTION_BENCH:
            panels = caption_panel(per_model, bmodels)
        else:
            dim_keys: list[str] = []
            for m in bmodels:
                for k, _ in dimensions(data[m][b]):
                    if k not in dim_keys:
                        dim_keys.append(k)
            dim_keys.sort(key=lambda k: (k != LEAD_DIM.get(b), k))
            panels = "".join(
                dim_table(k, {m: (data[m][b].get(k) or {}) for m in bmodels}, bmodels)
                for k in dim_keys)

        sections.append(
            f"<section class='bench'><details open>"
            f"<summary><span class='disc'></span>"
            f"<h2>{esc(BENCH_LABEL.get(b, b))}</h2>"
            f"<span class='bench-meta'>{meta}</span></summary>"
            f"<div class='panels'>{panels}</div></details></section>")

    model_line = models[0] if len(models) == 1 else f"{len(models)} models"
    return (PAGE
            .replace("__MODEL__", esc(model_line))
            .replace("__NBENCH__", str(len(benches_present)))
            .replace("__GENERATED__", esc(generated))
            .replace("__TILES__", tiles)
            .replace("__COMPARISON__", comparison)
            .replace("__SECTIONS__", "".join(sections)))


PAGE = """<div class="report viz-root">
<header class="masthead">
  <div class="eyebrow">Omni-modal evaluation</div>
  <h1>Omni-Bench Results</h1>
  <p class="dek">__MODEL__ · __NBENCH__ benchmarks · generated __GENERATED__</p>
</header>

__TILES__

<section class="comparison">
  <div class="section-label">Overall accuracy</div>
  <div class="chart-wrap">__COMPARISON__</div>
</section>

__SECTIONS__

<footer class="foot">Generated by <code>scripts/build_report.py</code> from <code>results/&lt;model&gt;/&lt;benchmark&gt;/summary.json</code>.</footer>
</div>

<style>
.report {
  --plane:#f6f5f2; --surface:#ffffff; --ink:#14130f; --ink-2:#57564f; --muted:#8a887f;
  --rule:#1c1b17; --hair:#e4e2da; --grid:#eceae3; --track:#eef0f3;
  --accent:#2a6fd0; --accent-soft:rgba(42,111,208,0.14);
  --s0:#2a78d6; --s1:#1baf7a; --s2:#eda100; --s3:#008300;
  --s4:#4a3aa7; --s5:#e34948; --s6:#e87ba4; --s7:#eb6834;
  --serif:"Iowan Old Style",Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--plane); font-family:var(--sans);
  max-width:1080px; margin:0 auto; padding:56px 28px 72px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
@media (prefers-color-scheme:dark){ .report{
  --plane:#100f0d; --surface:#1a1917; --ink:#f4f3ee; --ink-2:#c3c1b7; --muted:#8f8d84;
  --rule:#e8e6df; --hair:#2c2b27; --grid:#242320; --track:#26251f;
  --accent:#4f95ec; --accent-soft:rgba(79,149,236,0.20);
  --s0:#3987e5; --s1:#199e70; --s2:#c98500; --s3:#008300;
  --s4:#9085e9; --s5:#e66767; --s6:#d55181; --s7:#d95926; } }
:root[data-theme="dark"] .report{
  --plane:#100f0d; --surface:#1a1917; --ink:#f4f3ee; --ink-2:#c3c1b7; --muted:#8f8d84;
  --rule:#e8e6df; --hair:#2c2b27; --grid:#242320; --track:#26251f;
  --accent:#4f95ec; --accent-soft:rgba(79,149,236,0.20);
  --s0:#3987e5; --s1:#199e70; --s2:#c98500; --s3:#008300;
  --s4:#9085e9; --s5:#e66767; --s6:#d55181; --s7:#d95926; }
:root[data-theme="light"] .report{
  --plane:#f6f5f2; --surface:#ffffff; --ink:#14130f; --ink-2:#57564f; --muted:#8a887f;
  --rule:#1c1b17; --hair:#e4e2da; --grid:#eceae3; --track:#eef0f3;
  --accent:#2a6fd0; --accent-soft:rgba(42,111,208,0.14);
  --s0:#2a78d6; --s1:#1baf7a; --s2:#eda100; --s3:#008300;
  --s4:#4a3aa7; --s5:#e34948; --s6:#e87ba4; --s7:#eb6834; }

.report h1,.report h2{ font-family:var(--serif); font-weight:600; letter-spacing:-0.01em; text-wrap:balance; }
.masthead{ border-bottom:2px solid var(--rule); padding-bottom:20px; margin-bottom:28px; }
.eyebrow{ font-size:12px; letter-spacing:0.14em; text-transform:uppercase; color:var(--accent); font-weight:600; }
.masthead h1{ font-size:38px; margin:6px 0 8px; }
.dek{ color:var(--ink-2); font-size:15px; margin:0; }

.tiles{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:14px; margin-bottom:34px; }
.tile{ background:var(--surface); border:1px solid var(--hair); border-radius:12px; padding:16px 18px; }
.tile-label{ font-size:12.5px; font-weight:600; color:var(--ink-2); }
.tile-value{ font-size:34px; font-weight:650; margin:6px 0 4px; font-variant-numeric:tabular-nums; }
.tile-value .unit{ font-size:17px; color:var(--muted); margin-left:2px; }
.tile-sub{ font-size:12px; color:var(--muted); }
.metric-name{ display:inline-block; margin-right:6px; padding:1px 6px; border-radius:5px;
  background:var(--accent-soft); color:var(--accent); font-weight:600; }

.section-label,.panel-title{ font-size:12px; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }
.comparison{ margin-bottom:8px; }
.chart-wrap{ background:var(--surface); border:1px solid var(--hair); border-radius:12px;
  padding:16px 18px 8px; margin-top:10px; overflow-x:auto; }
.legend{ display:flex; flex-wrap:wrap; gap:14px; padding:2px 2px 12px; font-size:12.5px; color:var(--ink-2); }
.lg{ display:inline-flex; align-items:center; gap:6px; }
.sw{ width:11px; height:11px; border-radius:3px; display:inline-block; }
.sw.s0{background:var(--s0)}.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}.sw.s3{background:var(--s3)}
.sw.s4{background:var(--s4)}.sw.s5{background:var(--s5)}.sw.s6{background:var(--s6)}.sw.s7{background:var(--s7)}
.chart{ width:100%; height:auto; display:block; }
.chart .grid{ stroke:var(--grid); stroke-width:1; }
.chart .track{ fill:var(--track); }
.chart .fill{ fill:var(--accent); }
.chart .fill.s0{fill:var(--s0)}.chart .fill.s1{fill:var(--s1)}.chart .fill.s2{fill:var(--s2)}.chart .fill.s3{fill:var(--s3)}
.chart .fill.s4{fill:var(--s4)}.chart .fill.s5{fill:var(--s5)}.chart .fill.s6{fill:var(--s6)}.chart .fill.s7{fill:var(--s7)}
.chart .cat{ fill:var(--ink); font-size:13px; }
.chart .val{ fill:var(--ink); font-size:12px; font-weight:600; font-variant-numeric:tabular-nums; }
.chart .tick{ fill:var(--muted); font-size:11px; }

.bench{ margin-top:22px; }
.bench details{ background:var(--surface); border:1px solid var(--hair); border-radius:12px; }
.bench summary{ list-style:none; cursor:pointer; display:flex; align-items:baseline; gap:12px;
  padding:16px 18px; flex-wrap:wrap; }
.bench summary::-webkit-details-marker{ display:none; }
.bench summary:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:12px; }
.disc{ width:0; height:0; align-self:center; border-left:6px solid var(--muted);
  border-top:5px solid transparent; border-bottom:5px solid transparent;
  transition:transform 0.15s ease; flex:none; }
.bench details[open] .disc{ transform:rotate(90deg); }
.bench summary h2{ font-size:22px; margin:0; }
.bench-meta{ font-size:13px; color:var(--muted); display:inline-flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
.bench-meta .headline{ font-size:15px; font-weight:650; color:var(--accent); font-variant-numeric:tabular-nums; }
.bench-meta .sep{ opacity:0.45; }
.panels{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:22px 30px;
  align-items:start; padding:4px 18px 22px; }
.panel{ margin:0; }
.panel-title{ margin-bottom:8px; }

table.paper{ width:100%; border-collapse:collapse; font-size:13px; }
table.paper thead th{ font-weight:600; color:var(--ink-2); text-align:left; padding:0 10px 6px;
  border-bottom:1.5px solid var(--rule); font-size:11.5px; letter-spacing:0.04em; text-transform:uppercase; }
table.paper thead th.num{ text-align:right; }
table.paper thead th.unit-row{ text-align:right; border-bottom:1px solid var(--hair);
  padding-top:2px; font-size:10px; color:var(--muted); font-weight:500; text-transform:none; letter-spacing:0; }
table.paper tbody td{ padding:6px 10px; border-bottom:1px solid var(--hair); vertical-align:middle; }
table.paper tbody tr:last-child td{ border-bottom:1.5px solid var(--rule); }
table.paper td.cat{ color:var(--ink); max-width:200px; }
table.paper td.num{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
table.paper td.num.best{ color:var(--accent); }
table.paper td.n{ text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); font-size:11.5px; width:80px; }
.barcell{ width:38%; padding-left:0 !important; }
.barcell .track{ display:block; height:8px; border-radius:4px; background:var(--track); overflow:hidden; }
.barcell .fill{ display:block; height:100%; border-radius:4px; background:var(--accent); }
.pending{ color:var(--muted); font-size:13px; font-style:italic; margin:4px 0 0; }

.leaderboard{ margin-bottom:20px; }
.foot{ margin-top:40px; padding-top:16px; border-top:1px solid var(--hair); color:var(--muted); font-size:12px; }
.foot code{ font-size:11.5px; background:var(--surface); border:1px solid var(--hair); padding:1px 5px; border-radius:4px; }
@media (prefers-reduced-motion:reduce){ .disc{ transition:none; } }
@media (max-width:560px){ .masthead h1{ font-size:30px; } .barcell{ display:none; } }
</style>
"""


def render_report(results_dir: Path, out: Path | None = None) -> Path:
    """Build the HTML report; returns the output path. Safe to call from the CLI."""
    out = out or (results_dir / "report.html")
    data = load_results(results_dir)
    if not data:
        raise ValueError(f"No summary.json found under {results_dir}")
    out.write_text(build_html(data, _dt.date.today().isoformat()), encoding="utf-8")
    return out
