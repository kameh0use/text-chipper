#!/usr/bin/env python3
"""Render the four README charts from a chunkbench results.json.

    pip install matplotlib
    python benchmarks/make_charts.py benchmarks/results/<run>/results.json benchmarks/charts
    python benchmarks/make_charts.py benchmarks/results/<run> benchmarks/charts --svg

Charts (all for one lane, `single` by default, overlap 0):

    speed_overall.png       geometric-mean throughput over all workloads
    speed_per_workload.png  MiB/s for every library on every workload (annotated heat map)
    memory_single_core.png  mean peak memory of the whole Python process, split in two parts
    quality_scorecard.png   worst-case / average quality numbers per library

Every number is read from results.json; nothing is typed in by hand. The environment line at the
bottom of each chart (CPU, Python, rounds, commit) comes from the same file. These charts are the
README summary; `python -m chunkbench plot <results dir>` draws the detailed per-metric charts.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------- style

BLUE, BLUE_SOFT = "#2563eb", "#9db9f2"      # text_chipper, text_chipper (flat)
GREY, GREY_SOFT = "#8d8c86", "#cfcec8"      # everything else
INK, INK2, INK3 = "#14130f", "#5d5c57", "#8d8c86"
GRID, BG = "#e6e5e0", "#ffffff"

SHORT = {
    "text_chipper": "text_chipper",
    "text_chipper (flat)": "text_chipper\n(flat)",
    "tiktoken window": "tiktoken\nwindow",
    "chonkie TokenChunker": "chonkie\nToken",
    "chonkie RecursiveChunker": "chonkie\nRecursive",
    "semchunk": "semchunk",
    "LangChain": "LangChain",
    "semantic-text-splitter": "semantic-\ntext-splitter",
}
WORKLOAD_ORDER = [
    "prose_en", "markdown_doc", "code_heavy", "multilingual",
    "real_pubmed", "real_finance", "real_wikitexts", "real_chatlogs", "real_state_of_the_union",
    "stress_single_line", "stress_giant_list", "stress_cjk_unbroken",
]
KIND_ORDER = {"synthetic": 0, "real": 1, "stress": 2}
KIND_TITLE = {"synthetic": "SYNTHETIC", "real": "REAL-WORLD", "stress": "STRESS"}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "svg.fonttype": "path",
        "axes.edgecolor": GRID,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    })
    return plt


# --------------------------------------------------------------------------- data helpers


def load(path: Path) -> dict:
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def color_of(lib: str, soft: bool = False) -> str:
    if lib == "text_chipper":
        return BLUE_SOFT if soft else BLUE
    if lib.startswith("text_chipper"):
        return BLUE_SOFT
    return GREY_SOFT if soft else GREY


def size_label(mib: float) -> str:
    return f"{mib:.1f} MiB" if mib >= 0.95 else f"{mib * 1024:.0f} KiB"


def pct_text(v: float, digits: int = 1) -> str:
    """Percent with `digits` decimals; a value below 100 is never rounded up to 100."""
    if v is None or math.isnan(v):
        return "–"
    if v >= 100 - 1e-9:
        return "100%"
    r = round(v, digits)
    if r >= 100:
        r = 100 - 10 ** -digits
    return f"{r:.{digits}f}%"


def axes_in(fig, left: float, bottom: float, width: float, height: float):
    """Axes placed in inches (from the figure's bottom-left corner)."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def footer(meta: dict, lane: str) -> str:
    cpu = re.sub(r"\s*\d+-Core Processor", "", meta.get("cpu", "unknown CPU")).replace(" Processor", "")
    args = meta.get("args", {})
    lane_txt = "single-core lane" if lane == "single" else "all-cores lane"
    return (
        f"{cpu} ({meta.get('logical_cpus')} logical CPUs) · Python {meta.get('python')} · {lane_txt} · "
        f"median of {args.get('rounds')} cold runs · {args.get('chunk_size', 512)}-token chunks, "
        f"{meta.get('reference_tokenizer', 'cl100k_base')} · chunkbench {meta.get('chunkbench')} · "
        f"commit {meta.get('project_commit') or 'n/a'}"
    )


class Data:
    def __init__(self, raw: dict, lane: str, overlap: int = 0) -> None:
        self.meta = raw["meta"]
        self.lane = lane
        cells = [c for c in raw["cells"] if c["status"] == "ok" and c["lane"] == lane and c["overlap"] == overlap]
        if not cells:
            sys.exit(f"no successful cells for lane={lane!r}, overlap={overlap}")
        self.cells = cells
        self.libs_all = list(dict.fromkeys(c["library"] for c in cells))
        wls = list(dict.fromkeys(c["workload"] for c in cells))
        kind = {c["workload"]: c["input_kind"] for c in cells}
        self.size = {c["workload"]: c["input_mib"] for c in cells}
        self.kind = kind
        self.workloads = sorted(
            wls,
            key=lambda w: (KIND_ORDER.get(kind[w], 9), WORKLOAD_ORDER.index(w) if w in WORKLOAD_ORDER else 99, w),
        )
        self.speed = {(c["library"], c["workload"]): c["mib_per_s"] for c in cells}
        # libraries ordered by geometric-mean speed, fastest first: one order for every chart
        self.gm = {l: geomean([self.speed[(l, w)] for w in self.workloads if (l, w) in self.speed]) for l in self.libs_all}
        self.libs = sorted(self.libs_all, key=lambda l: -self.gm[l])

    def of(self, lib: str) -> list[dict]:
        return [c for c in self.cells if c["library"] == lib]


# --------------------------------------------------------------------------- common drawing


def title_block(fig, title: str, subtitle: str) -> None:
    """Title and subtitle at fixed distances from the top edge, so every chart lines up."""
    fh = fig.get_size_inches()[1]
    fig.text(0.035, 1 - 0.2 / fh, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.035, 1 - 0.66 / fh, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


def foot(fig, text: str, extra: str = "") -> None:
    if extra:
        fig.text(0.035, 0.052, extra, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.text(0.035, 0.018, text, fontsize=7.2, color=INK3, ha="left", va="bottom")


def save(plt, fig, out: Path, stem: str, svg: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200, facecolor=BG)
    if svg:
        fig.savefig(out / f"{stem}.svg", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out / (stem + '.png')}")


# --------------------------------------------------------------------------- 1. overall speed


def chart_speed_overall(plt, d: Data, out: Path, svg: bool) -> None:
    libs = d.libs
    n = len(libs)
    fig = plt.figure(figsize=(9, 1.25 + 0.46 * n + 1.0))
    ax = axes_in(fig, 2.6, 1.0, 5.9, 0.46 * n)
    ys = list(range(n))
    vals = [d.gm[l] for l in libs]
    ax.barh(ys, vals, height=0.62, color=[color_of(l) for l in libs])
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, max(vals) * 1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([l for l in libs], fontsize=10.5)
    for tick, lib in zip(ax.get_yticklabels(), libs):
        tick.set_fontweight("bold" if lib.startswith("text_chipper") else "normal")
        tick.set_color(INK if lib.startswith("text_chipper") else INK2)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK3, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("MiB/s of input text", fontsize=9, color=INK2, labelpad=6)
    best = d.gm["text_chipper"] if "text_chipper" in d.gm else max(vals)
    for y, lib, v in zip(ys, libs, vals):
        ax.text(v + max(vals) * 0.012, y, f"{v:.1f}", va="center", ha="left", fontsize=10.5,
                fontweight="bold" if lib.startswith("text_chipper") else "normal", color=INK)
        if lib != "text_chipper" and not lib.startswith("text_chipper") or lib == "text_chipper (flat)":
            ax.text(max(vals) * 1.17, y, f"{best / v:.1f}× slower", va="center", ha="right", fontsize=9, color=INK3)
    ax.set_xlim(0, max(vals) * 1.17)
    title_block(fig, "Speed: overall", f"Geometric mean of throughput over {len(d.workloads)} workloads. Higher is better.")
    foot(fig, footer(d.meta, d.lane))
    save(plt, fig, out, "speed_overall", svg)


# --------------------------------------------------------------------------- 2. per workload heat map


def chart_speed_per_workload(plt, d: Data, out: Path, svg: bool) -> None:
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("speed", ["#f1f4fb", "#b9ccf4", "#5b88e8", "#1e40af"])
    libs, ncol = d.libs, len(d.libs)

    # row layout: group header rows (0.55 high), workload rows (1.0), a gap, the geometric-mean row
    rows: list[dict] = []
    y = 0.0
    last_kind = None
    for w in d.workloads:
        if d.kind[w] != last_kind:
            rows.append({"type": "group", "y": y, "h": 0.6, "text": KIND_TITLE.get(d.kind[w], d.kind[w].upper())})
            y += 0.6
            last_kind = d.kind[w]
        rows.append({"type": "wl", "y": y, "h": 1.0, "w": w})
        y += 1.0
    y += 0.35
    rows.append({"type": "gm", "y": y, "h": 1.05})
    y += 1.05
    total_h = y

    row_in = 0.42
    fig_h = 1.7 + total_h * row_in + 0.8
    fig = plt.figure(figsize=(10.2, fig_h))
    left, right = 0.235, 0.985
    ax = fig.add_axes([left, 0.8 / fig_h, right - left, total_h * row_in / fig_h])
    ax.set_xlim(0, ncol)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    lo = math.log10(0.03)
    for r in rows:
        if r["type"] == "group":
            ax.text(-0.12, r["y"] + r["h"] / 2 + 0.12, r["text"], ha="right", va="center", fontsize=7.4,
                    fontweight="bold", color=INK3)
            continue
        if r["type"] == "wl":
            vals = {l: d.speed.get((l, r["w"])) for l in libs}
            ax.text(-0.78, r["y"] + 0.5, r["w"], ha="right", va="center", fontsize=9.3, color=INK)
            ax.text(-0.10, r["y"] + 0.5, size_label(d.size[r["w"]]), ha="right", va="center", fontsize=7.4, color=INK3)
        else:
            vals = {l: d.gm[l] for l in libs}
            ax.plot([-2.9, ncol], [r["y"] - 0.18, r["y"] - 0.18], color=GRID, lw=1.0, clip_on=False)
            ax.text(-0.10, r["y"] + 0.5, "geometric mean", ha="right", va="center", fontsize=9.3,
                    fontweight="bold", color=INK)
        known = [v for v in vals.values() if v]
        top = max(known)
        for j, l in enumerate(libs):
            v = vals[l]
            if not v:
                continue
            t = min(1.0, max(0.0, (math.log10(v / top) - lo) / (0 - lo)))
            fc = cmap(t)
            dark = (0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]) < 0.56
            is_best = v >= top - 1e-9
            ax.add_patch(Rectangle((j + 0.03, r["y"] + 0.04), 0.94, r["h"] - 0.08, fc=fc, ec="none", zorder=1))
            if is_best:
                ax.add_patch(Rectangle((j + 0.03, r["y"] + 0.04), 0.94, r["h"] - 0.08, fc="none", ec=INK, lw=1.6, zorder=3))
            ax.text(j + 0.5, r["y"] + r["h"] / 2 + 0.02, f"{v:.1f}", ha="center", va="center", fontsize=9.3,
                    fontweight="bold" if is_best or l == "text_chipper" else "normal",
                    color="white" if dark else INK, zorder=4)
    for j, l in enumerate(libs):
        ax.text(j + 0.5, -0.3, SHORT.get(l, l), ha="center", va="bottom", fontsize=8.8,
                fontweight="bold" if l.startswith("text_chipper") else "normal",
                color=INK if l.startswith("text_chipper") else INK2, linespacing=1.15, clip_on=False)

    title_block(fig, "Speed: every workload",
                "MiB/s of input text, higher is better. Darker = closer to the fastest library in that row; outlined = fastest.")
    foot(fig, footer(d.meta, d.lane))
    save(plt, fig, out, "speed_per_workload", svg)


# --------------------------------------------------------------------------- 3. memory


def chart_memory(plt, d: Data, out: Path, svg: bool) -> None:
    from matplotlib.patches import Patch

    stats = {}
    for l in d.libs:
        cs = d.of(l)
        peak = statistics.fmean(c["rss_peak_mb"] for c in cs)
        base = statistics.fmean(min(c["rss_after_init_mb"], c["rss_peak_mb"]) for c in cs)
        stats[l] = {"peak": peak, "base": base, "extra": peak - base, "max_extra": max(c["rss_delta_mb"] for c in cs)}
    libs = sorted(d.libs, key=lambda l: stats[l]["peak"])
    n = len(libs)
    fig = plt.figure(figsize=(9, 1.25 + 0.46 * n + 1.25))
    ax = axes_in(fig, 2.6, 1.3, 4.0, 0.46 * n)
    ys = list(range(n))
    for y, l in zip(ys, libs):
        s = stats[l]
        ours = l.startswith("text_chipper")
        ax.barh(y, s["base"], height=0.62, color=BLUE if l == "text_chipper" else (BLUE_SOFT if ours else GREY))
        ax.barh(y, s["extra"], left=s["base"], height=0.62, color=BLUE_SOFT if ours else GREY_SOFT)
    xmax = max(s["peak"] for s in stats.values())
    ax.set_xlim(0, xmax * 1.2)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(libs, fontsize=10.5)
    for tick, lib in zip(ax.get_yticklabels(), libs):
        tick.set_fontweight("bold" if lib.startswith("text_chipper") else "normal")
        tick.set_color(INK if lib.startswith("text_chipper") else INK2)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK3, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel("MB, resident memory of the whole Python process", fontsize=9, color=INK2, labelpad=6)
    side = ax.get_yaxis_transform()  # x in axes fraction, y in data units: a text column outside the plot
    ax.text(1.05, -0.95, "Largest extra on\na single workload", transform=side, ha="left", va="bottom", fontsize=8,
            color=INK2, linespacing=1.2, clip_on=False)
    for y, l in zip(ys, libs):
        s = stats[l]
        ours = l.startswith("text_chipper")
        ax.text(s["peak"] + xmax * 0.015, y, f"{s['peak']:.0f}", va="center", ha="left", fontsize=10.5,
                fontweight="bold" if ours else "normal")
        ax.text(1.05, y, f"+{s['max_extra']:.0f} MB", transform=side, va="center", ha="left", fontsize=9.5, color=INK2,
                clip_on=False)
    handles = [Patch(color=GREY, label="after imports and tokenizer load"), Patch(color=GREY_SOFT, label="extra while chunking (mean)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.6,
              labelcolor=INK2, handlelength=1.1, columnspacing=1.8)
    title_block(fig, "Memory: peak process size",
                f"Mean over {len(d.workloads)} workloads, interpreter included. Lower is better.")
    foot(fig, footer(d.meta, d.lane))
    save(plt, fig, out, "memory_single_core", svg)


# --------------------------------------------------------------------------- 4. quality scorecard


def chart_quality(plt, d: Data, out: Path, svg: bool) -> None:
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("q", ["#d8572a", "#f2b48f", "#f4f3ef", "#9db9f2", "#2563eb"])
    norm = TwoSlopeNorm(vmin=40, vcenter=85, vmax=100)

    def q(c, k):
        return (c.get("quality") or {}).get(k)

    def col_budget(cs):
        v = [100 - q(c, "over_limit_pct") for c in cs if q(c, "over_limit_pct") is not None]
        return min(v) if v else None, False

    def col_words(cs):
        v = [100 - q(c, "mid_word_cut_pct") for c in cs if q(c, "mid_word_cut_pct") is not None]
        return min(v) if v else None, False

    def col_text(cs):
        content = [q(c, "content_coverage_pct") for c in cs if q(c, "content_coverage_pct") is not None]
        body = [q(c, "coverage_pct") for c in cs if q(c, "coverage_pct") is not None]
        if not content:
            return None, False
        return min(content), (min(body) < min(content) - 0.01)

    def col_fences(cs):
        v = [q(c, "fence_intact_pct") for c in cs if q(c, "fence_intact_pct") is not None]
        return min(v) if v else None, False

    def col_clean(cs):
        v = [q(c, "clean_end_pct") for c in cs if q(c, "clean_end_pct") is not None]
        return statistics.fmean(v) if v else None, False

    columns = [
        ("Chunks within\nthe token limit", col_budget, "worst"),
        ("Words\nnever cut", col_words, "worst"),
        ("Source text\npreserved", col_text, "worst"),
        ("Code fences\nkept whole", col_fences, "worst"),
    ]
    libs, n, ncol = d.libs, len(d.libs), len(columns)
    speed_max = max(d.gm.values())
    grid = {l: [fn(d.of(l)) for _, fn, _ in columns] for l in libs}
    footnote_needed = any(flag for l in libs for _, flag in grid[l])

    cell_w, cell_h = 1.55, 0.62
    gap = 0.35
    fig_w = 3.1 + (ncol + 1) * cell_w + gap + 0.4
    bottom = 1.45
    fig_h = 2.05 + n * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = axes_in(fig, 3.1, bottom, (ncol + 1) * cell_w + gap, n * cell_h)
    ax.set_xlim(0, ncol + 1 + gap / cell_w)
    ax.set_ylim(n, 0)
    ax.axis("off")
    for i, l in enumerate(libs):
        ours = l.startswith("text_chipper")
        ax.text(-0.08, i + 0.5, l, ha="right", va="center", fontsize=10.2,
                fontweight="bold" if ours else "normal", color=INK if ours else INK2)
        for j, (v, flag) in enumerate(grid[l]):
            if v is None:
                ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, fc="#f4f3ef", ec="none"))
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", fontsize=10, color=INK3)
                continue
            fc = cmap(norm(v))
            dark = (0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]) < 0.5
            ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, fc=fc, ec="none"))
            mark = "†" if flag else ("‡" if (j == 1 and ours and v < 100 - 1e-9) else "")
            ax.text(j + 0.5, i + 0.5, pct_text(v) + mark, ha="center", va="center", fontsize=10.2,
                    fontweight="bold" if ours else "normal", color="white" if dark else INK)
    sx = ncol + gap / cell_w
    speed_cmap = LinearSegmentedColormap.from_list("sp", ["#eef2fb", "#9db9f2", "#2563eb"])
    for i, l in enumerate(libs):
        ours = l.startswith("text_chipper")
        fc = speed_cmap(d.gm[l] / speed_max)
        dark = (0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]) < 0.5
        ax.add_patch(Rectangle((sx + 0.03, i + 0.05), 0.94, 0.9, fc=fc, ec="none"))
        ax.text(sx + 0.5, i + 0.5, f"{d.gm[l]:.1f}", ha="center", va="center", fontsize=10.2,
                fontweight="bold" if ours else "normal", color="white" if dark else INK)
    ax.text(sx + 0.5, -0.62, "Speed\n(MiB/s)", ha="center", va="bottom", fontsize=8.8, color=INK, linespacing=1.2, clip_on=False)
    ax.text(sx + 0.5, -0.2, "geometric mean", ha="center", va="bottom", fontsize=7.2, color=INK3, clip_on=False)
    for j, (label, _, kind) in enumerate(columns):
        ax.text(j + 0.5, -0.62, label, ha="center", va="bottom", fontsize=8.8, color=INK, linespacing=1.2, clip_on=False)
        ax.text(j + 0.5, -0.2, "worst workload" if kind == "worst" else "average", ha="center", va="bottom", fontsize=7.2,
                color=INK3, clip_on=False)

    title_block(fig, "Chunk quality", "Left: share of good outcomes, worst workload of 12 (blue = clean, orange = problems). Right: speed.")
    extra = ""
    if footnote_needed:
        tc = next((c for c in d.of("text_chipper") if (c.get("quality") or {}).get("coverage_pct", 100) < 99.99), None)
        body = f" (body text only: {pct_text(q(tc, 'coverage_pct'), 2)} on {tc['workload']})" if tc else ""
        extra = ("† Counts heading text that text_chipper returns in each chunk's `headers` instead of the chunk text"
                 + body + ".")
    tcw = d.of("text_chipper")
    worst = max(tcw, key=lambda c: q(c, "mid_word_cut_pct") or 0) if tcw else None
    if worst and (q(worst, "mid_word_cut_pct") or 0) > 0:
        pct = q(worst, "mid_word_cut_pct")
        k = round(100 / pct)
        what = (f"one cut inside a word out of {k:,} cut points" if abs(k * pct / 100 - 1) < 0.01
                else f"{pct:.2f} % of cut points are inside a word")
        note = f"‡ text_chipper's worst workload ({worst['workload']}): {what}; no other workload has a cut inside a word."
        extra = (extra + "\n" + note) if extra else note
    foot(fig, footer(d.meta, d.lane), extra)
    save(plt, fig, out, "quality_scorecard", svg)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Render the README benchmark charts from results.json")
    ap.add_argument("results", type=Path, help="results.json, or the results directory that contains it")
    ap.add_argument("out", type=Path, nargs="?", default=here / "charts", help="output directory (default: benchmarks/charts)")
    ap.add_argument("--lane", choices=["single", "all"], default="single", help="which lane to chart (default: single)")
    ap.add_argument("--svg", action="store_true", help="also write SVG files")
    args = ap.parse_args(argv)

    try:
        plt = setup_matplotlib()
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")
    data = Data(load(args.results), args.lane)
    chart_speed_overall(plt, data, args.out, args.svg)
    chart_speed_per_workload(plt, data, args.out, args.svg)
    chart_memory(plt, data, args.out, args.svg)
    chart_quality(plt, data, args.out, args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
