"""Charts from results.json (PNG + SVG, light background, for the GitHub README).

One chart per metric: workloads on the y axis, one bar per library. Library colors are fixed by
library (never by rank) and use the validated 8-slot categorical order, so a library keeps its
color in every chart. The legend is always shown; values are in summary.csv for exact numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (file stem, title, field, x label, log scale, source: "cell" or "quality", only for lane)
CHARTS = [
    ("throughput", "Throughput, first call in a fresh process (higher is better)", "mib_per_s_fresh", "MiB/s of input text", True, "cell"),
    ("cpu_time", "CPU time per run (lower is better)", "cpu_ms_median", "ms of CPU (user+sys, all threads)", True, "cell"),
    ("memory", "Extra peak memory while chunking (lower is better)", "rss_delta_mb", "MB above the post-init baseline", False, "cell"),
    ("over_limit", "Chunks over the token budget (lower is better)", "over_limit_pct", "% of chunks > budget", False, "quality"),
    ("mid_word_cuts", "Cuts inside a word (lower is better)", "mid_word_cut_pct", "% of cut positions", False, "quality"),
    ("clean_ends", "Chunks ending at a sentence or line break (higher is better)", "clean_end_pct", "% of chunk ends", False, "quality"),
    ("answer_alignment", "Answer-span alignment, oracle IoU (higher is better)", "oracle_iou_mean", "token IoU", False, "quality"),
    ("code_fences_intact", "Small code fences kept whole (higher is better)", "fence_intact_pct", "% of fences", False, "quality"),
]


def _value(cell: dict, field: str, source: str):
    if source != "cell":
        return (cell.get("quality") or {}).get(field)
    value = cell.get(field)
    if value is None and field == "mib_per_s_fresh":  # results recorded before fresh timing existed
        value = cell.get("mib_per_s")
    return value


def render(results_dir: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c["status"] == "ok"]
    out = results_dir / "plots"
    out.mkdir(exist_ok=True)

    from .adapters import ADAPTERS

    order = [a.label for a in ADAPTERS.values() if any(c["library"] == a.label for c in cells)]
    color = {lib: SERIES[i % len(SERIES)] for i, lib in enumerate(order)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK})

    n = 0
    for lane in sorted({c["lane"] for c in cells}):
        for ov in sorted({c["overlap"] for c in cells}):
            for stem, title, field, xlabel, log, source in CHARTS:
                sub = [c for c in cells if c["lane"] == lane and c["overlap"] == ov and _value(c, field, source) is not None]
                if not sub:
                    continue
                wls = list(dict.fromkeys(c["workload"] for c in sub))
                libs = [lib for lib in order if any(c["library"] == lib for c in sub)]
                bar_h = 0.8 / len(libs)
                fig, ax = plt.subplots(figsize=(9, 0.55 * len(wls) * max(2, len(libs) * 0.55) + 1.8), facecolor=SURFACE)
                ax.set_facecolor(SURFACE)
                for li, lib in enumerate(libs):
                    ys, vs = [], []
                    for wi, wl in enumerate(wls):
                        v = next((_value(c, field, source) for c in sub if c["library"] == lib and c["workload"] == wl), None)
                        if v is not None:
                            ys.append(wi + (li - (len(libs) - 1) / 2) * bar_h)
                            vs.append(v)
                    ax.barh(ys, vs, height=bar_h * 0.85, color=color[lib], label=lib, edgecolor=SURFACE, linewidth=1)
                ax.set_yticks(range(len(wls)))
                ax.set_yticklabels(wls, color=INK2, fontsize=9)
                ax.invert_yaxis()
                if log:
                    ax.set_xscale("log")
                ax.set_xlabel(xlabel, color=INK2, fontsize=9)
                ax.set_title(f"{title}\nlane: {lane}, overlap: {ov}", loc="left", fontsize=11, color=INK)
                ax.grid(axis="x", color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                ax.tick_params(colors=INK2, length=0)
                for side in ("top", "right", "left"):
                    ax.spines[side].set_visible(False)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(libs)), frameon=False, fontsize=9, labelcolor=INK2)
                fig.tight_layout()
                base = out / f"{stem}__{lane}_ov{ov}"
                fig.savefig(f"{base}.png", dpi=200)
                fig.savefig(f"{base}.svg")
                plt.close(fig)
                n += 1
    print(f"wrote {n} charts to {out}")
    return 0
