"""Charts for the large-file benchmark (results of streaming_bench.py), in the style of the main charts.

Usage: python benchmarks/plot_streaming.py benchmarks/results/streaming_20261114/*.json [--out benchmarks/charts]

Timing and memory medians use only the runs WITHOUT --verify (verification builds extra objects).
Writes streaming_speed_{single,all}.{png,svg} and streaming_memory_{single,all}.{png,svg}.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8b85"
BLUE, BLUE_M, BLUE_L = "#2563eb", "#6f97ee", "#c3d4f7"
GRAY, GRAY_L = "#8d8b85", "#cfceca"

LABEL = {
    "tiktoken_window": "tiktoken window",
    "chonkie_token": "chonkie TokenChunker",
    "chonkie_recursive": "chonkie RecursiveChunker",
    "semchunk": "semchunk",
    "langchain": "LangChain",
    "semantic_text_splitter": "semantic-text-splitter",
}
OTHERS = list(LABEL)
FOOTER = ("GitHub Actions runner (4 logical CPUs) · Python 3.12 · WikiText-103 raw, {size:.1f} MiB · 512-token chunks, "
          "cl100k_base · median of 3 runs · one workflow run")

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return rows


def med(rows, lane, mode, field):
    vals = [r[field] for r in rows if r.get("lane") == lane and r["mode"] == mode
            and r.get("status") == "ok" and not r["verified"] and r.get(field) is not None]
    return st.median(vals) if vals else None


def frame(title, subtitle, footer, h):
    fig = plt.figure(figsize=(9, h), dpi=150, facecolor=SURFACE)
    fig.text(0.035, 1 - 0.45 / h, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.035, 1 - 0.95 / h, subtitle, fontsize=9.5, color=INK2, va="top")
    fig.text(0.035, 0.18 / h, footer, fontsize=7.2, color=MUTED, va="bottom")
    return fig


def save(fig, out, stem):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{stem}.{ext}", facecolor=SURFACE)
    plt.close(fig)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def speed_chart(rows, lane, out, size):
    items = [("text_chipper, stream", med(rows, lane, "text_chipper:stream", "mib_per_s"), BLUE, True),
             ("text_chipper, blocks", med(rows, lane, "text_chipper:blocks", "mib_per_s"), BLUE_L, True)]
    items += [(LABEL[k], med(rows, lane, f"{k}:blocks", "mib_per_s"), GRAY, False) for k in OTHERS]
    items = sorted([i for i in items if i[1] is not None], key=lambda i: -i[1])
    top = max(v for _, v, _, _ in items)
    lanetxt = "single-core lane" if lane == "single" else "all-cores lane (library defaults)"
    fig = frame(f"Large file: speed, {lane}-core" if lane == "single" else "Large file: speed, all cores",
                f"MiB/s of input text on one {size:.0f} MiB file, {lanetxt}. Higher is better.\n"
                "Time includes reading the file. Libraries without streaming get the file in ~8 MiB blocks.",
                FOOTER.format(size=size), 5.9)
    ax = fig.add_axes([0.30, 0.13, 0.56, 0.62])
    style_axes(ax)
    for i, (name, v, color, bold) in enumerate(items):
        ax.barh(i, v, height=0.62, color=color)
        ax.text(v + top * 0.015, i, f"{v:.1f}", va="center", fontsize=10.5, fontweight="bold" if bold else "normal")
        if i:
            ax.text(1.03, i, f"{top / v:.1f}× slower", va="center", fontsize=9, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, *_ in items], fontsize=10.5, color=INK2)
    for lbl, (_, _, _, bold) in zip(ax.get_yticklabels(), items):
        if bold:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("MiB/s of input text", fontsize=9, color=INK2)
    save(fig, out, f"streaming_speed_{lane}")


def memory_chart(rows, lane, out, size):
    def fmt(v):
        return f"{v:.0f} MB" if v < 1000 else f"{v / 1000:.1f} GB"

    groups = [("text_chipper", [("stream", med(rows, lane, "text_chipper:stream", "growth_mb"), BLUE),
                                ("blocks", med(rows, lane, "text_chipper:blocks", "growth_mb"), BLUE_M),
                                ("whole", med(rows, lane, "text_chipper:whole", "growth_mb"), BLUE_L)])]
    others = sorted(OTHERS, key=lambda k: med(rows, lane, f"{k}:blocks", "growth_mb") or 0)
    for k in others:
        groups.append((LABEL[k], [("blocks", med(rows, lane, f"{k}:blocks", "growth_mb"), GRAY),
                                  ("whole", med(rows, lane, f"{k}:whole", "growth_mb"), GRAY_L)]))
    n_bars = sum(len(b) for _, b in groups)
    h = 1.9 + 0.2 * n_bars + 0.28 * len(groups)
    lane_title = "single-core" if lane == "single" else "all cores"
    lane_txt = "single-core lane" if lane == "single" else "all-cores lane, library defaults"
    fig = frame(f"Large file: memory, {lane_title}",
                f"Growth of peak process memory while chunking one {size:.0f} MiB file, {lane_txt} (log scale). Lower is better.\n"
                "Blocks: ~8 MiB at a time. Whole: file read into one Python string, then one call.",
                FOOTER.format(size=size), h)
    ax = fig.add_axes([0.26, 0.9 / h, 0.60, 1 - (1.9 / h) - 0.04])
    style_axes(ax)
    xmin, xmax = 20, 30000
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    y, centers, names = 0, [], []
    for name, bars in groups:
        start = y
        for label, v, color in bars:
            if v is None:
                continue
            ax.barh(y, v - xmin, left=xmin, height=0.8, color=color)
            ax.text(v * 1.12, y, f"{fmt(v)}", va="center", fontsize=8.5,
                    fontweight="bold" if name == "text_chipper" else "normal")
            ax.text(xmin * 1.08, y, label, va="center", fontsize=7, color="white" if color in (BLUE, BLUE_M, GRAY) else INK2)
            y += 1
        centers.append((start + y - 1) / 2)
        names.append(name)
        y += 0.7
    ax.set_yticks(centers)
    ax.set_yticklabels(names, fontsize=10, color=INK2)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "text_chipper":
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.invert_yaxis()
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100 MB", "1 GB", "10 GB"])
    ax.minorticks_off()
    ax.set_xlabel("growth of peak memory", fontsize=9, color=INK2)
    save(fig, out, f"streaming_memory_{lane}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=str(Path(__file__).parent / "charts"))
    a = ap.parse_args()
    rows = load(a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    size = next(r["file_mib"] for r in rows if "file_mib" in r)
    speed_chart(rows, "single", out, size)
    speed_chart(rows, "all", out, size)
    memory_chart(rows, "single", out, size)
    memory_chart(rows, "all", out, size)
    print("wrote charts to", out)
