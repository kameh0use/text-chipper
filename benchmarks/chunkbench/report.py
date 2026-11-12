"""results.json -> summary.csv (flat, for your own charts) and report.md (tables for the README)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

CSV_COLUMNS = [
    "workload", "input_kind", "fmt", "lane", "overlap", "library", "status", "timing_mode", "spread_pct",
    "mib_per_s", "mib_per_s_fresh", "repeat_vs_fresh_speedup", "wall_ms_fresh_median", "wall_ms_median", "wall_ms_min", "wall_ms_stdev", "cpu_ms_median", "cpu_utilization",
    "cold_ms_median", "init_ms_median", "rss_peak_mb", "rss_delta_mb", "samples", "rounds_ok",
    "chunks", "tokens_max", "tokens_mean", "tokens_p5", "tokens_p50", "tokens_p95", "fill_ratio",
    "over_limit_chunks", "over_limit_pct", "worst_overshoot_pct", "tiny_chunk_pct",
    "located_pct", "coverage_pct", "content_coverage_pct", "lost_nonspace_chars", "lost_unexplained_chars", "heading_meta_chars", "duplication_pct", "replacement_char_chunks",
    "mid_word_cut_pct", "clean_end_pct", "orphan_heading_pct", "fence_intact_pct", "fences_measured",
    "overlap_tokens_median", "overlap_tokens_max", "overlap_violation_pct",
    "oracle_iou_mean", "oracle_recall_mean", "chunks_per_span_mean", "single_chunk_hit_pct",
    "context_tokens_per_span_mean", "gold_source", "error",
]


def _flat(cell: dict) -> dict:
    row = {k: cell.get(k) for k in CSV_COLUMNS}
    w = cell.get("wall_ms")
    if isinstance(w, dict):
        row.update(wall_ms_median=w["median"], wall_ms_min=w["min"], wall_ms_stdev=w["stdev"])
        mx = w.get("max")
        if mx is not None and w["median"]:
            row["spread_pct"] = 100.0 * (mx - w["min"]) / w["median"]
    fw = cell.get("fresh_wall_ms")
    if isinstance(fw, dict):
        row["wall_ms_fresh_median"] = fw["median"]
    row.update({k: v for k, v in (cell.get("quality") or {}).items() if k in CSV_COLUMNS})
    return row


def _fmt(v, digits=1, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{suffix}"
    return f"{v:,}{suffix}"


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def write_all(out_dir: Path) -> None:
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    meta, cells = data["meta"], data["cells"]
    rows = [_flat(c) for c in cells]

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    lines: list[str] = []
    add = lines.append
    add("# Benchmark report\n")
    add("## Environment\n")
    add(f"- CPU: {meta['cpu']} ({meta['logical_cpus']} logical CPUs), RAM: {meta.get('ram_gb')} GB")
    add(f"- OS: {meta['platform']}; Python {meta['python']} ({meta['implementation']})")
    add(f"- Reference tokenizer: `{meta['reference_tokenizer']}`; budget: {meta['args'].get('chunk_size')} tokens; "
        f"overlaps: {meta['args'].get('overlaps')}; rounds: {meta['args'].get('rounds')}")
    add("- Versions: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()))
    add(f"- Project commit: {meta.get('project_commit') or 'n/a'}; run at {meta['timestamp_utc']}; "
        f"load average before run: {meta.get('load_avg_1m')}\n")

    lanes = sorted({r["lane"] for r in rows})
    overlaps = sorted({r["overlap"] for r in rows})
    workloads = list(dict.fromkeys(r["workload"] for r in rows))

    has_fresh = any(r.get("mib_per_s_fresh") for r in rows)
    is_cold = any(r.get("timing_mode") == "cold" for r in rows)
    if is_cold:
        add("## How to read the speed numbers\n")
        add("- **MiB/s**: median over independent processes of the time of the *first and only* chunking call "
            "on the unmodified text. A fresh process cannot reuse a result or token-count cache, so these "
            "numbers are safe to quote. Import and tokenizer loading are excluded (`init_ms` shows them).")
        add("- **spread %**: (max - min) / median over the processes; a quick noise check.\n")
    if has_fresh:
        add("## How to read the speed numbers\n")
        add("- **fresh MiB/s** (quote this one): every timed call gets a different input string "
            "(the text plus a few unique bytes), so a library that caches by input cannot answer from memory.")
        add("- **repeat MiB/s**: the same string every call, after a warm-up. Libraries that memoise "
            "results or token counts look much faster here than they are on new documents.")
        add("- **fresh/repeat ms**: fresh median divided by repeat median. Around 1 means no caching effect; "
            "well above 1 means the repeat number is flattered by a cache.\n")
    add("## Speed summary (geometric mean of throughput over all workloads)\n")
    if is_cold:
        for lane in lanes:
            for ov in overlaps:
                sub = [r for r in rows if r["lane"] == lane and r["overlap"] == ov and r["status"] == "ok"]
                if not sub:
                    continue
                add(f"**lane `{lane}`, overlap {ov}**\n")
                add("| library | geomean MiB/s | workloads |")
                add("| --- | ---: | ---: |")
                for lib in dict.fromkeys(r["library"] for r in sub):
                    xs = [r["mib_per_s"] for r in sub if r["library"] == lib]
                    add(f"| {lib} | {_fmt(_geomean(xs))} | {len(xs)}/{len(workloads)} |")
                add("")
    for lane in ([] if is_cold else lanes):
        for ov in overlaps:
            sub = [r for r in rows if r["lane"] == lane and r["overlap"] == ov and r["status"] == "ok"]
            if not sub:
                continue
            add(f"**lane `{lane}`, overlap {ov}**\n")
            add("| library | geomean fresh MiB/s | geomean repeat MiB/s | workloads |")
            add("| --- | ---: | ---: | ---: |")
            libs = list(dict.fromkeys(r["library"] for r in sub))
            for lib in libs:
                xs = [r["mib_per_s"] for r in sub if r["library"] == lib]
                fs = [r.get("mib_per_s_fresh") for r in sub if r["library"] == lib]
                add(f"| {lib} | {_fmt(_geomean(fs))} | {_fmt(_geomean(xs))} | {len(xs)}/{len(workloads)} |")
            add("")

    speed_cols = [("MiB/s", "mib_per_s", 1), ("median ms", "wall_ms_median", 1), ("spread %", "spread_pct", 0)] if is_cold else [
        ("fresh MiB/s", "mib_per_s_fresh", 1), ("repeat MiB/s", "mib_per_s", 1),
        ("fresh/repeat ms", "repeat_vs_fresh_speedup", 2), ("median ms", "wall_ms_median", 1)]
    cols = speed_cols + [
        ("CPU ms", "cpu_ms_median", 0),
        ("CPU/wall", "cpu_utilization", 2), ("RSS Δ MB", "rss_delta_mb", 0), ("chunks", "chunks", 0),
        ("max tok", "tokens_max", 0), ("over %", "over_limit_pct", 2), ("body cov %", "coverage_pct", 3), ("content cov %", "content_coverage_pct", 3),
        ("located %", "located_pct", 1), ("mid-word %", "mid_word_cut_pct", 1), ("clean end %", "clean_end_pct", 1),
        ("fence ok %", "fence_intact_pct", 1), ("orphan h %", "orphan_heading_pct", 1),
        ("IoU", "oracle_iou_mean", 3), ("1-chunk %", "single_chunk_hit_pct", 1),
    ]
    add("## Results per workload\n")
    add("Columns are defined in `benchmarks/README.md`. `–` means not applicable or not measured. "
        "Failures are listed at the end.\n")
    for wl in workloads:
        for lane in lanes:
            for ov in overlaps:
                sub = [r for r in rows if r["workload"] == wl and r["lane"] == lane and r["overlap"] == ov]
                if not sub:
                    continue
                kind = sub[0]["input_kind"]
                add(f"### `{wl}` ({kind}, {sub[0]['fmt']}) — lane `{lane}`, overlap {ov}\n")
                add("| library | " + " | ".join(c[0] for c in cols) + " |")
                add("| --- | " + " | ".join("---:" for _ in cols) + " |")
                for r in sub:
                    if r["status"] != "ok":
                        add(f"| {r['library']} | " + " | ".join("error" if i == 0 else "" for i, _ in enumerate(cols)) + " |")
                        continue
                    add(f"| {r['library']} | " + " | ".join(_fmt(r.get(k), d) for _, k, d in cols) + " |")
                add("")

    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        add("## Failures\n")
        for r in failures:
            add(f"- `{r['library']}` on `{r['workload']}` ({r['lane']}, overlap {r['overlap']}): {r['error']}")
        add("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
