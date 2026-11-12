"""Orchestrator and CLI:  python -m chunkbench {check,fetch,run,report,plot}

`run` executes every (library x workload x lane x overlap) cell in a fresh subprocess, several
rounds in shuffled order, then judges the returned chunks with the independent reference tokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, adapters, corpora, metrics
from .tokenizer import get_reference

HERE = Path(__file__).resolve().parent.parent
MARK = "CHUNKBENCH_RESULT "


# --------------------------------------------------------------------------- environment


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        return None


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment(libs: list[str], args: argparse.Namespace | None = None) -> dict:
    dists = sorted({adapters.ADAPTERS[n].dist for n in libs if adapters.ADAPTERS[n].dist} | {"tiktoken", "numpy"})
    return {
        "chunkbench": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "load_avg_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        "versions": {d: adapters.version_of(d) for d in dists},
        "project_commit": _git_commit(),
        "args": {k: v for k, v in vars(args).items() if k != "fn"} if args else {},
    }


# --------------------------------------------------------------------------- cells


def _run_child(cell: dict, text_file: Path, emit: Path | None, args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable, "-m", "chunkbench.worker",
        "--adapter", cell["adapter"], "--text-file", str(text_file), "--fmt", cell["fmt"],
        "--chunk-size", str(args.chunk_size), "--overlap", str(cell["overlap"]), "--lane", cell["lane"],
        "--repeats", str(args.repeats), "--max-repeats", str(args.max_repeats), "--min-time", str(args.min_time),
        "--encoding", args.encoding, "--mode", args.mode,
    ]
    if getattr(args, "no_fresh", False):
        cmd.append("--no-fresh")
    if emit:
        cmd += ["--emit-chunks", str(emit)]
    env = dict(os.environ)
    env.update(adapters.thread_env(cell["lane"]))
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {args.timeout}s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    return {"ok": False, "error": f"worker crashed (exit {proc.returncode}): {proc.stderr.strip()[-400:]}"}


def _summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "p95": metrics._percentile(s, 0.95),
        "n": len(s),
    }


def cmd_run(args: argparse.Namespace) -> int:
    if args.rounds is None:
        args.rounds = 7 if args.mode == "cold" else 3
    adapters.load_plugins()
    libs = args.libs or adapters.DEFAULT_LIBS
    unknown = [n for n in libs if n not in adapters.ADAPTERS]
    if unknown:
        print(f"unknown libraries: {unknown}; choose from {list(adapters.ADAPTERS)}", file=sys.stderr)
        return 2

    workloads = corpora.build_workloads(args.size_mib, args.max_gold, args.seed, include_real=not args.no_real)
    if args.workloads:
        missing = [w for w in args.workloads if w not in workloads]
        if missing:
            print(f"unknown workloads: {missing}; available: {list(workloads)}", file=sys.stderr)
            return 2
        workloads = {k: workloads[k] for k in args.workloads}

    ref = get_reference(args.encoding)
    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    cache = out_dir / "_cache"
    cache.mkdir(parents=True)
    for w in workloads.values():
        (cache / f"{w.name}.txt").write_text(w.text, encoding="utf-8")

    cells = []
    for name in libs:
        for wl in workloads.values():
            for lane in args.lanes:
                for ov in args.overlaps:
                    if ov > 0 and not adapters.ADAPTERS[name].supports_overlap:
                        continue
                    cells.append({"adapter": name, "workload": wl.name, "fmt": wl.fmt, "lane": lane, "overlap": ov})
    print(f"{len(cells)} cells x {args.rounds} round(s); output: {out_dir}", file=sys.stderr)

    results: dict[tuple, dict] = {}
    for rnd in range(args.rounds):
        order = list(range(len(cells)))
        random.Random(args.seed + rnd).shuffle(order)  # shuffled so no library always runs first/last
        for k, idx in enumerate(order, 1):
            cell = cells[idx]
            key = (cell["adapter"], cell["workload"], cell["lane"], cell["overlap"])
            if key in results and results[key]["errors"] and not results[key]["rounds"]:
                continue  # failed or timed out in an earlier round: do not burn the timeout again
            emit = cache / ("chunks_%d.json" % idx) if rnd == 0 else None
            res = _run_child(cell, cache / f"{cell['workload']}.txt", emit, args)
            entry = results.setdefault(key, {**cell, "rounds": [], "errors": []})
            if res.get("ok"):
                entry["rounds"].append(res)
                if emit and "chunks_file" not in entry:
                    entry["chunks_file"] = str(emit)
            else:
                entry["errors"].append(res.get("error", "unknown error"))
            tag = "ok" if res.get("ok") else "ERROR " + str(res.get("error"))[:80]
            print(f"[round {rnd + 1}/{args.rounds}] {k}/{len(order)} {cell['adapter']:<22} {cell['workload']:<22} {cell['lane']:<6} ov={cell['overlap']:<3} {tag}", file=sys.stderr)

    final = []
    for key, entry in results.items():
        wl = workloads[entry["workload"]]
        row = {k: entry[k] for k in ("adapter", "workload", "fmt", "lane", "overlap")}
        row["library"] = adapters.ADAPTERS[entry["adapter"]].label
        row["input_mib"] = wl.size_bytes / corpora.MIB
        row["input_kind"] = wl.kind
        if not entry["rounds"]:
            row["status"] = "error"
            row["error"] = entry["errors"][0] if entry["errors"] else "no result"
            final.append(row)
            continue
        rounds = entry["rounds"]
        if args.mode == "cold":
            # one timed call per process; the samples are the per-process first calls
            wall = [r["cold_ms"] for r in rounds]
            cpu = [r.get("cold_cpu_ms", r["cold_ms"]) for r in rounds]
        else:
            wall = [x for r in rounds for x in r["wall_ms"]]
            cpu = [x for r in rounds for x in r["cpu_ms"]]
        wsum = _summarize(wall)
        fresh = [x for r in rounds for x in r.get("fresh_wall_ms", [])]
        if fresh:
            fsum = _summarize(fresh)
            row["fresh_wall_ms"] = fsum
            row["mib_per_s_fresh"] = row["input_mib"] / (fsum["median"] / 1000.0) if fsum["median"] else None
            # > 1 means the repeated-input number is flattered by caching (or warm-up); ~1 means not.
            row["repeat_vs_fresh_speedup"] = fsum["median"] / wsum["median"] if wsum["median"] else None
        row["timing_mode"] = args.mode
        row.update(
            status="ok",
            wall_ms=wsum,
            cpu_ms_median=statistics.median(cpu),
            cpu_utilization=(statistics.median(cpu) / wsum["median"]) if wsum["median"] else None,
            mib_per_s=row["input_mib"] / (wsum["median"] / 1000.0) if wsum["median"] else None,
            cold_ms_median=statistics.median(r["cold_ms"] for r in rounds),
            init_ms_median=statistics.median(r["init_ms"] for r in rounds),
            rss_peak_mb=max(r["rss_peak_mb"] for r in rounds),
            rss_delta_mb=statistics.median(max(0.0, r["rss_peak_mb"] - r["rss_after_init_mb"]) for r in rounds),
            rss_after_init_mb=statistics.median(r["rss_after_init_mb"] for r in rounds),
            thread_enforcement=rounds[0].get("thread_enforcement"),
            samples=len(wall),
            rounds_ok=len(rounds),
            rounds_failed=len(entry["errors"]),
            chunk_count_stable=len({r["n_chunks"] for r in rounds}) == 1,
        )
        if entry.get("chunks_file"):
            views = [tuple(v) for v in json.loads(Path(entry["chunks_file"]).read_text(encoding="utf-8"))]
            row["quality"] = metrics.evaluate(
                wl.text, views, chunk_size=args.chunk_size, overlap=row["overlap"], fmt=wl.fmt,
                gold=wl.gold, count=ref.count,
            )
            row["quality"]["gold_source"] = wl.gold_source if wl.gold else None
        final.append(row)

    meta = environment(libs, args)
    meta["reference_tokenizer"] = ref.name
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "cells": final}, indent=1, ensure_ascii=False))
    shutil.rmtree(cache, ignore_errors=True)

    from . import report

    report.write_all(out_dir)
    print(f"\nresults: {out_dir}/results.json\nreport:  {out_dir}/report.md\ntable:   {out_dir}/summary.csv", file=sys.stderr)
    failed = [r for r in final if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} cell(s) failed; see 'error' in results.json", file=sys.stderr)
    return 0 if not failed else 1


# --------------------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> int:
    """Smoke-test every adapter on a small document and validate the contract. Run this first:
    it surfaces API drift in a third-party library before any long benchmark starts."""
    adapters.load_plugins()
    ref = get_reference(args.encoding)
    sample = corpora.synthetic_markdown(60_000, 1)
    bad = 0
    print(f"{'library':<24} {'version':<14} {'plain':<7} {'markdown':<9} notes")
    for name in args.libs or adapters.DEFAULT_LIBS:
        spec = adapters.ADAPTERS[name]
        cols, notes = [], []
        for fmt in ("plain", "markdown"):
            try:
                runner = spec.build(adapters.Ctx(args.chunk_size, 0, fmt, args.encoding))
                views = runner.views(runner.run(sample))
                if not views:
                    raise RuntimeError("returned no chunks")
                joined = "".join(v.body for v in views)
                if len("".join(joined.split())) < 0.95 * len("".join(sample.split())):
                    notes.append(f"{fmt}: lost >5% of the text")
                over = sum(1 for v in views if ref.count(v.delivered) > args.chunk_size)
                if over:
                    notes.append(f"{fmt}: {over}/{len(views)} chunks over budget")
                cols.append("ok")
            except Exception as exc:  # noqa: BLE001
                cols.append("FAIL")
                notes.append(f"{fmt}: {type(exc).__name__}: {str(exc)[:110]}")
                bad += 1
        print(f"{spec.label:<24} {adapters.version_of(spec.dist):<14} {cols[0]:<7} {cols[1]:<9} {'; '.join(notes)}")
    return 1 if bad else 0


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="chunkbench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--libs", nargs="*", help=f"subset of: {', '.join(adapters.DEFAULT_LIBS)}")
        p.add_argument("--chunk-size", type=int, default=512)
        p.add_argument("--encoding", default="cl100k_base")

    p = sub.add_parser("check", help="smoke-test the adapters")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("fetch", help="download the public real-world corpora and pin them by SHA-256")
    p.add_argument("--update-lock", action="store_true")
    p.set_defaults(fn=lambda a: corpora.fetch_corpora(a.update_lock))

    p = sub.add_parser("run", help="run the benchmark")
    common(p)
    p.add_argument("--workloads", nargs="*")
    p.add_argument("--overlaps", nargs="*", type=int, default=[0], help="overlap values in tokens (default: 0)")
    p.add_argument("--lanes", nargs="*", choices=["single", "all"], default=["single", "all"])
    p.add_argument("--mode", choices=["cold", "steady"], default="cold",
                   help="cold (default): every sample is the first and only call of a fresh process, on the unmodified "
                        "text, so no cache can help. steady: warm repeat loop + salted fresh-input loop (slow)")
    p.add_argument("--rounds", type=int, default=None,
                   help="independent processes per cell, in shuffled order (default: 7 for cold, 3 for steady)")
    p.add_argument("--repeats", type=int, default=5, help="steady mode: minimum timed repeats per process")
    p.add_argument("--max-repeats", type=int, default=200, help="steady mode only")
    p.add_argument("--min-time", type=float, default=2.0, help="steady mode: keep repeating until this many seconds were timed")
    p.add_argument("--size-mib", type=float, default=1.0, help="size of each synthetic corpus")
    p.add_argument("--max-gold", type=int, default=300, help="answer spans per workload for the alignment metric")
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--timeout", type=int, default=120, help="seconds per process; a cell that times out is not retried")
    p.add_argument("--no-real", action="store_true", help="skip downloaded real corpora")
    p.add_argument("--no-fresh", action="store_true", help="skip the fresh-input timing samples (faster run)")
    p.add_argument("--out", default=str(HERE / "results"))
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="regenerate report.md / summary.csv from a results directory")
    p.add_argument("results_dir")
    p.set_defaults(fn=lambda a: (__import__("chunkbench.report", fromlist=["x"]).write_all(Path(a.results_dir)), 0)[1])

    p = sub.add_parser("plot", help="render PNG/SVG charts from a results directory")
    p.add_argument("results_dir")
    p.set_defaults(fn=lambda a: __import__("chunkbench.plot", fromlist=["x"]).render(Path(a.results_dir)))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
