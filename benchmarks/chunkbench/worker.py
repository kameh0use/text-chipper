"""Measurement worker. One fresh process per (library, workload, lane, round), so memory and CPU
numbers are never contaminated by another library, another workload or a previous run.

Protocol: prints exactly one line, `CHUNKBENCH_RESULT {json}`, on stdout.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import resource
import sys
import time
import traceback


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def pin_to_one_cpu() -> str:
    if hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpus[len(cpus) // 2]})
            return "affinity"
        except OSError:
            pass
    return "env-only"


_LINE_END = re.compile(r"(?<=\S)$", re.MULTILINE)
_INNER_SPACE = re.compile(r"(?<=\S) (?=\S)")
_SALT_EVERY = 10  # one inserted tag per ~10 words, i.e. every ~50-60 bytes


def fresh_variant(text: str, i: int) -> str:
    """A different string for every call, down to the sentence level.

    Repeating the *same* string lets a library that memoises (a result cache, or a cache of
    per-segment token counts) answer from memory after the first call, which measures the cache,
    not the chunker. Real pipelines see each document once. Salting only the end of the text, or
    of each line, still leaves most sentences of a long paragraph identical, so a per-sentence
    cache keeps hitting. Therefore every non-blank line gets a unique suffix and every 10th
    single space inside a line gets a unique tag. Blank lines, indentation and line structure
    are kept, so paragraphs, lists and fences are unchanged. The variant is built before the
    timer starts and is a few percent longer than the original."""
    n = 0

    def tag(_m: re.Match) -> str:
        nonlocal n
        n += 1
        return f" {i} " if n % _SALT_EVERY == 0 else " "

    return _LINE_END.sub(f" {i}", _INNER_SPACE.sub(tag, text)) + f"\n\n{i}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--fmt", required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--overlap", type=int, required=True)
    ap.add_argument("--lane", choices=["single", "all"], required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-repeats", type=int, default=200)
    ap.add_argument("--min-time", type=float, default=2.0)
    ap.add_argument("--no-fresh", action="store_true", help="steady mode only: skip the fresh-input samples")
    ap.add_argument("--mode", choices=["cold", "steady"], default="cold",
                    help="cold: ONE timed call on the unmodified text in this fresh process (default); "
                         "steady: warm repeat loop plus salted fresh-input loop")
    ap.add_argument("--emit-chunks", default="")
    ap.add_argument("--encoding", default="cl100k_base")
    args = ap.parse_args()

    out: dict = {"adapter": args.adapter, "lane": args.lane}
    try:
        out["thread_enforcement"] = pin_to_one_cpu() if args.lane == "single" else "none"
        t_import = time.perf_counter()
        from . import adapters

        adapters.load_plugins()
        spec = adapters.ADAPTERS[args.adapter]
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()

        ctx = adapters.Ctx(args.chunk_size, args.overlap, args.fmt, args.encoding)
        t0 = time.perf_counter()
        runner = spec.build(ctx)  # imports the library and loads its tokenizer
        out["init_ms"] = (time.perf_counter() - t0) * 1000.0
        out["import_total_ms"] = (time.perf_counter() - t_import) * 1000.0
        out["rss_after_init_mb"] = peak_rss_mb()

        # Cold call: the first call in a fresh process, on the unmodified text. Nothing can be
        # cached across processes, so this is the number that is safe to quote.
        gc.collect()
        c0 = time.process_time()
        t0 = time.perf_counter()
        native = runner.run(text)
        out["cold_ms"] = (time.perf_counter() - t0) * 1000.0
        out["cold_cpu_ms"] = (time.process_time() - c0) * 1000.0
        if args.mode == "cold":
            out["rss_peak_mb"] = peak_rss_mb()
            views = runner.views(native)
            if args.emit_chunks:
                with open(args.emit_chunks, "w", encoding="utf-8") as f:
                    json.dump([[v.delivered, v.body, list(v.headers)] for v in views], f, ensure_ascii=False)
            out["n_chunks"] = len(views)
            out["ok"] = True
            sys.stdout.write("CHUNKBENCH_RESULT " + json.dumps(out) + "\n")
            sys.stdout.flush()
            return 0
        del native

        wall_ms: list[float] = []
        cpu_ms: list[float] = []
        total = 0.0
        gc_was_enabled = gc.isenabled()
        try:
            while len(wall_ms) < args.max_repeats and (len(wall_ms) < args.repeats or total < args.min_time):
                gc.collect()
                gc.disable()  # same convention as timeit/pyperf: no GC pauses inside a sample
                c0 = time.process_time()
                t0 = time.perf_counter()
                native = runner.run(text)
                w = time.perf_counter() - t0
                c = time.process_time() - c0
                if gc_was_enabled:
                    gc.enable()
                wall_ms.append(w * 1000.0)
                cpu_ms.append(c * 1000.0)
                total += w
                if len(wall_ms) < args.max_repeats and (
                    len(wall_ms) < args.repeats or total < args.min_time
                ):
                    del native
        finally:
            if gc_was_enabled:
                gc.enable()

        out["rss_peak_mb"] = peak_rss_mb()  # read before any post-processing allocates more
        out["wall_ms"] = wall_ms
        out["cpu_ms"] = cpu_ms

        # Fresh-input samples: same work, but never the same string twice (see fresh_variant).
        if not args.no_fresh:
            fresh_ms: list[float] = []
            ftotal = 0.0
            # The cold call and the steady loop above used up i = 0..; offsets keep them unique.
            i = 1
            while len(fresh_ms) < args.max_repeats and (len(fresh_ms) < args.repeats or ftotal < args.min_time):
                variant = fresh_variant(text, i)
                i += 1
                gc.collect()
                gc.disable()
                t0 = time.perf_counter()
                res = runner.run(variant)
                w = time.perf_counter() - t0
                if gc_was_enabled:
                    gc.enable()
                del res, variant
                fresh_ms.append(w * 1000.0)
                ftotal += w
            out["fresh_wall_ms"] = fresh_ms

        views = runner.views(native)
        if args.emit_chunks:
            with open(args.emit_chunks, "w", encoding="utf-8") as f:
                json.dump([[v.delivered, v.body, list(v.headers)] for v in views], f, ensure_ascii=False)
        out["n_chunks"] = len(views)
        out["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - report every failure, including SystemExit
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc(limit=6)
    sys.stdout.write("CHUNKBENCH_RESULT " + json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
