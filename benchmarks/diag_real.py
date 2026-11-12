#!/usr/bin/env python3
"""Diagnostics on the real corpora (run `python -m chunkbench fetch` from benchmarks/ first).

    python benchmarks/diag_real.py            # from the repository root, or from benchmarks/
    python benchmarks/diag_real.py --chunk-size 512 --min-time 0.5

For every corpus it prints, per contestant, three timings:

  first       the first call after construction (warm-up / lazy init included)
  steady      median of repeated calls with the *same* string
  fresh-text  median of repeated calls where every call gets a *different* string

`steady` rewards any cache keyed on the input; `fresh-text` is what a pipeline that sees each
document once experiences. When steady is far below fresh-text (or below one plain tiktoken
pass), the contestant is answering from a cache.

For text_chipper it also breaks down tokenizer work per call site. That needs a build with the
`stats` feature (`maturin develop --release --features stats`). The control-flow *events*
(merge_fast_direct, merge_estimate_rejected, ...) are listed separately: they are not tokenizer
calls and their byte counts must not be added to the tokenizer totals.
"""
from __future__ import annotations

import argparse
import gc
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkbench import adapters, corpora  # noqa: E402
from chunkbench.worker import fresh_variant  # noqa: E402

MIB = 1024 * 1024

# Sites in backend::CountSite that mark a decision, not a tokenizer call.
EVENT_SITES = frozenset(
    {
        "merge_fast_direct", "merge_fast_prefix", "merge_exact_fallback", "merge_fast_disabled",
        "merge_estimate_rejected", "merge_prefix_candidate", "merge_candidate_too_large",
        "merge_exact_accept", "merge_exact_reject", "merge_early_reject",
        "merge_estimate_under_2plus", "merge_estimate_under_1", "merge_estimate_equal",
        "merge_estimate_over_1", "merge_estimate_over_2", "merge_estimate_over_3",
        "merge_estimate_over_4plus", "merge_margin1_would_accept", "merge_margin1_unsafe",
        "merge_margin0_would_accept", "merge_margin0_unsafe",
    }
)


def timed(fn, arg) -> float:
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        fn(arg)
        return (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()


def steady_ms(fn, text: str, min_time: float, repeats: int) -> float:
    fn(text)  # warm-up, excluded
    samples, total = [], 0.0
    while len(samples) < repeats or total < min_time:
        ms = timed(fn, text)
        samples.append(ms)
        total += ms / 1000.0
        if len(samples) >= 500:
            break
    return statistics.median(samples)


def fresh_ms(fn, text: str, min_time: float, repeats: int) -> float:
    samples, total, i = [], 0.0, 1
    while len(samples) < repeats or total < min_time:
        variant = fresh_variant(text, i)  # built outside the timer
        i += 1
        ms = timed(fn, variant)
        samples.append(ms)
        total += ms / 1000.0
        if len(samples) >= 500:
            break
    return statistics.median(samples)


def line(label: str, nbytes: int, first: float | None, steady: float | None, fresh: float | None) -> str:
    parts = []
    if first is not None:
        parts.append(f"first {first:8.2f} ms")
    if steady is not None:
        parts.append(f"steady {steady:7.2f} ms ({nbytes / MIB / (steady / 1000.0):6.1f} MiB/s)")
    if fresh is not None:
        parts.append(f"fresh-text {fresh:7.2f} ms")
    return f"  {label:<28} " + " | ".join(parts)


class EncodeCounter:
    """Counts what is passed to tiktoken's encode family while active."""

    NAMES = ("encode", "encode_ordinary", "encode_with_unstable", "encode_batch", "encode_ordinary_batch",
             "encode_to_numpy", "encode_single_token")

    def __init__(self) -> None:
        import tiktoken

        self.cls = type(tiktoken.get_encoding("cl100k_base"))
        self.calls = 0
        self.chars = 0
        self._orig: dict[str, object] = {}

    def __enter__(self) -> "EncodeCounter":
        for name in self.NAMES:
            orig = getattr(self.cls, name, None)
            if orig is None:
                continue
            self._orig[name] = orig

            def make(orig=orig, name=name):
                def wrapper(inst, text, *a, **kw):
                    self.calls += 1
                    self.chars += sum(len(t) for t in text) if isinstance(text, (list, tuple)) else len(text)
                    return orig(inst, text, *a, **kw)

                return wrapper

            try:
                setattr(self.cls, name, make())
            except (TypeError, AttributeError):  # immutable extension type: cannot instrument
                self._orig.pop(name, None)
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(self.cls, name, orig)


def tokenizer_report(chipper, text: str, tc) -> None:
    chipper.chunk(text)  # warm-up
    tc._reset_tokenizer_stats()
    t0 = time.perf_counter()
    chipper.chunk(text)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    stats = tc._tokenizer_stats()

    sites = {k: v for k, v in stats["sites"].items() if v["calls"]}
    tok = {k: v for k, v in sites.items() if k not in EVENT_SITES}
    events = {k: v for k, v in sites.items() if k in EVENT_SITES}
    calls = sum(v["calls"] for v in tok.values())
    nbytes = sum(v["input_bytes"] for v in tok.values())
    ms = sum(v["elapsed_ms"] for v in tok.values())
    size = len(text.encode("utf-8"))
    pct = 100.0 * ms / wall_ms if wall_ms else 0.0
    print(f"  text_chipper tokenizer work (one call): {calls:,} count() calls, "
          f"{nbytes / size:.2f}x the text's bytes, {ms:.2f} ms of {wall_ms:.2f} ms wall ({pct:.0f}%)")
    for name, v in sorted(tok.items(), key=lambda kv: -kv[1]["elapsed_ms"]):
        print(f"      {name:<24} {v['calls']:>6,} calls {v['input_bytes']:>12,} bytes {v['elapsed_ms']:>8.3f} ms")
    if events:
        ev = ", ".join(f"{k}={v['calls']}" for k, v in sorted(events.items(), key=lambda kv: -kv[1]["calls"]))
        print(f"      chunker events (no tokenizer call): {ev}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--min-time", type=float, default=0.3, help="seconds of timed samples per measurement")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--corpora", nargs="*", help="corpus stems, default: every real corpus found")
    args = ap.parse_args()

    import tiktoken
    import text_chipper as tc

    enc = tiktoken.get_encoding("cl100k_base")
    have_stats = hasattr(tc, "_tokenizer_stats")
    print(f"text_chipper {tc.__version__}; stats build: {'yes' if have_stats else 'no'}; "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'unset')}; python {platform.python_version()}")

    works = {w.name.removeprefix("real_"): w for w in corpora.real_workloads(0, 1)}
    if not works:
        print("no corpora found: run `python -m chunkbench fetch` from benchmarks/ first", file=sys.stderr)
        return 2
    names = args.corpora or sorted(works, key=lambda n: works[n].size_bytes)

    def chipper(**kw):
        return tc.TextChipper(max_length=args.chunk_size, overlap=0, encoding="cl100k_base", **kw)

    for name in names:
        text = works[name].text
        size = len(text.encode("utf-8"))
        ntok = len(enc.encode(text, disallowed_special=()))
        print(f"\n== {name}: {size:,} bytes, {ntok:,} tokens ({size / ntok:.2f} bytes/token)")

        floor = lambda t: enc.encode(t, disallowed_special=())  # noqa: E731
        print(line("tiktoken.encode (floor)", size, None, steady_ms(floor, text, args.min_time, args.repeats), None))

        contestants = [
            ("text_chipper (default)", chipper()),
            ("text_chipper strict=False", chipper(strict=False)),
            ("text_chipper flat (ctx off)", chipper(include_context=False)),
        ]
        default_chipper = contestants[0][1]
        for label, c in contestants:
            first = timed(c.chunk, text)
            print(line(label, size, first, steady_ms(c.chunk, text, args.min_time, args.repeats),
                       fresh_ms(c.chunk, text, args.min_time, args.repeats)))

        try:
            t0 = time.perf_counter()
            runner = adapters.ADAPTERS["chonkie_recursive"].build(adapters.Ctx(args.chunk_size, 0, "markdown"))
            build_ms = (time.perf_counter() - t0) * 1000.0
            first = timed(runner.run, text)
            print(line("chonkie RecursiveChunker", size, first, steady_ms(runner.run, text, args.min_time, args.repeats),
                       fresh_ms(runner.run, text, args.min_time, args.repeats)) + f"   (build {build_ms:.0f} ms)")
            with EncodeCounter() as ec:
                runner.run(text)
            print(f"    chonkie passed to tiktoken.encode*, per run: {ec.calls} calls, {ec.chars:,} chars = "
                  f"{100.0 * ec.chars / len(text):.0f}% of the text")
        except Exception as exc:  # noqa: BLE001 - diagnostics must keep going
            print(f"  chonkie RecursiveChunker     skipped: {type(exc).__name__}: {exc}")

        if have_stats:
            tokenizer_report(default_chipper, text, tc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
