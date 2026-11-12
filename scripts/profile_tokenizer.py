#!/usr/bin/env python3
"""Profile tokenizer calls made by text_chipper on the benchmark workloads of
benchmarks/chunkbench/corpora.py.

Build the extension with:
    maturin develop --release --features stats
Then run:
    python scripts/profile_tokenizer.py

The warm-up is excluded from counters and wall-clock timing. Backend statistics are
process-global and include tokenizer calls from the full chunk operation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from chunkbench.corpora import build_workloads  # noqa: E402
import text_chipper  # noqa: E402

# CountSite entries that mark a decision rather than a tokenizer call (see src/backend.rs).
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

DEFAULT_WORKLOADS = (
    "prose_en",
    "markdown_doc",
    "multilingual",
    "code_heavy",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workloads", nargs="*", default=list(DEFAULT_WORKLOADS))
    args = parser.parse_args()

    workloads = build_workloads(include_real=False)
    missing = [name for name in args.workloads if name not in workloads]
    if missing:
        parser.error(f"unknown workloads: {', '.join(missing)}")

    if not hasattr(text_chipper, "_reset_tokenizer_stats"):
        raise SystemExit(
            "Tokenizer stats are not enabled. Rebuild with: "
            "maturin develop --release --features stats"
        )

    for name in args.workloads:
        text = workloads[name].text
        chipper = text_chipper.TextChipper(
            max_length=args.chunk_size,
            overlap=0,
            encoding="cl100k_base",
        )

        # Warm tokenizer / allocator paths before resetting the counters.
        chipper.chunk(text)
        text_chipper._reset_tokenizer_stats()

        started = time.perf_counter()
        chunks = chipper.chunk(text)
        wall_ms = (time.perf_counter() - started) * 1000.0
        stats = text_chipper._tokenizer_stats()

        # The stats also hold control-flow events (merge_fast_direct, merge_estimate_rejected, ...)
        # that are decisions, not tokenizer calls; keep them out of the call and byte totals.
        real_sites = {k: v for k, v in stats["sites"].items() if k not in EVENT_SITES}
        events = {k: v for k, v in stats["sites"].items() if k in EVENT_SITES and v["calls"]}
        calls = sum(v["calls"] for v in real_sites.values())
        tokenizer_ms = sum(v["elapsed_ms"] for v in real_sites.values())
        input_bytes = sum(v["input_bytes"] for v in real_sites.values())
        pct = (100.0 * tokenizer_ms / wall_ms) if wall_ms else 0.0
        per_chunk = calls / max(1, len(chunks))

        print(f"\n== {name} ({len(text.encode('utf-8')) / (1024 * 1024):.2f} MiB)")
        print(
            f"chunks={len(chunks)} wall={wall_ms:.2f} ms "
            f"tokenizer={tokenizer_ms:.2f} ms ({pct:.1f}% of wall) "
            f"calls={calls} ({per_chunk:.2f}/chunk) "
            f"input={input_bytes / (1024 * 1024):.2f} MiB"
        )
        print("site                       calls   input MiB   time ms   wall %   us/call")
        for site, values in real_sites.items():
            site_calls = values["calls"]
            if site_calls == 0:
                continue
            site_ms = values["elapsed_ms"]
            site_pct = (100.0 * site_ms / wall_ms) if wall_ms else 0.0
            us_per_call = (site_ms * 1000.0 / site_calls) if site_calls else 0.0
            print(
                f"{site:<25} {site_calls:>7} "
                f"{values['input_bytes'] / (1024 * 1024):>10.2f} "
                f"{site_ms:>9.2f} {site_pct:>8.1f}% {us_per_call:>9.2f}"
            )

        if events:
            print("events (not tokenizer calls): " + ", ".join(f"{k}={v['calls']}" for k, v in events.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
