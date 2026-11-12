#!/usr/bin/env python3
"""Explain exactly which source characters text_chipper leaves out of its chunk bodies.

    python benchmarks/diag_lost.py                      # synthetic markdown_doc, 3 MiB (report default)
    python benchmarks/diag_lost.py --size-mib 1.0       # match a CI run (benchmark.yaml default)
    python benchmarks/diag_lost.py --file README.md     # any Markdown file
    python benchmarks/diag_lost.py --samples 25

It runs `text_chipper` with the benchmark's settings, finds every non-whitespace character that is
not inside some chunk body, and sorts those characters into:

  carried     the line is a heading whose title the next chunk lists in `headers`
  heading     the line is a heading but the next chunk does NOT list its title: really lost
  fence       inside a fenced code block
  other       ordinary text that no chunk contains: really lost

Only `heading` and `other` are real losses; `carried` is what `content cov %` credits.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from chunkbench import corpora, metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="chunk this file instead of the synthetic markdown_doc workload")
    ap.add_argument("--size-mib", type=float, default=3.0)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--encoding", default="cl100k_base")
    ap.add_argument("--samples", type=int, default=15, help="examples to print per category")
    args = ap.parse_args()

    import text_chipper

    if args.file:
        source = Path(args.file).read_text(encoding="utf-8")
    else:
        source = corpora.build_workloads(size_mib=args.size_mib, include_real=False)["markdown_doc"].text
    chipper = text_chipper.TextChipper(
        max_length=args.chunk_size, overlap=args.overlap, encoding=args.encoding, include_context=True
    )
    chunks = chipper.chunk(source)
    bodies = [c.text for c in chunks]
    headers = [tuple(c.headers) for c in chunks]

    loc = metrics.locate(source, bodies, ordered_overlap_ok=args.overlap > 0)
    found = [(i, sp) for i, sp in enumerate(loc.spans) if sp is not None]
    print(f"chunks: {len(chunks)}   located verbatim: {loc.n_located}/{len(chunks)}")
    if loc.n_located < len(chunks):
        print("  WARNING: some chunk bodies are not verbatim source slices; coverage is understated")

    n = len(source)
    diff = np.zeros(n + 1, dtype=np.int64)
    for _, (s, e) in found:
        diff[s] += 1
        diff[e] -= 1
    covered = np.cumsum(diff[:n]) > 0
    nonspace = np.fromiter((not c.isspace() for c in source), dtype=bool, count=n)
    total = int(nonspace.sum())
    lost_total = int((~covered & nonspace).sum())
    print(f"non-whitespace chars: {total:,}   not in any body: {lost_total:,} ({100 * lost_total / total:.3f}%)")

    sorted_starts = sorted(s for _, (s, _e) in found)
    start_to_idx = {sp[0]: i for i, sp in found}
    heading_at = {}  # line start -> (end, title)
    for hs, he, title in metrics._heading_lines(source):
        heading_at[hs] = (he, title)
    fences = metrics._fences(source)
    fence_starts = [f[0] for f in fences]

    def in_fence(pos: int) -> bool:
        k = bisect.bisect_right(fence_starts, pos) - 1
        return k >= 0 and fences[k][0] <= pos <= fences[k][1]

    cats: dict[str, int] = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)

    def note(cat: str, chars: int, text: str) -> None:
        cats[cat] += chars
        if len(samples[cat]) < args.samples:
            samples[cat].append(text)

    pos = 0
    heading_lines = sorted(heading_at)
    for line in source.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        pos = end
        miss = int((~covered[start:end] & nonspace[start:end]).sum())
        if miss == 0:
            continue
        txt = line.rstrip("\r\n")
        if in_fence(start):
            note("fence", miss, txt[:100])
            continue
        # A heading may span several lines (setext): look it up by any line inside its range.
        k = bisect.bisect_right(heading_lines, start) - 1
        h = heading_at.get(heading_lines[k]) if k >= 0 else None
        if h and start < h[0]:
            he, title = h
            j = bisect.bisect_left(sorted_starts, he)
            nxt = headers[start_to_idx[sorted_starts[j]]] if j < len(sorted_starts) else ()
            t = unicodedata.normalize("NFC", title)
            ok = any(t == x or t.startswith(x) for x in (unicodedata.normalize("NFC", y) for y in nxt) if x)
            note("carried" if ok else "heading", miss, f"{txt[:70]!r}   next chunk headers={list(nxt)}")
        else:
            note("other", miss, txt[:100])

    print()
    for cat in ("carried", "heading", "fence", "other"):
        print(f"{cat:8s} {cats[cat]:>9,} chars")
        for s in samples[cat]:
            print(f"    {s}")
    real = cats["heading"] + cats["fence"] + cats["other"]
    print(f"\nreal loss (heading + fence + other): {real:,} chars = {100 * real / total:.3f}% "
          f"-> content cov % = {100 - 100 * real / total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
