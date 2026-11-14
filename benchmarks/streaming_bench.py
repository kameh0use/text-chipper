"""Large-file benchmark: text_chipper streaming vs other ways of chunking one big file.

Uses the contestant adapters of the main harness (chunkbench/adapters.py), so every library is
configured exactly as in the main benchmark: cl100k_base, library defaults, same reference
tokenizer for verification. One run = one mode in one fresh process.

Modes (`<adapter>:<how>`; adapter names are the ones in chunkbench.adapters.ADAPTERS):

  text_chipper:stream   TextChipper.from_file_stream(path), chunks consumed one at a time
  <adapter>:whole       read the whole file into one str and call the library once
  <adapter>:blocks      feed ~8 MiB blocks cut at line boundaries, one call per block
                        (how a str-only library is realistically given a big file; chunks at
                        block edges are cut early)

Lanes (same meaning as in the main harness):
  single   thread env vars = 1 and the process is pinned to one CPU
  all      no restriction

Timing is end to end for the chunking work: reading the file plus the library's own
chunk/split call. Verification (`--verify`) runs between calls and is NOT timed: it counts the
tokens of every delivered chunk with the reference tokenizer and hashes the delivered text.
A verify run builds extra per-chunk objects, so take memory numbers from runs WITHOUT --verify.
Memory is the growth of the process peak RSS after warm-up (interpreter, library import and
tokenizer load are excluded).
"""
import argparse
import hashlib
import json
import os
import sys
import time

MIB = 1024 * 1024
BLOCK_CHARS = 8 * MIB  # overridable with --block-mib (used by the self-test)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", required=True, help="<adapter>:<stream|whole|blocks>")
    ap.add_argument("--lane", choices=["single", "all"], default="single")
    ap.add_argument("--fmt", choices=["plain", "markdown"], default="plain",
                    help="passed to adapters that have a Markdown-aware variant (default: plain)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--reference", default="cl100k_base", help="reference tokenizer for --verify")
    ap.add_argument("--verify", action="store_true", help="independent token count + SHA-256 (not timed)")
    ap.add_argument("--block-mib", type=float, default=8, help="block size for :blocks modes")
    ap.add_argument("--out", help="append one JSON line with the result to this file")
    return ap.parse_args()


class Clock:
    t = 0.0


def timed(fn, *a):
    s = time.perf_counter()
    try:
        return fn(*a)
    finally:
        Clock.t += time.perf_counter() - s


def read_all(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def iter_blocks(path, block_chars=BLOCK_CHARS):
    buf, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            n += len(line)
            if n >= block_chars:
                yield "".join(buf)
                buf, n = [], 0
    if buf:
        yield "".join(buf)


class Stats:
    def __init__(self, ref, size):
        self.ref, self.size = ref, size
        self.n = self.worst = self.over = 0
        self.digest = hashlib.sha256()

    def add_text(self, text):
        self.n += 1
        if self.ref is not None:
            k = self.ref.count(text)
            self.worst = max(self.worst, k)
            self.over += k > self.size
            self.digest.update(text.encode("utf-8", "replace"))

    def add_native(self, runner, native):
        if self.ref is None:
            self.n += len(native)
        else:
            for v in runner.views(native):
                self.add_text(v.delivered)


def materialize(x):
    return x if hasattr(x, "__len__") else list(x)


def main():
    args = parse_args()
    size = args.max_length
    name, _, how = args.mode.partition(":")

    # Thread limits must be in the environment before any library creates its thread pool.
    from chunkbench import adapters
    from chunkbench.worker import peak_rss_mb, pin_to_one_cpu

    enforcement = "none"
    if args.lane == "single":
        os.environ.update(adapters.thread_env("single"))
        enforcement = pin_to_one_cpu()
    adapters.load_plugins()

    if name not in adapters.ADAPTERS:
        sys.exit(f"unknown adapter {name!r}; known: {', '.join(adapters.ADAPTERS)}")
    if how not in {"stream", "whole", "blocks"} or (how == "stream" and name != "text_chipper"):
        sys.exit(f"unknown or unsupported mode {args.mode!r}")

    ctx = adapters.Ctx(size, 0, args.fmt, "cl100k_base")
    sample = "# Title\n\nSome text to warm up.\n\n```python\nx = 1\n```\n"
    runner = adapters.ADAPTERS[name].build(ctx)  # imports the library, loads its tokenizer
    materialize(runner.run(sample))
    chipper = None
    if how == "stream":
        from text_chipper import TextChipper

        # Same settings as the `text_chipper` adapter.
        chipper = TextChipper(max_length=size, overlap=0, encoding="cl100k_base", include_context=True)
        list(chipper.chunk(sample))

    ref = None
    if args.verify:
        from chunkbench.tokenizer import get_reference

        ref = get_reference(args.reference)
        ref.count(sample)

    stats = Stats(ref, size)
    base = peak_rss_mb()
    cpu0 = time.process_time()

    if how == "stream":
        it = iter(chipper.from_file_stream(args.path))
        while True:
            s = time.perf_counter()
            try:
                ch = next(it)
            except StopIteration:
                Clock.t += time.perf_counter() - s
                break
            Clock.t += time.perf_counter() - s
            stats.add_text(ch.contextualized_text)
    elif how == "whole":
        text = timed(read_all, args.path)
        native = timed(lambda t: materialize(runner.run(t)), text)
        stats.add_native(runner, native)
        del native, text
    else:
        blocks = iter_blocks(args.path, int(args.block_mib * MIB))
        while True:
            try:
                block = timed(next, blocks)
            except StopIteration:
                break
            native = timed(lambda t: materialize(runner.run(t)), block)
            stats.add_native(runner, native)
            del native, block

    cpu = time.process_time() - cpu0
    peak = peak_rss_mb()
    file_mib = os.path.getsize(args.path) / MIB
    result = {
        "mode": args.mode, "lane": args.lane, "status": "ok", "fmt": args.fmt,
        "thread_enforcement": enforcement,
        "file_mib": round(file_mib, 1), "max_length": size,
        "time_s": round(Clock.t, 2), "mib_per_s": round(file_mib / Clock.t, 1),
        "cpu_s": round(cpu, 2) if not args.verify else None,
        "peak_rss_mb": round(peak), "growth_mb": round(peak - base),
        "chunks": stats.n, "verified": args.verify,
        "largest_chunk_tokens": stats.worst if args.verify else None,
        "chunks_over_limit": stats.over if args.verify else None,
        "sha256": stats.digest.hexdigest() if args.verify else None,
    }
    print(f"mode={args.mode} lane={args.lane} ({enforcement}) file={file_mib:.0f} MiB max_length={size}")
    print(f"time={Clock.t:.1f}s throughput={file_mib / Clock.t:.1f} MiB/s cpu={cpu:.1f}s")
    print(f"chunks={stats.n} peak_rss={peak:.0f} MB growth={peak - base:.0f} MB")
    if args.verify:
        print(f"largest_chunk={stats.worst} tokens, over_limit={stats.over}, sha256={result['sha256'][:16]}...")
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
