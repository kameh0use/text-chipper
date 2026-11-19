"""Cross-tokenizer matrix benchmark: chunk with tokenizer A, recount with tokenizer B.

Tokenizers: cl100k_base, o200k_base, p50k_base, r50k_base, llama3, llama4, qwen, deepseek, BGE
(chunker AND consumer) and Claude (consumer only, via the token-counting API, sampled). See
chunkbench/crosstok.py for the metrics.

    pip install -r requirements.txt
    export HF_TOKEN=...              # needed for the gated Llama repos
    export ANTHROPIC_API_KEY=...     # optional, enables the Claude columns
    python benchmarks/cross_tokenizer.py
    python benchmarks/cross_tokenizer.py --headroom 0      # old behaviour: budget == limit
    python benchmarks/cross_tokenizer.py --guard           # + text_chipper chunked with the MAX over all tokenizers
    python benchmarks/cross_tokenizer.py --render benchmarks/cross_tokenizer_results/matrix/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunkbench import adapters, corpora, crosstok  # noqa: E402

DEFAULT_CLAUDE_MODELS = ["claude-sonnet-5-5"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-mib", type=float, default=0.25, help="size of each generated workload")
    ap.add_argument("--limit", type=int, default=512, help="hard token limit of the consumer")
    ap.add_argument("--headroom", type=int, default=2,
                    help="chunk budget = limit - headroom (2 = BGE's [CLS]/[SEP]; 0 reproduces the old run)")
    ap.add_argument("--chunk-size", type=int, default=None, help="override the chunk budget directly")
    ap.add_argument("--guard", action="store_true",
                    help="add a `guard` row: text_chipper chunked with guard tokenizers (budget enforced on the maximum "
                         "count over every loaded tokenizer). Needs a text_chipper build with guard_encodings.")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--no-real", action="store_true", help="skip downloaded real-world corpora")
    ap.add_argument("--libs", nargs="*", default=list(adapters.ADAPTERS), choices=list(adapters.ADAPTERS))
    ap.add_argument("--workloads", nargs="*")
    ap.add_argument("--tokenizers", nargs="*", default=crosstok.DEFAULT_TOKENIZERS, choices=list(crosstok.SPECS),
                    help="tokenizers to load (used as chunker tokenizers and as consumers)")
    ap.add_argument("--chunk-with", nargs="*", help="subset of --tokenizers used to configure the chunkers")
    ap.add_argument("--hf-repo", action="append", default=[], metavar="KEY=REPO",
                    help="use another Hugging Face repo for a key, e.g. llama3=your-org/llama-3-tokenizer-mirror")
    ap.add_argument("--tokenizer-dir", type=Path, default=HERE / "tokenizers")
    ap.add_argument("--lock", type=Path, default=HERE / "tokenizers" / "tokenizers.lock.json")
    ap.add_argument("--strict", action="store_true", help="fail if any tokenizer cannot be loaded")
    ap.add_argument("--require-real", action="store_true", help="fail unless at least one pinned real corpus is included")
    ap.add_argument("--assert-guard-zero", action="store_true",
                    help="acceptance mode: fail unless guard has results and zero overflow for every loaded tokenizer")
    ap.add_argument("--offline", action="store_true", help="never download tokenizer files")
    ap.add_argument("--claude-models", nargs="*", default=None,
                    help=f"Claude model ids to count with (default {DEFAULT_CLAUDE_MODELS} when ANTHROPIC_API_KEY is set)")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--claude-sample", type=int, default=25, help="chunks counted per workload x library x chunker tokenizer")
    ap.add_argument("--claude-libs", nargs="*", default=["text_chipper"])
    ap.add_argument("--focus", action="append", default=None, metavar="A:B",
                    help="chunker:consumer pairs for the per-workload tables (default cl100k_base:BGE and BGE:BGE)")
    ap.add_argument("--headline-lib", default="text_chipper")
    ap.add_argument("--out", type=Path, default=HERE / "cross_tokenizer_results" / "matrix")
    ap.add_argument("--render", type=Path, help="re-render report.md from an existing results.json and exit")
    return ap.parse_args(argv)


def write_outputs(out: Path, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "rows.csv").write_text(crosstok.rows_to_csv(res["results"]), encoding="utf-8")
    (out / "report.md").write_text(crosstok.render_report(res), encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.render:
        res = json.loads(args.render.read_text(encoding="utf-8"))
        (args.render.parent / "report.md").write_text(crosstok.render_report(res), encoding="utf-8")
        print(f"Wrote {args.render.parent / 'report.md'}")
        return 0

    budget = args.chunk_size if args.chunk_size is not None else args.limit - args.headroom
    overrides = dict(kv.split("=", 1) for kv in args.hf_repo)

    counters, files, meta, unavailable = crosstok.load_counters(
        args.tokenizers, args.tokenizer_dir, args.lock, strict=args.strict, offline=args.offline,
        repo_overrides=overrides)
    if not counters:
        print("no tokenizer could be loaded:", json.dumps(unavailable, indent=2), file=sys.stderr)
        return 2
    for k, why in unavailable.items():
        print(f"! tokenizer {k} unavailable: {why}", file=sys.stderr)
    if any(m.get("first_use") for m in meta.values()):
        print(f"! new tokenizer hashes were recorded in {args.lock}; commit that file with the results.", file=sys.stderr)

    chunk_with = [k for k in (args.chunk_with or list(counters)) if k in counters]

    claude, claude_note = {}, "Not run."
    if args.no_claude:
        claude_note = "Not run (`--no-claude`)."
    elif args.claude_models is None and not os.environ.get("ANTHROPIC_API_KEY"):
        claude_note = "Not run: set `ANTHROPIC_API_KEY` to enable the Claude columns."
    else:
        cache = args.out / "claude_count_cache.json"
        for model in (args.claude_models or DEFAULT_CLAUDE_MODELS):
            try:
                cc = crosstok.ClaudeCounter(model, cache_path=cache)
                claude[cc.key] = cc
            except Exception as exc:  # noqa: BLE001
                claude_note = f"Not run: {type(exc).__name__}: {exc}"
                if args.strict:
                    raise
                print(f"! Claude counter {model} unavailable: {exc}", file=sys.stderr)

    workloads = corpora.build_workloads(args.size_mib, seed=args.seed, include_real=not args.no_real)
    if args.workloads:
        missing = sorted(set(args.workloads) - set(workloads))
        if missing:
            print(f"unknown workload(s): {', '.join(missing)}", file=sys.stderr)
            return 2
        workloads = {n: workloads[n] for n in args.workloads}
    if args.require_real and not any(w.kind == "real" for w in workloads.values()):
        print("acceptance run requires at least one real corpus; fetch pinned corpora first", file=sys.stderr)
        return 2

    run = crosstok.run_matrix(
        list(workloads.values()), args.libs, chunk_with, counters, files, adapters_mod=adapters,
        budget=budget, limit=args.limit, seed=args.seed, claude=claude or None,
        claude_libs=args.claude_libs, claude_sample=args.claude_sample, guard=args.guard)

    if args.assert_guard_zero:
        if not args.guard:
            print("--assert-guard-zero requires --guard", file=sys.stderr)
            return 2
        guarded = [r for r in run["pooled"] if r["adapter"] == "text_chipper" and
                   r["chunk_tok"] == crosstok.GUARD_KEY and r["group"] == "all"]
        expected = set(counters)
        got = {r["count_tok"] for r in guarded}
        bad = [r for r in guarded if r["over"] != 0]
        if got != expected or bad or run["failures"]:
            print(f"guard acceptance failed: expected consumers={sorted(expected)}, got={sorted(got)}, "
                  f"overflow rows={[(r['count_tok'], r['over']) for r in bad]}, "
                  f"chunk failures={len(run['failures'])}", file=sys.stderr)
            return 1

    res = {
        "benchmark": "cross-tokenizer-matrix",
        "params": {"limit": args.limit, "chunk_size": budget, "headroom": args.limit - budget,
                   "size_mib": args.size_mib, "seed": args.seed, "include_real": not args.no_real,
                   "focus": args.focus or ["cl100k_base:BGE", "BGE:BGE"], "headline_lib": args.headline_lib,
                   "claude_sample": args.claude_sample, "claude_libs": args.claude_libs if claude else [],
                   "guard": args.guard},
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            **{d: adapters.version_of(d) for d in ("text-chipper", "chonkie", "semantic-text-splitter", "semchunk",
                                                   "langchain-text-splitters", "tiktoken", "tokenizers",
                                                   "transformers", "huggingface_hub", "anthropic")},
        },
        "tokenizers": meta,
        "unavailable": unavailable,
        "chunk_with": chunk_with,
        "claude_models": list(claude),
        "claude_note": claude_note,
        "adapters": list(args.libs),
        "libraries": {n: adapters.ADAPTERS[n].label for n in args.libs},
        "workloads": [{"name": w.name, "kind": w.kind, "bytes": w.size_bytes} for w in workloads.values()],
        **run,
    }
    write_outputs(args.out, res)
    print(f"\nWrote {args.out / 'report.md'}, results.json and rows.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
