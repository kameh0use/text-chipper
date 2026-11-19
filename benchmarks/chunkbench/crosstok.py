"""Cross-tokenizer matrix: chunk with tokenizer A, recount the same text with tokenizer B.

Why a matrix and not a single pair
  Chunkers are configured with ONE tokenizer, but the text is later consumed by another one: an
  embedding model (BGE, ...) or an LLM (Llama, Qwen, DeepSeek, Claude, ...). A chunk that is "512
  tokens" for cl100k_base is some other number for every other vocabulary. This module measures that
  gap for every (chunker tokenizer A) x (consumer tokenizer B) pair.

Terms
  limit      the hard limit of the consumer (e.g. 512 for BGE).
  budget     what the chunker is asked for (`chunk_size`). Default: limit - headroom.
  headroom   tokens the consumer adds on its own. BGE adds [CLS] and [SEP] (2). LLM tokenizers are
             counted without BOS/EOS because a chunk is normally one part of a longer prompt.
  overflow   a chunk whose count under B (specials included where `specials` is true) is > limit.
  excess     sum over chunks of max(0, count_B - limit): how many tokens a truncating consumer loses.
  fill       count_A / budget. A chunker can trivially avoid overflow by under-filling, so fill is
             reported next to every overflow number.

Everything here is import-light: third-party packages are imported lazily so that the statistics and
report code can be tested without tiktoken/tokenizers installed.
"""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import os
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from . import metrics

NATURAL_KINDS = ("synthetic", "real")
GROUPS = ("natural", "stress", "all")


def group_of(kind: str) -> str:
    return "natural" if kind in NATURAL_KINDS else "stress"


# --------------------------------------------------------------------------- tokenizer registry


@dataclass(frozen=True)
class TokSpec:
    key: str  # label used everywhere (also passed to the chunkers as the "encoding" name)
    kind: str  # "tiktoken" | "hf"
    encoding: str | None = None  # tiktoken encoding name
    repo: str | None = None  # Hugging Face repo that hosts tokenizer.json
    file: str | None = None  # local file name under the tokenizer directory
    specials: bool = False  # count model-added special tokens when this is the CONSUMER (embedders)
    gated: bool = False
    note: str = ""


SPECS: dict[str, TokSpec] = {
    s.key: s
    for s in [
        TokSpec("cl100k_base", "tiktoken", encoding="cl100k_base", note="GPT-3.5 / GPT-4"),
        TokSpec("o200k_base", "tiktoken", encoding="o200k_base", note="GPT-4o and later OpenAI models"),
        TokSpec("p50k_base", "tiktoken", encoding="p50k_base", note="Codex / older OpenAI models"),
        TokSpec("r50k_base", "tiktoken", encoding="r50k_base", note="GPT-2 / older OpenAI models"),
        TokSpec("llama3", "hf", repo="meta-llama/Meta-Llama-3-8B", file="llama3-tokenizer.json", gated=True,
                note="Llama 3 / 3.1 / 3.2 / 3.3 share one 128k vocabulary"),
        TokSpec("llama4", "hf", repo="meta-llama/Llama-4-Scout-17B-16E", file="llama4-tokenizer.json", gated=True,
                note="Llama 4 uses a different, larger vocabulary than Llama 3 (check vocab_size below)"),
        TokSpec("qwen", "hf", repo="Qwen/Qwen3-8B", file="qwen3-tokenizer.json",
                note="Qwen2.5 / Qwen3 family tokenizer"),
        TokSpec("deepseek", "hf", repo="deepseek-ai/DeepSeek-V3", file="deepseek-v3-tokenizer.json",
                note="DeepSeek-V3 / R1 tokenizer"),
        TokSpec("BGE", "hf", repo="BAAI/bge-large-en-v1.5", file="bge-large-en-v1.5-tokenizer.json", specials=True,
                note="embedding model; [CLS]/[SEP] are counted"),
    ]
}
DEFAULT_TOKENIZERS = list(SPECS)

# Pseudo chunker key: text_chipper configured with guard tokenizers, i.e. its budget is enforced on the
# LARGEST count over every loaded tokenizer. It is a row of the matrices, never a consumer column.
GUARD_KEY = "guard"
GUARD_ADAPTERS = ("text_chipper", "text_chipper_flat")


class Counter(Protocol):
    key: str
    specials: bool

    def count_many(self, texts: Sequence[str], specials: bool = False) -> list[int]: ...


class TiktokenCounter:
    def __init__(self, spec: TokSpec) -> None:
        import tiktoken

        self.key = spec.key
        self.specials = False
        self._enc = tiktoken.get_encoding(spec.encoding)
        self.vocab_size = self._enc.n_vocab

    def count_many(self, texts: Sequence[str], specials: bool = False) -> list[int]:
        # encode_ordinary: text such as "<|endoftext|>" is counted as ordinary text, like the old script.
        return [len(ids) for ids in self._enc.encode_ordinary_batch(list(texts))]


class HFCounter:
    def __init__(self, spec: TokSpec, path: Path) -> None:
        from tokenizers import Tokenizer

        self.key = spec.key
        self.specials = spec.specials
        self._tok = Tokenizer.from_file(str(path))
        # Some tokenizer.json files ship with truncation (e.g. max_length=512) switched on. That would
        # silently cap every count at the model limit and hide exactly the overflow we measure.
        self._tok.no_truncation()
        self._tok.no_padding()
        self.vocab_size = self._tok.get_vocab_size()

    def count_many(self, texts: Sequence[str], specials: bool = False) -> list[int]:
        texts = list(texts)
        if not texts:
            return []
        return [len(e.ids) for e in self._tok.encode_batch(texts, add_special_tokens=specials)]


class MaxCounter:
    """Element-wise maximum of several counters: what a guard-configured chunker budgets against.
    Plain counts only (model-added special tokens are covered by the budget headroom, as for every
    other chunker), so `specials` is accepted and ignored."""

    specials = False

    def __init__(self, members: Sequence[Counter]) -> None:
        self.members = list(members)

    def count_many(self, texts: Sequence[str], specials: bool = False) -> list[int]:
        per = [m.count_many(texts) for m in self.members]
        return [max(col) for col in zip(*per)] if per else [0] * len(texts)


def guard_ctx(adapters_mod: Any, budget: int, fmt: str, counters: dict, tokfiles: dict) -> Any:
    """Adapter context for the guard row: the first tiktoken encoding (or tokenizer file) is the primary
    tokenizer and every other loaded tokenizer is a guard."""
    encs = [SPECS[k].encoding for k in counters if k in SPECS and SPECS[k].kind == "tiktoken" and SPECS[k].encoding]
    files = [tokfiles[k] for k in counters if tokfiles.get(k)]
    if encs:
        return adapters_mod.Ctx(budget, 0, fmt, encs[0], None, tuple(encs[1:]), tuple(files))
    return adapters_mod.Ctx(budget, 0, fmt, adapters_mod.ENCODING, files[0] if files else None, (), tuple(files[1:]))


class ClaudeCounter:
    """Consumer-only counter backed by Anthropic's token-counting endpoint.

    There is no public offline tokenizer for current Claude models, so Claude can only be a consumer
    (column), never a chunker tokenizer (row), and every count costs one API call. Counts are cached on
    disk (keyed by model + text hash) and callers sample chunks instead of counting all of them.
    The fixed per-request message overhead is measured once with a one-token probe and subtracted.
    """

    specials = False
    RETRY_STATUS = (408, 409, 429, 500, 502, 503, 529)

    def __init__(self, model: str, cache_path: Path | None = None, client: Any = None, workers: int = 8,
                 max_retries: int = 6, sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model
        self.key = f"claude:{model}"
        self._cache_path = cache_path
        self._workers = workers
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = client
        self._cache: dict[str, int] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.overhead = self._probe_overhead()

    def _api(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def _raw(self, text: str) -> int | None:
        k = hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()
        if k in self._cache:
            return self._cache[k]
        for attempt in range(self._max_retries + 1):
            try:
                r = self._api().messages.count_tokens(model=self.model, messages=[{"role": "user", "content": text}])
                self._cache[k] = int(r.input_tokens)
                return self._cache[k]
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                if status in self.RETRY_STATUS and attempt < self._max_retries:
                    self._sleep(min(60, 2 ** attempt))
                    continue
                if status == 400:  # e.g. whitespace-only text is rejected: not countable, skip it
                    return None
                raise
        return None

    def _probe_overhead(self) -> int:
        raw = self._raw("a")
        return 0 if raw is None else max(0, raw - 1)  # "a" is a single token in every vocabulary

    def count_many(self, texts: Sequence[str], specials: bool = False) -> list[int | None]:  # type: ignore[override]
        with concurrent.futures.ThreadPoolExecutor(self._workers) as pool:
            raw = list(pool.map(self._raw, texts))
        self.flush()
        return [None if n is None else n - self.overhead for n in raw]

    def flush(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


# --------------------------------------------------------------------------- pinning (lock file)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lock(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_lock(path: Path, lock: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hf_download(repo: str, revision: str | None, dest: Path) -> str:
    """Download tokenizer.json into `dest`, return the commit hash it came from."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo, filename="tokenizer.json", revision=revision,
                             token=os.environ.get("HF_TOKEN") or None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    return Path(cached).parent.name  # <cache>/snapshots/<commit>/tokenizer.json


def resolve_hf_file(spec: TokSpec, tok_dir: Path, lock: dict[str, dict], *, repo: str | None = None,
                    offline: bool = False, download: Callable[[str, str | None, Path], str] = _hf_download
                    ) -> tuple[Path, dict, bool]:
    """Return (path, lock entry, first_use). Verifies the file against the lock when it has a hash.

    First use of a tokenizer (no hash in the lock) is trust-on-first-use: the hash and commit are
    recorded so later runs are pinned. Commit the lock file next to the results.
    """
    repo = repo or spec.repo
    path = tok_dir / spec.file
    entry = dict(lock.get(spec.key, {}))
    if entry.get("repo") not in (None, repo):  # user pointed this key at another repo: do not reuse pins
        entry = {}
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"{path} is missing and --offline is set")
        try:
            commit = download(repo, entry.get("revision"), path)
        except Exception as exc:  # noqa: BLE001
            hint = " (gated repo: accept the licence on huggingface.co and set HF_TOKEN)" if spec.gated else ""
            raise RuntimeError(f"could not download {repo}/tokenizer.json{hint}: {exc}") from exc
        entry.setdefault("revision", commit)
    digest = sha256_file(path)
    first_use = "sha256" not in entry
    if not first_use and entry["sha256"] != digest:
        raise RuntimeError(f"{path.name}: sha256 {digest} does not match the pinned {entry['sha256']}")
    entry["sha256"] = digest
    entry["repo"] = repo
    return path, entry, first_use


def load_counters(keys: Sequence[str], tok_dir: Path, lock_path: Path, *, strict: bool = False,
                  offline: bool = False, repo_overrides: dict[str, str] | None = None):
    """Build a counter per tokenizer key. Unavailable tokenizers are reported, not fatal, unless strict."""
    lock = load_lock(lock_path)
    before = json.dumps(lock, sort_keys=True)
    counters: dict[str, Counter] = {}
    files: dict[str, str | None] = {}
    meta: dict[str, dict] = {}
    unavailable: dict[str, str] = {}
    for key in keys:
        spec = SPECS[key]
        try:
            if spec.kind == "tiktoken":
                c: Counter = TiktokenCounter(spec)
                meta[key] = {"source": f"tiktoken:{spec.encoding}", "vocab_size": c.vocab_size, "note": spec.note}
                files[key] = None
            else:
                path, entry, first_use = resolve_hf_file(spec, tok_dir, lock, repo=(repo_overrides or {}).get(key),
                                                         offline=offline)
                c = HFCounter(spec, path)
                entry["vocab_size"] = c.vocab_size
                lock[key] = entry
                meta[key] = {"source": f"hf:{entry['repo']}", "revision": entry.get("revision", "?"),
                             "sha256": entry["sha256"], "vocab_size": c.vocab_size, "first_use": first_use,
                             "note": spec.note}
                files[key] = str(path)
            counters[key] = c
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            unavailable[key] = f"{type(exc).__name__}: {exc}"
    if json.dumps(lock, sort_keys=True) != before:
        save_lock(lock_path, lock)
    return counters, files, meta, unavailable


# --------------------------------------------------------------------------- statistics


def pct(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, -(-len(s) * q // 100) - 1))
    return float(s[int(k)])


def chunk_stats(n_a: Sequence[int], n_b: Sequence[int], budget: int, limit: int) -> dict:
    """All per-chunk statistics for one (workload, library, A, B) cell. `n_a` is the plain count under
    the chunker's tokenizer, `n_b` the consumer's count (specials included if it adds them)."""
    n = len(n_b)
    excess = sum(x - limit for x in n_b if x > limit)
    over = sum(x > limit for x in n_b)
    deltas = [b - a for a, b in zip(n_a, n_b)]
    fills = [a / budget for a in n_a]
    return {
        "chunks": n,
        "plain_over": sum(a > budget for a in n_a),  # chunker broke its OWN budget (recount under A)
        "over": over,
        "over_pct": 100 * over / max(1, n),
        "excess": excess,  # tokens a truncating consumer would drop
        "tokens_b": sum(n_b),
        "excess_pct": 100 * excess / max(1, sum(n_b)),
        "p95_nb": pct(n_b, 95),
        "p99_nb": pct(n_b, 99),
        "max_nb": max(n_b, default=0),
        "med_delta": statistics.median(deltas) if deltas else 0.0,
        "p95_delta": pct(deltas, 95),
        "max_delta": max(deltas, default=0),
        "fill_med": statistics.median(fills) if fills else 0.0,
        "fill_mean": statistics.fmean(fills) if fills else 0.0,
        "under_half_pct": 100 * sum(f < 0.5 for f in fills) / max(1, n),
    }


class Pool:
    """Chunks of several workloads pooled together, plus the per-workload overflow rates (macro view)."""

    def __init__(self) -> None:
        self.n_a: list[int] = []
        self.n_b: list[int] = []
        self.macro: list[float] = []

    def add(self, n_a: Sequence[int], n_b: Sequence[int], over_pct: float) -> None:
        self.n_a += list(n_a)
        self.n_b += list(n_b)
        if n_a:
            self.macro.append(over_pct)

    def stats(self, budget: int, limit: int) -> dict:
        s = chunk_stats(self.n_a, self.n_b, budget, limit)
        s["workloads"] = len(self.macro)
        s["macro_over_pct"] = statistics.fmean(self.macro) if self.macro else 0.0
        return s


# --------------------------------------------------------------------------- the run


def run_matrix(workloads: Sequence[Any], adapter_names: Sequence[str], a_keys: Sequence[str],
               counters: dict[str, Counter], tokfiles: dict[str, str | None], *, adapters_mod: Any,
               budget: int, limit: int, seed: int = 0, claude: dict[str, ClaudeCounter] | None = None,
               claude_libs: Sequence[str] = (), claude_sample: int = 25, guard: bool = False,
               log: Callable[[str], None] = print) -> dict:
    rows: list[dict] = []
    quality_rows: list[dict] = []
    pools: dict[tuple, Pool] = {}
    failures: list[dict] = []
    dead: set[tuple[str, str]] = set()

    def record(wl, an, a, b, n_a, n_b, sampled):
        st = chunk_stats(n_a, n_b, budget, limit)
        rows.append({"workload": wl.name, "kind": wl.kind, "adapter": an, "chunk_tok": a, "count_tok": b,
                     "sampled": sampled, **st})
        for g in ("all", group_of(wl.kind)):
            pools.setdefault((an, a, b, g), Pool()).add(n_a, n_b, st["over_pct"])

    for wl in workloads:
        for an in adapter_names:
            adapter = adapters_mod.ADAPTERS[an]
            keys = list(a_keys) + ([GUARD_KEY] if guard and an in GUARD_ADAPTERS else [])
            for a in keys:
                if (an, a) in dead:
                    continue
                if a == GUARD_KEY:
                    ctx = guard_ctx(adapters_mod, budget, wl.fmt, counters, tokfiles)
                    counter_a: Any = MaxCounter(list(counters.values()))
                else:
                    ctx = adapters_mod.Ctx(budget, 0, wl.fmt, a, tokfiles.get(a))
                    counter_a = counters[a]
                try:
                    runner = adapter.build(ctx)
                    views = runner.views(runner.run(wl.text))
                    texts = [v.delivered for v in views]
                except Exception as exc:  # noqa: BLE001
                    dead.add((an, a))
                    failures.append({"adapter": an, "chunk_tok": a, "workload": wl.name,
                                     "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                    log(f"  ! {adapter.label} cannot chunk with {a}: {type(exc).__name__}: {str(exc)[:120]}")
                    continue
                n_a = counter_a.count_many(texts)  # plain count: what the chunker was budgeting
                if a == GUARD_KEY:
                    quality = metrics.evaluate(
                        wl.text,
                        [(v.delivered, v.body, v.headers) for v in views],
                        chunk_size=budget,
                        overlap=0,
                        fmt=wl.fmt,
                        gold=wl.gold,
                        count=lambda text: counter_a.count_many([text])[0],
                    )
                    quality_rows.append({"workload": wl.name, "kind": wl.kind, **quality})
                for b, cb in counters.items():
                    record(wl, an, a, b, n_a, cb.count_many(texts, specials=cb.specials), False)
                if claude and an in claude_libs and texts:
                    rng = random.Random(f"{seed}|{wl.name}|{an}|{a}")
                    idx = sorted(rng.sample(range(len(texts)), min(len(texts), claude_sample)))
                    sub = [texts[i] for i in idx]
                    for ck, cc in claude.items():
                        got = cc.count_many(sub)
                        pairs = [(n_a[i], x) for i, x in zip(idx, got) if x is not None]
                        if pairs:
                            record(wl, an, a, ck, [p[0] for p in pairs], [p[1] for p in pairs], True)
                log(f"{wl.name:<24} {adapter.label:<24} {a:<12} {len(texts):>5} chunks")

    pooled = []
    for (an, a, b, g), pool in sorted(pools.items()):
        pooled.append({"adapter": an, "chunk_tok": a, "count_tok": b, "group": g, **pool.stats(budget, limit)})
    return {"results": rows, "pooled": pooled, "quality": quality_rows, "failures": failures, "guard": guard}


# --------------------------------------------------------------------------- report


def _md_table(headers: Sequence[str], body: Sequence[Sequence[str]], left: int = 1) -> str:
    align = ["---"] * left + ["---:"] * (len(headers) - left)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _index(res: dict) -> dict[tuple, dict]:
    return {(r["adapter"], r["chunk_tok"], r["count_tok"], r["group"]): r for r in res["pooled"]}


def _matrix(idx, adapter, a_keys, b_keys, field, fmt) -> str:
    body = []
    for a in a_keys:
        cells = []
        for b in b_keys:
            r = idx.get((adapter, a, b, "natural"))
            v = fmt(r[field]) if r else "–"
            cells.append(f"**{v}**" if a == b and r else v)
        body.append([f"`{a}`"] + cells)
    return _md_table(["chunked with ↓ / counted with →"] + [f"`{b}`" for b in b_keys], body)


def render_report(res: dict) -> str:
    p = res["params"]
    labels = res["libraries"]
    idx = _index(res)
    tok_keys = list(res["tokenizers"])
    a_keys = [k for k in res["chunk_with"] if k in tok_keys]
    guard_on = bool(res.get("guard"))
    m_keys = a_keys + ([GUARD_KEY] if guard_on else [])  # matrix rows: the guard row comes last
    b_keys = [k for k in tok_keys]
    claude_keys = sorted({r["count_tok"] for r in res["pooled"] if r["count_tok"].startswith("claude:")})
    libs = res["adapters"]
    head = p.get("headline_lib") if p.get("headline_lib") in libs else (libs[0] if libs else "")
    out: list[str] = ["# Cross-tokenizer matrix benchmark", ""]

    out += [
        f"Consumer limit **{p['limit']}** tokens, chunk budget **{p['chunk_size']}** (headroom {p['headroom']}). "
        "Each returned `delivered` string is recounted with every tokenizer below. "
        "Rows of the matrices = the tokenizer the chunker was configured with, columns = the tokenizer that "
        "consumes the text. Diagonal cells (bold) are the chunker judged by its own tokenizer.",
        "",
        "**Reading the numbers.** *Overflow* = share of chunks above the limit. *Excess* = tokens a truncating "
        "consumer would lose, as % of all tokens it saw (so 2 lost tokens and 292 lost tokens are not the same). "
        "*Fill* = chunk size / budget; a chunker can avoid overflow by under-filling, so check fill. "
        "Aggregates use the **natural** workloads (synthetic + real); the `stress_*` workloads are shown separately "
        "because they would dominate any pooled rate.",
        "",
        "## Tokenizers",
        "",
    ]
    rows = []
    for k, m in res["tokenizers"].items():
        rows.append([f"`{k}`", m["source"], f"`{m.get('revision', '-')[:12]}`" if m.get("revision") else "-",
                     f"`{m['sha256'][:12]}`" if m.get("sha256") else "-", str(m.get("vocab_size", "?")),
                     m.get("note", "") + (" **(first use: hash recorded, commit the lock file)**" if m.get("first_use") else "")])
    out += [_md_table(["Key", "Source", "Commit", "SHA-256", "Vocab", "Note"], rows, left=6), ""]
    if res.get("unavailable"):
        out += ["**Not run** (tokenizer unavailable):", ""]
        out += [f"- `{k}`: {v}" for k, v in res["unavailable"].items()]
        out += [""]

    # 1. self-consistency
    out += ["## 1. Does the chunker respect its own budget?", "",
            "Diagonal (A = A) pairs, natural workloads. Cell = `budget violations · median fill · chunks under 50% fill`. "
            "A violation is a chunk that is over budget when recounted with the chunker's own tokenizer; "
            "it should be 0. For BGE the recount excludes [CLS]/[SEP], the overflow tables below include them.", ""]
    body = []
    for lib in libs:
        cells = []
        for a in a_keys:
            r = idx.get((lib, a, a, "natural"))
            cells.append(f"{r['plain_over']} · {r['fill_med']:.2f} · {r['under_half_pct']:.1f}%" if r else "–")
        body.append([labels[lib]] + cells)
    out += [_md_table(["Library"] + [f"`{a}`" for a in a_keys], body), ""]

    # 2. headline matrices
    if head:
        out += [f"## 2. Cross-tokenizer matrix: {labels[head]}", "",
                "### Overflow rate (% of chunks over the limit)", "",
                _matrix(idx, head, m_keys, b_keys, "over_pct", _f), "",
                "### Excess (% of tokens a truncating consumer loses)", "",
                _matrix(idx, head, m_keys, b_keys, "excess_pct", lambda x: _f(x, 2)), "",
                "### Worst chunk (max token count under the consumer)", "",
                _matrix(idx, head, m_keys, b_keys, "max_nb", lambda x: str(int(x))), "",
                "### Median count difference (consumer − chunker tokens)", "",
                _matrix(idx, head, m_keys, b_keys, "med_delta", lambda x: _f(x)), "",
                "### p95 count difference (consumer − chunker tokens)", "",
                _matrix(idx, head, m_keys, b_keys, "p95_delta", lambda x: _f(x)), ""]

    # 3. library comparison
    out += ["## 3. Library comparison", "",
            "Natural workloads, pooled. *Self* = A = B. *Cross* = every A ≠ B pair (Claude excluded). "
            "The worst pair is the (chunker → consumer) combination with the highest overflow rate.", ""]
    body = []
    for lib in libs:
        selfc = [r for r in res["pooled"] if r["adapter"] == lib and r["group"] == "natural"
                 and r["chunk_tok"] == r["count_tok"]]
        cross = [r for r in res["pooled"] if r["adapter"] == lib and r["group"] == "natural"
                 and r["chunk_tok"] != r["count_tok"] and r["chunk_tok"] != GUARD_KEY
                 and not r["count_tok"].startswith("claude:")]
        sc = sum(r["chunks"] for r in selfc)
        cc = sum(r["chunks"] for r in cross)
        worst = max(cross, key=lambda r: r["over_pct"], default=None)
        body.append([
            labels[lib],
            _f(100 * sum(r["over"] for r in selfc) / sc) if sc else "–",
            _f(100 * sum(r["over"] for r in cross) / cc) if cc else "–",
            _f(100 * sum(r["excess"] for r in cross) / max(1, sum(r["tokens_b"] for r in cross)), 2) if cc else "–",
            f"`{worst['chunk_tok']}`→`{worst['count_tok']}` ({_f(worst['over_pct'])}%)" if worst else "–",
        ])
    out += [_md_table(["Library", "Self overflow %", "Cross overflow %", "Cross excess %", "Worst pair"], body), ""]

    # 3b. guard tokenizers
    if guard_on:
        out += ["## 3b. Guard tokenizers", "",
                "The `guard` row chunks with the **maximum** count over every tokenizer above (Python: `guard_encodings` / "
                "`guard_tokenizer_files`), with the same budget. Overflow is pooled over all consumer tokenizers above "
                "(natural workloads); Claude cannot be a guard member, so it is not part of the guarantee. "
                "Median fill shows what the guard costs: a chunker can avoid overflow by under-filling.", ""]
        for lib in [l for l in libs if l in GUARD_ADAPTERS]:
            body = []
            for a in m_keys:
                rs = [r for r in res["pooled"] if r["adapter"] == lib and r["group"] == "natural"
                      and r["chunk_tok"] == a and not r["count_tok"].startswith("claude:")]
                if not rs:
                    continue
                n, over = sum(r["chunks"] for r in rs), sum(r["over"] for r in rs)
                body.append([f"`{a}`", str(n), str(over), _f(100 * over / max(1, n)),
                             str(int(max(r["max_nb"] for r in rs))), f"{rs[0]['fill_med']:.2f}",
                             _f(rs[0]["under_half_pct"])])
            if body:
                out += [f"### {labels[lib]}", "",
                        _md_table(["Chunked with", "Chunks", "Over limit", "Overflow %", "Worst chunk", "Median fill",
                                   "Under 50% fill %"], body), ""]

        quality = [r for r in res.get("quality", []) if r.get("kind") in NATURAL_KINDS]
        if quality:
            out += ["### Guard chunk quality", "",
                    "These source-fidelity and boundary metrics are computed on the guarded chunks themselves. "
                    "Token overflow remains a hard acceptance gate; quality metrics are optimization targets.", ""]
            body = []
            for r in quality:
                fmt = lambda value: _f(value) if value is not None else "–"
                body.append([f"`{r['workload']}`", str(r["chunks"]), fmt(r.get("content_coverage_pct")),
                             fmt(r.get("located_pct")), fmt(r.get("mid_word_cut_pct")),
                             fmt(r.get("clean_end_pct")), fmt(r.get("fence_intact_pct")),
                             fmt(r.get("fill_ratio", 0) * 100), fmt(r.get("tiny_chunk_pct"))])
            out += [_md_table(["Workload", "Chunks", "Content coverage %", "Located %", "Mid-word cuts %",
                               "Clean ends %", "Fences intact %", "Mean fill %", "Tiny chunks %"], body), ""]

    # 4. focus pairs per workload
    out += ["## 4. Per workload (overflow %)", ""]
    for pair in p.get("focus", []):
        a, b = pair.split(":")
        if a not in tok_keys or b not in tok_keys:
            continue
        out += [f"### `{a}` → `{b}`", ""]
        cell = {(r["workload"], r["adapter"]): r for r in res["results"]
                if r["chunk_tok"] == a and r["count_tok"] == b and not r["sampled"]}
        body = []
        for w in res["workloads"]:
            body.append([f"{w['name']}" + (" *(stress)*" if w["kind"] == "stress" else "")]
                        + [_f(cell[(w["name"], lib)]["over_pct"]) if (w["name"], lib) in cell else "–" for lib in libs])
        for g, name in (("natural", "**natural, pooled**"),):
            body.append([name] + [_f(idx[(lib, a, b, g)]["over_pct"]) if (lib, a, b, g) in idx else "–" for lib in libs])
        body.append(["**natural, macro-average**"]
                    + [_f(idx[(lib, a, b, "natural")]["macro_over_pct"]) if (lib, a, b, "natural") in idx else "–" for lib in libs])
        out += [_md_table(["Workload"] + [labels[l] for l in libs], body), ""]

    # 5. Claude
    out += ["## 5. Claude (token-counting API, sampled)", ""]
    if claude_keys:
        out += [f"Chunks are sampled (`{p.get('claude_sample')}` per workload, library and chunker tokenizer; seeded) and "
                "counted with Anthropic's count_tokens endpoint, with the fixed per-request overhead subtracted. "
                "Claude can only be a consumer: there is no offline tokenizer to chunk with. The endpoint is documented "
                "as an estimate, and different Claude models can use different tokenizers, so the model id is part of "
                "the column name.", ""]
        for ck in claude_keys:
            body = []
            for a in m_keys:
                for lib in p.get("claude_libs", []):
                    r = idx.get((lib, a, ck, "natural"))
                    if r:
                        body.append([f"`{a}`", labels.get(lib, lib), str(r["chunks"]), _f(r["over_pct"]),
                                     _f(r["excess_pct"], 2), _f(r["med_delta"]), _f(r["p95_delta"]), str(int(r["max_nb"]))])
            out += [f"### `{ck}`", "",
                    _md_table(["Chunked with", "Library", "n", "Overflow %", "Excess %", "Median Δ", "p95 Δ", "Max count"],
                              body, left=2), ""]
    else:
        out += [res.get("claude_note", "Not run."), ""]

    if res.get("failures"):
        out += ["## Failures", "", "A library that failed to chunk with a tokenizer is skipped for that tokenizer.", ""]
        seen = set()
        body = []
        for f in res["failures"]:
            k = (f["adapter"], f["chunk_tok"])
            if k not in seen:
                seen.add(k)
                body.append([labels.get(f["adapter"], f["adapter"]), f"`{f['chunk_tok']}`", f["error"].replace("|", "\\|")])
        out += [_md_table(["Library", "Chunker tokenizer", "Error"], body, left=3), ""]

    out += ["## Limits", "",
            "This measures token counts and hard-limit overflow, not embedding or retrieval quality, and not the "
            "model's behaviour past the limit (most embedders truncate; most LLM APIs reject). "
            "Chunks are counted without BOS/EOS for LLM tokenizers; add 1 token per prompt if you need it. "
            "One seed and 0.25 MiB per synthetic workload: rates on the small real corpora (tens of chunks) are noisy. "
            "Raw per-workload rows, pooled statistics and package versions are in `results.json` / `rows.csv`.", ""]
    return "\n".join(out)


def rows_to_csv(rows: Sequence[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
