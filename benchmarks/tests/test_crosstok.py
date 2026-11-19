"""Self-tests for the cross-tokenizer matrix. No tokenizer packages or network needed."""
import json
import re
import tempfile
from pathlib import Path

import stubs  # noqa: F401  (registers stub adapters)
from chunkbench import adapters, crosstok
from chunkbench.corpora import Workload

_TOK = re.compile(r"\w+|[^\w\s]")


class RegexCounter:
    """Counts word/punctuation tokens. `extra` tokens are added per text when specials are requested,
    like BGE's [CLS]/[SEP]; `split` > 0 makes long words count as several tokens (a 'different vocabulary')."""

    def __init__(self, key, specials=False, extra=2, split=0):
        self.key, self.specials, self._extra, self._split = key, specials, extra, split

    def count_many(self, texts, specials=False):
        out = []
        for t in texts:
            n = 0
            for m in _TOK.findall(t):
                n += 1 if not self._split else max(1, -(-len(m) // self._split))
            out.append(n + (self._extra if specials else 0))
        return out


TEXT = " ".join(f"alpha{i % 7} beta, gamma." for i in range(900))


def wl(name="w", kind="synthetic"):
    return Workload(name, "plain", TEXT, kind)


def test_percentile_and_stats():
    assert crosstok.pct([], 95) == 0.0
    assert crosstok.pct(list(range(1, 101)), 95) == 95
    s = crosstok.chunk_stats([10, 10, 10, 4], [10, 13, 14, 4], budget=10, limit=12)
    assert s["over"] == 2 and s["excess"] == (13 - 12) + (14 - 12)
    assert s["max_nb"] == 14 and s["max_delta"] == 4 and s["plain_over"] == 0
    assert s["under_half_pct"] == 25.0 and abs(s["fill_mean"] - 0.85) < 1e-9
    assert crosstok.chunk_stats([], [], 10, 12)["over_pct"] == 0


def test_headroom_explains_the_old_bge_overflow():
    plain, bge = RegexCounter("plain"), RegexCounter("BGE", specials=True)
    counters = {"plain": plain, "BGE": bge}
    kw = dict(adapters_mod=adapters, limit=50, log=lambda *_: None)
    # budget == limit: every full window overflows by exactly the 2 special tokens
    old = crosstok.run_matrix([wl()], ["stub_window"], ["plain"], counters, {}, budget=50, **kw)
    r = next(r for r in old["pooled"] if r["count_tok"] == "BGE" and r["group"] == "natural")
    assert r["over_pct"] > 90 and r["max_delta"] == 2
    # budget == limit - 2: nothing overflows and the chunker never broke its own budget
    new = crosstok.run_matrix([wl()], ["stub_window"], ["plain"], counters, {}, budget=48, **kw)
    r = next(r for r in new["pooled"] if r["count_tok"] == "BGE" and r["group"] == "natural")
    assert r["over"] == 0 and r["plain_over"] == 0 and r["fill_med"] > 0.9


def test_different_vocabulary_overflows_and_stress_is_separated():
    counters = {"a": RegexCounter("a"), "b": RegexCounter("b", split=4)}
    run = crosstok.run_matrix([wl("nat", "synthetic"), wl("str", "stress")], ["stub_window"], ["a"], counters, {},
                              adapters_mod=adapters, budget=40, limit=40, log=lambda *_: None)
    by = {(r["count_tok"], r["group"]): r for r in run["pooled"]}
    assert by[("a", "natural")]["over"] == 0
    assert by[("b", "natural")]["over_pct"] > 50 and by[("b", "natural")]["excess"] > 0
    assert by[("b", "all")]["chunks"] == by[("b", "natural")]["chunks"] + by[("b", "stress")]["chunks"]


def test_failing_adapter_is_recorded_not_fatal():
    def boom(ctx):
        raise RuntimeError("tokenizer file not supported")

    adapters.register_test_adapter(adapters.Adapter("stub_boom", "stub boom", None, "", "", False, boom))
    counters = {"a": RegexCounter("a")}
    run = crosstok.run_matrix([wl("x"), wl("y")], ["stub_boom", "stub_window"], ["a"], counters, {},
                              adapters_mod=adapters, budget=40, limit=40, log=lambda *_: None)
    assert len(run["failures"]) == 1 and run["failures"][0]["adapter"] == "stub_boom"
    assert {r["adapter"] for r in run["results"]} == {"stub_window"}


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def count_tokens(self, model, messages):
        self.calls += 1
        text = messages[0]["content"]
        if not text.strip():
            err = RuntimeError("blank")
            err.status_code = 400
            raise err
        return type("R", (), {"input_tokens": len(_TOK.findall(text)) + 7})()  # 7 = fake message overhead


def test_claude_counter_overhead_cache_and_blank():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "c.json"
        client = FakeClient()
        c = crosstok.ClaudeCounter("m", cache_path=cache, client=client)
        assert c.overhead == 7
        assert c.count_many(["one two three", "   "]) == [3, None]
        calls = client.calls
        c2 = crosstok.ClaudeCounter("m", cache_path=cache, client=FakeClient())
        assert c2.count_many(["one two three"]) == [3] and client.calls == calls  # served from the cache


def test_claude_retries_on_rate_limit():
    class Flaky(FakeClient):
        def count_tokens(self, model, messages):
            self.calls += 1
            if self.calls in (2, 3):
                e = RuntimeError("slow down")
                e.status_code = 429
                raise e
            return type("R", (), {"input_tokens": 8})()

    slept = []
    c = crosstok.ClaudeCounter("m", client=Flaky(), sleep=slept.append, workers=1)
    assert c.count_many(["x y"]) == [1] and len(slept) == 2


def test_claude_in_run_is_sampled_and_deterministic():
    counters = {"a": RegexCounter("a")}
    claude = {"claude:m": crosstok.ClaudeCounter("m", client=FakeClient())}
    kw = dict(adapters_mod=adapters, budget=20, limit=20, claude=claude, claude_libs=["stub_window"],
              claude_sample=5, seed=1, log=lambda *_: None)
    r1 = crosstok.run_matrix([wl()], ["stub_window"], ["a"], counters, {}, **kw)
    r2 = crosstok.run_matrix([wl()], ["stub_window"], ["a"], counters, {}, **kw)
    row = next(r for r in r1["results"] if r["count_tok"] == "claude:m")
    assert row["sampled"] and row["chunks"] == 5 and r1["results"] == r2["results"]


def test_lock_pinning_tofu_and_mismatch():
    spec = crosstok.SPECS["qwen"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / spec.file).write_text("{}")
        path, entry, first = crosstok.resolve_hf_file(spec, d, {}, offline=True)
        assert first and entry["sha256"] == crosstok.sha256_file(path)
        _, _, first2 = crosstok.resolve_hf_file(spec, d, {"qwen": entry}, offline=True)
        assert not first2
        bad = dict(entry, sha256="0" * 64)
        try:
            crosstok.resolve_hf_file(spec, d, {"qwen": bad}, offline=True)
            raise AssertionError("expected a hash mismatch")
        except RuntimeError as exc:
            assert "does not match" in str(exc)
        try:
            crosstok.resolve_hf_file(crosstok.SPECS["deepseek"], d, {}, offline=True)
            raise AssertionError("expected missing file")
        except FileNotFoundError:
            pass


def test_download_failure_mentions_gating():
    def deny(repo, rev, dest):
        raise OSError("401")

    with tempfile.TemporaryDirectory() as d:
        try:
            crosstok.resolve_hf_file(crosstok.SPECS["llama3"], Path(d), {}, download=deny)
            raise AssertionError("expected failure")
        except RuntimeError as exc:
            assert "HF_TOKEN" in str(exc)


def test_report_renders_all_sections():
    counters = {"a": RegexCounter("a"), "b": RegexCounter("b", split=4, specials=True)}
    run = crosstok.run_matrix([wl("nat"), wl("str", "stress")], ["stub_window", "stub_chars"], ["a", "b"], counters,
                              {}, adapters_mod=adapters, budget=40, limit=42, log=lambda *_: None)
    res = {"params": {"limit": 42, "chunk_size": 40, "headroom": 2, "focus": ["a:b"], "headline_lib": "stub_window",
                      "claude_sample": 5, "claude_libs": []},
           "tokenizers": {"a": {"source": "t", "vocab_size": 1}, "b": {"source": "t", "vocab_size": 2, "revision": "abc",
                                                                       "sha256": "f" * 64, "first_use": True}},
           "unavailable": {"llama3": "RuntimeError: gated"}, "chunk_with": ["a", "b"], "claude_note": "Not run.",
           "adapters": ["stub_window", "stub_chars"], "libraries": {"stub_window": "stub window", "stub_chars": "stub chars"},
           "workloads": [{"name": "nat", "kind": "synthetic", "bytes": 1}, {"name": "str", "kind": "stress", "bytes": 1}],
           **run}
    md = crosstok.render_report(json.loads(json.dumps(res)))
    for needle in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "`a` → `b`", "*(stress)*", "gated", "first use", "Limits"):
        assert needle in md, needle
    assert "| –" not in md.split("## 3.")[1].split("## 4.")[0]
    assert crosstok.rows_to_csv(res["results"]).startswith("workload,kind,adapter")


# --------------------------------------------------------------------------- guard row


def test_max_counter_is_the_elementwise_maximum():
    a, b = RegexCounter("a"), RegexCounter("b", split=3)
    texts = ["alpha1 beta, gamma.", "x y z", "supercalifragilistic word"]
    got = crosstok.MaxCounter([a, b]).count_many(texts)
    assert got == [max(x, y) for x, y in zip(a.count_many(texts), b.count_many(texts))]
    assert any(g > x for g, x in zip(got, a.count_many(texts)))  # the splitting counter dominates somewhere
    assert crosstok.MaxCounter([]).count_many(texts) == [0, 0, 0]


def test_guard_ctx_uses_first_tiktoken_as_primary_and_every_file_as_guard():
    counters = {"cl100k_base": None, "o200k_base": None, "qwen": None, "BGE": None}
    files = {"qwen": "/t/qwen.json", "BGE": "/t/bge.json"}
    ctx = crosstok.guard_ctx(adapters, 510, "markdown", counters, files)
    assert (ctx.encoding, ctx.tokenizer_file) == ("cl100k_base", None)
    assert ctx.guard_encodings == ("o200k_base",)
    assert ctx.guard_files == ("/t/qwen.json", "/t/bge.json")
    only_hf = crosstok.guard_ctx(adapters, 510, "plain", {"qwen": None, "BGE": None}, files)
    assert only_hf.tokenizer_file == "/t/qwen.json" and only_hf.guard_files == ("/t/bge.json",)


def test_guard_row_is_added_only_for_guard_adapters_and_never_pollutes_section_3():
    seen = []

    def build(ctx):
        seen.append(ctx)
        from stubs import _views, _window

        return adapters.Runner(_window(ctx.chunk_size), _views)

    adapters.register_test_adapter(adapters.Adapter("stub_guarded", "stub guarded", None, "", "", False, build))
    old = crosstok.GUARD_ADAPTERS
    crosstok.GUARD_ADAPTERS = ("stub_guarded",)
    try:
        counters = {"a": RegexCounter("a"), "b": RegexCounter("b", split=4, specials=True)}
        run = crosstok.run_matrix([wl("nat")], ["stub_guarded", "stub_chars"], ["a", "b"], counters, {"b": "/t/b.json"},
                                  adapters_mod=adapters, budget=40, limit=42, guard=True, log=lambda *_: None)
        res = {"params": {"limit": 42, "chunk_size": 40, "headroom": 2, "focus": [], "headline_lib": "stub_guarded",
                          "claude_sample": 5, "claude_libs": []},
               "tokenizers": {"a": {"source": "t"}, "b": {"source": "t"}}, "unavailable": {},
               "chunk_with": ["a", "b"], "claude_note": "Not run.", "adapters": ["stub_guarded", "stub_chars"],
               "libraries": {"stub_guarded": "stub guarded", "stub_chars": "stub chars"},
               "workloads": [{"name": "nat", "kind": "synthetic", "bytes": 1}], **run}
        md = crosstok.render_report(json.loads(json.dumps(res)))
    finally:
        crosstok.GUARD_ADAPTERS = old
    guard_rows = [r for r in run["results"] if r["chunk_tok"] == crosstok.GUARD_KEY]
    assert guard_rows and {r["adapter"] for r in guard_rows} == {"stub_guarded"}  # stub_chars gets no guard row
    assert run["quality"] and run["quality"][0]["chunks"] > 0
    assert run["quality"][0]["content_coverage_pct"] == 100.0
    assert any(c.tokenizer_file == "/t/b.json" or "/t/b.json" in c.guard_files for c in seen)
    assert "## 3b. Guard tokenizers" in md and "`guard`" in md
    section3 = md.split("## 3.")[1].split("## 3b.")[0]
    assert "`guard`" not in section3  # guard rows are excluded from the pooled library comparison
