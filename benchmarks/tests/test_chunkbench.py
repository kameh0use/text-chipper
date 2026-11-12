import json
import sys

import pytest

from chunkbench import corpora, metrics, runner
from chunkbench.tokenizer import RegexRef

REF = RegexRef()


def ev(source, bodies, size=20, overlap=0, fmt="plain", gold=()):
    return metrics.evaluate(source, [(b, b) for b in bodies], chunk_size=size, overlap=overlap, fmt=fmt,
                            gold=list(gold), count=REF.count)


def test_coverage_and_loss():
    src = "alpha beta gamma. delta epsilon zeta. eta theta iota."
    full = ev(src, ["alpha beta gamma.", "delta epsilon zeta.", "eta theta iota."])
    assert full["coverage_pct"] == 100 and full["located_pct"] == 100
    lossy = ev(src, ["alpha beta gamma.", "eta theta iota."])
    assert lossy["coverage_pct"] < 100 and lossy["lost_nonspace_chars"] > 0


def test_unlocated_chunk_is_reported():
    m = ev("hello world", ["hello", "w�rld"])
    assert m["located_pct"] == 50 and m["replacement_char_chunks"] == 1


def test_mid_word_cut_and_clean_end():
    src = "The quick brown fox. Jumps over."
    good = ev(src, ["The quick brown fox.", "Jumps over."])
    assert good["mid_word_cut_pct"] == 0 and good["clean_end_pct"] == 100
    bad = ev(src, ["The quick bro", "wn fox. Jumps over."])
    assert bad["mid_word_cut_pct"] == 100 and bad["clean_end_pct"] == 0


def test_budget_violation_detected():
    m = ev("a b c d e f g h", ["a b c d e f g h"], size=4)
    assert m["over_limit_chunks"] == 1 and m["tokens_max"] == 8


def test_fence_intact_and_orphan_heading():
    src = "# T\n\ntext here.\n\n```py\nx = 1\n```\n\nafter."
    intact = ev(src, ["# T\n\ntext here.", "```py\nx = 1\n```", "after."], size=40, fmt="markdown")
    assert intact["fence_intact_pct"] == 100 and intact["orphan_heading_pct"] == 0
    split = ev(src, ["# T", "text here.\n\n```py", "x = 1\n```\n\nafter."], size=40, fmt="markdown")
    assert split["fence_intact_pct"] == 0 and split["orphan_heading_pct"] > 0


def test_overlap_measurement():
    src = "a b c d e f g h i j"
    m = ev(src, ["a b c d e f", "e f g h i j"], size=10, overlap=2)
    assert m["overlap_tokens_median"] == 2 and m["overlap_violation_pct"] == 0
    m = ev(src, ["a b c d e f", "d e f g h i j"], size=10, overlap=2)
    assert m["overlap_violation_pct"] == 100


def test_alignment_iou():
    src = "one two three. four five six. seven eight nine."
    s = src.index("four")
    gold = [(s, s + len("four five six."))]
    tight = ev(src, ["one two three.", "four five six.", "seven eight nine."], gold=gold)
    assert tight["oracle_iou_mean"] == pytest.approx(1.0) and tight["single_chunk_hit_pct"] == 100
    loose = ev(src, ["one two three. four five six. seven eight nine."], gold=gold)
    assert loose["oracle_iou_mean"] < 0.6


def test_corpora_are_deterministic():
    a = corpora.build_workloads(0.05, 20, 1, include_real=False)
    b = corpora.build_workloads(0.05, 20, 1, include_real=False)
    assert {k: v.text for k, v in a.items()} == {k: v.text for k, v in b.items()}
    assert set(a) >= {"prose_en", "markdown_doc", "multilingual", "code_heavy"}
    assert all(w.gold for n, w in a.items() if n.startswith(("prose", "markdown")))


def _run(tmp_path, *extra):
    argv = ["run", "--libs", "stub_window", "stub_chars", "stub_lossy", "--encoding", "regex-test",
            "--chunk-size", "64", "--workloads", "prose_en", "markdown_doc", "--size-mib", "0.05",
            "--rounds", "2", "--repeats", "2", "--min-time", "0", "--max-gold", "30", "--no-real",
            "--out", str(tmp_path), *extra]
    assert runner.main(argv) == 0
    (d,) = list(tmp_path.iterdir())
    return d


@pytest.fixture
def run_dir(tmp_path):
    """A run with the default timing mode (cold)."""
    return _run(tmp_path)


@pytest.fixture
def steady_run_dir(tmp_path):
    """A run in steady mode, which also records fresh-input timings."""
    return _run(tmp_path, "--mode", "steady")


def test_end_to_end(run_dir):
    data = json.loads((run_dir / "results.json").read_text())
    assert data["meta"]["reference_tokenizer"] == "regex-test"
    cells = {(c["adapter"], c["workload"], c["lane"]): c for c in data["cells"]}
    assert len(cells) == 3 * 2 * 2
    good = cells[("stub_window", "prose_en", "single")]
    assert good["status"] == "ok" and good["rounds_ok"] == 2 and good["mib_per_s"] > 0
    # cold mode (the default): one first-and-only call per process, so there are no fresh-input samples
    assert good["timing_mode"] == "cold" and good["wall_ms"]["n"] == 2
    assert "mib_per_s_fresh" not in good and "fresh_wall_ms" not in good
    assert good["quality"]["over_limit_chunks"] == 0 and good["quality"]["coverage_pct"] > 99.9
    assert cells[("stub_chars", "prose_en", "all")]["quality"]["over_limit_pct"] > 0
    assert cells[("stub_chars", "prose_en", "all")]["quality"]["mid_word_cut_pct"] > 50
    lossy = cells[("stub_lossy", "prose_en", "all")]["quality"]
    assert lossy["coverage_pct"] < 90
    for name in ("results.json", "summary.csv", "report.md"):
        assert (run_dir / name).exists()
    assert "stub window" in (run_dir / "report.md").read_text()
    report = (run_dir / "report.md").read_text()
    assert "first and only" in report and "fresh MiB/s" not in report
    assert not (run_dir / "_cache").exists()


def test_steady_mode_records_fresh_input_timings(steady_run_dir):
    data = json.loads((steady_run_dir / "results.json").read_text())
    cells = {(c["adapter"], c["workload"], c["lane"]): c for c in data["cells"]}
    good = cells[("stub_window", "prose_en", "single")]
    assert good["status"] == "ok" and good["timing_mode"] == "steady" and good["mib_per_s"] > 0
    assert good["mib_per_s_fresh"] > 0 and good["fresh_wall_ms"]["n"] >= 2 and good["repeat_vs_fresh_speedup"] > 0
    assert "fresh MiB/s" in (steady_run_dir / "report.md").read_text()


def test_failing_adapter_is_reported_not_fatal(tmp_path, monkeypatch):
    argv = ["run", "--libs", "tiktoken_window", "--encoding", "regex-test", "--workloads", "prose_en",
            "--size-mib", "0.02", "--rounds", "1", "--repeats", "1", "--min-time", "0", "--no-real",
            "--lanes", "all", "--out", str(tmp_path)]
    rc = runner.main(argv)  # tiktoken is imported with the regex-test encoding -> may or may not be installed
    (d,) = list(tmp_path.iterdir())
    cell = json.loads((d / "results.json").read_text())["cells"][0]
    assert cell["status"] in {"ok", "error"}
    if cell["status"] == "error":
        assert rc == 1 and cell["error"]


def test_plots_render(run_dir):
    assert runner.main(["plot", str(run_dir)]) == 0
    assert (run_dir / "plots" / "throughput__single_ov0.png").stat().st_size > 5000


def test_fresh_variants_share_no_line_and_keep_blank_lines():
    from chunkbench.worker import fresh_variant

    text = "alpha beta\n\ngamma\ndelta"
    a, b = fresh_variant(text, 1), fresh_variant(text, 2)
    assert a != b
    la, lb = {x for x in a.splitlines() if x.strip()}, {x for x in b.splitlines() if x.strip()}
    assert not (la & lb)
    assert a.splitlines()[1] == ""  # the blank line between paragraphs survives


def test_fresh_variant_changes_the_inside_of_long_lines():
    from chunkbench.worker import fresh_variant

    line = " ".join(f"w{k}" for k in range(300))
    a, b = fresh_variant(line, 1), fresh_variant(line, 2)
    assert a.count(" 1 ") >= 25 and not (set(a.split()) & {"2"}) and a != b
    assert "\n" not in a.split("\n\n")[0]  # no line structure was introduced


# ---- content coverage: headings carried as metadata ------------------------------------------

HEAD_SRC = "# Guide\n\nalpha beta gamma.\n\n## Install\n\ndelta epsilon zeta.\n\nSetext\n------\n\neta theta iota.\n"


def ev3(source, views, size=40):
    return metrics.evaluate(source, [(d, b, h) for d, b, h in views], chunk_size=size, overlap=0,
                            fmt="markdown", gold=[], count=REF.count)


def test_heading_carried_in_headers_counts_as_content():
    views = [
        ("alpha beta gamma.", "alpha beta gamma.", ["Guide"]),
        ("delta epsilon zeta.", "delta epsilon zeta.", ["Guide", "Install"]),
        ("eta theta iota.", "eta theta iota.", ["Guide", "Setext"]),
    ]
    m = ev3(HEAD_SRC, views)
    assert m["coverage_pct"] < 100 and m["lost_nonspace_chars"] > 0       # strict body coverage
    assert m["content_coverage_pct"] == 100 and m["lost_unexplained_chars"] == 0
    assert m["heading_meta_chars"] == m["lost_nonspace_chars"]


def test_heading_missing_from_headers_stays_lost():
    views = [
        ("alpha beta gamma.", "alpha beta gamma.", ["Guide"]),
        ("delta epsilon zeta.", "delta epsilon zeta.", ["Guide"]),          # "Install" dropped
        ("eta theta iota.", "eta theta iota.", ["Guide", "Setext"]),
    ]
    m = ev3(HEAD_SRC, views)
    assert m["content_coverage_pct"] < 100
    assert m["lost_unexplained_chars"] == len("##Install")             # only that heading is lost


def test_truncated_header_prefix_is_accepted_but_body_loss_is_not_hidden():
    views = [
        ("alpha beta gamma.", "alpha beta gamma.", ["Gu"]),                  # truncated title
        ("delta epsilon zeta.", "delta epsilon zeta.", ["Guide", "Install"]),
        ("eta theta iota.", "eta theta iota.", ["Guide", "Setext"]),
    ]
    assert ev3(HEAD_SRC, views)["content_coverage_pct"] == 100
    views[1] = ("delta epsilon zeta.", "delta epsilon", ["Guide", "Install"])  # real body loss
    m = ev3(HEAD_SRC, views)
    assert m["content_coverage_pct"] < 100 and m["lost_unexplained_chars"] >= len("zeta.")


def test_code_fence_hash_lines_are_not_headings():
    src = "```\n# not a heading\n```\n\ntext here.\n"
    m = ev3(src, [("text here.", "text here.", ["not a heading"])])
    assert m["content_coverage_pct"] < 100          # fenced text is not credited to metadata


def test_two_tuple_views_still_work():
    m = ev("alpha beta gamma.", ["alpha beta gamma."])
    assert m["content_coverage_pct"] == 100


def test_atx_trailing_hash_rules_and_two_line_setext():
    assert metrics._atx_title("# Learn C#") == "Learn C#"
    assert metrics._atx_title("## Title ##") == "Title"
    assert metrics._atx_title("###") == ""
    src = "Two line\ntitle here\n====\n\nbody text.\n"
    assert [t for *_, t in metrics._heading_lines(src)] == ["Two line title here"]
    views = [("body text.", "body text.", ["Two line title here"])]
    assert ev3(src, views)["content_coverage_pct"] == 100
    src = "# Learn C#\n\nbody text.\n"
    assert ev3(src, [("body text.", "body text.", ["Learn C#"])])["content_coverage_pct"] == 100
