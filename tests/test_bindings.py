"""End-to-end tests for the Python bindings.

The Rust test-suite exercises the engine directly; these tests exercise what users actually
call (the pyo3 layer) and cross-check the multi-threaded paths against the single-threaded
streaming reader.

Run after building the extension into the active environment:

    maturin develop --release
    pytest tests/ -v
"""

import random

import pytest

text_chipper = pytest.importorskip("text_chipper")
TextChipper = text_chipper.TextChipper


# --------------------------------------------------------------------------- helpers


def key(c):
    """Every observable field of a chunk, for exact comparisons."""
    return (
        c.text,
        c.start_index,
        c.end_index,
        c.start_byte,
        c.end_byte,
        c.orig_start_index,
        c.orig_end_index,
        c.orig_start_byte,
        c.orig_end_byte,
        c.token_count,
        tuple(c.headers),
    )


_WORDS = (
    "alpha beta gamma delta retrieval chunk token processing documents header context "
    "naive cafe Zurich Berlin the of a an x"
).split()


def make_markdown(target_bytes, seed=0):
    """Deterministic, irregular Markdown (headings, paragraphs, lists, fenced code)."""
    rnd = random.Random(seed)
    parts = []
    size = 0
    while size < target_bytes:
        kind = rnd.randrange(10)
        words = " ".join(rnd.choice(_WORDS) for _ in range(rnd.randrange(3, 90)))
        if kind < 2:
            piece = "#" * rnd.randrange(1, 4) + " " + words[:40] + "\n\n"
        elif kind < 6:
            piece = words + "\n\n"
        elif kind < 8:
            items = [
                "- " + " ".join(rnd.choice(_WORDS) for _ in range(rnd.randrange(2, 20)))
                for _ in range(rnd.randrange(2, 7))
            ]
            piece = "\n".join(items) + "\n\n"
        elif kind == 8:
            piece = "```\n" + words + "\n\nmore code\n```\n\n"
        else:
            piece = words + "\n"
        parts.append(piece)
        size += len(piece)
    return "".join(parts)


# --------------------------------------------------------------------------- basics


def test_offsets_roundtrip_and_budget():
    doc = make_markdown(60_000, seed=1)
    chip = TextChipper(max_length=64, overlap=8, encoding="chars", normalize="none")
    chunks = chip.chunk(doc)
    assert chunks
    raw = doc.encode("utf-8")
    for c in chunks:
        assert doc[c.start_index : c.end_index] == c.text
        assert raw[c.start_byte : c.end_byte].decode("utf-8") == c.text
        assert (c.orig_start_index, c.orig_end_index) == (c.start_index, c.end_index)
        # strict mode (default): the delivered text, context included, fits the budget
        assert c.token_count <= 64
        assert chip.count_tokens(c.contextualized_text) == c.token_count


def test_headers_are_attached_as_context():
    # Two sections that are too big to share one chunk, so the second chunk is its own section and
    # carries the full heading path. (A tiny document would be merged into one chunk whose headers
    # are only the common ancestor, with the lower heading kept in the body.)
    doc = (
        "# Guide\n\n" + "Intro sentence. " * 8
        + "\n\n## Setup\n\n" + "Install the package and run it. " * 4 + "\n"
    )
    chunks = TextChipper(max_length=200, overlap=10, encoding="chars").chunk(doc)
    assert len(chunks) >= 2
    assert chunks[0].headers == ["Guide"]
    assert any(c.headers == ["Guide", "Setup"] for c in chunks)
    assert chunks[-1].contextualized_text.startswith("[ctx:Guide > Setup] ")

    no_ctx = TextChipper(
        max_length=200, overlap=10, encoding="chars", include_context=False
    ).chunk(doc)
    assert all(c.headers == [] for c in no_ctx)


def test_normalization_keeps_original_offsets():
    doc = "cafe\u0301 au lait. " * 200  # 'e' + combining acute: NFC composes it
    chip = TextChipper(max_length=60, overlap=5, encoding="chars", normalize="nfc")
    for c in chip.chunk(doc):
        assert "\u0301" not in c.text
        assert doc[c.orig_start_index : c.orig_end_index].strip()
        assert c.orig_end_index <= len(doc)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_length=10, overlap=10),  # overlap must be smaller than max_length
        dict(encoding="no_such_encoding"),
        dict(normalize="bogus"),
    ],
)
def test_invalid_configuration_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        TextChipper(**kwargs)


def test_invalid_errors_argument(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    chip = TextChipper(max_length=50, overlap=5, encoding="chars")
    with pytest.raises(ValueError):
        chip.from_file(str(p), errors="nonsense")


# --------------------------------------------------------------------------- batches


def test_chunk_batch_matches_chunk_on_the_parallel_branch():
    # >= 4 texts and >= 64 KiB in total: this is the rayon branch of chunk_batch.
    texts = [make_markdown(40_000, seed=s) for s in range(6)]
    chip = TextChipper(max_length=96, overlap=12, encoding="words")
    batch = chip.chunk_batch(texts)
    assert len(batch) == len(texts)
    for t, got in zip(texts, batch):
        assert [key(c) for c in got] == [key(c) for c in chip.chunk(t)]


def test_chunk_batch_small_branch():
    chip = TextChipper(max_length=50, overlap=5, encoding="chars")
    assert chip.chunk_batch([]) == []
    out = chip.chunk_batch(["one short text", "# H\n\nanother one"])
    assert len(out) == 2 and all(out)


# --------------------------------------------------------------------------- files


def test_from_file_stream_and_from_files_agree(tmp_path):
    paths = []
    for s in range(5):
        p = tmp_path / f"doc{s}.md"
        p.write_bytes(make_markdown(30_000, seed=10 + s).encode("utf-8"))
        paths.append(str(p))
    chip = TextChipper(max_length=80, overlap=10, encoding="chars")

    many = chip.from_files(paths)
    assert len(many) == len(paths)
    for path, got in zip(paths, many):
        expected = [key(c) for c in chip.from_file(path)]
        assert [key(c) for c in got] == expected
        assert [key(c) for c in chip.from_file_stream(path)] == expected


def test_invalid_utf8_strict_vs_replace(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"valid start \xff\xfe broken bytes and more text after it\n")
    chip = TextChipper(max_length=50, overlap=5, encoding="chars")
    with pytest.raises(OSError):
        chip.from_file(str(p))
    chunks = chip.from_file(str(p), errors="replace")
    assert chunks and "\ufffd" in "".join(c.text for c in chunks)


def test_from_files_reports_the_first_error_in_input_order(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("hello world, this is fine.\n")
    # Slow-to-fail file placed BEFORE a fast-to-fail (missing) one. With scheduling-dependent
    # error selection the missing file would win some of the time.
    bad_utf8 = tmp_path / "bad.txt"
    bad_utf8.write_bytes((b"plain text line\n" * 200_000) + b"\xff\xfe\n")
    missing = tmp_path / "missing.txt"
    chip = TextChipper(max_length=100, overlap=10, encoding="chars")
    for _ in range(10):
        with pytest.raises(OSError, match="invalid UTF-8"):
            chip.from_files([str(good), str(bad_utf8), str(missing)])


# --------------------------------------------------------------------------- big documents


def test_large_document_multithreaded_path_matches_streaming(tmp_path):
    """chunk() on a >= 4 MiB document runs on several threads (when more than one is
    available); the streaming reader is always single-threaded. Results must be identical."""
    doc = make_markdown(6 * 1024 * 1024, seed=99)
    assert len(doc.encode("utf-8")) >= 4 * 1024 * 1024
    chip = TextChipper(max_length=256, overlap=20, encoding="chars", normalize="nfc")

    parallel = chip.chunk(doc)

    path = tmp_path / "big.md"
    path.write_bytes(doc.encode("utf-8"))
    sequential = list(chip.from_file_stream(str(path)))

    assert len(parallel) == len(sequential) > 1000
    for i, (a, b) in enumerate(zip(parallel, sequential)):
        assert key(a) == key(b), f"chunk {i} differs"


# --------------------------------------------------------------------------- error handling

# One grapheme cluster made of four characters: with encoding="chars" it needs 4 tokens.
UNSPLITTABLE = "\U0001F469\U0001F3FD\u200d\U0001F4BB"


def _tiny_chipper(**kwargs):
    return TextChipper(
        max_length=1,
        overlap=0,
        encoding="chars",
        normalize="none",
        include_context=False,
        **kwargs,
    )


def test_chunk_raises_when_a_grapheme_cannot_fit_in_strict_mode():
    with pytest.raises(ValueError, match="grapheme"):
        _tiny_chipper().chunk(UNSPLITTABLE)


def test_chunk_batch_raises_instead_of_returning_empty_or_partial_results():
    chip = _tiny_chipper()
    # Sequential branch (fewer than 4 texts).
    with pytest.raises(ValueError, match="grapheme"):
        chip.chunk_batch(["ab", UNSPLITTABLE, "cd"])
    # Parallel branch (>= 4 texts and >= 64 KiB in total).
    texts = ["abc " * 5_000] * 4 + [UNSPLITTABLE]
    with pytest.raises(ValueError, match="grapheme"):
        chip.chunk_batch(texts)


def test_non_strict_mode_emits_an_oversized_grapheme_instead_of_raising():
    chunks = _tiny_chipper(strict=False).chunk(UNSPLITTABLE)
    assert [c.text for c in chunks] == [UNSPLITTABLE]


# --------------------------------------------------------------------------- flat mode


def test_include_context_false_never_reports_headers():
    doc = "# Title\n\n" + "some body text that needs several chunks. " * 30
    chip = TextChipper(
        max_length=40,
        overlap=0,
        encoding="chars",
        normalize="none",
        include_context=False,
    )
    chunks = chip.chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert list(c.headers) == []
        assert c.contextualized_text == c.text
