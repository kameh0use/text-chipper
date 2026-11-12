"""One adapter per contestant. Review these first: they are the only place where a library is
configured, so they are where bias could hide.

Rules every adapter follows
  * Same task for everyone: split the input into chunks of at most `chunk_size` cl100k_base tokens.
  * Same tokenizer: cl100k_base (tiktoken vocabulary) everywhere.
  * Library defaults for everything else (separators, trimming, merging). Nothing is tuned.
  * Use the library's own documented route for token-bounded splitting. Where a library offers a
    Markdown-aware splitter, it is used for Markdown workloads (`fmt == "markdown"`).
  * Only the library's own `chunk`/`split` call is timed; conversion of the native result to
    strings happens outside the timed region (`views`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

ENCODING = "cl100k_base"


@dataclass
class Ctx:
    chunk_size: int
    overlap: int
    fmt: str  # "plain" | "markdown"
    encoding: str = ENCODING


@dataclass
class View:
    """What a chunk looks like to a consumer: `delivered` is what would be embedded / sent to a
    model (it must satisfy the budget); `body` is the source text it covers."""

    delivered: str
    body: str
    # Section titles the library carries as metadata instead of in the body (for `text_chipper`:
    # `Chunk.headers`). Used only by the `content_coverage_pct` metric.
    headers: tuple[str, ...] = ()


class Runner:
    def __init__(self, run: Callable[[str], Any], views: Callable[[Any], list[View]]) -> None:
        self.run = run
        self.views = views


@dataclass
class Adapter:
    name: str
    label: str
    dist: str | None  # PyPI distribution name, for the version stamp
    docs: str
    description: str
    supports_overlap: bool
    build: Callable[[Ctx], Runner]


def _plain_views(raw: Any) -> list[View]:
    return [View(s, s) for s in raw]


# --------------------------------------------------------------------------- text_chipper


def _text_chipper(ctx: Ctx, include_context: bool) -> Runner:
    import text_chipper

    chipper = text_chipper.TextChipper(
        max_length=ctx.chunk_size,
        overlap=ctx.overlap,
        encoding=ctx.encoding,
        include_context=include_context,
    )
    return Runner(
        chipper.chunk,
        lambda raw: [View(c.contextualized_text, c.text, tuple(c.headers)) for c in raw],
    )


# --------------------------------------------------------------------------- tiktoken window


def _tiktoken_window(ctx: Ctx) -> Runner:
    import tiktoken

    enc = tiktoken.get_encoding(ctx.encoding)
    step = ctx.chunk_size - ctx.overlap

    def run(text: str) -> list[str]:
        ids = enc.encode(text, disallowed_special=())
        return [enc.decode(ids[i : i + ctx.chunk_size]) for i in range(0, len(ids), step)]

    return Runner(run, _plain_views)


# --------------------------------------------------------------------------- chonkie


def _chonkie_tokenizer(ctx: Ctx):
    import tiktoken

    return tiktoken.get_encoding(ctx.encoding)


def _chonkie_build(cls_name: str, ctx: Ctx, **extra: Any):
    import chonkie

    cls = getattr(chonkie, cls_name)
    errors: list[str] = []
    # chonkie accepts a tokenizer name, a tokenizer object or a counter depending on version.
    for tok in (_chonkie_tokenizer(ctx), ctx.encoding):
        for kw in ("tokenizer", "tokenizer_or_token_counter"):
            try:
                return cls(**{kw: tok}, chunk_size=ctx.chunk_size, **extra)
            except TypeError as exc:
                errors.append(f"{kw}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kw}={type(tok).__name__}: {exc}")
    raise RuntimeError(f"could not construct chonkie.{cls_name}: " + " | ".join(errors[:4]))


def _chonkie_token(ctx: Ctx) -> Runner:
    chunker = _chonkie_build("TokenChunker", ctx, chunk_overlap=ctx.overlap)
    return Runner(chunker.chunk, lambda raw: _plain_views([c.text for c in raw]))


def _chonkie_recursive(ctx: Ctx) -> Runner:
    chunker = _chonkie_build("RecursiveChunker", ctx)
    return Runner(chunker.chunk, lambda raw: _plain_views([c.text for c in raw]))


# --------------------------------------------------------------------------- semantic-text-splitter


def _semantic_text_splitter(ctx: Ctx) -> Runner:
    import semantic_text_splitter as sts

    cls = sts.MarkdownSplitter if ctx.fmt == "markdown" else sts.TextSplitter
    # "gpt-4" resolves to cl100k_base inside tiktoken-rs.
    splitter = cls.from_tiktoken_model("gpt-4", ctx.chunk_size, overlap=ctx.overlap)
    return Runner(splitter.chunks, _plain_views)


# --------------------------------------------------------------------------- semchunk


def _semchunk(ctx: Ctx) -> Runner:
    import semchunk
    import tiktoken

    # Default settings, including its own token-count memoisation (memoize=True). That cache is
    # per chunker object, so the cold mode (a fresh process per sample) is the fair way to time it.
    chunker = semchunk.chunkerify(tiktoken.get_encoding(ctx.encoding), ctx.chunk_size)
    overlap = ctx.overlap or None  # semchunk: >= 1 means an absolute number of tokens

    def run(text: str) -> list[str]:
        return chunker(text, overlap=overlap)

    return Runner(run, _plain_views)


# --------------------------------------------------------------------------- langchain


def _langchain(ctx: Ctx) -> Runner:
    from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

    cls = MarkdownTextSplitter if ctx.fmt == "markdown" else RecursiveCharacterTextSplitter
    splitter = cls.from_tiktoken_encoder(
        encoding_name=ctx.encoding,
        chunk_size=ctx.chunk_size,
        chunk_overlap=ctx.overlap,
        disallowed_special=(),
    )
    return Runner(splitter.split_text, _plain_views)


ADAPTERS: dict[str, Adapter] = {
    a.name: a
    for a in [
        Adapter(
            "text_chipper",
            "text_chipper",
            "text-chipper",
            "",
            "Default configuration (Markdown-aware, strict budget incl. header context).",
            True,
            lambda ctx: _text_chipper(ctx, True),
        ),
        Adapter(
            "text_chipper_flat",
            "text_chipper (flat)",
            "text-chipper",
            "",
            "include_context=False: plain token windows, no Markdown parsing.",
            True,
            lambda ctx: _text_chipper(ctx, False),
        ),
        Adapter(
            "chonkie_token",
            "chonkie TokenChunker",
            "chonkie",
            "https://docs.chonkie.ai/",
            "chonkie.TokenChunker (fixed token windows).",
            True,
            _chonkie_token,
        ),
        Adapter(
            "chonkie_recursive",
            "chonkie RecursiveChunker",
            "chonkie",
            "https://docs.chonkie.ai/",
            "chonkie.RecursiveChunker with default rules (no overlap option).",
            False,
            _chonkie_recursive,
        ),
        Adapter(
            "semantic_text_splitter",
            "semantic-text-splitter",
            "semantic-text-splitter",
            "https://github.com/benbrandt/text-splitter",
            "TextSplitter (plain) / MarkdownSplitter (markdown), tiktoken cl100k_base.",
            True,
            _semantic_text_splitter,
        ),
        Adapter(
            "semchunk",
            "semchunk",
            "semchunk",
            "https://github.com/isaacus-dev/semchunk",
            "semchunk.chunkerify(tiktoken cl100k_base, chunk_size); plain-text splitter, no Markdown mode.",
            True,
            _semchunk,
        ),
        Adapter(
            "langchain",
            "LangChain",
            "langchain-text-splitters",
            "https://python.langchain.com/docs/how_to/split_by_token/",
            "RecursiveCharacterTextSplitter / MarkdownTextSplitter .from_tiktoken_encoder.",
            True,
            _langchain,
        ),
        Adapter(
            "tiktoken_window",
            "tiktoken window",
            "tiktoken",
            "https://github.com/openai/tiktoken",
            "Naive baseline: encode once, slice fixed token windows, decode each window.",
            True,
            _tiktoken_window,
        ),
    ]
}

DEFAULT_LIBS = list(ADAPTERS)


def register_test_adapter(adapter: Adapter) -> None:
    """Used by the self-tests to plug in stub contestants."""
    ADAPTERS[adapter.name] = adapter


def version_of(dist: str | None) -> str:
    if not dist:
        return "-"
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def thread_env(lane: str) -> dict[str, str]:
    """Environment overrides for the single-thread lane (the CPU affinity pin is applied by the
    worker where the OS supports it)."""
    if lane != "single":
        return {}
    keys = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    env = {k: "1" for k in keys}
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def in_test_mode() -> bool:
    return os.environ.get("CHUNKBENCH_TEST") == "1"


def load_plugins() -> None:
    """Import extra adapter modules named in CHUNKBENCH_PLUGIN (comma separated). Self-tests only."""
    import importlib

    for name in filter(None, os.environ.get("CHUNKBENCH_PLUGIN", "").split(",")):
        importlib.import_module(name)
