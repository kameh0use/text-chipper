"""Measure how many source characters a plain (non-overlapping) chunker actually loses on CJK text.

Why: the benchmark's `lost_nonspace_chars` counts every character of a chunk that cannot be
matched verbatim, so one replacement character removes the whole chunk from coverage. This
script counts only the characters that really disappear from the delivered text.

Use it for splitters whose chunks are consecutive, non-overlapping slices of the source
(tiktoken window, chonkie TokenChunker, and so on). Do NOT use it for text_chipper's default
mode, whose chunk bodies omit heading text on purpose (that text is returned in `headers`).

    from cjk_loss_check import loss_report
    print(loss_report(source_text, [view.delivered for view in views]))
"""
from __future__ import annotations

REPL = "�"


def _nonspace(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def loss_report(source: str, chunks: list[str]) -> dict:
    """Return the chunk-level and character-level damage for one chunker run."""
    if REPL in source:
        raise ValueError("source already contains U+FFFD; the check cannot tell damage apart")
    damaged = [c for c in chunks if REPL in c]
    joined = "".join(chunks)
    total = _nonspace(source)
    kept = _nonspace(joined.replace(REPL, ""))
    lost = total - kept
    return {
        "chunks": len(chunks),
        "chunks_with_replacement_char": len(damaged),
        "pct_chunks_with_replacement_char": round(100 * len(damaged) / max(1, len(chunks)), 2),
        "replacement_chars": sum(c.count(REPL) for c in chunks),
        "source_nonspace_chars": total,
        "lost_nonspace_chars": lost,
        "pct_chars_lost": round(100 * lost / max(1, total), 3),
    }
