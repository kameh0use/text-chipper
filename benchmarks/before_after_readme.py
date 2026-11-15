"""Before/after on a real Markdown file: fixed token window vs text-chipper.

Writes every chunk from both methods to a Markdown report and flags chunks
that contain an unclosed fenced code block. Also reports chunks that contain
U+FFFD replacement characters, which is what a token window produces when it
cuts a multi-byte character.

Usage (from the folder where you installed tiktoken and text-chipper):
    python before_after.py path/to/doc.md
    python before_after.py path/to/README.md --max-tokens 256 --out report.md

Use a real document from your own project. Do not edit the output by hand.

Limit: a chunk that lies entirely inside a code block, with no fence markers,
is not flagged. The check only sees fence lines (lines starting with ``` or ~~~).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import tiktoken
import text_chipper

ENCODING = "cl100k_base"
REPLACEMENT = "�"

# A fence line: up to three spaces of indentation, then at least three backticks or tildes.
FENCE_LINE = re.compile(r"^ {0,3}(`{3,}|~{3,})", re.MULTILINE)


def window_chunks(text: str, max_tokens: int, enc) -> list[str]:
    """Fixed token windows: encode once, slice the tokens, decode each slice."""
    ids = enc.encode(text, disallowed_special=())
    return [enc.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def chipper_chunks(text: str, max_tokens: int) -> list[tuple[str, list[str], int, str]]:
    """text-chipper in its default mode: Markdown-aware, strict budget.

    Returns (delivered text, headers, token count, body). The body is checked for
    code fences: the delivered text starts with a "[ctx:...]" label glued onto the
    first line, which hides a fence opened on that line.
    """
    chipper = text_chipper.TextChipper(max_length=max_tokens, overlap=0, encoding=ENCODING)
    return [(c.contextualized_text, list(c.headers), c.token_count, c.text) for c in chipper.chunk(text)]


def splits_code_fence(chunk: str) -> bool:
    """An odd number of fence lines means the chunk starts or ends inside a code block."""
    return len(FENCE_LINE.findall(chunk)) % 2 == 1


def fenced(text: str) -> str:
    """Wrap text in a code fence longer than any backtick run inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def flags(chunk: str, body: str | None = None) -> str:
    out = []
    if splits_code_fence(body if body is not None else chunk):
        out.append("⚠ splits a code fence")
    if REPLACEMENT in chunk:
        out.append("⚠ contains U+FFFD")
    return (" " + " · ".join(out)) if out else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("before_after.md"))
    args = ap.parse_args()

    text = args.path.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding(ENCODING)

    window = window_chunks(text, args.max_tokens, enc)
    chipper = chipper_chunks(text, args.max_tokens)

    def count_flags(chunks):
        fences = sum(splits_code_fence(c) for c in chunks)
        repl = sum(REPLACEMENT in c for c in chunks)
        return fences, repl

    w_fences, w_repl = count_flags(window)
    c_fences = sum(splits_code_fence(body) for _, _, _, body in chipper)
    c_repl = sum(REPLACEMENT in delivered for delivered, _, _, _ in chipper)

    lines = [
        f"# Before / after: `{args.path.name}`",
        "",
        f"Budget: {args.max_tokens} tokens, `{ENCODING}`, no overlap.",
        "",
        f"- Fixed token window: {len(window)} chunks, {w_fences} split a code fence, {w_repl} contain U+FFFD",
        f"- text-chipper (default): {len(chipper)} chunks, {c_fences} split a code fence, {c_repl} contain U+FFFD",
        "",
        "## Fixed token window",
        "",
    ]
    for i, chunk in enumerate(window, 1):
        lines += [f"### Chunk {i} · {len(enc.encode(chunk))} tokens{flags(chunk)}", "", fenced(chunk), ""]

    lines += ["## text-chipper (default)", ""]
    for i, (chunk, headers, tokens, body) in enumerate(chipper, 1):
        heading = " > ".join(headers) if headers else "(no heading)"
        lines += [f"### Chunk {i} · {tokens} tokens · headers: {heading}{flags(chunk, body)}", "", fenced(chunk), ""]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"window:       {len(window)} chunks, {w_fences} split a code fence, {w_repl} contain U+FFFD")
    print(f"text-chipper: {len(chipper)} chunks, {c_fences} split a code fence, {c_repl} contain U+FFFD")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
