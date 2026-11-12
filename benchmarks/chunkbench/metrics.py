"""Library-independent quality metrics.

Everything here is computed from (a) the source text and (b) the strings a library returned.
No library's own offsets or token counts are used: chunks are located in the source by exact
substring search, and tokens are counted with the reference tokenizer.

All metrics are reported for every library; none is a single "score". Higher is not always better,
see the README for how to read each one.
"""
from __future__ import annotations

import bisect
import re
import statistics
import unicodedata
from dataclasses import dataclass

import numpy as np

# Characters that mean "word" for the mid-word-cut metric. Scripts written without spaces are
# excluded because "inside a word" is not defined for them by whitespace.
_NO_SPACE_SCRIPTS = re.compile(
    "[฀-໿က-႟ក-៿⺀-鿿぀-ヿ가-힯豈-﫿＀-￯]"
)
_SENTENCE_END = re.compile(r"[.!?…。！？][\"'”’»)\]]*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
SETEXT_LINES = 2  # a setext title is at most this many lines
_FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})")


def _is_word_char(ch: str) -> bool:
    return (ch.isalnum() or ch == "_") and not _NO_SPACE_SCRIPTS.match(ch)


@dataclass
class Located:
    spans: list[tuple[int, int] | None]  # one per chunk, in order
    n_located: int


def locate(source: str, bodies: list[str], ordered_overlap_ok: bool) -> Located:
    """Find each chunk body in the source, left to right, by exact substring search.

    A chunk that is not found verbatim (altered, re-encoded, or containing replacement characters)
    gets `None`; that is itself a fidelity finding and is reported as `located_pct`.
    """
    spans: list[tuple[int, int] | None] = []
    last_start = -1
    last_end = 0
    n = 0
    for body in bodies:
        if not body:
            spans.append(None)
            continue
        lo = last_start + 1 if ordered_overlap_ok else last_end
        idx = source.find(body, max(lo, 0))
        if idx < 0 and not ordered_overlap_ok:
            idx = source.find(body, max(last_start + 1, 0))
        if idx < 0:
            spans.append(None)
            continue
        spans.append((idx, idx + len(body)))
        last_start, last_end = idx, idx + len(body)
        n += 1
    return Located(spans, n)


def _union_nonspace(source: str, spans: list[tuple[int, int]]) -> tuple[int, int]:
    """(covered non-whitespace chars, total non-whitespace chars)."""
    mask = np.fromiter((not c.isspace() for c in source), dtype=np.int8, count=len(source))
    prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    covered = 0
    cur_s = cur_e = None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += int(prefix[cur_e] - prefix[cur_s])
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += int(prefix[cur_e] - prefix[cur_s])
    return covered, int(prefix[-1])


def _fences(source: str) -> list[tuple[int, int]]:
    """Fenced code blocks as (start, end) char offsets (end = end of the closing fence line)."""
    out = []
    pos = 0
    open_start = None
    fence_char = ""
    fence_len = 0
    for line in source.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if open_start is None:
            m = _FENCE_OPEN.match(stripped)
            if m and not (m.group(2)[0] == "`" and "`" in stripped[m.end():]):
                open_start, fence_char, fence_len = pos + len(m.group(1)), m.group(2)[0], len(m.group(2))
        else:
            t = stripped.strip()
            if t and set(t) == {fence_char} and len(t) >= fence_len:
                out.append((open_start, pos + len(stripped)))
                open_start = None
        pos += len(line)
    return out


def _atx_title(content: str) -> str:
    """Title of an ATX heading line, trimmed the way CommonMark (and text_chipper) do it: the
    optional closing `#` run is dropped only when preceded by whitespace (so `# Learn C#` keeps
    its `#`)."""
    t = content.strip().lstrip("#").strip()
    stripped = t.rstrip("#")
    if stripped != t and (not stripped or stripped[-1].isspace()):
        return stripped.strip()
    return t


def _heading_lines(source: str) -> list[tuple[int, int, str]]:
    """Headings outside code fences as (start, end, title) char offsets of their source lines.

    Covers ATX headings and setext headings of up to two title lines (the title lines plus the
    underline; the title is the trimmed lines joined by single spaces). This is deliberately a
    small re-implementation: it must not depend on any library under test.
    """
    fences = _fences(source)
    fi = 0
    out: list[tuple[int, int, str]] = []
    pos = 0
    prev: list[tuple[int, int, str]] = []  # consecutive non-blank candidate title lines (max 2)
    for line in source.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        pos = end
        while fi < len(fences) and fences[fi][1] < start:
            fi += 1
        if fi < len(fences) and fences[fi][0] <= start <= fences[fi][1]:
            prev = []
            continue
        content = line.rstrip("\r\n")
        if _HEADING.match(content):
            out.append((start, end, _atx_title(content)))
            prev = []
            continue
        if prev and re.fullmatch(r" {0,3}(=+|-+)[ \t]*", content):
            title = " ".join(t for _, _, t in prev)
            out.append((prev[0][0], end, title))
            prev = []
            continue
        if content.strip():
            prev = (prev + [(start, end, content.strip())])[-SETEXT_LINES:]
        else:
            prev = []
    return out


def _content_coverage(
    source: str,
    views: list[tuple],
    found: list[tuple[int, tuple[int, int]]],
    spans: list[tuple[int, int]],
) -> dict:
    """Coverage that also counts heading lines a library carries as chunk metadata.

    A source character is *covered by a body* if some chunk body contains it. A heading line whose
    characters are not in any body still counts as preserved when the first chunk that starts after
    it lists the heading title in its `headers`. The heading's own syntax (`#`, setext underline)
    is markup, not content, so it is counted together with the title. Everything else that is
    missing stays missing, and a heading that a library truncated or dropped from `headers`
    stays lost.
    """
    n = len(source)
    diff = np.zeros(n + 1, dtype=np.int64)
    for s, e in spans:
        diff[s] += 1
        diff[e] -= 1
    covered = np.cumsum(diff[:n]) > 0
    nonspace = np.fromiter((not c.isspace() for c in source), dtype=bool, count=n)
    total = int(nonspace.sum())
    body_covered = int((covered & nonspace).sum())

    starts = [s for _, (s, _e) in found]
    order = sorted(range(len(found)), key=lambda k: starts[k])
    sorted_starts = [starts[k] for k in order]
    meta_chars = 0
    for hs, he, title in _heading_lines(source):
        lost = int((~covered[hs:he] & nonspace[hs:he]).sum())
        if lost == 0 or not title:
            continue
        j = bisect.bisect_left(sorted_starts, he)
        if j >= len(order):
            continue
        view = views[found[order[j]][0]]
        headers = view[2] if len(view) > 2 else ()
        t = unicodedata.normalize("NFC", title)
        if any(t == h or t.startswith(h) for h in (unicodedata.normalize("NFC", x) for x in headers) if h):
            meta_chars += lost
    return {
        "heading_meta_chars": meta_chars,
        "content_coverage_pct": 100.0 * (body_covered + meta_chars) / total if total else 100.0,
        "lost_unexplained_chars": total - body_covered - meta_chars,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def evaluate(
    source: str,
    views: list[tuple[str, str]],
    *,
    chunk_size: int,
    overlap: int,
    fmt: str,
    gold: list[tuple[int, int]],
    count,
) -> dict:
    """Compute every quality metric for one library on one workload.

    `views` is a list of (delivered, body); `count` is the reference token counter.
    """
    n = len(views)
    m: dict = {"chunks": n}
    if n == 0:
        return m

    # A view is (delivered, body) or (delivered, body, headers); `headers` is optional metadata.
    delivered = [v[0] for v in views]
    bodies = [v[1] for v in views]

    # ---- budget (on what would actually be delivered to the model)
    toks = [count(d) for d in delivered]
    over = [t for t in toks if t > chunk_size]
    m["tokens_max"] = max(toks)
    m["tokens_mean"] = statistics.fmean(toks)
    m["tokens_p5"] = _percentile(toks, 0.05)
    m["tokens_p50"] = _percentile(toks, 0.50)
    m["tokens_p95"] = _percentile(toks, 0.95)
    m["fill_ratio"] = m["tokens_mean"] / chunk_size
    m["over_limit_chunks"] = len(over)
    m["over_limit_pct"] = 100.0 * len(over) / n
    m["worst_overshoot_pct"] = 100.0 * (max(toks) - chunk_size) / chunk_size if over else 0.0
    body_tail = toks[:-1]
    m["tiny_chunk_pct"] = 100.0 * sum(1 for t in body_tail if t < 0.1 * chunk_size) / max(1, len(body_tail))

    # ---- fidelity (is the source preserved?)
    loc = locate(source, bodies, ordered_overlap_ok=overlap > 0)
    m["located_pct"] = 100.0 * loc.n_located / n
    found = [(i, sp) for i, sp in enumerate(loc.spans) if sp is not None]
    spans = [sp for _, sp in found]
    if spans:
        covered, total = _union_nonspace(source, spans)
        m["coverage_pct"] = 100.0 * covered / total if total else 100.0
        m["lost_nonspace_chars"] = total - covered
        m.update(_content_coverage(source, views, found, spans))
        text_chars = sum(e - s for s, e in spans)
        union_chars = sum(e - s for s, e in _merge(spans))
        m["duplication_pct"] = 100.0 * (text_chars - union_chars) / union_chars if union_chars else 0.0
    m["replacement_char_chunks"] = sum(1 for b in bodies if "�" in b and "�" not in source)

    # ---- boundary quality
    if len(spans) > 1:
        ends = [e for _, e in spans[:-1]]
        cut_positions = set(ends) | {s for s, _ in spans[1:]}
        inside = 0
        considered = 0
        for p in cut_positions:
            if 0 < p < len(source):
                considered += 1
                if _is_word_char(source[p - 1]) and _is_word_char(source[p]):
                    inside += 1
        m["mid_word_cut_pct"] = 100.0 * inside / considered if considered else 0.0

        clean = 0
        for _, (s, e) in found[:-1]:
            raw = source[s:e]
            if _SENTENCE_END.search(raw.rstrip()) or raw.endswith("\n") or source[e : e + 1] == "\n":
                clean += 1
        m["clean_end_pct"] = 100.0 * clean / (len(found) - 1)

    # ---- Markdown structure
    if fmt == "markdown":
        orphan = 0
        for body in bodies[:-1]:
            lines = [ln for ln in body.splitlines() if ln.strip()]
            if lines and _HEADING.match(lines[-1]):
                orphan += 1
        m["orphan_heading_pct"] = 100.0 * orphan / max(1, n - 1)
        starts = [s for s, _ in spans]
        small = []
        for fs, fe in _fences(source):
            if count(source[fs:fe]) <= chunk_size // 2:
                small.append((fs, fe))
        if small and spans:
            intact = 0
            for fs, fe in small:
                j = bisect.bisect_right(starts, fs + 3) - 1
                # Look back a few chunks: with overlap several chunks can start before `fs`.
                ok = False
                for k in range(j, max(-1, j - 4), -1):
                    s, e = spans[k]
                    if s <= fs + 3 and e >= fe - 1:
                        ok = True
                        break
                intact += ok
            m["fences_measured"] = len(small)
            m["fence_intact_pct"] = 100.0 * intact / len(small)

    # ---- overlap behaviour (token-level, between consecutive located chunks)
    if len(found) > 1:
        ov_tokens = []
        for (_, (s1, e1)), (_, (s2, e2)) in zip(found[:-1], found[1:]):
            ov_tokens.append(count(source[s2:e1]) if e1 > s2 else 0)
        m["overlap_tokens_median"] = statistics.median(ov_tokens)
        m["overlap_tokens_max"] = max(ov_tokens)
        m["overlap_violation_pct"] = 100.0 * sum(1 for t in ov_tokens if t > overlap) / len(ov_tokens)

    # ---- answer-span alignment (oracle: a perfect retriever returns every chunk touching the span)
    if gold and spans:
        m.update(_alignment(source, spans, gold, count))
    return m


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _alignment(source: str, spans: list[tuple[int, int]], gold: list[tuple[int, int]], count) -> dict:
    starts = [s for s, _ in spans]
    ious, recalls, per, single, ctx_tokens = [], [], [], 0, []
    for gs, ge in gold:
        hi = bisect.bisect_left(starts, ge)  # chunks starting before the excerpt ends
        sel = [(s, e) for s, e in spans[max(0, hi - 40) : hi] if e > gs]
        if not sel:
            ious.append(0.0)
            recalls.append(0.0)
            per.append(0)
            continue
        cs = min(s for s, _ in sel)
        ce = max(e for _, e in sel)
        inter_s, inter_e = max(gs, cs), min(ge, ce)
        ex_tokens = max(1, count(source[gs:ge]))
        inter_tokens = count(source[inter_s:inter_e]) if inter_e > inter_s else 0
        union_tokens = count(source[min(gs, cs) : max(ge, ce)])
        ious.append(inter_tokens / max(1, union_tokens))
        recalls.append(min(1.0, inter_tokens / ex_tokens))
        per.append(len(sel))
        ctx_tokens.append(count(source[cs:ce]))
        if len(sel) == 1 and sel[0][0] <= gs + 3 and sel[0][1] >= ge - 1:
            single += 1
    return {
        "gold_spans": len(gold),
        "oracle_iou_mean": statistics.fmean(ious),
        "oracle_recall_mean": statistics.fmean(recalls),
        "chunks_per_span_mean": statistics.fmean(per),
        "single_chunk_hit_pct": 100.0 * single / len(gold),
        "context_tokens_per_span_mean": statistics.fmean(ctx_tokens) if ctx_tokens else 0.0,
    }
