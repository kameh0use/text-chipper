use std::sync::Arc;

use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};

use super::backend::CountSite;
use super::{Backend, Chunk};

#[derive(Debug, Clone, Copy, Default)]
pub struct Origin {
    pub byte: usize,
    pub chr: usize,
}

#[inline]
fn floor_cb(t: &str, mut i: usize) -> usize {
    if i >= t.len() {
        return t.len();
    }
    while !t.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[inline]
fn ceil_cb(t: &str, mut i: usize) -> usize {
    if i >= t.len() {
        return t.len();
    }
    while !t.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[derive(Debug, Default)]
pub struct CharCursor {
    byte: usize,
    chr: usize,
}

impl CharCursor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn at(&mut self, text: &str, byte: usize) -> usize {
        let byte = floor_cb(text, byte);
        if byte >= self.byte {
            self.chr += text[self.byte..byte].chars().count();
        } else {
            self.chr -= text[byte..self.byte].chars().count();
        }
        self.byte = byte;
        self.chr
    }
}

pub fn empty_headers() -> Arc<Vec<String>> {
    Arc::new(Vec::new())
}

/// Count Unicode scalar values with a constant-time ASCII fast path.
#[inline]
pub(super) fn char_len(text: &str) -> usize {
    if text.is_ascii() {
        text.len()
    } else {
        text.chars().count()
    }
}
/// Build grapheme boundaries, using byte boundaries for ASCII text.
#[inline]
fn build_grapheme_boundaries(text: &str) -> Vec<usize> {
    if text.is_ascii() {
        (0..=text.len()).collect()
    } else {
        text.grapheme_indices(true)
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect()
    }
}
#[inline]
fn is_grapheme_boundary(t: &str, i: usize) -> bool {
    if i == 0 || i >= t.len() {
        return true;
    }
    // Fast path: two adjacent ASCII bytes are a boundary unless they are CR LF.
    let b = t.as_bytes();
    if b[i - 1] < 0x80 && b[i] < 0x80 {
        return !(b[i - 1] == b'\r' && b[i] == b'\n');
    }
    GraphemeCursor::new(i, t.len(), true)
        .is_boundary(t, 0)
        .unwrap_or(true)
}

/// Trim whitespace, but never trim *into* a grapheme cluster: " \u{301}x" must not become
/// "\u{301}x" (the combining mark would start the chunk mid-cluster).
pub(super) fn trim_bounds(t: &str, s: usize, e: usize) -> (usize, usize) {
    let sl = &t[s..e];
    let tr = sl.trim_start();
    let mut s2 = s + (sl.len() - tr.len());
    if s2 > s && s2 < e && !is_grapheme_boundary(t, s2) {
        // keep the whitespace: the cluster starts at `s`
        s2 = s;
    }
    let tr = t[s2..e].trim_end();
    let mut e2 = s2 + tr.len();
    if e2 < e && !is_grapheme_boundary(t, e2) {
        e2 = e;
    }
    (s2, e2)
}

#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut Vec<Chunk>,
    text: &str,
    origin: Origin,
    cursor: &mut CharCursor,
    s: usize,
    e: usize,
    tokens: usize,
    headers: &Arc<Vec<String>>,
) {
    let cs = cursor.at(text, s);
    let ce = cursor.at(text, e);
    out.push(Chunk {
        text: text[s..e].to_string(),
        start_char: origin.chr + cs,
        end_char: origin.chr + ce,
        start_byte: origin.byte + s,
        end_byte: origin.byte + e,
        token_count: tokens,
        headers: Arc::clone(headers),
    });
}

/// Abbreviations (compared case-insensitively, without the final dot) after which a period
/// normally does NOT end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "cf", "al", "fig",
    "figs", "eq", "eqs", "no", "nos", "vol", "vols", "pp", "inc", "ltd", "co", "corp", "approx",
    "dept", "gen", "gov", "hon", "rev", "sen", "rep", "col", "capt", "lt", "sgt", "mt", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "do\u{e7}",
    "yrd", "bkz", "\u{f6}rn",
];

/// Does the `.` at byte `dot` (which is followed by whitespace) really end a sentence?
/// Rejects list / section numbers at the start of a line ("2. item", "1.2. Title"),
/// single-letter initials ("J. Smith"), lettered list markers ("a. item") and the common
/// abbreviations above ("Dr.", "e.g.", "Fig."). "No." only counts as an abbreviation when a
/// digit follows ("No. 5").
fn dot_ends_sentence(t: &str, dot: usize) -> bool {
    let head = &t[..dot];
    let ws = head
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let word = head[ws..].trim_start_matches([
        '(', '[', '{', '"', '\'', '\u{201c}', '\u{2018}', '\u{ab}', '*', '_',
    ]);
    if word.is_empty() {
        return true;
    }
    let line_start = head[..ws]
        .trim_end_matches([' ', '\t'])
        .chars()
        .next_back()
        .is_none_or(|c| c == '\n');
    let numeric = word.chars().all(|c| c.is_ascii_digit() || c == '.')
        && word.chars().any(|c| c.is_ascii_digit());
    if line_start && numeric {
        return false;
    }
    let mut it = word.chars();
    if let (Some(first), None) = (it.next(), it.next()) {
        if first.is_alphabetic() && (first.is_uppercase() || line_start) {
            return false;
        }
    }
    if word.len() <= 6 {
        for a in ABBREVIATIONS {
            if word.eq_ignore_ascii_case(a) {
                if *a == "no" || *a == "nos" {
                    return !t[dot + 1..]
                        .trim_start()
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit());
                }
                return false;
            }
        }
    }
    true
}

/// Sentence terminators that are unambiguous in any context: no following space is needed.
/// Covers CJK / full-width forms, the Devanagari danda (shared by Hindi, Marathi, Nepali,
/// Sanskrit, Bengali, Gujarati, Odia, Punjabi ...), Arabic / Urdu, Armenian, Hebrew, Ethiopic,
/// Myanmar, Khmer, Tibetan and Mongolian full stops, and double punctuation marks.
fn is_sure_terminator(c: char) -> bool {
    matches!(
        c,
        '\u{3002}'
            | '\u{ff01}'
            | '\u{ff1f}'
            | '\u{ff1b}'
            | '\u{ff0e}'
            | '\u{ff61}'
            | '\u{0964}'
            | '\u{0965}'
            | '\u{06d4}'
            | '\u{061f}'
            | '\u{0589}'
            | '\u{05c3}'
            | '\u{1362}'
            | '\u{1367}'
            | '\u{1368}'
            | '\u{104b}'
            | '\u{17d4}'
            | '\u{17d5}'
            | '\u{0f0d}'
            | '\u{0f0e}'
            | '\u{1803}'
            | '\u{1809}'
            | '\u{203c}'
            | '\u{2047}'
            | '\u{2048}'
            | '\u{2049}'
            | '\u{037e}'
    )
}

/// Terminators that only end a sentence when followed by whitespace (they also occur inside
/// numbers, URLs, abbreviations ...).
fn is_soft_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | ';' | ':' | '\u{2026}' | '\u{061b}')
}

/// Closing quotes / brackets that belong to the sentence they follow: `He said "stop."`.
fn is_closer(c: char) -> bool {
    matches!(
        c,
        ')' | ']'
            | '}'
            | '"'
            | '\''
            | '\u{201d}'
            | '\u{2019}'
            | '\u{00bb}'
            | '\u{203a}'
            | '\u{300d}'
            | '\u{300f}'
            | '\u{ff09}'
            | '\u{3011}'
            | '\u{3009}'
            | '\u{300b}'
            | '\u{ff3d}'
    )
}

/// Is a sentence cut valid at char boundary `p`, i.e. just after a terminator plus any
/// closing quotes / brackets that follow it?
fn sentence_end_at(t: &str, p: usize) -> bool {
    let nx = t[p..].chars().next();
    if nx.is_some_and(is_closer) {
        return false; // the cut belongs after the closer
    }
    let mut term: Option<(usize, char)> = None;
    // At most 8 closers are skipped, which keeps a flood of quote characters linear.
    for (i, c) in t[..p].char_indices().rev().take(9) {
        if is_closer(c) {
            continue;
        }
        term = Some((i, c));
        break;
    }
    let Some((ti, tc)) = term else {
        return false;
    };
    if is_sure_terminator(tc) {
        return !nx.is_some_and(is_sure_terminator);
    }
    if is_soft_terminator(tc) {
        let ws = nx.is_none_or(char::is_whitespace);
        return ws && (tc != '.' || nx.is_none() || dot_ends_sentence(t, ti));
    }
    false
}

fn is_wordish(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Does char boundary `p` fall inside a word (or inside a grapheme cluster)? `gb` are the
/// grapheme boundaries of `t` (sorted), so every lookup is O(log n). Marks, viramas, ZWJ/ZWNJ
/// and tone marks stay attached to their base letter because the test works on grapheme
/// clusters; apostrophes and middle dots between letters (`don't`) and a hyphen between
/// letters (cut AFTER it, never before) keep the word together.
fn inside_word(t: &str, gb: &[usize], p: usize) -> bool {
    if p == 0 || p >= t.len() {
        return false;
    }
    let (Some(l), Some(r)) = (t[..p].chars().next_back(), t[p..].chars().next()) else {
        return false;
    };
    if l.is_ascii() && r.is_ascii() && is_wordish(l) && is_wordish(r) {
        return true;
    }
    let i = gb.partition_point(|&b| b < p);
    if gb.get(i) != Some(&p) {
        return true; // inside a grapheme cluster
    }
    let first_at = |pos: usize| t[pos..].chars().next();
    let lg_start = if i > 0 { gb[i - 1] } else { 0 };
    let lw = first_at(lg_start).is_some_and(is_wordish);
    let rw = is_wordish(r);
    if lw && rw {
        return true;
    }
    let connector = |c: char| matches!(c, '\'' | '\u{2019}' | '\u{00b7}');
    if lw && (connector(r) || r == '-') {
        // grapheme after the connector starts at the next boundary
        if let Some(&next) = gb.get(i + 1) {
            if next < t.len() && first_at(next).is_some_and(is_wordish) {
                return true;
            }
        }
    }
    if connector(l)
        && rw
        && lg_start == p - l.len_utf8()
        && i >= 2
        && first_at(gb[i - 2]).is_some_and(is_wordish)
    {
        return true;
    }
    false
}

/// Pull `end` back to the best boundary within the last 10 % of [start, end):
/// paragraph > line > sentence > word. Returns `end` if nothing suitable.
fn snap_end(t: &str, gb: &[usize], start: usize, end: usize) -> usize {
    if end <= start || end >= t.len() {
        return end;
    }
    let lower = ceil_cb(t, start + (end - start) * 9 / 10);
    let (mut para, mut line, mut sent, mut word) = (None, None, None, None);
    // A cut right after a terminator is only legal on a grapheme boundary: "\u{3002}\u{200d}"
    // (or a terminator followed by a combining mark) is one cluster and must not be split.
    let on_gb = |p: usize| gb.binary_search(&p).is_ok();

    for (i, ch) in t[start..end].char_indices().rev() {
        let p = start + i + ch.len_utf8(); // cut point right after `ch`
        if p < lower {
            break;
        }
        if ch == '\n' {
            let prev_nl = p >= start + 2 && t.as_bytes()[p - 2] == b'\n';
            if !on_gb(p) {
                // never expected after a control character; stay safe anyway
            } else if prev_nl {
                para = Some(p);
                break;
            } else if line.is_none() {
                line = Some(p);
            }
        } else if sent.is_none()
            && (is_sure_terminator(ch) || is_soft_terminator(ch) || is_closer(ch))
            && on_gb(p)
            && sentence_end_at(t, p)
        {
            sent = Some(p);
        }
        if word.is_none() && !inside_word(t, gb, p) {
            word = Some(p);
        }
    }
    para.or(line)
        .or(sent)
        .or(word)
        .filter(|&b| b > start)
        .unwrap_or(end)
}

// Boundary strength (lower = stronger). Also used as the separator level.
const R_PARA: u8 = 0;
/// Start of a top-level Markdown list item (stronger than a line break inside an item).
const R_ITEM: u8 = 1;
/// Start of a nested list item.
const R_SUBITEM: u8 = 2;
const R_LINE: u8 = 3;
const R_SENT: u8 = 4;
#[allow(dead_code)]
const R_WORD: u8 = 5;
const R_CHAR: u8 = 6;

/// Structural value of a boundary. Higher values mean that keeping the boundary
/// is more important than squeezing a few extra tokens into the chunk.
/// This is deliberately language-independent: it scores document structure, not words.
#[inline]
fn boundary_value(rank: u8) -> i32 {
    match rank {
        R_PARA => 24,
        R_ITEM => 21,
        R_SUBITEM => 18,
        R_LINE => 14,
        R_SENT => 9,
        R_WORD => 4,
        _ => 0,
    }
}

/// Score a candidate cut by balancing structural quality against unused capacity.
/// Integer arithmetic keeps this hot path cheap and deterministic.
#[inline]
fn boundary_score(rank: u8, used: usize, packed: usize) -> i32 {
    if packed == 0 {
        return i32::MIN;
    }
    let fill = (used.saturating_mul(100) / packed).min(100) as i32;
    boundary_value(rank) * 10 - (100 - fill) * 10
}

/// Small look-ahead adjustment for the next structural boundary.
///
/// A candidate is more attractive when the next paragraph/item/line boundary can be
/// reached in the following chunk without leaving that chunk almost empty. This is
/// deliberately a tie-breaker-sized term: it must not override a materially stronger
/// current boundary.
#[inline]
fn lookahead_bonus(distance_tokens: usize, budget: usize) -> i32 {
    if budget == 0 || distance_tokens == 0 {
        return 0;
    }
    if distance_tokens > budget {
        return -12;
    }
    let fill = (distance_tokens.saturating_mul(100) / budget).min(100) as i32;
    (fill * 24 / 100) - 12
}

/// Byte offsets (relative to the text being split) of list-item starts, supplied by the
/// structured chunker. They are boundary *hints*: every offset is a line start, so using or
/// ignoring one never cuts inside a line. Both vectors must be sorted ascending.
#[derive(Debug, Default, Clone)]
pub(crate) struct ListCuts {
    pub(crate) top: Vec<usize>,
    pub(crate) nested: Vec<usize>,
}

/// The cuts strictly inside `(a, b)` from a sorted offset list.
fn cuts_in(cuts: &[usize], a: usize, b: usize) -> Vec<usize> {
    let lo = cuts.partition_point(|&c| c <= a);
    let hi = cuts.partition_point(|&c| c < b).max(lo);
    cuts[lo..hi].to_vec()
}

/// A contiguous byte range of the text that fits the budget on its own.
/// `rank` is the strength of the boundary that follows it.
#[derive(Clone, Copy, Debug)]
struct Atom {
    start: usize,
    end: usize,
    tokens: usize,
    rank: u8,
}

/// Cut points (strictly inside (a, b)) for one separator level.
/// Paragraph/line cuts go AFTER the newline run (it stays with the previous segment, which
/// matches how BPE pre-tokenizers glue punctuation and newlines). Sentence/word cuts go
/// BEFORE the whitespace (it stays with the next word, again like the tokenizer does), so the
/// sum of per-segment token counts is (almost) exactly the count of the concatenation.
fn find_cuts(t: &str, a: usize, b: usize, level: u8) -> Vec<usize> {
    let bytes = t.as_bytes();
    let mut cuts = Vec::new();
    match level {
        R_PARA => {
            let mut i = a;
            while i < b {
                if bytes[i] == b'\n' {
                    let mut k = i + 1;
                    let mut nl = 1;
                    while k < b && matches!(bytes[k], b' ' | b'\t' | b'\r' | b'\n') {
                        if bytes[k] == b'\n' {
                            nl += 1;
                        }
                        k += 1;
                    }
                    if nl >= 2 && k < b {
                        cuts.push(k);
                    }
                    i = k;
                } else {
                    i += 1;
                }
            }
        }
        R_LINE => {
            for (i, &byte) in bytes.iter().enumerate().take(b).skip(a) {
                if byte == b'\n' && i + 1 < b {
                    cuts.push(i + 1);
                }
            }
        }
        R_SENT => {
            for (i, ch) in t[a..b].char_indices() {
                if !(is_sure_terminator(ch) || is_soft_terminator(ch) || is_closer(ch)) {
                    continue;
                }
                let p = a + i + ch.len_utf8();
                if p >= b {
                    continue;
                }
                if sentence_end_at(t, p) {
                    cuts.push(p);
                }
            }
        }
        _ => {
            let mut prev_ws = false;
            for (i, ch) in t[a..b].char_indices() {
                let ws = ch.is_whitespace();
                if ws && !prev_ws && i > 0 {
                    cuts.push(a + i);
                }
                prev_ws = ws;
            }
        }
    }
    cuts.retain(|&c| is_grapheme_boundary(t, c));
    cuts
}

struct Splitter<'a> {
    text: &'a str,
    backend: &'a Backend,
    budget: usize,
    overlap: usize,
    snap: bool,
    ratio: f64, // bytes per token, adapted as we go
    grapheme_boundaries: Vec<usize>,
    lists: ListCuts,
    /// Exact token count of the complete `text`, when the caller already tokenized that exact
    /// string. Consumed by the first top-level `collect` so the block is not counted twice.
    known_total: Option<usize>,
}

impl<'a> Splitter<'a> {
    /// Grapheme boundaries of `text[a..b]` (relative to `a`), derived from this splitter's
    /// already-built vector instead of segmenting the slice again. Returns `None` when the
    /// result could differ from a fresh `build_grapheme_boundaries(&text[a..b])`: for ASCII
    /// slices (which use per-byte boundaries) and when `a` or `b` is not itself a boundary.
    /// Starting and ending at boundaries, the segmentation of the slice is identical.
    fn shared_boundaries(&self, a: usize, b: usize) -> Option<Vec<usize>> {
        if a >= b || self.text[a..b].is_ascii() {
            return None;
        }
        let gb = &self.grapheme_boundaries;
        let lo = gb.binary_search(&a).ok()?;
        let hi = gb.binary_search(&b).ok()?;
        if hi <= lo {
            return None;
        }
        Some(gb[lo..=hi].iter().map(|&x| x - a).collect())
    }

    fn floor_boundary(&self, byte: usize) -> usize {
        let i = self.grapheme_boundaries.partition_point(|&b| b <= byte);
        self.grapheme_boundaries[i.saturating_sub(1)]
    }

    fn next_boundary(&self, byte: usize) -> usize {
        let i = self.grapheme_boundaries.partition_point(|&b| b <= byte);
        self.grapheme_boundaries
            .get(i)
            .copied()
            .unwrap_or(self.text.len())
    }

    /// Largest `end` such that count(text[start..end]) <= budget.
    /// Uses proportional guesses to bracket, then bisection. Always returns end > start.
    fn max_fit(&mut self, start: usize) -> (usize, usize) {
        let t = self.text;
        let n = t.len();
        let budget = self.budget;
        let mut lo = start;
        let mut lo_c = 0usize;
        let mut hi: Option<usize> = None;

        // Tolerance: with snapping the tail gets pulled back to a boundary anyway, so an exact
        // byte-level fit is wasted work. Each probe re-encodes a whole window, so fewer is faster.
        let tol_tokens = if self.snap { (budget / 32).max(1) } else { 0 };
        let tol_bytes = if self.snap {
            ((self.ratio * (budget / 64).max(1) as f64) as usize).max(1)
        } else {
            (self.ratio as usize).max(1)
        };

        // Bound the window: some tokenizers count a single huge pre-token (e.g. a whitespace
        // flood) in quadratic time, and the proportional extrapolation below could otherwise
        // jump straight to the end of the whole text. Capping the window at 8 bytes per token
        // (at least 2 KiB) keeps the total work linear in the input: the cost per window is
        // bounded, so it grows with the number of windows only. A window cut short by the cap
        // is still within budget and simply followed by another window.
        let cap = budget.saturating_mul(8).max(2048);
        let limit = n.min(start.saturating_add(cap));

        let guess = ((budget as f64) * self.ratio).ceil() as usize;
        let mut cur = self.floor_boundary(start.saturating_add(guess.max(1)).min(limit));

        for iter in 0..96 {
            if cur <= lo {
                cur = self.next_boundary(lo);
            }
            if let Some(h) = hi {
                if cur >= h {
                    cur = self.floor_boundary(lo + (h - lo) / 2);
                    if cur <= lo {
                        break;
                    }
                }
            }
            let c = self
                .backend
                .count_at(&t[start..cur], CountSite::WindowProbe);
            if c <= budget {
                lo = cur;
                lo_c = c;
                if cur >= limit || c + tol_tokens >= budget {
                    break;
                }
            } else {
                hi = Some(cur);
            }
            if let Some(h) = hi {
                if h - lo <= tol_bytes || self.next_boundary(lo) >= h {
                    break;
                }
            }

            let span = lo - start;
            let est = if lo_c > 0 {
                span as f64 / lo_c as f64
            } else {
                self.ratio
            };
            let rem = budget.saturating_sub(lo_c);
            let mut g = lo + ((rem as f64) * est) as usize;
            match hi {
                None => {
                    if g <= lo {
                        g = lo + ((est * 2.0) as usize).max(1);
                    }
                    if iter >= 6 {
                        g = lo + span.max(1); // exponential growth
                    }
                    g = g.min(limit);
                }
                Some(h) => {
                    if g <= lo || g >= h || iter >= 6 {
                        g = lo + (h - lo) / 2;
                    }
                }
            }
            cur = self.floor_boundary(g);
        }

        if lo == start {
            lo = self.next_boundary(start);
            lo_c = self.backend.count_at(&t[start..lo], CountSite::WindowProbe);
        }
        if lo_c > 0 {
            self.ratio = ((lo - start) as f64 / lo_c as f64).max(0.25);
        }
        (lo, lo_c)
    }

    /// Trim [start, end) and make sure the *delivered* (trimmed) text really fits the budget.
    /// Returns (s, e, tokens, end_used).
    fn settle(
        &self,
        start: usize,
        mut end: usize,
        mut known: Option<usize>,
    ) -> (usize, usize, usize, usize) {
        let t = self.text;
        for _ in 0..16 {
            let (s, e) = trim_bounds(t, start, end);
            if s >= e {
                return (s, e, 0, end);
            }
            let tc = match known.take() {
                Some(c) if s == start && e == end => c,
                _ => self.backend.count_at(&t[s..e], CountSite::Settle),
            };
            if tc <= self.budget {
                return (s, e, tc, end);
            }
            let cut = ((e - s) * (tc - self.budget) / tc).max(1);
            let ne = self.floor_boundary(e.saturating_sub(cut));
            if ne <= start {
                break;
            }
            end = ne;
        }
        // last resort: one character
        let ne = self.next_boundary(start);
        let (s, e) = trim_bounds(t, start, ne);
        let tc = if s < e {
            self.backend.count_at(&t[s..e], CountSite::Settle)
        } else {
            0
        };
        (s, e, tc, ne)
    }

    /// Recursively split [a, b) along paragraph > line > sentence > word boundaries until every
    /// piece fits the budget. Every piece is counted exactly once, and huge
    /// ranges are not even counted: their size already proves they must be split.
    fn collect(
        &mut self,
        a: usize,
        b: usize,
        level: u8,
        rank_after: u8,
        known_over: bool,
        atoms: &mut Vec<Atom>,
    ) {
        if a >= b {
            return;
        }
        let t = self.text;
        // Only the very first call covers the whole text; `take()` makes the hint single-use.
        let known = if a == 0 && b == t.len() {
            self.known_total.take()
        } else {
            None
        };
        if !known_over {
            let len = b - a;
            let surely_over = (len as f64) > 4.0 * (self.budget as f64) * self.ratio;
            if !surely_over {
                // `known` equals `count(&t[a..b])` (same string), so decisions are unchanged.
                let mut c = match known {
                    Some(k) => k,
                    None => self.backend.count_at(&t[a..b], CountSite::Atomize),
                };
                if c > 0 && len >= 16 {
                    self.ratio = (0.8 * self.ratio + 0.2 * (len as f64 / c as f64)).max(0.25);
                }
                // An item that fits only without its trailing newline must stay one atom.
                // Deliberately narrow: only a few stripped whitespace bytes are forgiven, so a
                // large whitespace run can never be counted as free.
                if c > self.budget {
                    let (ts, te) = trim_bounds(t, a, b);
                    if ts < te && len - (te - ts) <= 4 {
                        let c2 = self.backend.count_at(&t[ts..te], CountSite::Atomize);
                        if c2 <= self.budget {
                            c = c2.max(1);
                        }
                    }
                }
                if c <= self.budget {
                    atoms.push(Atom {
                        start: a,
                        end: b,
                        tokens: c,
                        rank: rank_after,
                    });
                    return;
                }
            }
        }
        if level >= R_CHAR {
            self.windows(a, b, rank_after, atoms);
            return;
        }
        let mut bounds = match level {
            R_ITEM => cuts_in(&self.lists.top, a, b),
            R_SUBITEM => cuts_in(&self.lists.nested, a, b),
            _ => find_cuts(t, a, b, level),
        };
        if bounds.is_empty() {
            self.collect(a, b, level + 1, rank_after, true, atoms);
            return;
        }
        bounds.push(b);
        let mut s = a;
        for e in bounds {
            let rank = if e == b { rank_after } else { level };
            self.collect(s, e, level + 1, rank, false, atoms);
            s = e;
        }
    }

    /// Last resort for ranges without any separator (CJK without punctuation, base64, URLs...).
    fn windows(&mut self, a: usize, b: usize, rank_after: u8, atoms: &mut Vec<Atom>) {
        let t = self.text;
        let first = atoms.len();
        let grapheme_boundaries = self
            .shared_boundaries(a, b)
            .unwrap_or_else(|| build_grapheme_boundaries(&t[a..b]));
        let mut sub = Splitter {
            text: &t[a..b],
            backend: self.backend,
            budget: self.budget,
            overlap: 0,
            snap: self.snap,
            ratio: self.ratio,
            grapheme_boundaries,
            lists: ListCuts::default(),
            known_total: None,
        };
        sub.windows_rel(a, atoms);
        self.ratio = sub.ratio;
        if atoms.len() > first {
            if let Some(last) = atoms.last_mut() {
                last.rank = rank_after;
            }
        }
    }

    fn windows_rel(&mut self, base: usize, atoms: &mut Vec<Atom>) {
        let t = self.text;
        let n = t.len();
        let mut start = 0usize;
        while start < n {
            let (end, c) = self.max_fit(start);
            let mut res = None;
            if end < n && self.snap {
                let sn = snap_end(t, &self.grapheme_boundaries, start, end);
                if sn > start && sn < end {
                    let r = self.settle(start, sn, None);
                    if r.0 < r.1 && r.2 <= self.budget {
                        res = Some(r);
                    }
                }
            }
            let (_s, _e, tc, used) = match res {
                Some(r) => r,
                None => self.settle(start, end, Some(c)),
            };
            // A whitespace-only window trims to nothing, but it is still part of the text a
            // merged chunk delivers when it sits between two non-blank atoms. Weighing it 0
            // lets `merge` pack unbounded whitespace for free; the strict gate then re-splits
            // that chunk and its tail starts before the next chunk's overlap start.
            let tc = if _s >= _e {
                self.backend
                    .count_at(&t[start..used], CountSite::Atomize)
                    .clamp(1, self.budget)
            } else {
                tc
            };
            atoms.push(Atom {
                start: base + start,
                end: base + used,
                tokens: tc,
                rank: R_CHAR,
            });
            start = used;
        }
    }

    /// Pack atoms into chunks. With `snap`, a pruned set of structural cut candidates is scored
    /// using current boundary strength, fill, and the next strong boundary. The chosen cut is
    /// exactly token-validated before emission. Overlap repeats whole trailing atoms.
    fn merge(
        &self,
        atoms: &[Atom],
        origin: Origin,
        cursor: &mut CharCursor,
        headers: &Arc<Vec<String>>,
        out: &mut Vec<Chunk>,
    ) {
        let t = self.text;
        let n = atoms.len();
        let budget = self.budget;

        // Prefix sums make token estimates O(1). Per-rank indexes let each chunk
        // inspect only a small representative set rather than every possible cut.
        let mut prefix_tokens = vec![0usize; n + 1];
        let mut cuts_by_rank: [Vec<usize>; R_WORD as usize + 1] =
            std::array::from_fn(|_| Vec::new());
        let mut strong_cuts = Vec::new();
        for (idx, atom) in atoms.iter().enumerate() {
            prefix_tokens[idx + 1] = prefix_tokens[idx].saturating_add(atom.tokens);
            let cut = idx + 1;
            if let Some(cuts) = cuts_by_rank.get_mut(atom.rank as usize) {
                cuts.push(cut);
            }
            if atom.rank <= R_LINE {
                strong_cuts.push(cut);
            }
        }

        // Reuse these small buffers for every output chunk.
        let mut candidates: Vec<(i32, usize, usize)> = Vec::with_capacity(12);
        let mut scored: Vec<(i32, usize, usize)> = Vec::with_capacity(12);

        let mut i = 0usize;
        let mut last_j = 0usize;

        while i < n {
            // The chunk end must advance past the previous one (guarantees termination).
            let min_end = if last_j > i { last_j + 1 } else { i + 1 };

            // Bound the byte length of a merged chunk. Some tokenizers count one huge
            // pre-token (a whitespace flood) in quadratic time and every chunk is re-counted
            // exactly, so merging dozens of cheap-looking whitespace windows into one
            // 64 KiB chunk made the exact recount dominate. 32 bytes per token (at least
            // 16 KiB) is far beyond any natural text, so normal chunking is unaffected.
            let byte_cap = budget.saturating_mul(32).max(16 * 1024);
            let mut j = i;
            let mut sum = 0usize;
            while j < n
                && (j < min_end
                    || (sum + atoms[j].tokens <= budget
                        && atoms[j].end - atoms[i].start <= byte_cap))
            {
                sum += atoms[j].tokens;
                j += 1;
            }

            let mut chosen_j = j;
            if j < n && self.snap {
                candidates.clear();
                scored.clear();

                // Greedy end is always a candidate.
                candidates.push((boundary_score(atoms[j - 1].rank, sum, sum), j, sum));
                let lo_k = min_end.max(i + 1);

                // Candidate pruning: retain at most two recent representatives for
                // paragraph/item/subitem/line boundaries and one for sentence/word.
                // At most 11 structural/greedy candidates are scored, regardless of
                // how many atoms are present in this packed range.
                for rank in 0..=R_WORD {
                    let cuts = &cuts_by_rank[rank as usize];
                    let mut pos = cuts.partition_point(|&cut| cut < j);
                    let keep = if rank <= R_LINE { 2 } else { 1 };
                    let mut kept = 0usize;

                    while pos > 0 && kept < keep {
                        pos -= 1;
                        let cut = cuts[pos];
                        if cut < lo_k {
                            break;
                        }
                        if !candidates.iter().any(|&(_, existing, _)| existing == cut) {
                            let used = prefix_tokens[cut].saturating_sub(prefix_tokens[i]);
                            candidates.push((boundary_score(rank, used, sum), cut, used));
                        }
                        kept += 1;
                    }
                }

                // Ensure there is always a minimum-progress fallback in the shortlist.
                if !candidates.iter().any(|&(_, cut, _)| cut == lo_k) {
                    let used = prefix_tokens[lo_k].saturating_sub(prefix_tokens[i]);
                    candidates.push((boundary_score(atoms[lo_k - 1].rank, used, sum), lo_k, used));
                }

                // Look ahead only for the pruned set. Binary search finds the next
                // paragraph/item/line boundary; prefix sums calculate its token distance
                // without another tokenizer call.
                for &(base_score, cut, used) in &candidates {
                    let next_pos = strong_cuts.partition_point(|&strong_cut| strong_cut <= cut);
                    let bonus = if let Some(&next_cut) = strong_cuts.get(next_pos) {
                        let distance = prefix_tokens[next_cut].saturating_sub(prefix_tokens[cut]);
                        lookahead_bonus(distance, budget)
                    } else {
                        0
                    };
                    scored.push((base_score + bonus * 2, cut, used));
                }
                scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

                // Validate candidates in preference order. Every accepted chunk gets
                // one exact count; tokenizer edge effects trigger another attempt only
                // among this bounded shortlist.
                let mut selected = None;
                for &(_, cut, _) in &scored {
                    let (s, e) = trim_bounds(t, atoms[i].start, atoms[cut - 1].end);
                    if s >= e {
                        continue;
                    }
                    let exact_tokens = self.backend.count_at(&t[s..e], CountSite::Candidate);
                    if exact_tokens <= budget {
                        selected = Some((cut, s, e, exact_tokens));
                        break;
                    }
                }

                if let Some((cut, s, e, exact_tokens)) = selected {
                    emit(out, t, origin, cursor, s, e, exact_tokens, headers);
                    chosen_j = cut;
                } else {
                    // Unusual tokenizer edge effects can invalidate every pruned estimate.
                    // Retain the previous strict shrink fallback for this rare path.
                    let mut cut = j;
                    loop {
                        let (s, e) = trim_bounds(t, atoms[i].start, atoms[cut - 1].end);
                        if s >= e {
                            break;
                        }
                        let exact_tokens = self.backend.count_at(&t[s..e], CountSite::Candidate);
                        if exact_tokens <= budget || cut - i <= 1 || cut <= min_end {
                            emit(out, t, origin, cursor, s, e, exact_tokens, headers);
                            chosen_j = cut;
                            break;
                        }
                        cut -= 1;
                    }
                }
            } else {
                // The final snapped tail is exactly validated too. When snapping is
                // disabled, preserve the prior additive-count fast path for well-below-
                // budget chunks and recount only at the edge.
                let mut cut = j;
                loop {
                    let (s, e) = trim_bounds(t, atoms[i].start, atoms[cut - 1].end);
                    if s >= e {
                        break;
                    }
                    if !self.snap && sum + 2 < budget {
                        emit(out, t, origin, cursor, s, e, sum, headers);
                        chosen_j = cut;
                        break;
                    }
                    let exact_tokens = self.backend.count_at(&t[s..e], CountSite::Candidate);
                    if exact_tokens <= budget || cut - i <= 1 || cut <= min_end {
                        emit(out, t, origin, cursor, s, e, exact_tokens, headers);
                        chosen_j = cut;
                        break;
                    }
                    sum = sum.saturating_sub(atoms[cut - 1].tokens);
                    cut -= 1;
                }
            }

            last_j = chosen_j;
            if chosen_j >= n {
                break;
            }

            let mut ni = chosen_j;
            if self.overlap > 0 {
                let mut acc = 0usize;
                let mut k = chosen_j;
                while k > i + 1 {
                    let tk = atoms[k - 1].tokens;
                    if acc + tk > self.overlap {
                        break;
                    }
                    acc += tk;
                    k -= 1;
                }
                // the next chunk must still have room for atom `chosen_j`
                while k < chosen_j && acc + atoms[chosen_j].tokens > budget {
                    acc -= atoms[k].tokens;
                    k += 1;
                }
                // Atom counts are only approximately additive (BPE) and the delivered text is
                // trimmed, so verify that the overlapped chunk can really hold atom `chosen_j`.
                // If it cannot, shrink the overlap. Otherwise the next chunk is over budget,
                // gets re-split by the strict gate, and its tail starts before the next chunk's
                // overlap start (starts go backwards).
                while k < chosen_j && acc + atoms[chosen_j].tokens + 2 >= budget {
                    let (s, e) = trim_bounds(t, atoms[k].start, atoms[chosen_j].end);
                    if s >= e || self.backend.count_at(&t[s..e], CountSite::Overlap) <= budget {
                        break;
                    }
                    acc -= atoms[k].tokens;
                    k += 1;
                }
                ni = k;
            }
            i = ni;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn split_by_tokens(
    text: &str,
    origin: Origin,
    budget: usize,
    overlap: usize,
    snap: bool,
    backend: &Backend,
    headers: &Arc<Vec<String>>,
    out: &mut Vec<Chunk>,
) {
    split_by_tokens_with_lists(
        text,
        origin,
        budget,
        overlap,
        snap,
        backend,
        headers,
        ListCuts::default(),
        out,
    );
}

/// [`split_by_tokens`] with Markdown list-item boundaries as extra, stronger-than-line cut
/// levels: paragraph > top-level item > nested item > line > sentence > word.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_by_tokens_with_lists(
    text: &str,
    origin: Origin,
    budget: usize,
    overlap: usize,
    snap: bool,
    backend: &Backend,
    headers: &Arc<Vec<String>>,
    lists: ListCuts,
    out: &mut Vec<Chunk>,
) {
    split_by_tokens_with_lists_known(
        text, origin, budget, overlap, snap, backend, headers, lists, None, out,
    );
}

/// Like [`split_by_tokens_with_lists`], with an optional exact token count of the whole
/// `text` that the caller already has. Pass `None` unless the count is for exactly this
/// string (a lower bound from a prefix probe is not acceptable).
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_by_tokens_with_lists_known(
    text: &str,
    origin: Origin,
    budget: usize,
    overlap: usize,
    snap: bool,
    backend: &Backend,
    headers: &Arc<Vec<String>>,
    lists: ListCuts,
    known_total: Option<usize>,
    out: &mut Vec<Chunk>,
) {
    if text.trim().is_empty() {
        return;
    }
    let mut cursor = CharCursor::new();

    if budget == 0 || budget == usize::MAX {
        let (s, e) = trim_bounds(text, 0, text.len());
        let tc = backend.count_at(&text[s..e], CountSite::Public);
        emit(out, text, origin, &mut cursor, s, e, tc, headers);
        return;
    }

    let mut sp = Splitter {
        text,
        backend,
        budget,
        overlap: overlap.min(budget.saturating_sub(1)),
        snap,
        ratio: backend.initial_ratio(),
        grapheme_boundaries: build_grapheme_boundaries(text),
        lists,
        known_total,
    };
    let mut atoms: Vec<Atom> = Vec::new();
    sp.collect(0, text.len(), R_PARA, R_CHAR, false, &mut atoms);
    sp.merge(&atoms, origin, &mut cursor, headers, out);
}

// Unit tests for adaptive boundary selection.

#[cfg(test)]
mod adaptive_boundary_tests {
    use super::*;

    #[test]
    fn stronger_boundaries_score_higher_at_equal_fill() {
        assert!(boundary_score(R_PARA, 90, 100) > boundary_score(R_LINE, 90, 100));
        assert!(boundary_score(R_LINE, 90, 100) > boundary_score(R_SENT, 90, 100));
        assert!(boundary_score(R_SENT, 90, 100) > boundary_score(R_WORD, 90, 100));
        assert!(boundary_score(R_WORD, 90, 100) > boundary_score(R_CHAR, 90, 100));
    }

    #[test]
    fn fuller_chunk_can_outscore_a_weak_boundary() {
        assert!(boundary_score(R_CHAR, 100, 100) > boundary_score(R_PARA, 70, 100));
    }

    #[test]
    fn boundary_score_handles_empty_packed_range() {
        assert_eq!(boundary_score(R_PARA, 0, 0), i32::MIN);
    }

    #[test]
    fn lookahead_prefers_reachable_well_filled_structural_cut() {
        assert!(lookahead_bonus(80, 100) > lookahead_bonus(20, 100));
        assert!(lookahead_bonus(120, 100) < lookahead_bonus(80, 100));
        assert_eq!(lookahead_bonus(0, 100), 0);
        assert_eq!(lookahead_bonus(20, 0), 0);
    }
}

#[cfg(test)]
mod boundary_share_tests {
    use super::*;

    /// A caller-supplied exact count of the whole text only saves a tokenizer call; it must
    /// never change the chunks.
    #[test]
    fn known_total_hint_does_not_change_output() {
        let backend = Backend::Chars;
        let headers = empty_headers();
        let text = "Alpha beta gamma. Delta epsilon zeta.\n\nEta theta iota kappa lambda mu. Nu xi omicron pi rho sigma tau.";
        for budget in [8usize, 16, 32, 200] {
            let run = |known: Option<usize>| {
                let mut out = Vec::new();
                split_by_tokens_with_lists_known(
                    text,
                    Origin { byte: 0, chr: 0 },
                    budget,
                    0,
                    true,
                    &backend,
                    &headers,
                    ListCuts::default(),
                    known,
                    &mut out,
                );
                out.into_iter()
                    .map(|c| (c.text, c.token_count))
                    .collect::<Vec<_>>()
            };
            let exact = backend.count_at(text, CountSite::Other);
            assert_eq!(run(None), run(Some(exact)), "budget={budget}");
        }
    }

    #[test]
    fn shared_slice_boundaries_match_fresh_segmentation() {
        let text = "\u{65e5}\u{672c}\u{8a9e} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} e\u{301}ab \u{1F1E9}\u{1F1EA}\u{1F1EB}\u{1F1F7} r\n\n\u{7d42}\u{308f}\u{308a}\u{3002}x\r\ny";
        let backend = Backend::Chars;
        let sp = Splitter {
            text,
            backend: &backend,
            budget: 8,
            overlap: 0,
            snap: false,
            ratio: 1.0,
            grapheme_boundaries: build_grapheme_boundaries(text),
            lists: ListCuts::default(),
            known_total: None,
        };
        let gb = sp.grapheme_boundaries.clone();
        for (i, &a) in gb.iter().enumerate() {
            for &b in gb.iter().skip(i + 1) {
                let sub = &text[a..b];
                match sp.shared_boundaries(a, b) {
                    Some(shared) => {
                        assert_eq!(shared, build_grapheme_boundaries(sub), "a={a} b={b}")
                    }
                    None => assert!(sub.is_ascii(), "a={a} b={b}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod sentence_boundary_tests {
    use super::*;

    fn cuts(s: &str) -> Vec<usize> {
        find_cuts(s, 0, s.len(), R_SENT)
    }

    #[test]
    fn list_and_section_numbers_are_not_sentence_ends() {
        assert!(cuts("2. sit learning resume").is_empty());
        assert!(cuts("1.2. Title words").is_empty());
        assert!(cuts("a. lettered item").is_empty());
        assert!(cuts("first line\n  10. indented number").is_empty());
    }

    #[test]
    fn abbreviations_and_initials_are_not_sentence_ends() {
        for s in [
            "Dr. Smith left",
            "See e.g. this",
            "Mr. Jones came",
            "J. R. R. Tolkien wrote",
            "Fig. 3 shows",
            "No. 5 is here",
            "Jan. 5th was cold",
        ] {
            assert!(cuts(s).is_empty(), "unexpected sentence cut in {s:?}");
        }
    }

    #[test]
    fn real_sentence_ends_are_kept() {
        assert_eq!(cuts("It ended. Then more"), vec![9]);
        assert_eq!(cuts("In 1999. Then"), vec![8]);
        assert_eq!(cuts("No. I disagree"), vec![3]);
        assert_eq!(cuts("Really? Yes! Done."), vec![7, 12]);
    }
}

#[cfg(test)]
mod script_boundary_tests {
    use super::*;

    fn cuts(s: &str) -> Vec<usize> {
        find_cuts(s, 0, s.len(), R_SENT)
    }

    fn after(s: &str, c: char) -> usize {
        s.find(c).unwrap() + c.len_utf8()
    }

    #[test]
    fn sentence_terminators_of_many_scripts() {
        let cases: [(&str, char); 10] = [
            ("\u{92f}\u{939} \u{935}\u{93e}\u{915}\u{94d}\u{92f} \u{939}\u{948}\u{964} \u{926}\u{942}\u{938}\u{930}\u{93e} \u{939}\u{948}\u{964}", '\u{964}'),
            ("\u{531}\u{57d}\u{561} \u{576}\u{561}\u{56d}\u{561}\u{564}\u{561}\u{57d}\u{578}\u{582}\u{569}\u{575}\u{578}\u{582}\u{576} \u{567}\u{589} \u{531}\u{57d}\u{561} \u{56f}\u{56b}\u{57d}\u{561}\u{589}", '\u{589}'),
            ("\u{647}\u{644} \u{647}\u{630}\u{627} \u{633}\u{624}\u{627}\u{644}\u{61f} \u{646}\u{639}\u{645} \u{647}\u{630}\u{627} \u{62c}\u{648}\u{627}\u{628}", '\u{61f}'),
            ("\u{6cc}\u{6c1} \u{6c1}\u{6d2}\u{6d4} \u{648}\u{6c1} \u{6c1}\u{6d2} \u{627}\u{6cc}\u{6a9}", '\u{6d4}'),
            ("\u{12ed}\u{1205} \u{12d3}\u{1228}\u{134d}\u{1270} \u{1290}\u{1308}\u{122d} \u{1290}\u{12cd}\u{1362} \u{12ed}\u{1205} \u{1201}\u{1208}\u{1270}\u{129b}\u{12cd} \u{1290}\u{12cd}", '\u{1362}'),
            ("\u{1019}\u{103c}\u{1014}\u{103a}\u{1019}\u{102c}\u{104b} \u{1019}\u{103c}\u{1014}\u{103a} \u{1019}\u{102c}", '\u{104b}'),
            ("\u{1780}\u{17d2}\u{1789}\u{17bb}\u{17c6}\u{17d4} \u{1780}\u{17d2}\u{1789}\u{17bb}\u{17c6} \u{1785}\u{17b6}\u{1794}\u{17cb}", '\u{17d4}'),
            ("\u{f56}\u{f7c}\u{f51}\u{f0b}\u{f66}\u{f90}\u{f51}\u{f0b}\u{f0d} \u{f56}\u{f7c}\u{f51}\u{f0b}\u{f66}\u{f90}\u{f51}\u{f0b} \u{f56}\u{f7c}", '\u{f0d}'),
            ("\u{4eca}\u{5929}\u{5f88}\u{597d}\u{3002}\u{660e}\u{5929}\u{4e5f}\u{597d}\u{540e}\u{5929}", '\u{3002}'),
            ("\u{5e9}\u{5dc}\u{5d5}\u{5dd} \u{5e2}\u{5d5}\u{5dc}\u{5dd}\u{5c3} \u{5e9}\u{5dc}\u{5d5}\u{5dd} \u{5e2}\u{5d5}\u{5dc}\u{5dd}", '\u{5c3}'),
        ];
        for (s, term) in cases {
            assert_eq!(
                cuts(s),
                vec![after(s, term)],
                "terminator {term:?} in {s:?}"
            );
        }
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        let s = "\u{4ed6}\u{8bf4}\u{300c}\u{4f60}\u{597d}\u{3002}\u{300d}\u{7136}\u{540e}\u{79bb}\u{5f00}\u{4e86}";
        assert_eq!(cuts(s), vec![after(s, '\u{300d}')]);
        let s = "He said \"stop.\" Then left";
        assert_eq!(cuts(s), vec![s.rfind('"').unwrap() + 1]);
    }

    fn iw(t: &str, p: usize) -> bool {
        inside_word(t, &build_grapheme_boundaries(t), p)
    }

    #[test]
    fn snap_end_never_cuts_between_terminator_and_zwj() {
        let t = "\u{8a9e}\u{306e}\u{30c8}\u{3002}\u{200d}\u{1f1e9}\u{1f1ea}";
        let gb = build_grapheme_boundaries(t);
        let cut = snap_end(t, &gb, 0, 15);
        assert!(gb.contains(&cut), "cut {cut} is inside a grapheme cluster");
    }

    #[test]
    fn words_stay_together_across_scripts() {
        assert!(iw("don't", 3));
        assert!(iw("don't", 4));
        assert!(!iw("a b", 1));
        assert!(iw("state-of", 5)); // never cut before a hyphen between letters
        assert!(!iw("state-of", 6)); // cutting after it is fine
        assert!(iw("foo_bar", 3));
        // Thai: no spaces, tone marks stay with their letters
        let thai = "\u{e20}\u{e32}\u{e29}\u{e32}\u{e44}\u{e17}\u{e22}";
        for (i, _) in thai.char_indices().skip(1) {
            assert!(iw(thai, i), "inside Thai word at {i}");
        }
        // Devanagari conjunct: never "inside" at a non-boundary, and a space is a boundary
        let hi = "\u{915}\u{94d}\u{937}\u{924} \u{939}\u{93f}";
        let sp = hi.find(' ').unwrap();
        assert!(!iw(hi, sp));
        assert!(iw(hi, "\u{915}\u{94d}".len()));
    }
}
