use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

use super::{
    backend::{record_event, Backend, CountSite},
    block::{Block, Headers},
    chunk_types::{context_prefix, fit_context_headers, Chunk, ChunkConfig},
    markdown::BlockParser,
    split_engine::{
        char_len, empty_headers, split_by_tokens_with_lists_known, trim_bounds, ListCuts, Origin,
    },
    ChipError,
};

/// Upper bound for the chunker's retained pending text plus deferred heading prefix.
const MAX_PENDING_STATE_BYTES: usize = 2 * 1024 * 1024;
/// Only cache short, repeatedly occurring structural fragments. This keeps both hashing
/// overhead and retained memory bounded for varied documents.
/// Blocks up to this size are always counted whole by `count_block_text`; larger ones may be
/// answered with a prefix lower bound. A count is therefore exact iff `len <= this`.
const BLOCK_PROBE_MIN_BYTES: usize = 16 * 1024;
const SMALL_COUNT_CACHE_MAX_TEXT_BYTES: usize = 128;
const SMALL_COUNT_CACHE_MAX_ENTRIES: usize = 256;

// Pending-chunk state and the SmartChunker (block-level packing).

#[derive(Clone)]
pub(crate) struct Pending {
    pub(crate) text: String,
    start_byte: usize,
    start_char: usize,
    /// Estimated body-only tokens (exact right after an exact count or for a single block).
    tokens: usize,
    /// Exact token count of the delivered text (context prefix + body) when known.
    exact_total: Option<usize>,
    /// Block joins accumulated since the last exact count; bounds the estimate's drift.
    joins: usize,
    headers: Headers,
    // Source heading prefix is held separately until a cross-section merge is actually made.
    // This keeps the original source text of ordinary single-section chunks.
    pub(crate) prefix: String,
    prefix_start_byte: usize,
    prefix_start_char: usize,
}

/// Incremental, block-aware chunker: feed it lines with [`SmartChunker::push_line`], then call
/// [`SmartChunker::finish`]. It drives the Markdown [`BlockParser`], packs blocks into chunks
/// and splits oversized blocks.
///
/// Used by the streaming file reader and by the whole-document path when
/// `include_context = true`. With `include_context = false` the parser still recognizes
/// fenced code and lists but does not track headings; neither the whole-document API nor
/// `FileChunker` uses this chunker in that mode (they use flat windows, see [`ChunkConfig`]).
pub struct SmartChunker {
    cfg: ChunkConfig,
    backend: Arc<Backend>,
    pub(crate) parser: BlockParser,
    pub(crate) pending: Option<Pending>,
    peak_pending_bytes: usize,
    ctx_cache: Option<(Headers, usize)>,
    // The parser shares one Headers Arc across consecutive blocks in the same section.
    // Cache its fitted form so long sections do not tokenize the same context repeatedly.
    fit_cache: Option<(Headers, Headers)>,
    scratch: Vec<Block>,
    /// Exact token counts for short repeated blocks and separators. A bounded per-chunker
    /// cache avoids re-tokenizing the same Markdown fragments without a process-global lock.
    small_count_cache: RefCell<HashMap<String, usize>>,
}

impl SmartChunker {
    pub fn new(cfg: ChunkConfig, backend: Arc<Backend>) -> Self {
        // Heading tracking (and with it `Chunk::headers`) exists only in context mode.
        let parse_headings = cfg.include_context;
        SmartChunker {
            cfg,
            backend,
            parser: BlockParser::with_headings(parse_headings),
            pending: None,
            peak_pending_bytes: 0,
            ctx_cache: None,
            fit_cache: None,
            scratch: Vec::new(),
            small_count_cache: RefCell::new(HashMap::with_capacity(64)),
        }
    }

    pub(crate) fn peak_pending_bytes(&self) -> usize {
        self.peak_pending_bytes
    }

    pub fn push_line(&mut self, line: &str, out: &mut Vec<Chunk>) -> Result<(), ChipError> {
        let mut blocks = std::mem::take(&mut self.scratch);
        self.parser.push_line(line, &mut blocks)?;
        for b in blocks.drain(..) {
            self.handle_block(b, out)?;
        }
        self.scratch = blocks;
        Ok(())
    }

    pub fn finish(&mut self, out: &mut Vec<Chunk>) -> Result<(), ChipError> {
        let mut blocks = std::mem::take(&mut self.scratch);
        self.parser.finish(&mut blocks);
        for b in blocks.drain(..) {
            self.handle_block(b, out)?;
        }
        self.scratch = blocks;
        self.flush_pending(out)?;
        Ok(())
    }

    /// Earliest source position that a not-yet-emitted chunk may still reference.
    /// Includes the pending cross-block merge, the parser's open block, and its
    /// separator/heading gap. The gap must be included because a later merge can
    /// pull those source characters back into a chunk.
    pub(crate) fn earliest_future_start(&self) -> (usize, usize) {
        let (mut byte, mut chr) = self.parser.earliest_future_start();
        if let Some(pending) = &self.pending {
            byte = byte.min(pending.start_byte).min(pending.prefix_start_byte);
            chr = chr.min(pending.start_char).min(pending.prefix_start_char);
        }
        (byte, chr)
    }

    /// Count short structural fragments once per chunker. Cache hits are exact because the
    /// key is the complete UTF-8 string, not a hash-only fingerprint. The cache is deliberately
    /// bounded and excludes longer prose/candidate strings, where hashing and retention would
    /// cost more than the occasional tokenizer call.
    fn count_small_cached_at(&self, text: &str, site: CountSite) -> usize {
        if text.len() > SMALL_COUNT_CACHE_MAX_TEXT_BYTES {
            return self.backend.count_at(text, site);
        }
        if let Some(&tokens) = self.small_count_cache.borrow().get(text) {
            return tokens;
        }
        let tokens = self.backend.count_at(text, site);
        let mut cache = self.small_count_cache.borrow_mut();
        if cache.len() >= SMALL_COUNT_CACHE_MAX_ENTRIES {
            cache.clear();
        }
        cache.insert(text.to_owned(), tokens);
        tokens
    }

    fn fit_headers(&mut self, headers: &Headers, max_context_tokens: usize) -> Headers {
        if let Some((source, fitted)) = &self.fit_cache {
            if Arc::ptr_eq(source, headers) {
                return Arc::clone(fitted);
            }
        }
        let fitted = fit_context_headers(headers, max_context_tokens, &self.backend);
        self.fit_cache = Some((Arc::clone(headers), Arc::clone(&fitted)));
        fitted
    }

    fn ctx_tokens(&mut self, headers: &Headers) -> usize {
        if let Some((h, n)) = &self.ctx_cache {
            if Arc::ptr_eq(h, headers) {
                return *n;
            }
        }
        let n = self
            .backend
            .count_at(&context_prefix(headers), CountSite::Context);
        self.ctx_cache = Some((Arc::clone(headers), n));
        n
    }

    /// Context-prefix tokens without touching the cache (usable while `pending` is borrowed).
    fn ctx_tokens_ro(&self, headers: &Headers) -> usize {
        if !self.cfg.include_context || headers.is_empty() {
            return 0;
        }
        if let Some((h, n)) = &self.ctx_cache {
            if Arc::ptr_eq(h, headers) {
                return *n;
            }
        }
        self.backend
            .count_at(&context_prefix(headers), CountSite::Context)
    }

    fn budget_for(&mut self, headers: &Headers) -> usize {
        let max = self.cfg.max_tokens;
        if max == 0 {
            return usize::MAX;
        }
        let ctx = if self.cfg.include_context && !headers.is_empty() {
            self.ctx_tokens(headers)
        } else {
            0
        };
        max.saturating_sub(ctx).max(1)
    }

    /// Count exactly what downstream embedding code receives, including context.
    fn count_delivered_at(&self, headers: &Headers, text: &str, site: CountSite) -> usize {
        if self.cfg.include_context && !headers.is_empty() {
            let mut full = context_prefix(headers);
            full.push_str(text);
            self.backend.count_at(&full, site)
        } else {
            self.backend.count_at(text, site)
        }
    }

    /// Final safety gate. The fast splitter uses body-only estimates; this gate re-tokenizes
    /// the actual delivered string and, if necessary, splits at UTF-8 character boundaries.
    /// It never relies on the sum of independently-tokenized fragments (which is not additive
    /// for BPE tokenizers).
    pub(crate) fn finalize(&self, c: Chunk, out: &mut Vec<Chunk>) -> Result<(), ChipError> {
        self.finalize_with_count(c, out, None)
    }

    /// `known_total` is supplied only when an upstream operation has already counted the
    /// exact delivered string. This avoids repeating a tokenizer pass for intact blocks.
    fn finalize_with_count(
        &self,
        mut c: Chunk,
        out: &mut Vec<Chunk>,
        known_total: Option<usize>,
    ) -> Result<(), ChipError> {
        let max = self.cfg.max_tokens;
        if !self.cfg.strict || max == 0 {
            // Chunks that carry headers only know their body count (or 0), so recount them
            // to report the delivered (context + body) size. Header-less chunks keep the
            // splitter's count, which is an estimate for chunks well below the limit;
            // only strict mode guarantees an exact `token_count`.
            if let Some(k) = known_total {
                c.token_count = k;
            } else if self.cfg.include_context && !c.headers.is_empty() {
                c.token_count =
                    self.count_delivered_at(&c.headers, &c.text, CountSite::FinalizeInput);
            }
            out.push(c);
            return Ok(());
        }

        // `token_count` is part of the public result and is exact in strict mode. Reuse an
        // upstream exact count when available; otherwise perform one authoritative count here.
        // Do not substitute atom sums, which would make the reported token_count approximate.
        // Count the actual delivered representation first. Most chunks fit with their
        // context, so this ordering avoids a separate context+first-scalar tokenization pass.
        // Only when the full contextualized chunk is over budget do we test whether context
        // itself prevents even the first scalar from fitting and should be dropped.
        // `known_total`, when present, is an exact count of the delivered string
        // (context prefix + body), whether or not headers are attached.
        let mut total = match known_total {
            Some(k) => k,
            None => self.count_delivered_at(&c.headers, &c.text, CountSite::FinalizeInput),
        };
        if total <= max {
            c.token_count = total;
            out.push(c);
            return Ok(());
        }
        if !c.headers.is_empty() {
            if let Some((_, first)) = c.text.grapheme_indices(true).next() {
                let first_end = first.len();
                if self.count_delivered_at(
                    &c.headers,
                    &c.text[..first_end],
                    CountSite::FinalizeHeaderCheck,
                ) > max
                {
                    c.headers = empty_headers();
                    total = self.count_delivered_at(&c.headers, &c.text, CountSite::FinalizeRetry);
                    if total <= max {
                        c.token_count = total;
                        out.push(c);
                        return Ok(());
                    }
                }
            }
        }

        let text = c.text.as_str();
        let mut boundaries: Vec<usize> = text.grapheme_indices(true).map(|(i, _)| i).collect();
        boundaries.push(text.len());
        if boundaries.len() <= 1 {
            return Ok(());
        }
        let mut bi = 0usize;
        while bi + 1 < boundaries.len() {
            // Find the longest suffix from this start whose *combined* representation fits.
            let mut best: Option<(usize, usize)> = None;
            // Scan the (already approximately-sized) candidate window. BPE token counts are
            // not strictly monotonic as bytes are appended, so binary search is not sound.
            for end_i in (bi + 1)..boundaries.len() {
                let end = boundaries[end_i];
                let (ts, te) = trim_bounds(text, boundaries[bi], end);
                if ts >= te {
                    best = Some((end_i, 0));
                    continue;
                }
                let n =
                    self.count_delivered_at(&c.headers, &text[ts..te], CountSite::FinalizeRetry);
                if n <= max {
                    best = Some((end_i, n));
                }
            }
            let (mut end_i, mut n) = match best {
                Some(v) if v.0 > bi => v,
                _ => {
                    // A complete Unicode grapheme can itself exceed an exceptionally small
                    // token budget. There is no lossless non-empty chunk satisfying the invariant.
                    // Return a typed error rather than panic or emit an oversized chunk.
                    let one_end = boundaries[bi + 1];
                    let (ts, te) = trim_bounds(text, boundaries[bi], one_end);
                    if ts >= te {
                        bi += 1;
                        continue;
                    }
                    let one_n = self.count_delivered_at(
                        &c.headers,
                        &text[ts..te],
                        CountSite::FinalizeRetry,
                    );
                    if one_n > max {
                        return Err(ChipError::ChunkTooLarge {
                            required_tokens: one_n,
                            max_tokens: max,
                            start_byte: c.start_byte + boundaries[bi],
                        });
                    }
                    (bi + 1, one_n)
                }
            };

            // Optional word snap, but only if the snapped candidate remains non-empty and fits.
            if self.cfg.snap_words && end_i + 1 < boundaries.len() {
                let end_byte = boundaries[end_i];
                if let Some(rel) = text[boundaries[bi]..end_byte].rfind(char::is_whitespace) {
                    let snapped = boundaries[bi] + rel;
                    if snapped > boundaries[bi] {
                        let (ts, te) = trim_bounds(text, boundaries[bi], snapped);
                        if ts < te {
                            let sn = self.count_delivered_at(
                                &c.headers,
                                &text[ts..te],
                                CountSite::FinalizeSnap,
                            );
                            if sn > 0 && sn <= max {
                                if let Ok(idx) = boundaries.binary_search(&snapped) {
                                    end_i = idx;
                                    n = sn;
                                }
                            }
                        }
                    }
                }
            }

            let raw_s = boundaries[bi];
            let raw_e = boundaries[end_i];
            let (ts, te) = trim_bounds(text, raw_s, raw_e);
            if ts < te {
                let prefix_chars = char_len(&text[..ts]);
                let body_chars = char_len(&text[ts..te]);
                let start_byte = c.start_byte + ts;
                let end_byte = c.start_byte + te;
                let start_char = c.start_char + prefix_chars;
                let end_char = start_char + body_chars;
                let piece = Chunk {
                    text: text[ts..te].to_string(),
                    start_char,
                    end_char,
                    start_byte,
                    end_byte,
                    token_count: n,
                    headers: Arc::clone(&c.headers),
                };
                // `n` was counted against this exact trimmed slice above; do not tokenize it again.
                if n <= max {
                    out.push(piece);
                } else {
                    return Err(ChipError::ChunkTooLarge {
                        required_tokens: n,
                        max_tokens: max,
                        start_byte,
                    });
                }
            }
            bi = end_i.max(bi + 1);
        }
        Ok(())
    }

    fn flush_pending(&mut self, out: &mut Vec<Chunk>) -> Result<(), ChipError> {
        if let Some(p) = self.pending.take() {
            let end_char = p.start_char + char_len(&p.text);
            let end_byte = p.start_byte + p.text.len();
            let known_total = p.exact_total;
            let c = Chunk {
                text: p.text,
                start_char: p.start_char,
                end_char,
                start_byte: p.start_byte,
                end_byte,
                token_count: p.exact_total.unwrap_or(p.tokens),
                headers: p.headers,
            };
            self.finalize_with_count(c, out, known_total)?;
        }
        Ok(())
    }

    /// The parser closes a block at every blank line, so a multi-paragraph list item arrives as
    /// several blocks ("4. next", then "   continuation paragraph\n5. last": the continuation
    /// block may also carry the NEXT item, because only blank lines end a block). When the
    /// pending chunk is full, flushing it would cut the item in half even though the item fits.
    ///
    /// If `b` starts with an indented continuation of the pending chunk's last top-level list
    /// item and that whole item fits the budget, this flushes only the text BEFORE the item,
    /// keeps the item pending, merges the continuation part of `b` into it and then handles the
    /// rest of `b` normally. Returns `None` when `b` was fully handled, or gives `b` back
    /// untouched when the rule does not apply (no list item, item too big, other section...).
    fn keep_list_item_together(
        &mut self,
        b: Block,
        out: &mut Vec<Chunk>,
    ) -> Result<Option<Block>, ChipError> {
        let max = self.cfg.max_tokens;
        if max == 0 || b.is_code || !b.text.starts_with([' ', '\t']) {
            return Ok(Some(b));
        }
        let Some(p) = self.pending.as_ref() else {
            return Ok(Some(b));
        };
        if p.headers.as_slice() != b.headers.as_slice() {
            return Ok(Some(b));
        }
        let items = crate::markdown::list_parser::parse_list_items(&p.text);
        let Some(tail_start) = items
            .iter()
            .rev()
            .find(|i| i.depth == 0)
            .map(|i| i.start_byte)
        else {
            return Ok(Some(b));
        };
        let head_len = p.text[..tail_start].trim_end().len();
        if head_len == 0 {
            return Ok(Some(b)); // the pending chunk already starts with this item
        }
        let tail_len = p.text.len() - tail_start;
        let mut cand = String::with_capacity(tail_len + b.sep_before.len() + b.text.len());
        cand.push_str(&p.text[tail_start..]);
        cand.push_str(&b.sep_before);
        cand.push_str(&b.text);
        let join_len = tail_len + b.sep_before.len();
        // Where does the item end? It must reach into `b` (indented continuation lines).
        let item_end = match crate::markdown::list_parser::parse_list_items(&cand).first() {
            Some(f) if f.depth == 0 && f.start_byte == 0 && f.end_byte > join_len => {
                f.end_byte.min(cand.len())
            }
            _ => return Ok(Some(b)),
        };
        let item_text = cand[..item_end].trim_end();
        if item_text.len() <= join_len
            || self.count_delivered_at(&p.headers, item_text, CountSite::ListItemCheck) > max
        {
            return Ok(Some(b));
        }
        // Part of `b` that belongs to the item (relative to b.text), and the rest.
        let b1_len = item_text.len() - join_len;
        let rest_start = b.text[b1_len..]
            .find(|c: char| !c.is_whitespace())
            .map(|i| b1_len + i);

        // Flush everything before the item; the item stays pending.
        let Some(p) = self.pending.take() else {
            return Ok(Some(b));
        };
        let head_text = &p.text[..head_len];
        let head_chunk = Chunk {
            text: head_text.to_string(),
            start_char: p.start_char,
            end_char: p.start_char + char_len(head_text),
            start_byte: p.start_byte,
            end_byte: p.start_byte + head_len,
            token_count: self
                .backend
                .count_at(head_text, CountSite::PendingListSplit),
            headers: Arc::clone(&p.headers),
        };
        let tail_text = p.text[tail_start..].to_string();
        let tail_tokens = self
            .backend
            .count_at(&tail_text, CountSite::PendingListSplit);
        let tail_byte = p.start_byte + tail_start;
        let tail_char = p.start_char + char_len(&p.text[..tail_start]);
        self.pending = Some(Pending {
            text: tail_text,
            start_byte: tail_byte,
            start_char: tail_char,
            tokens: tail_tokens,
            exact_total: if p.headers.is_empty() {
                Some(tail_tokens)
            } else {
                None
            },
            joins: 0,
            headers: p.headers,
            prefix: String::new(),
            prefix_start_byte: tail_byte,
            prefix_start_char: tail_char,
        });
        self.finalize_with_count(head_chunk, out, None)?;

        match rest_start {
            None => self.handle_block(b, out)?,
            Some(k) => {
                let b2 = Block {
                    text: b.text[k..].to_string(),
                    start_byte: b.start_byte + k,
                    start_char: b.start_char + char_len(&b.text[..k]),
                    sep_before: b.text[b1_len..k].to_string(),
                    headers: Arc::clone(&b.headers),
                    is_code: false,
                };
                let b1 = Block {
                    text: b.text[..b1_len].to_string(),
                    ..b
                };
                self.handle_block(b1, out)?;
                self.handle_block(b2, out)?;
            }
        }
        Ok(None)
    }

    /// Token count of a whole block. For very large blocks a prefix is counted first, with
    /// doubling sizes: as soon as a prefix alone exceeds `max_tokens` the block certainly
    /// does not fit, and that (lower-bound) count is returned instead of tokenizing the whole
    /// text. Callers only compare the result with a budget <= `max_tokens` or use it as an
    /// estimate that is re-verified with an exact count, so a lower bound above the limit is
    /// equivalent. Some tokenizers are superlinear on huge pre-tokens (whitespace floods).
    fn count_block_text(&self, text: &str) -> usize {
        let max = self.cfg.max_tokens;
        if max == 0 || max == usize::MAX || text.len() <= BLOCK_PROBE_MIN_BYTES {
            return self.count_small_cached_at(text, CountSite::BlockWhole);
        }
        // Probe only small prefixes: if the block does fit, the full count below is needed
        // anyway and large probes would just add to it (superlinear tokenizers).
        let mut size = max.saturating_mul(4).max(2048);
        let probe_limit = size.max(8 * 1024);
        while size <= probe_limit && size.saturating_mul(2) <= text.len() {
            let mut end = size;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            let c = self.backend.count_at(&text[..end], CountSite::BlockPrefix);
            if c > max {
                return c;
            }
            size = size.saturating_mul(2);
        }
        self.count_small_cached_at(text, CountSite::BlockWhole)
    }

    pub fn handle_block(&mut self, mut b: Block, out: &mut Vec<Chunk>) -> Result<(), ChipError> {
        if b.sep_before.len() > crate::markdown::MAX_PARSER_GAP_BYTES {
            return Err(ChipError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "block separator exceeds the parser byte limit",
            )));
        }
        // Public callers can construct Blocks directly. Reject oversized values rather
        // than split_off() a giant String and temporarily duplicate its allocation; the
        // streaming parser already emits source-contiguous fragments within this bound.
        if b.text.len() > crate::markdown::MAX_OPEN_BLOCK_BYTES {
            return Err(ChipError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "block exceeds the {}-byte chunker input limit",
                    crate::markdown::MAX_OPEN_BLOCK_BYTES
                ),
            )));
        }

        // Context is useful metadata, but a very long heading must not reduce the body
        // allowance to one token. Reserve at most one quarter of the total budget for it.
        // The deepest heading is retained preferentially by fit_context_headers().
        if self.cfg.include_context && self.cfg.max_tokens > 0 {
            // Small budgets still get a chance to carry a compact context label.
            // Reserve at least five tokens where possible, but always leave body capacity.
            let context_cap = (self.cfg.max_tokens / 4)
                .max(5)
                .min(self.cfg.max_tokens.saturating_sub(1));
            b.headers = self.fit_headers(&b.headers, context_cap);
        }

        // Without merging, preserve the original block/header semantics.
        if !self.cfg.merge_blocks {
            self.flush_pending(out)?;
            let budget = self.budget_for(&b.headers);
            let b_tokens = self.count_block_text(&b.text);
            if b_tokens > budget {
                self.split_block(&b, budget, b_tokens, out)?;
            } else {
                let end_char = b.start_char + char_len(&b.text);
                let end_byte = b.start_byte + b.text.len();
                let known_total = if b.headers.is_empty() {
                    Some(b_tokens)
                } else {
                    None
                };
                let c = Chunk {
                    text: b.text,
                    start_char: b.start_char,
                    end_char,
                    start_byte: b.start_byte,
                    end_byte,
                    token_count: b_tokens,
                    headers: b.headers,
                };
                self.finalize_with_count(c, out, known_total)?;
            }
            return Ok(());
        }

        // Every block is tokenized exactly once; the count feeds both the merge estimate
        // and, when the block starts a new pending chunk, that chunk's own count.
        let b_tokens = self.count_block_text(&b.text);

        // A section boundary need not force a chunk boundary. When two adjacent blocks have
        // different heading paths, their longest common prefix is still truthful context for
        // both. The heading line is carried in `sep_before`, so a merged body remains a
        // contiguous slice of the original source and its offsets stay meaningful.
        if let Some(p) = self.pending.as_ref() {
            let section_changed = p.headers.as_slice() != b.headers.as_slice();
            let common_len = if self.cfg.include_context {
                p.headers
                    .iter()
                    .zip(b.headers.iter())
                    .take_while(|(a, z)| a == z)
                    .count()
            } else {
                0
            };
            let common: Headers = if common_len == p.headers.len() {
                Arc::clone(&p.headers)
            } else {
                Arc::new(p.headers[..common_len].to_vec())
            };
            let carry_prefix = section_changed && !p.prefix.is_empty();
            let prefix_len = if section_changed { p.prefix.len() } else { 0 };
            let candidate_len = prefix_len
                .saturating_add(p.text.len())
                .saturating_add(b.sep_before.len())
                .saturating_add(b.text.len());
            let retained_prefix_len = if carry_prefix { 0 } else { p.prefix.len() };
            let can_build_candidate = candidate_len <= MAX_PENDING_STATE_BYTES
                && candidate_len.saturating_add(retained_prefix_len) <= MAX_PENDING_STATE_BYTES;
            if !can_build_candidate {
                record_event(CountSite::MergeCandidateTooLarge, candidate_len);
            } else if carry_prefix {
                // Counts all candidate materializations caused by a deferred heading prefix,
                // whether the cheap token estimate succeeds or falls back to exact counting.
                record_event(CountSite::MergePrefixCandidate, candidate_len);
            }
            let max = self.cfg.max_tokens;
            let ctx_t = self.ctx_tokens_ro(&common);

            // Do not allocate a concatenated candidate unless the complete pending
            // state (including a deferred heading prefix) stays within its byte cap.
            if can_build_candidate {
              'merge: {
                // Calculate the cheap estimate before allocating the candidate.
                let mut fast_merge: Option<(usize, Option<usize>, usize)> = None;
                // Retain the estimate before the +2 safety margin for diagnostics only
                // (read only in `stats` builds).
                #[cfg(feature = "stats")]
                let mut rejected_base_estimate: Option<usize> = None;

                if max > 0 {
                    let sep_t = if b.sep_before.is_empty() {
                        0
                    } else {
                        self.count_small_cached_at(&b.sep_before, CountSite::MergeSeparator)
                    };

                    let prefix_t = if carry_prefix {
                        self.count_small_cached_at(&p.prefix, CountSite::MergePrefix)
                    } else {
                        0
                    };

                    let body_est = p.tokens + prefix_t + sep_t + b_tokens;
                    // A separator that ends in a newline starts a fresh pre-token in the
                    // BPE tokenizers, so the parts add up and the per-join slack is unnecessary.
                    let join_cost = if b.sep_before.ends_with('\n') && self.backend.joins_are_additive() {
                        0
                    } else {
                        1
                    };
                    let joins = p.joins + join_cost + usize::from(carry_prefix);

                    if ctx_t + body_est + joins < max {
                        fast_merge = Some((body_est, None, joins));
                    } else {
                        #[cfg(feature = "stats")]
                        {
                            rejected_base_estimate = Some(ctx_t + body_est + joins);
                        }
                        record_event(CountSite::MergeEstimateRejected, candidate_len);
                        // Even if every join cost nothing and BPE saved up to two tokens per
                        // join, the candidate cannot fit: reject without tokenizing it.
                        if ctx_t + body_est > max + 2 * (joins + 1) {
                            record_event(CountSite::MergeEarlyReject, candidate_len);
                            break 'merge;
                        }
                    }
                } else {
                    // Unlimited-token mode skips the estimate but still needs exact counts
                    // to maintain the pending token metadata.
                    record_event(CountSite::MergeFastDisabled, candidate_len);
                }

                // Fast path: append directly to the existing pending String.
                if let Some((tokens, exact_total, joins)) = fast_merge {
                    if !carry_prefix {
                        record_event(
                            CountSite::MergeFastDirect,
                            b.sep_before.len().saturating_add(b.text.len()),
                        );
                        if let Some(p) = self.pending.as_mut() {
                            p.text
                                .reserve(b.sep_before.len().saturating_add(b.text.len()));
                            p.text.push_str(&b.sep_before);
                            p.text.push_str(&b.text);
                            p.tokens = tokens;
                            p.exact_total = exact_total;
                            p.joins = joins;
                            p.headers = common;

                            self.peak_pending_bytes = self
                                .peak_pending_bytes
                                .max(p.text.len().saturating_add(p.prefix.len()));
                        }

                        return Ok(());
                    }

                    // The estimate succeeded, but the deferred heading prefix must be
                    // materialized into a candidate before the pending state can be updated.
                    record_event(CountSite::MergeFastPrefix, candidate_len);
                }

                // Exact-count path: retain the original candidate construction.
                let mut candidate = String::with_capacity(candidate_len);

                if section_changed {
                    candidate.push_str(&p.prefix);
                }

                candidate.push_str(&p.text);
                candidate.push_str(&b.sep_before);
                candidate.push_str(&b.text);

                let mut merged = fast_merge;

                if merged.is_none() {
                    record_event(CountSite::MergeExactFallback, candidate_len);
                    let candidate_tokens =
                        self.count_delivered_at(&common, &candidate, CountSite::MergeCandidate);

                    // Compare the pre-margin estimate with the exact delivered-token count.
                    // These diagnostics run only in stats builds and do not affect decisions.
                    #[cfg(feature = "stats")]
                    if let Some(base_estimate) = rejected_base_estimate {
                        let delta_site = if base_estimate < candidate_tokens {
                            match candidate_tokens - base_estimate {
                                1 => CountSite::MergeEstimateUnder1,
                                _ => CountSite::MergeEstimateUnder2Plus,
                            }
                        } else {
                            match base_estimate - candidate_tokens {
                                0 => CountSite::MergeEstimateEqual,
                                1 => CountSite::MergeEstimateOver1,
                                2 => CountSite::MergeEstimateOver2,
                                3 => CountSite::MergeEstimateOver3,
                                _ => CountSite::MergeEstimateOver4Plus,
                            }
                        };
                        record_event(delta_site, candidate_len);

                        // Retrospective simulation: would +1 / +0 have passed the
                        // estimate gate, and if so, would that have accepted an
                        // actually-over-budget candidate?
                        if base_estimate.saturating_add(1) <= max {
                            record_event(CountSite::MergeMargin1WouldAccept, candidate_len);
                            if candidate_tokens > max {
                                record_event(CountSite::MergeMargin1Unsafe, candidate_len);
                            }
                        }
                        if base_estimate <= max {
                            record_event(CountSite::MergeMargin0WouldAccept, candidate_len);
                            if candidate_tokens > max {
                                record_event(CountSite::MergeMargin0Unsafe, candidate_len);
                            }
                        }
                    }

                    if max == 0 || candidate_tokens <= max {
                        record_event(CountSite::MergeExactAccept, candidate_len);
                        merged = Some((
                            candidate_tokens.saturating_sub(ctx_t),
                            Some(candidate_tokens),
                            0,
                        ));
                    } else {
                        record_event(CountSite::MergeExactReject, candidate_len);
                    }
                }

                if let Some((tokens, exact_total, joins)) = merged {
                    if let Some(p) = self.pending.as_mut() {
                        if section_changed && !p.prefix.is_empty() {
                            p.start_byte = p.prefix_start_byte;
                            p.start_char = p.prefix_start_char;
                            p.prefix.clear();
                        }

                        p.text = candidate;
                        p.tokens = tokens;
                        p.exact_total = exact_total;
                        p.joins = joins;
                        p.headers = common;

                        self.peak_pending_bytes = self
                            .peak_pending_bytes
                            .max(p.text.len().saturating_add(p.prefix.len()));
                    }

                    return Ok(());
                }
              }
            }
        }

        // Keep a multi-paragraph list item together: move its already-pending part into a
        // fresh pending chunk and let this continuation block merge into it.
        let b = match self.keep_list_item_together(b, out)? {
            Some(b) => b,
            None => return Ok(()),
        };

        // The previous pending chunk did not fit with this block (or no pending chunk exists).
        self.flush_pending(out)?;
        let budget = self.budget_for(&b.headers);
        if b_tokens > budget {
            self.split_block(&b, budget, b_tokens, out)?;
            return Ok(());
        }
        let trimmed_prefix = b.sep_before.trim_start_matches(char::is_whitespace);
        let has_heading_prefix = trimmed_prefix.starts_with('#');
        let (prefix, prefix_start_byte, prefix_start_char) = if has_heading_prefix {
            let leading_bytes = b.sep_before.len() - trimmed_prefix.len();
            let leading_chars = char_len(&b.sep_before[..leading_bytes]);
            (
                trimmed_prefix.to_string(),
                b.start_byte
                    .saturating_sub(b.sep_before.len())
                    .saturating_add(leading_bytes),
                b.start_char
                    .saturating_sub(char_len(&b.sep_before))
                    .saturating_add(leading_chars),
            )
        } else {
            (String::new(), b.start_byte, b.start_char)
        };
        self.pending = Some(Pending {
            text: b.text,
            start_byte: b.start_byte,
            start_char: b.start_char,
            tokens: b_tokens,
            exact_total: if b.headers.is_empty() {
                Some(b_tokens)
            } else {
                None
            },
            joins: 0,
            headers: b.headers,
            prefix,
            prefix_start_byte,
            prefix_start_char,
        });
        if let Some(pending) = &self.pending {
            self.peak_pending_bytes = self
                .peak_pending_bytes
                .max(pending.text.len().saturating_add(pending.prefix.len()));
        }
        Ok(())
    }

    fn split_block(
        &self,
        b: &Block,
        budget: usize,
        b_tokens: usize,
        out: &mut Vec<Chunk>,
    ) -> Result<(), ChipError> {
        // `count_block_text` returns an exact count only when it tokenized the whole block;
        // above BLOCK_PROBE_MIN_BYTES it may return a prefix lower bound instead.
        let known_total = (b.text.len() <= BLOCK_PROBE_MIN_BYTES).then_some(b_tokens);
        let origin = Origin {
            byte: b.start_byte,
            chr: b.start_char,
        };
        let mut pieces = Vec::new();
        split_by_tokens_with_lists_known(
            &b.text,
            origin,
            budget,
            self.cfg.overlap_tokens,
            self.cfg.snap_words,
            &self.backend,
            &b.headers,
            list_cuts(b),
            known_total,
            &mut pieces,
        );
        for piece in pieces {
            let known_total = if self.cfg.snap_words && piece.headers.is_empty() {
                // With snapping enabled, split_engine only emits chunks after an exact count
                // against the delivered body slice. The strict gate can reuse that count.
                Some(piece.token_count)
            } else {
                // Context changes the tokenization of the delivered string; verify it whole.
                None
            };
            self.finalize_with_count(piece, out, known_total)?;
        }
        Ok(())
    }
}

/// List-item start offsets inside a block, used to prefer item boundaries when an oversized
/// block has to be split. Fenced code blocks yield none.
fn list_cuts(b: &Block) -> ListCuts {
    let mut cuts = ListCuts::default();
    if b.is_code {
        return cuts;
    }
    for item in crate::markdown::list_parser::parse_list_items(&b.text) {
        if item.start_byte == 0 {
            continue;
        }
        if item.depth == 0 {
            cuts.top.push(item.start_byte);
        } else {
            cuts.nested.push(item.start_byte);
        }
    }
    cuts
}

// ============================================================================
// Intra-document parallelism
// ============================================================================
//
// The chunker is a deterministic left-to-right fold over the parser's blocks, and the ONLY
// piece of state that influences its output is `SmartChunker::pending` (the other fields are
// memoization caches keyed by header identity, or statistics). That makes the following
// scheme exact rather than approximate:
//
//   1. Parse the whole (normalized) document into blocks. The parser never depends on the
//      chunker, so this yields exactly the blocks the interleaved sequential loop would see.
//   2. Cut the block list into contiguous segments and run each one on its own thread,
//      *speculatively* starting from an empty `pending`. Segment 0 really does start from an
//      empty state, so its result is final.
//   3. Stitch sequentially. For segment k the true entry state is known (the end state of
//      segment k-1). Re-run the first few blocks of segment k from that true state, and after
//      every block compare the true `pending` with the speculative run's snapshot after the
//      same block. Both runs are deterministic functions of (state, remaining blocks), so
//      once the states are *equal* the rest of the speculative output IS the true output and
//      is spliced in. Greedy packing is self-synchronizing in practice, so this happens after
//      a handful of blocks; if it does not (or snapshots run out) the true run simply
//      continues through the whole segment. The result is therefore always identical to the
//      sequential result; only the amount of parallel speed-up depends on resynchronization.
//
// Errors follow the same rule: the first error in document order wins, a speculative error is
// only trusted after a successful resync, and a parser error is reported only if no chunking
// error occurs in the blocks that precede it (exactly the sequential interleaving).
pub(crate) mod parallel {
    use std::sync::Arc;

    use rayon::prelude::*;

    use super::{Pending, SmartChunker};
    use crate::{
        backend::Backend,
        block::Block,
        chunk_types::{Chunk, ChunkConfig},
        markdown::BlockParser,
        ChipError,
    };

    /// Normalized documents smaller than this always take the sequential path.
    pub(crate) const DEFAULT_MIN_BYTES: usize = 2 * 1024 * 1024;
    /// Inspect only a bounded prefix before materializing the full block vector. Dense tiny
    /// paragraphs are better handled by the streaming fold, which does not retain every Block
    /// allocation at once.
    const DENSE_PREFLIGHT_SAMPLE_BYTES: usize = 64 * 1024;
    const DENSE_PREFLIGHT_MIN_BLANK_LINES: usize = 128;
    const DENSE_PREFLIGHT_MAX_AVG_BYTES_PER_PARAGRAPH: usize = 32;
    /// Smallest segment worth a task of its own.
    pub(crate) const DEFAULT_MIN_SEGMENT_BYTES: usize = 256 * 1024;
    /// Tiny-block documents do not amortize speculative segment runs and stitching.
    /// Require at least this many source bytes per parsed block before using the
    /// context-aware parallel fold. This is independent of language and block kind.
    const MIN_AVG_BLOCK_BYTES_FOR_PARALLEL: usize = 32;
    /// Fewer pieces than this are not worth fanning out in the no-context path.
    const MIN_PIECES_FOR_PARALLEL_FINALIZE: usize = 64;
    /// Resync bookkeeping limits per speculative segment (memory bound).
    const MAX_SNAPSHOTS_PER_SEGMENT: usize = 1024;
    const MAX_SNAPSHOT_BYTES_PER_SEGMENT: usize = 2 * 1024 * 1024;

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ParallelOpts {
        /// Use the parallel path only for normalized documents of at least this many bytes.
        pub(crate) min_bytes: usize,
        /// Lower bound for the bytes handled by one segment.
        pub(crate) min_segment_bytes: usize,
        /// 0 = automatic (2 x rayon threads, and never parallel on a single thread);
        /// otherwise exactly this many segments at most, regardless of the thread count.
        pub(crate) max_segments: usize,
    }

    impl Default for ParallelOpts {
        fn default() -> Self {
            ParallelOpts {
                min_bytes: DEFAULT_MIN_BYTES,
                min_segment_bytes: DEFAULT_MIN_SEGMENT_BYTES,
                max_segments: 0,
            }
        }
    }

    impl ParallelOpts {
        /// Never take the parallel path (reference behaviour for tests).
        #[cfg(test)]
        pub(crate) fn never() -> Self {
            ParallelOpts {
                min_bytes: usize::MAX,
                min_segment_bytes: DEFAULT_MIN_SEGMENT_BYTES,
                max_segments: 0,
            }
        }

        /// Force the parallel path with up to `segments` segments (tests).
        #[cfg(test)]
        pub(crate) fn forced(segments: usize) -> Self {
            ParallelOpts {
                min_bytes: 0,
                min_segment_bytes: 1,
                max_segments: segments,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    #[allow(dead_code)]
    pub(crate) struct ParallelStats {
        pub(crate) segments: usize,
        /// Seams where the true state re-synchronized with the speculative run.
        pub(crate) resynced: usize,
        /// Blocks that had to be processed a second time by the sequential stitch.
        pub(crate) restitched_blocks: usize,
    }

    /// Cheap, allocation-free admission check for documents dominated by very short
    /// paragraphs. It intentionally samples only the prefix: this is a scheduling heuristic,
    /// not a parser, and a false positive only selects the exact sequential implementation.
    pub(crate) fn has_dense_tiny_paragraph_prefix(normalized: &str) -> bool {
        let mut sample_end = normalized.len().min(DENSE_PREFLIGHT_SAMPLE_BYTES);
        while !normalized.is_char_boundary(sample_end) {
            sample_end -= 1;
        }
        let sample = &normalized[..sample_end];
        if sample.is_empty() {
            return false;
        }

        let blank_lines = sample
            .split_inclusive('\n')
            .filter(|line| {
                let line = line.strip_suffix('\n').unwrap_or(line);
                let line = line.strip_suffix('\r').unwrap_or(line);
                line.trim().is_empty()
            })
            .count();

        blank_lines >= DENSE_PREFLIGHT_MIN_BLANK_LINES
            && sample.len() / blank_lines.max(1) < DENSE_PREFLIGHT_MAX_AVG_BYTES_PER_PARAGRAPH
    }

    /// Number of segments to use for the block-fold path, or `None` for "stay sequential".
    pub(crate) fn plan(normalized_len: usize, opts: &ParallelOpts) -> Option<usize> {
        if normalized_len < opts.min_bytes {
            return None;
        }
        let cap = if opts.max_segments > 0 {
            opts.max_segments
        } else {
            let threads = rayon::current_num_threads();
            if threads < 2 {
                return None;
            }
            threads.saturating_mul(2)
        };
        let by_size = normalized_len / opts.min_segment_bytes.max(1);
        let n = cap.min(by_size);
        if n < 2 {
            None
        } else {
            Some(n)
        }
    }

    /// Whether the no-context path should finalize its pieces in parallel.
    pub(crate) fn plan_finalize(normalized_len: usize, pieces: usize, opts: &ParallelOpts) -> bool {
        if normalized_len < opts.min_bytes {
            return false;
        }
        if opts.max_segments > 0 {
            return pieces >= 2;
        }
        rayon::current_num_threads() >= 2 && pieces >= MIN_PIECES_FOR_PARALLEL_FINALIZE
    }

    /// Complete field-by-field comparison of two chunker states. Destructuring without `..`
    /// makes adding a field to `Pending` a compile error here, so the comparison cannot
    /// silently become incomplete. Headers are compared by value (identity only matters to
    /// the memoization caches).
    fn same_state(a: &Option<Pending>, b: &Option<Pending>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                let Pending {
                    text: x_text,
                    start_byte: x_start_byte,
                    start_char: x_start_char,
                    tokens: x_tokens,
                    exact_total: x_exact_total,
                    joins: x_joins,
                    headers: x_headers,
                    prefix: x_prefix,
                    prefix_start_byte: x_prefix_start_byte,
                    prefix_start_char: x_prefix_start_char,
                } = x;
                let Pending {
                    text: y_text,
                    start_byte: y_start_byte,
                    start_char: y_start_char,
                    tokens: y_tokens,
                    exact_total: y_exact_total,
                    joins: y_joins,
                    headers: y_headers,
                    prefix: y_prefix,
                    prefix_start_byte: y_prefix_start_byte,
                    prefix_start_char: y_prefix_start_char,
                } = y;
                x_start_byte == y_start_byte
                    && x_start_char == y_start_char
                    && x_tokens == y_tokens
                    && x_exact_total == y_exact_total
                    && x_joins == y_joins
                    && x_prefix_start_byte == y_prefix_start_byte
                    && x_prefix_start_char == y_prefix_start_char
                    && x_headers.as_slice() == y_headers.as_slice()
                    && x_text == y_text
                    && x_prefix == y_prefix
            }
            _ => false,
        }
    }

    fn state_bytes(p: &Option<Pending>) -> usize {
        match p {
            Some(p) => p.text.len() + p.prefix.len() + 64,
            None => 64,
        }
    }

    /// Result of running one segment from an empty state.
    struct SegRun {
        /// Chunks emitted while handling the segment's blocks (the end state is NOT flushed).
        chunks: Vec<Chunk>,
        /// `snaps[i]` = (chunks emitted so far, pending state) right after block `i`.
        /// Only a prefix of the segment is recorded (see the MAX_* limits).
        snaps: Vec<(usize, Option<Pending>)>,
        /// State after the last block; meaningful only when `err` is `None`.
        end_state: Option<Pending>,
        /// Error raised by the block with index `snaps.len() <= e`, if any.
        err: Option<ChipError>,
    }

    fn run_segment(
        blocks: &[Block],
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
        record_snapshots: bool,
    ) -> SegRun {
        let mut ch = SmartChunker::new(cfg.clone(), Arc::clone(backend));
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut snaps: Vec<(usize, Option<Pending>)> = Vec::new();
        let mut snap_bytes = 0usize;
        let mut recording = record_snapshots;
        for b in blocks {
            if let Err(e) = ch.handle_block(b.clone(), &mut chunks) {
                return SegRun {
                    chunks,
                    snaps,
                    end_state: None,
                    err: Some(e),
                };
            }
            if recording {
                let sz = state_bytes(&ch.pending);
                if snaps.len() >= MAX_SNAPSHOTS_PER_SEGMENT
                    || snap_bytes.saturating_add(sz) > MAX_SNAPSHOT_BYTES_PER_SEGMENT
                {
                    recording = false;
                } else {
                    snap_bytes += sz;
                    snaps.push((chunks.len(), ch.pending.clone()));
                }
            }
        }
        SegRun {
            chunks,
            snaps,
            end_state: ch.pending.take(),
            err: None,
        }
    }

    /// Split blocks into at most `nseg` contiguous, roughly byte-balanced segments.
    fn split_segments(blocks: Vec<Block>, nseg: usize) -> Vec<Vec<Block>> {
        let total: usize = blocks
            .iter()
            .map(|b| b.text.len() + b.sep_before.len())
            .sum();
        let target = (total / nseg.max(1)).max(1);
        let mut segs: Vec<Vec<Block>> = Vec::with_capacity(nseg);
        let mut cur: Vec<Block> = Vec::new();
        let mut acc = 0usize;
        for b in blocks {
            acc += b.text.len() + b.sep_before.len();
            cur.push(b);
            if acc >= target && segs.len() + 1 < nseg {
                segs.push(std::mem::take(&mut cur));
                acc = 0;
            }
        }
        if !cur.is_empty() {
            segs.push(cur);
        }
        segs
    }

    /// Parallel equivalent of the `include_context` branch of the sequential driver
    /// (`push_line` for every line, then `finish`). `normalized` must be non-empty.
    pub(crate) fn chunk_context_parallel(
        normalized: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
        nseg: usize,
    ) -> Result<(Vec<Chunk>, ParallelStats), ChipError> {
        // Check density before collecting Blocks: for hundreds of thousands of tiny
        // paragraphs, parsing the whole document into a Vec<Block> would itself dominate the
        // runtime even though the later fold falls back to sequential processing. Such inputs
        // take the ordinary streaming path, which retains only the current parser/chunker
        // state.
        if has_dense_tiny_paragraph_prefix(normalized) {
            let mut ch = SmartChunker::new(cfg.clone(), Arc::clone(backend));
            let mut out: Vec<Chunk> = Vec::new();
            for line in normalized.split_inclusive('\n') {
                ch.push_line(line, &mut out)?;
            }
            ch.finish(&mut out)?;
            return Ok((
                out,
                ParallelStats {
                    segments: 1,
                    ..ParallelStats::default()
                },
            ));
        }

        // ---- 1. parse (sequential, mirrors SmartChunker::push_line / finish) ----------------
        let mut parser = BlockParser::with_headings(cfg.include_context);
        let mut blocks: Vec<Block> = Vec::new();
        let mut parse_err: Option<ChipError> = None;
        for line in normalized.split_inclusive('\n') {
            let mark = blocks.len();
            if let Err(e) = parser.push_line(line, &mut blocks) {
                // The sequential driver drops blocks produced by the failing call unhandled.
                blocks.truncate(mark);
                parse_err = Some(e);
                break;
            }
        }
        if parse_err.is_none() {
            parser.finish(&mut blocks);
        }

        // ---- 2. admission check and speculative parallel runs -------------------------------
        // The block-fold strategy speculatively processes each segment and may need to
        // reprocess its prefix while stitching. On documents made of hundreds of thousands
        // of tiny paragraphs, that overhead can dominate the useful work and make scaling
        // superlinear. Keep those dense-block inputs on the ordinary sequential fold.
        //
        // Measure source bytes represented by parsed blocks (body + preceding separator),
        // rather than language-specific text features. The threshold is deliberately a
        // coarse admission heuristic only; it never changes parsing or chunk boundaries.
        let represented_bytes = blocks.iter().fold(0usize, |sum, block| {
            sum.saturating_add(block.text.len())
                .saturating_add(block.sep_before.len())
        });
        let min_parallel_bytes = blocks
            .len()
            .saturating_mul(MIN_AVG_BLOCK_BYTES_FOR_PARALLEL);
        if !blocks.is_empty() && represented_bytes < min_parallel_bytes {
            let mut ch = SmartChunker::new(cfg.clone(), Arc::clone(backend));
            let mut out: Vec<Chunk> = Vec::new();
            for block in blocks {
                ch.handle_block(block, &mut out)?;
            }
            // Match the streaming sequential driver's error order: process every block
            // emitted before the parser error, but do not flush pending state afterwards.
            if let Some(e) = parse_err {
                return Err(e);
            }
            ch.flush_pending(&mut out)?;
            return Ok((
                out,
                ParallelStats {
                    segments: 1,
                    ..ParallelStats::default()
                },
            ));
        }

        let segs = split_segments(blocks, nseg);
        let runs: Vec<SegRun> = segs
            .par_iter()
            .enumerate()
            .map(|(i, s)| run_segment(s, cfg, backend, i > 0))
            .collect();

        // ---- 3. sequential stitch ------------------------------------------------------------
        let mut stats = ParallelStats {
            segments: segs.len(),
            ..ParallelStats::default()
        };
        let mut out: Vec<Chunk> = Vec::new();
        let mut state: Option<Pending> = None;

        for (k, mut run) in runs.into_iter().enumerate() {
            if k == 0 {
                // Segment 0 starts from the true initial state: its result is final.
                if let Some(e) = run.err {
                    return Err(e);
                }
                out = std::mem::take(&mut run.chunks);
                state = run.end_state.take();
                continue;
            }

            let seg_blocks = &segs[k];
            let mut ch = SmartChunker::new(cfg.clone(), Arc::clone(backend));
            ch.pending = state.take();
            let mut true_out: Vec<Chunk> = Vec::new();
            let mut consumed = 0usize;
            let mut resume_at: Option<usize> = None;
            loop {
                // Is the true state after `consumed` blocks equal to the speculative state
                // after the same number of blocks?
                let spec_chunks: Option<usize> = if consumed == 0 {
                    if ch.pending.is_none() {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    match run.snaps.get(consumed - 1) {
                        Some((cnt, st)) if same_state(&ch.pending, st) => Some(*cnt),
                        _ => None,
                    }
                };
                if spec_chunks.is_some() {
                    resume_at = spec_chunks;
                    break;
                }
                if consumed == seg_blocks.len() {
                    break;
                }
                // Not synchronized yet: process the next block from the true state.
                ch.handle_block(seg_blocks[consumed].clone(), &mut true_out)?;
                consumed += 1;
            }
            stats.restitched_blocks += consumed;
            out.append(&mut true_out);
            match resume_at {
                Some(cnt) => {
                    stats.resynced += 1;
                    // From here on the true run is identical to the speculative run, including
                    // any error it hit (that block index is >= `consumed`).
                    if let Some(e) = run.err.take() {
                        return Err(e);
                    }
                    out.extend(run.chunks.drain(cnt..));
                    state = run.end_state.take();
                }
                None => {
                    // The true run covered the whole segment on its own.
                    state = ch.pending.take();
                }
            }
        }

        // ---- 4. tail --------------------------------------------------------------------------
        if let Some(e) = parse_err {
            // Sequential semantics: `finish()` (and its final flush) is never reached.
            return Err(e);
        }
        let mut ch = SmartChunker::new(cfg.clone(), Arc::clone(backend));
        ch.pending = state;
        ch.flush_pending(&mut out)?;
        Ok((out, stats))
    }

    /// Order-preserving parallel version of `for piece in pieces { gate.finalize(piece, out)? }`.
    /// `finalize` is a pure function of (piece, config, backend), so the pieces are independent.
    /// The FIRST error in piece order is returned, exactly like the sequential loop (rayon's own
    /// `collect::<Result<_, _>>()` would pick an arbitrary error).
    pub(crate) fn finalize_pieces(
        pieces: Vec<Chunk>,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
        out: &mut Vec<Chunk>,
    ) -> Result<(), ChipError> {
        let results: Vec<Result<Vec<Chunk>, ChipError>> = pieces
            .into_par_iter()
            .with_min_len(8)
            .map_init(
                || SmartChunker::new(cfg.clone(), Arc::clone(backend)),
                |gate, piece| {
                    let mut v: Vec<Chunk> = Vec::with_capacity(1);
                    gate.finalize(piece, &mut v).map(|()| v)
                },
            )
            .collect();
        for r in results {
            out.extend(r?);
        }
        Ok(())
    }
}

// Whole-document orchestration shares the chunker lifecycle and finalization rules.

mod whole_document {
    use std::sync::Arc;

    use super::parallel::{self, ParallelOpts};
    use super::SmartChunker;
    use crate::{
        backend::Backend,
        chunk_types::{Chunk, ChunkConfig},
        normalization::{NormBase, NormMap, OrigSpan},
        split_engine::{empty_headers, split_by_tokens, Origin},
        ChipError,
    };

    /// Chunk text while retaining a mapping from normalized offsets back to the original input.
    /// Very large documents are processed on several threads; the result is identical to the
    /// sequential one (see the `parallel` module for the argument).
    pub(crate) fn smart_chunk_mapped_profile(
        text: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
    ) -> Result<(Vec<Chunk>, Vec<OrigSpan>, usize), ChipError> {
        smart_chunk_mapped_profile_with(text, cfg, backend, &ParallelOpts::default())
    }

    pub(crate) fn smart_chunk_mapped_profile_with(
        text: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
        opts: &ParallelOpts,
    ) -> Result<(Vec<Chunk>, Vec<OrigSpan>, usize), ChipError> {
        let mut map = NormMap::default();
        let normalized = cfg
            .normalization
            .apply_mapped(text, NormBase::default(), &mut map);
        let mut out = Vec::new();
        if normalized.is_empty() {
            return Ok((out, Vec::new(), map.segs.len()));
        }

        // Two different algorithms, selected by `include_context` (documented on ChunkConfig):
        //  * context mode: Markdown block parser + block packing (parallel for large inputs);
        //  * flat mode: plain token windows over the whole normalized text, no Markdown parsing.
        if cfg.include_context {
            if let Some(nseg) = parallel::plan(normalized.len(), opts) {
                out = parallel::chunk_context_parallel(&normalized, cfg, backend, nseg)?.0;
            } else {
                let mut chunker = SmartChunker::new(cfg.clone(), Arc::clone(backend));
                for line in normalized.split_inclusive('\n') {
                    chunker.push_line(line, &mut out)?;
                }
                chunker.finish(&mut out)?;
            }
        } else {
            // Flat mode: the Markdown structure is deliberately ignored here, so fences, lists
            // and tables may be cut and no headers are attached (`empty_headers()` below).
            let mut pieces = Vec::new();
            split_by_tokens(
                &normalized,
                Origin::default(),
                cfg.max_tokens,
                cfg.overlap_tokens,
                cfg.snap_words,
                backend,
                &empty_headers(),
                &mut pieces,
            );
            if parallel::plan_finalize(normalized.len(), pieces.len(), opts) {
                parallel::finalize_pieces(pieces, cfg, backend, &mut out)?;
            } else {
                let gate = SmartChunker::new(cfg.clone(), Arc::clone(backend));
                for piece in pieces {
                    gate.finalize(piece, &mut out)?;
                }
            }
        }
        let spans = out.iter().map(|c| map.span(c)).collect();
        Ok((out, spans, map.segs.len()))
    }

    /// Chunk `text` and also return, for every chunk, its span in the *original* text
    /// (before normalization). See [`ChunkConfig`] for the two chunking modes and the budget
    /// guarantee. Errors are returned, never swallowed.
    pub fn smart_chunk_mapped(
        text: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
    ) -> Result<(Vec<Chunk>, Vec<OrigSpan>), ChipError> {
        let (chunks, spans, _) = smart_chunk_mapped_profile(text, cfg, backend)?;
        Ok((chunks, spans))
    }

    /// Chunk `text`; offsets in the result refer to the normalized text. See
    /// [`smart_chunk_mapped`] for original-text spans and [`ChunkConfig`] for semantics.
    pub fn smart_chunk(
        text: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
    ) -> Result<Vec<Chunk>, ChipError> {
        Ok(smart_chunk_mapped(text, cfg, backend)?.0)
    }
}

pub(crate) use whole_document::smart_chunk_mapped_profile;
#[cfg(test)]
pub(crate) use whole_document::smart_chunk_mapped_profile_with;
pub use whole_document::{smart_chunk, smart_chunk_mapped};
