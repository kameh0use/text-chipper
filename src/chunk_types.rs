use std::sync::Arc;

use super::{
    backend::Backend, block::Headers, normalization::Normalization, split_engine::empty_headers,
    ChipError,
};

/// Chunking options.
///
/// # `include_context` selects the chunking mode
///
/// * `true` (default): the Markdown block parser runs. Headings are tracked and attached to
///   every chunk as `headers`; the compact label `[ctx:A > B] ` produced by [`context_prefix`]
///   counts against `max_tokens` (at most about a quarter of the budget is reserved for it).
///   Fenced code blocks and list items are kept intact when they fit. Other constructs
///   (tables, block quotes, HTML blocks, indented code) are ordinary blank-line-separated text
///   blocks with no special handling.
/// * `false` ("flat mode"): [`crate::smart_chunk`] and [`crate::smart_chunk_mapped`] do
///   **not** parse Markdown. The normalized text is cut into token windows (with
///   `overlap_tokens` honoured) that prefer paragraph, line and word boundaries. Fences,
///   lists and tables are not protected, headings are ordinary text, `headers` is empty and
///   `merge_blocks` has no effect.
///   [`crate::FileChunker`] uses the same whole-document path, so file results equal
///   `smart_chunk` on the file's text at any size. Because that needs the whole file in
///   memory, flat-mode files larger than [`crate::FLAT_FILE_MAX_BYTES`] (32 MiB) are
///   rejected with an error; stream big files with `include_context = true` instead.
///
/// # `strict` selects the budget guarantee
///
/// * `true` (default): every chunk is re-tokenized exactly as delivered (context prefix
///   plus body) and must fit `max_tokens`; `token_count` is exact. If the context prefix
///   alone prevents the first grapheme from fitting, the headers of that chunk are dropped.
///   If a single grapheme cluster still exceeds the budget, chunking fails with
///   [`ChipError::ChunkTooLarge`].
/// * `false`: the final exact check is skipped. This is faster, but there is no guarantee
///   that every chunk fits and `token_count` may be an estimate for chunks without headers.
///   An unsplittable grapheme cluster larger than the budget becomes its own oversized chunk.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Token budget per chunk (context prefix included). `0` disables the limit.
    pub max_tokens: usize,
    /// Tokens repeated at the start of the next chunk when text has to be split. Must be
    /// smaller than `max_tokens` (see [`ChunkConfig::validate`]).
    pub overlap_tokens: usize,
    /// Unicode normalization applied before chunking; offsets are mapped back to the input.
    pub normalization: Normalization,
    /// Prefer cutting at word/sentence boundaries instead of at an arbitrary token when a
    /// hard split is unavoidable.
    pub snap_words: bool,
    /// Chunking mode: structure-aware with header context (`true`) or flat (`false`).
    /// See the type-level documentation.
    pub include_context: bool,
    /// Pack adjacent Markdown blocks into one chunk while they fit. When `false`, every
    /// block starts a new chunk. Only relevant when `include_context` is `true`.
    pub merge_blocks: bool,
    /// Budget guarantee: exact re-count of every delivered chunk (`true`) or best effort
    /// (`false`). See the type-level documentation.
    pub strict: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        ChunkConfig {
            max_tokens: 512,
            overlap_tokens: 50,
            normalization: Normalization::Nfc,
            snap_words: true,
            include_context: true,
            merge_blocks: true,
            strict: true,
        }
    }
}

impl ChunkConfig {
    /// Check option combinations that can never work (currently: overlap must be smaller
    /// than the budget).
    pub fn validate(&self) -> Result<(), ChipError> {
        if self.max_tokens > 0 && self.overlap_tokens >= self.max_tokens {
            return Err(ChipError::InvalidConfig(format!(
                "overlap ({}) must be smaller than max_length ({})",
                self.overlap_tokens, self.max_tokens
            )));
        }
        Ok(())
    }
}

/// The text prepended to a chunk's body to form its contextualized text, e.g.
/// `[ctx:Guide > Install] `. The prefix is counted against the token budget.
pub fn context_prefix(headers: &[String]) -> String {
    // Compact, tokenizer-friendly label: a verbose wrapper would consume a disproportionate
    // share of tiny budgets before any heading text was included.
    format!("[ctx:{}] ", headers.join(" > "))
}

/// Keep section metadata useful without allowing long headings to consume nearly the
/// entire chunk budget. Prefer the deepest section names; if even the deepest name is
/// too long, retain its longest character-safe prefix that fits the context allowance.
pub(crate) fn fit_context_headers(
    headers: &Headers,
    max_context_tokens: usize,
    backend: &Backend,
) -> Headers {
    if headers.is_empty() || max_context_tokens == 0 {
        return empty_headers();
    }
    let all_tokens = backend.count(&context_prefix(headers));
    if all_tokens <= max_context_tokens {
        return Arc::clone(headers);
    }

    // Ancestors are progressively dropped only when necessary. This preserves the most
    // specific section label, which is usually the most useful retrieval context. Retain
    // the deepest-only count: the truncation path below needs that exact same representation.
    let mut deepest_tokens = if headers.len() == 1 {
        Some(all_tokens)
    } else {
        None
    };
    for first in 1..headers.len() {
        let candidate = Arc::new(headers[first..].to_vec());
        let tokens = backend.count(&context_prefix(&candidate));
        if first == headers.len() - 1 {
            deepest_tokens = Some(tokens);
        }
        if tokens <= max_context_tokens {
            return candidate;
        }
    }

    let deepest = headers.last().map(String::as_str).unwrap_or("");
    if deepest.is_empty() {
        return empty_headers();
    }

    // Do not tokenize every character prefix: that makes a long heading quadratic.
    // Estimate a fitting prefix from the observed token/character ratio, then shrink
    // geometrically if BPE boundary effects still leave it over budget. Every emitted
    // candidate is authoritatively counted, while total work stays linear in heading size.
    let full_tokens = deepest_tokens.unwrap_or(all_tokens);
    let mut keep_chars = if full_tokens <= max_context_tokens {
        deepest.chars().count()
    } else {
        ((deepest.chars().count() as u128 * max_context_tokens as u128)
            / full_tokens.max(1) as u128) as usize
    };
    keep_chars = keep_chars.max(1).min(deepest.chars().count());

    loop {
        let end = deepest
            .char_indices()
            .nth(keep_chars)
            .map(|(i, _)| i)
            .unwrap_or(deepest.len());
        if end == 0 {
            return empty_headers();
        }
        let candidate = Arc::new(vec![deepest[..end].to_string()]);
        if backend.count(&context_prefix(&candidate)) <= max_context_tokens {
            return candidate;
        }
        if keep_chars <= 1 {
            return empty_headers();
        }
        keep_chars = (keep_chars / 2).max(1);
    }
}

/// A chunk of the normalized input.
///
/// Offsets (`start_*`, `end_*`) refer to the text *after* normalization; use
/// [`crate::smart_chunk_mapped`] to also obtain spans in the original input.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub token_count: usize,
    pub headers: Arc<Vec<String>>,
}

impl Chunk {
    pub fn context(&self) -> String {
        self.headers.join(" > ")
    }

    pub fn contextualized_text(&self) -> String {
        if self.headers.is_empty() {
            self.text.clone()
        } else {
            format!("{}{}", context_prefix(&self.headers), self.text)
        }
    }
}
