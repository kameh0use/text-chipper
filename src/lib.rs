//! `text_chipper`: token-budgeted chunking for Markdown and plain text.
//!
//! The crate takes a `&str` and returns chunks that fit a token budget, never cut inside a
//! grapheme cluster, and map back to the original input even when Unicode normalization is
//! applied. The Python bindings are exposed through [`TextChipper`].
//!
//! # Scope
//!
//! * **Input**: Markdown and plain text only. There is no HTML, PDF or DOCX loader and no
//!   code AST (function/class boundaries). Convert such inputs to Markdown or text first.
//!   The file helpers ([`read_text_file`], [`FileChunker`]) read UTF-8 text files only.
//!
//! # Budget guarantee
//!
//! * `strict = true` (default): every chunk is re-tokenized exactly as it will be delivered
//!   (context prefix included) and `token_count` is exact. If a single grapheme cluster alone
//!   exceeds the budget, chunking fails with [`ChipError::ChunkTooLarge`].
//! * `strict = false`: that final exact check is skipped. This is faster, but there is no
//!   guarantee that every chunk fits, and `token_count` may be an estimate. An unsplittable
//!   grapheme cluster larger than the budget is emitted as its own chunk instead of an error.
//!
//! # Two chunking modes (`include_context`)
//!
//! * `include_context = true` (default): the Markdown block parser runs. Headings are tracked
//!   and attached to chunks as `headers`; fenced code blocks and list items are kept intact
//!   when they fit; adjacent blocks are packed together (`merge_blocks`). Blocks are the
//!   blank-line-separated pieces of text: tables, block quotes, HTML blocks and indented code
//!   get no special handling. [`BlockParser`] is a chunking aid, not a complete CommonMark
//!   parser.
//! * `include_context = false` ("flat mode"): [`smart_chunk`] and [`smart_chunk_mapped`] do
//!   **not** parse Markdown at all. The normalized text is cut into token windows that prefer
//!   paragraph, line and word boundaries; fences, lists and tables are not protected and
//!   `headers` is always empty. [`FileChunker`] gives the same chunks as `smart_chunk` on the
//!   file's text, at any size; flat-mode files above 32 MiB are rejected with an error
//!   (stream them with `include_context = true`). See [`ChunkConfig`].

use pyo3::prelude::*;

mod backend;
mod block;
mod chunk_types;
mod chunker;
mod errors;
mod file;
mod markdown;
mod normalization;
mod python;
mod split_engine;

pub use backend::{Backend, Tokenizer};
pub use block::BlockKind;
pub use block::{Block, Headers};
pub use chunk_types::{context_prefix, Chunk, ChunkConfig};
pub use chunker::SmartChunker;
pub use chunker::{smart_chunk, smart_chunk_mapped};
pub use errors::ChipError;
pub use file::{read_text_file, FileChunker, FLAT_FILE_MAX_BYTES};
pub use markdown::list_parser::{parse_list_items, ListItem, ListMarker};
pub use markdown::BlockParser;
pub use normalization::{NormBase, NormMap, Normalization, OrigSpan};
pub use python::{TextChipper, TextChunk, TextStreamIter};
pub use split_engine::{split_by_tokens, CharCursor, Origin};

// Optional so that Rust users who depend on this crate can keep their own global allocator.
// The Python wheel is built with the default features, which include it.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[pymodule]
fn text_chipper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}

// Test modules live in `src/tests/`. `#[path]` keeps each module at the crate root so that
// `use super::*` and `crate::...` paths inside them stay valid.
#[cfg(test)]
#[path = "tests/unit.rs"]
mod tests;
#[cfg(test)]
#[path = "tests/merge_margin.rs"]
mod merge_margin_tests;
#[cfg(test)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]
#[path = "tests/oracle_list.rs"]
mod oracle_tests;
#[cfg(test)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]
#[path = "tests/oracle_block.rs"]
mod oracle_block_tests;
#[cfg(test)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]
#[path = "tests/parallel.rs"]
mod parallel_tests;
#[cfg(test)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]
#[path = "tests/robustness.rs"]
mod robustness_tests;
#[cfg(all(test, feature = "diagnostics"))]
#[path = "tests/release_suite.rs"]
mod release_suite;
