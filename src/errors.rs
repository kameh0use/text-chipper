use std::fmt;

/// Every failure is reported through this type; the Python bindings raise it as an exception
/// (`ValueError`, `RuntimeError` or `OSError`) and never return partial or empty results.
#[derive(Debug)]
pub enum ChipError {
    /// An option combination that can never work (e.g. overlap >= max_length).
    InvalidConfig(String),
    /// Unknown tokenizer encoding name.
    UnknownEncoding(String),
    /// The tokenizer backend failed to load or run.
    Backend(String),
    /// File access or invalid UTF-8 (when `errors="strict"`).
    Io(std::io::Error),
    /// Strict mode only: a single grapheme cluster needs more tokens than the budget allows,
    /// so no valid chunk exists at `start_byte`.
    ChunkTooLarge {
        required_tokens: usize,
        max_tokens: usize,
        start_byte: usize,
    },
}

impl fmt::Display for ChipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChipError::InvalidConfig(m) => write!(f, "invalid configuration: {m}"),
            ChipError::UnknownEncoding(n) => write!(
                f,
                "unknown encoding '{n}' (expected one of: cl100k_base, o200k_base, \
                 p50k_base, r50k_base, chars, words)"
            ),
            ChipError::Backend(m) => write!(f, "tokenizer backend error: {m}"),
            ChipError::Io(e) => write!(f, "I/O error: {e}"),
            ChipError::ChunkTooLarge { required_tokens, max_tokens, start_byte } => write!(f, "cannot split input without breaking a Unicode grapheme: it requires {required_tokens} tokens, exceeding max_tokens={max_tokens} (byte offset {start_byte})"),
        }
    }
}

impl std::error::Error for ChipError {}

impl From<std::io::Error> for ChipError {
    fn from(e: std::io::Error) -> Self {
        ChipError::Io(e)
    }
}
