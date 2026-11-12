//! Public block data model emitted by the Markdown block parser.
//!
//! Kept separate from parser state so chunking and file-streaming layers can
//! depend on the block representation without depending on parsing internals.

use std::sync::Arc;

/// Shared heading context attached to a parsed block.
pub type Headers = Arc<Vec<String>>;

/// A source-ordered Markdown block emitted by [`crate::BlockParser`].
#[derive(Debug, Clone)]
pub struct Block {
    pub text: String,
    pub start_byte: usize,
    pub start_char: usize,
    pub sep_before: String,
    pub headers: Headers,
    pub is_code: bool,
}

impl Block {
    /// Return the construct indicated by this block's first line.
    ///
    /// Classification is deliberately lightweight and does not imply full
    /// CommonMark parsing. ATX headings are normally consumed as section
    /// context by `BlockParser`, so `Heading` is mainly useful for standalone
    /// or externally constructed `Block` values.
    pub fn kind(&self) -> crate::block::BlockKind {
        crate::block::classify_block_start(&self.text, self.is_code)
    }

    /// Return the list items structurally recognized inside this block.
    /// Fenced code blocks are never scanned as Markdown lists.
    pub fn list_items(&self) -> Vec<crate::markdown::list_parser::ListItem> {
        if self.is_code {
            Vec::new()
        } else {
            crate::markdown::list_parser::parse_list_items(&self.text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(text: &str, is_code: bool) -> Block {
        Block {
            text: text.to_string(),
            start_byte: 0,
            start_char: 0,
            sep_before: String::new(),
            headers: Arc::new(Vec::new()),
            is_code,
        }
    }

    #[test]
    fn list_items_are_available_from_non_code_blocks_only() {
        let parsed = block("- one\n  - nested\n", false).list_items();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].depth, 1);
        assert!(block("- not markdown\n", true).list_items().is_empty());
    }
}

// Lightweight block classification belongs with the block data model.

// Lightweight classification of a block's opening line.
//
// This is intentionally not a complete CommonMark block parser. It identifies
// the leading construct recognized by the current chunking pipeline while
// leaving the original block text and public `Block` layout unchanged.

/// The Markdown construct indicated by a block's first line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockKind {
    Paragraph,
    Heading,
    FencedCode,
    BlockQuote,
    List,
    ThematicBreak,
}

/// Classify a block using only its first source line.
///
/// `is_code` is supplied by the parser and takes precedence because fenced
/// code blocks may begin with an info string rather than code content.
pub(crate) fn classify_block_start(text: &str, is_code: bool) -> BlockKind {
    if is_code {
        return BlockKind::FencedCode;
    }

    let first = text.lines().next().unwrap_or("");
    let line = first.trim_start_matches([' ', '\t']);
    // Tab stops count as in CommonMark, consistent with the list scanner.
    let indent = crate::markdown::scanners::indent_width(first.as_bytes());
    if indent > 3 {
        return BlockKind::Paragraph;
    }

    if crate::markdown::scanners::atx_heading(first).is_some() {
        return BlockKind::Heading;
    }
    if is_thematic_break(line) {
        return BlockKind::ThematicBreak;
    }

    let bytes = line.as_bytes();
    // CommonMark: the space after `>` is optional.
    if bytes.first() == Some(&b'>') {
        return BlockKind::BlockQuote;
    }
    if is_list_marker(bytes) {
        return BlockKind::List;
    }

    BlockKind::Paragraph
}

fn is_thematic_break(line: &str) -> bool {
    let bytes = line.as_bytes();
    let Some(&marker) = bytes.first() else {
        return false;
    };
    if !matches!(marker, b'-' | b'_' | b'*') {
        return false;
    }

    let mut count = 0usize;
    for &byte in bytes {
        if byte == marker {
            count += 1;
        } else if byte != b' ' && byte != b'\t' {
            return false;
        }
    }
    count >= 3
}

fn is_list_marker(bytes: &[u8]) -> bool {
    match bytes.first().copied() {
        Some(b'-' | b'+' | b'*') => {
            bytes.len() == 1 || bytes.get(1).is_some_and(u8::is_ascii_whitespace)
        }
        Some(b'0'..=b'9') => {
            let digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
            if digits == 0 || digits > 9 || digits >= bytes.len() {
                return false;
            }
            matches!(bytes[digits], b'.' | b')')
                && (digits + 1 == bytes.len()
                    || bytes.get(digits + 1).is_some_and(u8::is_ascii_whitespace))
        }
        _ => false,
    }
}

#[cfg(test)]
mod block_kind_tests {
    use super::*;

    #[test]
    fn classifies_supported_block_openers() {
        assert_eq!(
            classify_block_start("plain text", false),
            BlockKind::Paragraph
        );
        assert_eq!(
            classify_block_start("## heading", false),
            BlockKind::Heading
        );
        assert_eq!(classify_block_start("```rust", true), BlockKind::FencedCode);
        assert_eq!(
            classify_block_start("> quoted", false),
            BlockKind::BlockQuote
        );
        assert_eq!(
            classify_block_start(">no space", false),
            BlockKind::BlockQuote
        );
        assert_eq!(classify_block_start("  - item", false), BlockKind::List);
        assert_eq!(classify_block_start("3) item", false), BlockKind::List);
        assert_eq!(
            classify_block_start("  * * *", false),
            BlockKind::ThematicBreak
        );
    }

    #[test]
    fn rejects_near_miss_markers() {
        assert_eq!(
            classify_block_start("    > indented", false),
            BlockKind::Paragraph
        );
        assert_eq!(
            classify_block_start("\t- tab-indented is code-level", false),
            BlockKind::Paragraph
        );
        assert_eq!(classify_block_start("10. item", false), BlockKind::List);
        assert_eq!(
            classify_block_start("1234567890. item", false),
            BlockKind::Paragraph
        );
        assert_eq!(classify_block_start("--", false), BlockKind::Paragraph);
        assert_eq!(
            classify_block_start("-not a list", false),
            BlockKind::Paragraph
        );
    }
}
