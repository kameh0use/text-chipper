use self::scanners::SourcePos;
use crate::block::{Block, Headers};
use crate::ChipError;
use std::io;
use std::sync::Arc;

/// Maximum bytes retained in one open parser block before it is emitted as a
/// source-contiguous fragment. Splits always occur at UTF-8 boundaries.
pub(crate) const MAX_OPEN_BLOCK_BYTES: usize = 1024 * 1024;
/// Maximum bytes retained as heading/blank-line separators.
pub(crate) const MAX_PARSER_GAP_BYTES: usize = 1024 * 1024;
/// A metadata (front matter) block that has not closed after this many lines / bytes is
/// abandoned: parsing continues normally after it.
const FRONT_MATTER_MAX_LINES: usize = 200;
const FRONT_MATTER_MAX_BYTES: usize = 32 * 1024;
/// A setext heading title is at most this long (two lines, this many characters). Longer
/// paragraphs followed by `---` / `===` are ordinary text, not headings, so no body text is
/// ever swallowed into heading context by a stray underline.
const SETEXT_MAX_LINES: usize = 2;
const SETEXT_MAX_CHARS: usize = 160;

struct OpenBlock {
    text: String,
    start_byte: usize,
    start_char: usize,
    sep_before: String,
    is_code: bool,
    headers: Headers,
}

/// Incremental, line-oriented block splitter used for chunking. It is **not** a CommonMark
/// parser. It splits text at blank lines, recognizes fenced code, ATX/short setext headings and
/// YAML/TOML front matter, and keeps exact source offsets with bounded memory. Block quotes,
/// HTML blocks, indented code, tables and link reference definitions are ordinary text blocks,
/// and lists, quotes and thematic breaks do not interrupt a paragraph. List structure inside a
/// block is found by [`crate::parse_list_items`]. Agreement with `pulldown-cmark` on the CommonMark
/// corpus is measured by `oracle_block_tests` (see the README).
pub struct BlockParser {
    parse_headings: bool,
    source_pos: SourcePos,
    stack: Vec<(usize, String)>,
    headers: Headers,
    /// True after a heading is parsed until a non-empty body block consumes it.
    has_pending_heading: bool,
    fence: Option<(char, usize)>,
    /// Inside a front matter block: (delimiter char, lines consumed so far).
    front_matter: Option<(char, usize)>,
    cur: Option<OpenBlock>,
    gap: String,
    gap_valid: bool,
}

impl Default for BlockParser {
    fn default() -> Self {
        Self::new()
    }
}

fn opening_fence(content: &str) -> Option<(char, usize)> {
    scanners::opening_fence(content)
}

fn is_closing_fence(content: &str, fc: char, fl: usize) -> bool {
    scanners::closing_fence(content, fc, fl)
}

impl BlockParser {
    /// Create a parser that recognizes Markdown headings (the default).
    pub fn new() -> Self {
        Self::with_headings(true)
    }

    /// Create a parser with heading tracking switched on or off. With `false`, headings are
    /// treated as ordinary text and no header context is produced; code fences and the other
    /// block splitting rules still apply.
    pub fn with_headings(parse_headings: bool) -> Self {
        BlockParser {
            parse_headings,
            source_pos: SourcePos::ZERO,
            stack: Vec::new(),
            headers: Arc::new(Vec::new()),
            has_pending_heading: false,
            fence: None,
            front_matter: None,
            cur: None,
            gap: String::new(),
            gap_valid: true,
        }
    }

    fn start_block(&mut self, line: &str, is_code: bool, out: &mut Vec<Block>) {
        self.has_pending_heading = false;
        let sep_before = if self.gap_valid {
            std::mem::take(&mut self.gap)
        } else {
            self.gap.clear();
            String::new()
        };
        self.cur = Some(OpenBlock {
            text: String::new(),
            start_byte: self.source_pos.byte,
            start_char: self.source_pos.character,
            sep_before,
            is_code,
            headers: Arc::clone(&self.headers),
        });
        self.append_to_open_block(line, out);
    }

    /// Emit a heading-only block when a heading had no body text before the next
    /// heading or end-of-document, so neither the heading text nor its `headers`
    /// path is silently dropped.
    fn flush_empty_heading(&mut self, out: &mut Vec<Block>) {
        if !self.has_pending_heading {
            return;
        }
        self.has_pending_heading = false;
        if !self.gap_valid || self.gap.is_empty() {
            return;
        }
        let raw_gap = std::mem::take(&mut self.gap);
        let trimmed_start = raw_gap.trim_start_matches(char::is_whitespace);
        let leading_bytes = raw_gap.len() - trimmed_start.len();
        let trimmed_full = trimmed_start.trim_end();
        if trimmed_full.is_empty() {
            self.gap = raw_gap;
            return;
        }
        let content_end = leading_bytes + trimmed_full.len();
        let mut gap_slice = raw_gap;
        let trailing_gap = gap_slice.split_off(content_end);
        let text = gap_slice.split_off(leading_bytes);
        let sep_before = gap_slice;

        let total_gap_bytes = sep_before.len() + text.len() + trailing_gap.len();
        let total_gap_chars = crate::split_engine::char_len(&sep_before)
            + crate::split_engine::char_len(&text)
            + crate::split_engine::char_len(&trailing_gap);
        let start_byte = self.source_pos.byte.saturating_sub(total_gap_bytes) + sep_before.len();
        let start_char = self.source_pos.character.saturating_sub(total_gap_chars)
            + crate::split_engine::char_len(&sep_before);

        self.gap = trailing_gap;
        self.gap_valid = true;

        out.push(Block {
            text,
            start_byte,
            start_char,
            sep_before,
            headers: Arc::clone(&self.headers),
            is_code: false,
        });
    }

    fn close_block(&mut self, out: &mut Vec<Block>) {
        if let Some(mut b) = self.cur.take() {
            let kept = b.text.trim_end().len();
            self.gap = b.text.split_off(kept);
            self.gap_valid = true;
            if !b.text.is_empty() {
                out.push(Block {
                    text: b.text,
                    start_byte: b.start_byte,
                    start_char: b.start_char,
                    sep_before: b.sep_before,
                    headers: b.headers,
                    is_code: b.is_code,
                });
            }
        }
    }

    fn append_gap(&mut self, line: &str) -> Result<(), ChipError> {
        if self.gap.len().saturating_add(line.len()) > MAX_PARSER_GAP_BYTES {
            return Err(ChipError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Markdown separator/heading gap exceeds the {MAX_PARSER_GAP_BYTES}-byte parser limit"),
            )));
        }
        self.gap.push_str(line);
        Ok(())
    }

    fn emit_current_fragment(&mut self, out: &mut Vec<Block>) {
        let Some(block) = self.cur.as_mut() else {
            return;
        };
        if block.text.is_empty() {
            return;
        }
        let text = std::mem::take(&mut block.text);
        let start_byte = block.start_byte;
        let start_char = block.start_char;
        let start_advance = SourcePos {
            byte: text.len(),
            character: crate::split_engine::char_len(&text),
        };
        let fragment = Block {
            text,
            start_byte,
            start_char,
            sep_before: std::mem::take(&mut block.sep_before),
            headers: Arc::clone(&block.headers),
            is_code: block.is_code,
        };
        block.start_byte += start_advance.byte;
        block.start_char += start_advance.character;
        out.push(fragment);
    }

    /// Append in bounded pieces. This avoids temporarily copying an entire giant
    /// physical line into OpenBlock before discovering that the block is too large.
    fn append_to_open_block(&mut self, mut text: &str, out: &mut Vec<Block>) {
        while !text.is_empty() {
            let current_len = self.cur.as_ref().map_or(0, |block| block.text.len());
            if current_len == MAX_OPEN_BLOCK_BYTES {
                self.emit_current_fragment(out);
                continue;
            }
            let room = MAX_OPEN_BLOCK_BYTES - current_len;
            let mut take = room.min(text.len());
            while !text.is_char_boundary(take) {
                take -= 1;
            }
            if take == 0 {
                // room is at least one byte, and a UTF-8 scalar is at most four bytes;
                // a full block is emitted on the next iteration to make room.
                self.emit_current_fragment(out);
                continue;
            }
            if take < text.len() {
                if let Some(newline) = text[..take].rfind('\n') {
                    if newline + 1 >= take / 2 {
                        take = newline + 1;
                    }
                }
            }
            let block = self.cur.as_mut().expect("open block exists");
            let required = block.text.len() + take;
            if block.text.capacity() < required {
                const GROWTH_STEP: usize = 64 * 1024;
                let target = required
                    .saturating_add(GROWTH_STEP - 1)
                    .checked_div(GROWTH_STEP)
                    .unwrap_or(0)
                    .saturating_mul(GROWTH_STEP)
                    .min(MAX_OPEN_BLOCK_BYTES);
                block
                    .text
                    .reserve_exact(target.saturating_sub(block.text.len()));
            }
            block.text.push_str(&text[..take]);
            text = &text[take..];
            if !text.is_empty()
                && self
                    .cur
                    .as_ref()
                    .is_some_and(|block| block.text.len() == MAX_OPEN_BLOCK_BYTES)
            {
                self.emit_current_fragment(out);
            }
        }
    }
    pub fn push_line(&mut self, line: &str, out: &mut Vec<Block>) -> Result<(), ChipError> {
        // The scanner keeps source coordinates in bytes and exposes the content
        // without its LF/CRLF terminator. No line String is allocated here.
        let line_view = scanners::LineView::new(line, self.source_pos);
        let content = line_view.content;

        // YAML (`---` ... `---`/`...`) or TOML (`+++` ... `+++`) front matter at the very start
        // of the document is metadata: kept verbatim as one block, never parsed for headings.
        if let Some((delim, n)) = self.front_matter {
            let t = content.trim_end();
            // An ATX heading right after a blank line inside a still-open metadata block is a
            // document heading, not YAML: otherwise an unterminated `---` block would swallow
            // every heading up to the 200-line limit.
            let heading_after_blank = n > 1
                && scanners::atx_heading(content).is_some()
                && self
                    .cur
                    .as_ref()
                    .is_some_and(|b| b.text.ends_with("\n\n") || b.text.ends_with("\n\r\n"));
            let looks_like_metadata = !heading_after_blank
                && (n > 1
                    || (!t.is_empty()
                        && scanners::atx_heading(content).is_none()
                        && if delim == '-' {
                            t.contains(':')
                        } else {
                            t.contains('=')
                        }));
            if looks_like_metadata {
                self.append_to_open_block(line, out);
                let closes =
                    (delim == '-' && (t == "---" || t == "...")) || (delim == '+' && t == "+++");
                if closes
                    || n + 1 >= FRONT_MATTER_MAX_LINES
                    || self.open_bytes() > FRONT_MATTER_MAX_BYTES
                {
                    self.close_block(out);
                    self.front_matter = None;
                } else {
                    self.front_matter = Some((delim, n + 1));
                }
                self.source_pos = line_view.range.end;
                return Ok(());
            }
            // "---" followed by a blank line / heading / plain text: a thematic break, not
            // metadata. Un-flag the opener block and parse this line normally.
            if n <= 1 {
                if let Some(b) = self.cur.as_mut() {
                    b.is_code = false;
                }
            }
            self.front_matter = None;
        } else if self.parse_headings && self.source_pos.byte == 0 && self.cur.is_none() {
            let t = content.trim_start_matches('\u{feff}').trim_end();
            if t == "---" || t == "+++" {
                let delim = t.chars().next().unwrap_or('-');
                self.start_block(line, true, out);
                self.front_matter = Some((delim, 1));
                self.source_pos = line_view.range.end;
                return Ok(());
            }
        }

        let heading = if self.fence.is_none() && self.parse_headings {
            match scanners::atx_heading(content) {
                Some((_, _)) if content.len() > MAX_PARSER_GAP_BYTES => {
                    return Err(ChipError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Markdown heading exceeds the {MAX_PARSER_GAP_BYTES}-byte parser limit"
                        ),
                    )));
                }
                Some((level, title)) => Some((level, title.to_string())),
                None => None,
            }
        } else {
            None
        };
        if let Some((fc, fl)) = self.fence {
            self.append_to_open_block(line, out);
            if is_closing_fence(content, fc, fl) {
                self.close_block(out);
                self.fence = None;
            }
        } else if let Some(f) = opening_fence(content) {
            // Flush the preceding block immediately. Holding it until a fence closes
            // made an unterminated fence retain an unbounded Vec<Block>.
            self.close_block(out);
            self.fence = Some(f);
            self.start_block(line, true, out);
        } else if let Some((level, title)) = heading {
            self.close_block(out);
            self.flush_empty_heading(out);
            self.stack.retain(|h| h.0 < level);
            self.stack.push((level, title));
            self.headers = Arc::new(
                self.stack
                    .iter()
                    .filter(|h| !h.1.is_empty())
                    .map(|h| h.1.clone())
                    .collect(),
            );
            self.has_pending_heading = true;
            self.append_gap(line)?;
        } else if let Some(level) = self.setext_level(content) {
            // "Title\n=====": the open paragraph is the heading, not body text. Its source
            // (separator before it, the title lines, the underline) goes to the gap exactly
            // like an ATX heading line, so offsets stay contiguous.
            let b = self
                .cur
                .take()
                .expect("setext_level checked the open block");
            self.flush_empty_heading(out);
            let title = scanners::setext_title(&b.text);
            let mut gap = b.sep_before;
            gap.push_str(&b.text);
            gap.push_str(line);
            if gap.len() > MAX_PARSER_GAP_BYTES {
                return Err(ChipError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Markdown separator/heading gap exceeds the {MAX_PARSER_GAP_BYTES}-byte parser limit"),
                )));
            }
            self.gap = gap;
            self.gap_valid = true;
            self.stack.retain(|h| h.0 < level);
            self.stack.push((level, title));
            self.headers = Arc::new(
                self.stack
                    .iter()
                    .filter(|h| !h.1.is_empty())
                    .map(|h| h.1.clone())
                    .collect(),
            );
            self.has_pending_heading = true;
        } else if content.trim().is_empty() {
            self.close_block(out);
            self.append_gap(line)?;
        } else if self.cur.is_some() {
            self.append_to_open_block(line, out);
        } else {
            self.start_block(line, false, out);
        }
        self.source_pos = line_view.range.end;
        Ok(())
    }

    /// Level of the setext heading that `content` completes, if it underlines the open
    /// paragraph.
    fn setext_level(&self, content: &str) -> Option<usize> {
        if !self.parse_headings || self.fence.is_some() {
            return None;
        }
        let level = scanners::setext_underline(content)?;
        let cur = self.cur.as_ref()?;
        if cur.is_code || cur.text.is_empty() || !scanners::can_be_setext_content(&cur.text) {
            return None;
        }
        // A real title is short. Anything longer is a paragraph followed by a stray underline.
        // (The cheap length test first keeps repeated underlines after a huge block O(1).)
        if cur.text.len() > SETEXT_MAX_CHARS * 4 {
            return None;
        }
        let title = cur.text.trim_end();
        if title.lines().count() > SETEXT_MAX_LINES || title.chars().count() > SETEXT_MAX_CHARS {
            return None;
        }
        Some(level)
    }

    pub(crate) fn open_bytes(&self) -> usize {
        self.cur.as_ref().map_or(0, |block| block.text.len())
    }

    pub(crate) fn gap_bytes(&self) -> usize {
        self.gap.len()
    }

    /// Earliest source coordinate that an unfinished parser state may still reference.
    /// Keep parser internals private while allowing the streaming chunker to retain
    /// the corresponding prefix of its source buffer.
    pub(crate) fn earliest_future_start(&self) -> (usize, usize) {
        let mut earliest = self.source_pos.saturating_sub(&self.gap);
        if let Some(cur) = &self.cur {
            earliest = earliest.min(
                SourcePos {
                    byte: cur.start_byte,
                    character: cur.start_char,
                }
                .saturating_sub(&cur.sep_before),
            );
        }
        (earliest.byte, earliest.character)
    }

    pub fn finish(&mut self, out: &mut Vec<Block>) {
        self.close_block(out);
        self.flush_empty_heading(out);
        self.fence = None;
    }
}

// Markdown-related source scanners and structural list parsing live together
// because they share parsing rules and source-coordinate conventions.

pub(crate) mod scanners {
    //! Allocation-free byte scanners for the block parser.
    //!
    //! These helpers recognize only the Markdown constructs currently supported by
    //! `BlockParser`; they are not a complete CommonMark block parser.

    // Source-coordinate primitives are colocated with the Markdown scanners.

    use crate::split_engine::char_len;

    //
    // Byte offsets are suitable for slicing UTF-8 source strings; character offsets
    // count Unicode scalar values and are retained for the public chunk metadata.

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) struct SourcePos {
        pub(crate) byte: usize,
        pub(crate) character: usize,
    }

    impl SourcePos {
        pub(crate) const ZERO: Self = Self {
            byte: 0,
            character: 0,
        };

        #[inline]
        pub(crate) fn advance(self, text: &str) -> Self {
            Self {
                byte: self.byte + text.len(),
                character: self.character + char_len(text),
            }
        }

        #[inline]
        pub(crate) fn saturating_sub(self, text: &str) -> Self {
            Self {
                byte: self.byte.saturating_sub(text.len()),
                character: self.character.saturating_sub(char_len(text)),
            }
        }

        #[inline]
        pub(crate) fn min(self, other: Self) -> Self {
            Self {
                byte: self.byte.min(other.byte),
                character: self.character.min(other.character),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct SourceRange {
        pub(crate) start: SourcePos,
        pub(crate) content_end: SourcePos,
        pub(crate) end: SourcePos,
    }

    impl SourceRange {
        pub(crate) fn for_line(raw: &str, start: SourcePos, content_end_byte: usize) -> Self {
            let content = &raw[..content_end_byte];
            Self {
                start,
                content_end: start.advance(content),
                end: start.advance(raw),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct LineView<'a> {
        pub content: &'a str,
        pub range: SourceRange,
    }

    impl<'a> LineView<'a> {
        /// Build a line view from one line as supplied by `BufRead::read_line` or
        /// `str::lines`-style streaming. Byte offsets are relative to the source.
        pub(crate) fn new(raw: &'a str, start: SourcePos) -> Self {
            let mut end = raw.len();
            if raw.as_bytes().last() == Some(&b'\n') {
                end -= 1;
            }
            if end > 0 && raw.as_bytes()[end - 1] == b'\r' {
                end -= 1;
            }
            Self {
                content: &raw[..end],
                range: SourceRange::for_line(raw, start, end),
            }
        }
    }

    /// Return the byte offset of the first non-space byte, accepting at most three
    /// leading ASCII spaces. Tabs are not interchangeable with spaces here.
    #[inline]
    fn up_to_three_spaces(bytes: &[u8]) -> Option<usize> {
        let mut i = 0;
        while i < bytes.len() && bytes[i] == b' ' && i < 4 {
            i += 1;
        }
        (i <= 3).then_some(i)
    }

    /// Recognize the currently supported ATX heading form without allocating.
    /// The returned title is a slice of the original line.
    pub(crate) fn atx_heading(content: &str) -> Option<(usize, &str)> {
        let bytes = content.as_bytes();
        let indent = up_to_three_spaces(bytes)?;
        let rest = &content[indent..];
        let hashes = rest.as_bytes().iter().take_while(|&&b| b == b'#').count();
        if !(1..=6).contains(&hashes) {
            return None;
        }
        let after = &rest[hashes..];
        if !(after.is_empty() || after.as_bytes()[0] == b' ' || after.as_bytes()[0] == b'\t') {
            return None;
        }

        let trimmed = after.trim();
        let without_closing = trimmed.trim_end_matches('#');
        let title = if without_closing.len() == trimmed.len() {
            trimmed
        } else if without_closing.is_empty() || without_closing.ends_with(char::is_whitespace) {
            without_closing.trim_end()
        } else {
            trimmed
        };
        Some((hashes, title))
    }

    /// Recognize a setext underline (`====` -> level 1, `----` -> level 2): up to three leading
    /// spaces, one repeated marker, optional trailing blanks. Whether it really underlines a
    /// paragraph is decided by the caller (it needs the open block).
    pub(crate) fn setext_underline(content: &str) -> Option<usize> {
        let bytes = content.as_bytes();
        let indent = up_to_three_spaces(bytes)?;
        let rest = content[indent..].trim_end_matches([' ', '\t']);
        let first = *rest.as_bytes().first()?;
        if first != b'=' && first != b'-' {
            return None;
        }
        if !rest.bytes().all(|b| b == first) {
            return None;
        }
        Some(if first == b'=' { 1 } else { 2 })
    }

    /// Can `text` (an open, non-code block) be the content of a setext heading? Blocks that
    /// start like a list item, quote, table row or indented code are left alone so that
    /// "- item\n---" keeps its current meaning.
    pub(crate) fn can_be_setext_content(text: &str) -> bool {
        let first = text.lines().next().unwrap_or("");
        let bytes = first.as_bytes();
        if up_to_three_spaces(bytes).is_none() {
            return false;
        }
        let t = first.trim_start_matches(' ');
        let Some(c) = t.chars().next() else {
            return false;
        };
        if matches!(c, '-' | '*' | '+' | '>' | '|' | '#' | '\t') {
            return false;
        }
        let digits = t.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && matches!(t.as_bytes().get(digits), Some(b'.') | Some(b')')) {
            return false;
        }
        true
    }

    /// Heading title of a setext paragraph: its lines trimmed and joined with single spaces.
    pub(crate) fn setext_title(text: &str) -> String {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Recognize a fenced-code opener. Returns the ASCII marker and run length.
    pub(crate) fn opening_fence(content: &str) -> Option<(char, usize)> {
        let bytes = content.as_bytes();
        let indent = up_to_three_spaces(bytes)?;
        let rest = &bytes[indent..];
        let marker = *rest.first()?;
        if marker != b'`' && marker != b'~' {
            return None;
        }
        let run = rest.iter().take_while(|&&b| b == marker).count();
        if run < 3 {
            return None;
        }
        if marker == b'`' && rest[run..].contains(&b'`') {
            return None;
        }
        Some((marker as char, run))
    }

    /// Recognize a closing fence for an already-open fenced block.
    pub(crate) fn closing_fence(content: &str, fence_char: char, fence_len: usize) -> bool {
        let bytes = content.as_bytes();
        let indent = match up_to_three_spaces(bytes) {
            Some(i) => i,
            None => return false,
        };
        let marker = fence_char as u8;
        let rest = &bytes[indent..];
        let run = rest.iter().take_while(|&&b| b == marker).count();
        run >= fence_len && content[indent + run..].trim().is_empty()
    }

    /// CommonMark tab-stop indentation width. This is a measurement helper, not a
    /// complete list/container indentation parser.
    pub(crate) fn indent_width(bytes: &[u8]) -> usize {
        let mut width = 0usize;
        for &b in bytes {
            match b {
                b' ' => width += 1,
                b'\t' => width += 4 - (width % 4),
                _ => break,
            }
        }
        width
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn line_view_tracks_lf_crlf_and_bare_lines() {
            let lf = LineView::new(
                "abc\n",
                SourcePos {
                    byte: 10,
                    character: 10,
                },
            );
            assert_eq!(lf.content, "abc");
            assert_eq!(
                (
                    lf.range.start.byte,
                    lf.range.content_end.byte,
                    lf.range.end.byte
                ),
                (10, 13, 14)
            );
            assert_eq!(
                (
                    lf.range.start.character,
                    lf.range.content_end.character,
                    lf.range.end.character
                ),
                (10, 13, 14)
            );
            let crlf = LineView::new(
                "é\r\n",
                SourcePos {
                    byte: 20,
                    character: 10,
                },
            );
            assert_eq!(crlf.content, "é");
            assert_eq!((crlf.range.content_end.byte, crlf.range.end.byte), (22, 24));
            assert_eq!(
                (crlf.range.content_end.character, crlf.range.end.character),
                (11, 13)
            );
            let bare = LineView::new(
                "x",
                SourcePos {
                    byte: 4,
                    character: 4,
                },
            );
            assert_eq!(bare.content, "x");
        }

        #[test]
        fn atx_heading_rules_and_escaped_marker() {
            assert_eq!(atx_heading("### title"), Some((3, "title")));
            assert_eq!(atx_heading("   # title"), Some((1, "title")));
            assert_eq!(atx_heading("    # not heading"), None);
            assert_eq!(atx_heading("####### too many"), None);
            assert_eq!(atx_heading("#not-heading"), None);
            assert_eq!(atx_heading(r"\# escaped"), None);
        }

        #[test]
        fn fence_rules() {
            assert_eq!(opening_fence("  ```rust"), Some(('`', 3)));
            assert_eq!(opening_fence("~~~"), Some(('~', 3)));
            assert_eq!(opening_fence("    ```"), None);
            assert_eq!(opening_fence("```info`bad"), None);
            assert!(closing_fence("  ````  ", '`', 3));
            assert!(!closing_fence("  ``  ", '`', 3));
            assert!(!closing_fence("  ``` trailing", '`', 3));
            assert!(!closing_fence("    ```", '`', 3));
        }

        #[test]
        fn tabs_advance_to_four_column_stops() {
            assert_eq!(indent_width(b"    x"), 4);
            assert_eq!(indent_width(b"\tx"), 4);
            assert_eq!(indent_width(b" \tx"), 4);
            assert_eq!(indent_width(b"  \tx"), 4);
        }
    }

    #[cfg(test)]
    mod source_tests {
        use super::*;

        #[test]
        fn advances_bytes_and_unicode_scalar_offsets() {
            let start = SourcePos {
                byte: 7,
                character: 4,
            };
            assert_eq!(
                start.advance("é🙂"),
                SourcePos {
                    byte: 13,
                    character: 6
                }
            );
        }

        #[test]
        fn subtraction_and_min_are_componentwise() {
            let pos = SourcePos {
                byte: 10,
                character: 8,
            };
            assert_eq!(
                pos.saturating_sub("é"),
                SourcePos {
                    byte: 8,
                    character: 7
                }
            );
            assert_eq!(
                pos.min(SourcePos {
                    byte: 12,
                    character: 3
                }),
                SourcePos {
                    byte: 10,
                    character: 3
                }
            );
        }
    }
}

pub mod list_parser {
    //! Lightweight, source-preserving structural scanner for Markdown list items.
    //!
    //! This scanner preserves source byte ranges and recognizes nested list items,
    //! indented continuation lines, and lazy paragraph continuation. It does not
    //! copy source text and is not a complete CommonMark block parser.

    /// The marker used to start a list item.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ListMarker {
        Bullet(char),
        Ordered { delimiter: char, number: u64 },
    }

    /// Source range and nesting metadata for one list item.
    ///
    /// Byte ranges are half-open and always lie on UTF-8 boundaries. Line numbers
    /// are zero-based; `end_line` is exclusive. A parent's range includes its
    /// nested descendants and continuation lines.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ListItem {
        pub marker: ListMarker,
        pub depth: usize,
        pub parent: Option<usize>,
        pub marker_indent: usize,
        pub content_indent: usize,
        pub start_byte: usize,
        pub end_byte: usize,
        pub start_line: usize,
        pub end_line: usize,
    }

    #[derive(Clone, Copy)]
    struct MarkerInfo {
        marker: ListMarker,
        indent: usize,
        content_indent: usize,
        has_content: bool,
        content_start: usize,
    }

    fn is_thematic_break_line(line: &str) -> bool {
        let trimmed = line.trim_start_matches(' ');
        let mut chars = trimmed.chars().filter(|&c| c != ' ' && c != '\t');
        let Some(marker) = chars.next() else {
            return false;
        };
        matches!(marker, '-' | '_' | '*')
            && chars.all(|c| c == marker)
            && trimmed.chars().filter(|&c| c == marker).count() >= 3
    }

    fn marker_info(line: &str) -> Option<MarkerInfo> {
        // Keep thematic-break detection local to this scanner. Block classification
        // applies different indentation rules and must not suppress valid candidates.
        if is_thematic_break_line(line) {
            return None;
        }
        let bytes = line.as_bytes();
        let mut i = 0;
        let mut indent = 0;
        while i < bytes.len() {
            match bytes[i] {
                b' ' => {
                    indent += 1;
                    i += 1;
                }
                b'\t' => {
                    indent += 4 - (indent % 4);
                    i += 1;
                }
                _ => break,
            }
        }

        let marker_start = i;
        let marker = match *bytes.get(i)? {
            b'-' => {
                i += 1;
                ListMarker::Bullet('-')
            }
            b'+' => {
                i += 1;
                ListMarker::Bullet('+')
            }
            b'*' => {
                i += 1;
                ListMarker::Bullet('*')
            }
            b'0'..=b'9' => {
                let digits_start = i;
                while bytes.get(i).is_some_and(u8::is_ascii_digit) && i - digits_start < 10 {
                    i += 1;
                }
                let digits = i - digits_start;
                if digits == 0 || digits > 9 {
                    return None;
                }
                let delimiter = *bytes.get(i)?;
                if delimiter != b'.' && delimiter != b')' {
                    return None;
                }
                let number = line[digits_start..i].parse().ok()?;
                i += 1;
                ListMarker::Ordered {
                    delimiter: delimiter as char,
                    number,
                }
            }
            _ => return None,
        };

        let marker_width = indent + (i - marker_start);
        if i < bytes.len() && bytes[i] != b' ' && bytes[i] != b'\t' {
            return None;
        }

        let mut whitespace_columns = 0;
        while i < bytes.len() {
            match bytes[i] {
                b' ' => {
                    whitespace_columns += 1;
                    i += 1;
                }
                b'\t' => {
                    whitespace_columns += 4 - ((marker_width + whitespace_columns) % 4);
                    i += 1;
                }
                _ => break,
            }
        }
        // CommonMark treats an empty marker as having one column of padding. For
        // unusually wide padding, retain a conservative one-column content indent.
        let padding = if whitespace_columns == 0 || whitespace_columns > 4 {
            1
        } else {
            whitespace_columns
        };
        let content_indent = marker_width + padding;

        Some(MarkerInfo {
            marker,
            indent,
            content_indent,
            has_content: i < bytes.len(),
            content_start: i,
        })
    }

    #[inline]
    fn leading_indent(line: &str) -> usize {
        crate::markdown::scanners::indent_width(line.as_bytes())
    }

    /// Remove up to `cols` columns of leading indentation (tab stops of four).
    fn dedent_columns(line: &str, cols: usize) -> &str {
        let mut width = 0usize;
        let mut cut = 0usize;
        for (i, b) in line.bytes().enumerate() {
            match b {
                b' ' if width < cols => {
                    width += 1;
                    cut = i + 1;
                }
                b'\t' if width + (4 - width % 4) <= cols => {
                    width += 4 - width % 4;
                    cut = i + 1;
                }
                _ => break,
            }
        }
        &line[cut..]
    }

    /// Strip one or more CommonMark block-quote prefixes without allocating.
    /// The returned slice remains a view into the original UTF-8 source line.
    fn strip_blockquote_prefixes(mut line: &str) -> (&str, usize) {
        let mut depth = 0usize;
        loop {
            let bytes = line.as_bytes();
            let mut i = 0usize;
            let mut spaces = 0usize;
            while i < bytes.len() && bytes[i] == b' ' && spaces < 4 {
                i += 1;
                spaces += 1;
            }
            if spaces > 3 || bytes.get(i) != Some(&b'>') {
                return (line, depth);
            }
            i += 1;
            if matches!(bytes.get(i), Some(b' ' | b'\t')) {
                i += 1;
            }
            line = &line[i..];
            depth += 1;
        }
    }

    fn interrupts_lazy_paragraph(line: &str) -> bool {
        let trimmed = line.trim_start_matches([' ', '\t']);
        trimmed.starts_with('>')
            || crate::markdown::scanners::atx_heading(trimmed).is_some()
            || crate::markdown::scanners::opening_fence(trimmed).is_some()
            || crate::block::classify_block_start(trimmed, false)
                == crate::block::BlockKind::ThematicBreak
    }

    /// Parse list item boundaries and nesting from Markdown source.
    ///
    /// The function does not copy source text. Each returned item refers to a
    /// half-open byte range in `source`; nested items point to their parent by
    /// index in the returned vector.
    pub fn parse_list_items(source: &str) -> Vec<ListItem> {
        let mut items: Vec<ListItem> = Vec::new();
        // Quote depth is kept parallel to items so lazy/blank continuations inside
        // a block quote remain attached to the corresponding list item.
        let mut item_quote_depth: Vec<usize> = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        let mut byte_offset = 0usize;
        let mut line_number = 0usize;
        let mut previous_nonblank = false;
        // Open fenced code: (fence char, fence length, item indentation to strip, quote depth).
        // Lines inside a fence never start or end list items.
        let mut fence: Option<(char, usize, usize, usize)> = None;

        for raw_line in source.split_inclusive('\n') {
            let content = raw_line.strip_suffix('\n').unwrap_or(raw_line);
            let content = content.strip_suffix('\r').unwrap_or(content);
            // List markers and fences may live inside one or more block quotes.
            // Keep the original line for byte ranges, but scan its container content.
            let (logical, quote_depth) = strip_blockquote_prefixes(content);
            let line_end = byte_offset + raw_line.len();

            if let Some((fence_char, fence_len, strip, fence_quote_depth)) = fence {
                for &idx in &active {
                    items[idx].end_byte = line_end;
                    items[idx].end_line = line_number + 1;
                }
                let fence_line = if fence_quote_depth > 0 && quote_depth >= fence_quote_depth {
                    logical
                } else {
                    content
                };
                if crate::markdown::scanners::closing_fence(
                    dedent_columns(fence_line, strip),
                    fence_char,
                    fence_len,
                ) {
                    fence = None;
                }
                previous_nonblank = true;
                byte_offset = line_end;
                line_number += 1;
                continue;
            }
            let mut strip = active.last().map_or(0, |&i| items[i].content_indent);
            if leading_indent(logical) < strip
                && crate::markdown::scanners::opening_fence(logical).is_some()
            {
                // A fence outside the item's content column closes the list.
                active.clear();
                strip = 0;
            }
            if leading_indent(logical) >= strip {
                if let Some((fence_char, fence_len)) =
                    crate::markdown::scanners::opening_fence(dedent_columns(logical, strip))
                {
                    fence = Some((fence_char, fence_len, strip, quote_depth));
                    for &idx in &active {
                        items[idx].end_byte = line_end;
                        items[idx].end_line = line_number + 1;
                    }
                    previous_nonblank = true;
                    byte_offset = line_end;
                    line_number += 1;
                    continue;
                }
            }

            if let Some(info) = marker_info(logical) {
                // CommonMark does not allow an empty list item to interrupt an
                // open paragraph. An ordered list can interrupt a paragraph only
                // when it starts at 1 (for the period delimiter); larger period-
                // delimited starts remain paragraph text. Other delimiters retain
                // this parser's existing ordered-list behavior.
                // Keep this check at top level so valid nested markers are unaffected.
                let interrupts_paragraph = active.is_empty()
                    && previous_nonblank
                    && (!info.has_content
                        || matches!(
                            info.marker,
                            ListMarker::Ordered {
                                delimiter: '.',
                                number,
                            } if number != 1
                        ));
                if interrupts_paragraph {
                    previous_nonblank = true;
                    byte_offset = line_end;
                    line_number += 1;
                    continue;
                }

                let parent_pos = active
                    .iter()
                    .rposition(|&idx| info.indent >= items[idx].content_indent);
                // Four or more columns of indentation cannot start a list item here.
                // Directly after a non-blank line inside an active item the text is
                // lazy paragraph continuation; after a blank line it closes the list.
                if info.indent >= 4 && parent_pos.is_none() {
                    if !active.is_empty() && previous_nonblank {
                        for &idx in &active {
                            items[idx].end_byte = line_end;
                            items[idx].end_line = line_number + 1;
                        }
                    } else {
                        active.clear();
                    }
                    previous_nonblank = !logical.trim().is_empty();
                } else {
                    let parent = parent_pos.map(|p| active[p]);
                    if let Some(p) = parent_pos {
                        active.truncate(p + 1);
                    } else {
                        active.clear();
                    }
                    let depth = parent.map_or(0, |p| items[p].depth + 1);
                    let idx = items.len();
                    items.push(ListItem {
                        marker: info.marker,
                        depth,
                        parent,
                        marker_indent: info.indent,
                        content_indent: info.content_indent,
                        start_byte: byte_offset,
                        end_byte: line_end,
                        start_line: line_number,
                        end_line: line_number + 1,
                    });
                    item_quote_depth.push(quote_depth);
                    for &ancestor in &active {
                        items[ancestor].end_byte = line_end;
                        items[ancestor].end_line = line_number + 1;
                    }
                    active.push(idx);

                    // A list marker may itself contain another list marker on the
                    // same physical line (for example `- - item` or `1. - 2. item`).
                    // Model these as nested items sharing the same source start.
                    let mut nested_tail = &logical[info.content_start..];
                    let mut indent_base = info.content_indent;
                    let mut parent_idx = idx;
                    while let Some(nested) = marker_info(nested_tail) {
                        if !nested.has_content || nested.indent >= 4 {
                            break;
                        }
                        let child = items.len();
                        items.push(ListItem {
                            marker: nested.marker,
                            depth: items[parent_idx].depth + 1,
                            parent: Some(parent_idx),
                            marker_indent: indent_base + nested.indent,
                            content_indent: indent_base + nested.content_indent,
                            start_byte: byte_offset,
                            end_byte: line_end,
                            start_line: line_number,
                            end_line: line_number + 1,
                        });
                        item_quote_depth.push(quote_depth);
                        active.push(child);
                        parent_idx = child;
                        nested_tail = &nested_tail[nested.content_start..];
                        indent_base = items[child].content_indent;
                        if nested_tail.is_empty() {
                            break;
                        }
                    }

                    // Fences can begin in a list item's content rather than at the
                    // physical line's first non-quote character.
                    if let Some((fence_char, fence_len)) =
                        crate::markdown::scanners::opening_fence(&logical[info.content_start..])
                    {
                        fence = Some((fence_char, fence_len, info.content_indent, quote_depth));
                    }
                    previous_nonblank = !logical.trim().is_empty();
                }
            } else if !active.is_empty() {
                let blank = logical.trim().is_empty();
                let current = *active.last().unwrap();
                let indent = leading_indent(logical);
                let indented_continuation = indent >= items[current].content_indent;
                let quoted_continuation =
                    item_quote_depth[current] > 0 && quote_depth >= item_quote_depth[current];
                let lazy_continuation =
                    !blank && previous_nonblank && !interrupts_lazy_paragraph(logical);
                if blank {
                    // Blank lines only count once later content shows the item goes on
                    // (the extension then covers them); trailing blanks stay outside.
                    previous_nonblank = false;
                } else if indented_continuation || quoted_continuation || lazy_continuation {
                    for &idx in &active {
                        items[idx].end_byte = line_end;
                        items[idx].end_line = line_number + 1;
                    }
                    previous_nonblank = true;
                } else {
                    active.clear();
                    previous_nonblank = false;
                }
            } else {
                previous_nonblank = !logical.trim().is_empty();
            }
            byte_offset = line_end;
            line_number += 1;
        }

        items
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_nested_lists_and_parent_ranges() {
            let source = "- outer\n  - inner\n    1. deep\n  - sibling\n- second\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 5);
            assert_eq!(
                items.iter().map(|i| i.depth).collect::<Vec<_>>(),
                vec![0, 1, 2, 1, 0]
            );
            assert_eq!(items[1].parent, Some(0));
            assert_eq!(items[2].parent, Some(1));
            assert_eq!(items[3].parent, Some(0));
            assert_eq!(
                &source[items[0].start_byte..items[0].end_byte],
                "- outer\n  - inner\n    1. deep\n  - sibling\n"
            );
        }

        #[test]
        fn includes_indented_and_lazy_continuation_lines() {
            let source = "- first paragraph\n  continued indented\nand lazy continuation\n- next\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 2);
            assert_eq!(
                &source[items[0].start_byte..items[0].end_byte],
                "- first paragraph\n  continued indented\nand lazy continuation\n"
            );
        }

        #[test]
        fn handles_ordered_delimiters_tabs_and_unicode_ranges() {
            // A leading tab expands to four columns and cannot start a top-level list.
            assert!(parse_list_items("\t12) öłe\n").is_empty());

            // A tab after an ordered marker is valid padding. Keep this isolated
            // from the nested-list assertion so failures identify the exact rule.
            let tabbed = "12)\töłe\n";
            let tabbed_items = parse_list_items(tabbed);
            assert_eq!(tabbed_items.len(), 1);
            assert_eq!(
                tabbed_items[0].marker,
                ListMarker::Ordered {
                    delimiter: ')',
                    number: 12
                }
            );
            assert!(tabbed.is_char_boundary(tabbed_items[0].start_byte));
            assert!(tabbed.is_char_boundary(tabbed_items[0].end_byte));

            // Nested indentation and UTF-8 byte ranges are checked independently.
            let source = "éłik\n12) öłe\n    - alt\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].depth, 0);
            assert_eq!(items[1].depth, 1);
            assert!(source.is_char_boundary(items[0].start_byte));
            assert!(source.is_char_boundary(items[0].end_byte));
        }

        #[test]
        fn commonmark_nested_markers_and_blockquote_containers() {
            // Markers may be nested on the same physical line.
            let items = parse_list_items("- - foo\n");
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].depth, 0);
            assert_eq!(items[1].depth, 1);
            assert_eq!(items[1].parent, Some(0));

            let items = parse_list_items("1. - 2. foo\n");
            assert_eq!(items.len(), 3);
            assert_eq!(
                items.iter().map(|item| item.depth).collect::<Vec<_>>(),
                vec![0, 1, 2]
            );

            // Lists can begin inside block quotes and retain their container across
            // quoted blank lines and lazy continuation lines.
            let items = parse_list_items("> - foo\n- bar\n");
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].start_byte, 0);

            let items = parse_list_items("   > > 1.  one\n>>\n>>     two\n");
            assert_eq!(items.len(), 1);
            let items = parse_list_items(">>- one\n>>\n  >  > two\n");
            assert_eq!(items.len(), 1);

            let items = parse_list_items("> 1. > Blockquote\ncontinued here.\n");
            assert_eq!(items.len(), 1);
            let items = parse_list_items("> 1. > Blockquote\n> continued here.\n");
            assert_eq!(items.len(), 1);
        }

        #[test]
        fn list_item_fence_is_detected_after_marker_content() {
            let source = "- a\n- ```\n  b\n\n\n  ```\n- c\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 3);
            assert_eq!(&source[items[2].start_byte..items[2].end_byte], "- c\n");
        }

        #[test]
        fn wide_marker_indent_four_is_lazy_continuation_not_dead_branch() {
            // `100. ` has content_indent 5, so a 4-column marker lookalike cannot
            // nest; right after a non-blank line it continues the paragraph.
            let source = "100. x\n    - y\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 1);
            assert_eq!(&source[items[0].start_byte..items[0].end_byte], source);

            // After a blank line the same text no longer belongs to the item.
            let source = "100. x\n\n    - y\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].end_line, 1);
        }

        #[test]
        fn trailing_blank_lines_are_excluded_but_interior_ones_kept() {
            let source = "- a\n\n- b\n\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 2);
            assert_eq!(&source[items[0].start_byte..items[0].end_byte], "- a\n");
            assert_eq!(&source[items[1].start_byte..items[1].end_byte], "- b\n");

            let source = "- a\n\n  more\n\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 1);
            assert_eq!(
                &source[items[0].start_byte..items[0].end_byte],
                "- a\n\n  more\n"
            );
        }

        #[test]
        fn fenced_code_inside_items_is_not_scanned_for_markers() {
            let source = "- item\n  ```\n  - not an item\n  ```\n- next\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 2);
            assert_eq!(
                &source[items[0].start_byte..items[0].end_byte],
                "- item\n  ```\n  - not an item\n  ```\n"
            );
            assert_eq!(items[1].depth, 0);
        }

        #[test]
        fn top_level_fence_hides_list_lookalikes() {
            assert!(parse_list_items("```\n- x\n- y\n```\n").is_empty());
            assert_eq!(parse_list_items("```\n- x\n```\n- real\n").len(), 1);
        }

        #[test]
        fn ordered_item_fence_uses_content_indent() {
            let source = "1. a\n\n   ```\n   - x\n   ```\n";
            let items = parse_list_items(source);
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].end_byte, source.len());
        }

        #[test]
        fn paragraph_interrupt_rules_prevent_false_positive_lists() {
            // Empty markers cannot interrupt an open paragraph.
            for source in ["foo\n*\n", "foo bar\n1.\n", "*foo bar\n*\n"] {
                assert!(
                    parse_list_items(source).is_empty(),
                    "unexpected list item for {source:?}"
                );
            }

            // An ordered list starting at 14 cannot interrupt a paragraph.
            let source = "The number of windows in my house is\n14.  The number of doors is 6.\n";
            assert!(parse_list_items(source).is_empty());

            // A bullet or an ordered list starting at 1 may interrupt a paragraph
            // when it has actual item content.
            assert_eq!(parse_list_items("paragraph\n- item\n").len(), 1);
            assert_eq!(parse_list_items("paragraph\n1. item\n").len(), 1);

            // Empty markers remain valid when they begin a block after a blank line.
            assert_eq!(parse_list_items("paragraph\n\n*\n").len(), 1);
            assert_eq!(parse_list_items("paragraph\n\n1.\n").len(), 1);
        }

        #[test]
        fn does_not_treat_marker_lookalikes_as_list_items() {
            let source = "-not a list\n1234567890. too many digits\ntext\n";
            assert!(parse_list_items(source).is_empty());
            assert!(parse_list_items("    - indented code, not a top-level list\n").is_empty());
            assert!(parse_list_items("- - -\n").is_empty());
        }
    }
}

#[cfg(test)]
mod front_matter_and_setext_tests {
    use super::*;

    fn parse(src: &str) -> Vec<Block> {
        let mut p = BlockParser::new();
        let mut out = Vec::new();
        for l in src.split_inclusive('\n') {
            p.push_line(l, &mut out).unwrap();
        }
        p.finish(&mut out);
        out
    }

    fn headers_of(b: &Block) -> Vec<&str> {
        b.headers.iter().map(|s| s.as_str()).collect()
    }

    #[test]
    fn yaml_front_matter_with_blank_line_is_one_metadata_block() {
        let blocks = parse("---\ntitle: x\n\ntags: [a]\n---\n\nBody text\n");
        assert_eq!(blocks.len(), 2, "{blocks:?}");
        assert!(blocks[0].text.starts_with("---") && blocks[0].text.contains("tags: [a]"));
        assert!(blocks[1].headers.is_empty());
        assert_eq!(blocks[1].text, "Body text");
    }

    #[test]
    fn toml_front_matter_is_metadata() {
        let blocks = parse("+++\ntitle = \"x\"\n+++\n\n# Real\n\nBody\n");
        let last = blocks.last().unwrap();
        assert_eq!(headers_of(last), vec!["Real"]);
        assert!(blocks[0].text.contains("title = "));
    }

    #[test]
    fn thematic_break_at_start_is_not_front_matter() {
        let blocks = parse("---\n\n# Title\n\nBody\n");
        let last = blocks.last().unwrap();
        assert_eq!(headers_of(last), vec!["Title"]);
        let blocks = parse("---\n# Title\nBody\n");
        assert_eq!(headers_of(blocks.last().unwrap()), vec!["Title"]);
    }

    #[test]
    fn unterminated_front_matter_does_not_swallow_the_document() {
        let mut doc = String::from("---\nkey: value\n");
        for i in 0..300 {
            doc.push_str(&format!("line {i}\n"));
        }
        doc.push_str("\n# After\n\nBody\n");
        let blocks = parse(&doc);
        assert_eq!(headers_of(blocks.last().unwrap()), vec!["After"]);
    }

    #[test]
    fn unterminated_front_matter_with_blank_line_does_not_swallow_heading() {
        let blocks = parse("---\nkey: v\n\n# Real\n\nMARK");
        assert_eq!(
            headers_of(blocks.last().unwrap()),
            vec!["Real"],
            "{blocks:?}"
        );
        assert_eq!(blocks.last().unwrap().text, "MARK");
    }

    #[test]
    fn short_setext_heading_still_works() {
        let blocks = parse("Title\n=====\n\nBody\n");
        assert_eq!(headers_of(blocks.last().unwrap()), vec!["Title"]);
        let blocks = parse("Sub\n---\n\nBody\n");
        assert_eq!(headers_of(blocks.last().unwrap()), vec!["Sub"]);
    }

    #[test]
    fn long_paragraph_before_underline_is_text_not_heading() {
        let long = "word ".repeat(100);
        let blocks = parse(&format!("{long}\n---\nafter\n"));
        assert!(blocks.iter().all(|b| b.headers.is_empty()), "{blocks:?}");
        let joined: String = blocks.iter().map(|b| b.text.as_str()).collect();
        assert!(joined.contains("word word") && joined.contains("after"));
    }

    #[test]
    fn three_line_paragraph_before_underline_is_text_not_heading() {
        let blocks = parse("one\ntwo\nthree\n---\n");
        assert!(blocks.iter().all(|b| b.headers.is_empty()), "{blocks:?}");
    }

    #[test]
    fn consecutive_empty_headings_are_preserved() {
        let blocks = parse("### Sub\n## Next\n\nBody\n");
        assert_eq!(blocks.len(), 2, "{blocks:?}");
        assert_eq!(blocks[0].text, "### Sub");
        assert_eq!(headers_of(&blocks[0]), vec!["Sub"]);
        assert_eq!(blocks[1].text, "Body");
        assert_eq!(headers_of(&blocks[1]), vec!["Next"]);
    }
}
