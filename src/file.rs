use std::borrow::Cow;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::Arc;

use super::{
    backend::Backend,
    chunk_types::{Chunk, ChunkConfig},
    chunker::smart_chunk_mapped_profile,
    chunker::SmartChunker,
    normalization::{NormBase, NormMap, Normalization, OrigSpan},
    split_engine::char_len,
    ChipError,
};

const BOM: char = '\u{feff}';
/// Largest file accepted with `include_context = false` (flat mode).
///
/// Flat mode is a whole-document algorithm (it needs global grapheme and cut tables), so
/// `FileChunker` reads the file whole and the result is identical to `smart_chunk` on the same
/// text, whatever the file size. Peak memory is roughly 10-12x the file size for ASCII text
/// (the grapheme-boundary table alone is 8 bytes per byte), and `from_files` may process
/// several files at once. Larger files are rejected with an explicit error instead of being
/// chunked by a different algorithm; use `include_context = true` (streamed, bounded memory)
/// or split the file first.
pub const FLAT_FILE_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Maximum physical line accepted by the streaming reader. Oversized lines fail
/// explicitly instead of growing the reusable input buffer without a bound.
pub(crate) const MAX_INPUT_LINE_BYTES: usize = 1024 * 1024;

fn read_line_bounded<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    max: usize,
) -> io::Result<usize> {
    loop {
        let (taken, newline, eof) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                (0, false, true)
            } else {
                let through_newline = available
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(available.len(), |index| index + 1);
                let remaining = max.saturating_add(1).saturating_sub(buf.len());
                let taken = through_newline.min(remaining);
                buf.extend_from_slice(&available[..taken]);
                (taken, buf.last() == Some(&b'\n'), false)
            }
        };
        reader.consume(taken);
        if buf.len() > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("physical input line exceeds the {max}-byte streaming limit"),
            ));
        }
        if eof || newline || taken == 0 {
            return Ok(buf.len());
        }
        // If the buffer reached the limit, the next iteration checks whether the
        // line actually ended at EOF/newline or whether one more byte exceeds it.
    }
}

fn invalid_utf8(path: &str, e: impl std::fmt::Display) -> ChipError {
    ChipError::Io(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid UTF-8 in '{path}': {e} (use errors=\"replace\" to substitute)"),
    ))
}

/// Read a complete file into one `String`. This convenience helper is intentionally
/// whole-file and is not the bounded-memory API; use `FileChunker` for streaming.
pub fn read_text_file(path: &str, lossy: bool) -> Result<String, ChipError> {
    let bytes = std::fs::read(path)?;
    let mut text = if lossy {
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        String::from_utf8(bytes).map_err(|e| invalid_utf8(path, e))?
    };
    if text.starts_with(BOM) {
        text.replace_range(..BOM.len_utf8(), "");
    }
    Ok(text)
}

/// Chunks a UTF-8 text file. Iterate with [`FileChunker::next_chunk`].
///
/// With `include_context = true` the file is streamed line by line (bounded memory; a single
/// physical line may not exceed 1 MiB). With `include_context = false` (flat mode) the file is
/// read whole and chunked exactly like [`crate::smart_chunk`] on the same text; files larger
/// than [`FLAT_FILE_MAX_BYTES`] fail with an `InvalidData` I/O error. Both modes therefore
/// give the same chunks for the same text regardless of file size.
pub struct FileChunker {
    path: String,
    reader: BufReader<File>,
    chunker: SmartChunker,
    norm: Normalization,
    lossy: bool,
    queue: VecDeque<Chunk>,
    fast_queue: Option<VecDeque<(Chunk, OrigSpan)>>,
    scratch: Vec<Chunk>,
    buf: Vec<u8>,
    first_line: bool,
    done: bool,
    map: NormMap,
    base: NormBase,
    peak_norm_map_segments: usize,
    peak_parser_open_bytes: usize,
    peak_pending_bytes: usize,
    peak_parser_gap_bytes: usize,
    peak_input_line_bytes: usize,
}

impl FileChunker {
    /// Open `path` for chunking. `lossy = true` replaces invalid UTF-8 with U+FFFD, otherwise
    /// invalid input is an `InvalidData` I/O error. A leading BOM is stripped.
    pub fn open(
        path: &str,
        cfg: &ChunkConfig,
        backend: &Arc<Backend>,
        lossy: bool,
    ) -> Result<Self, ChipError> {
        cfg.validate()?;
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let cfg = cfg.clone();
        let norm = cfg.normalization;
        let (fast_queue, fast_peak_norm_map_segments) = if !cfg.include_context {
            if file_len > FLAT_FILE_MAX_BYTES {
                return Err(ChipError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "'{path}' is {file_len} bytes; flat mode (include_context=false) reads the \
                         whole file and accepts at most {FLAT_FILE_MAX_BYTES} bytes. Use \
                         include_context=true to stream it, or split the file."
                    ),
                )));
            }
            let text = read_text_file(path, lossy)?;
            let (chunks, spans, peak_segments) = smart_chunk_mapped_profile(&text, &cfg, backend)?;
            (Some(chunks.into_iter().zip(spans).collect()), peak_segments)
        } else {
            (None, 0)
        };
        let streaming = fast_queue.is_none();
        let reader_capacity = if streaming { 1 << 20 } else { 8 * 1024 };
        Ok(FileChunker {
            path: path.to_string(),
            reader: BufReader::with_capacity(reader_capacity, file),
            chunker: SmartChunker::new(cfg, Arc::clone(backend)),
            norm,
            lossy,
            queue: VecDeque::new(),
            fast_queue,
            scratch: Vec::new(),
            // Reserve the exact configured maximum up front so Vec's geometric
            // growth cannot silently double the line-buffer capacity.
            buf: if streaming {
                Vec::with_capacity(MAX_INPUT_LINE_BYTES + 1)
            } else {
                Vec::new()
            },
            first_line: true,
            done: false,
            map: NormMap::default(),
            base: NormBase::default(),
            peak_norm_map_segments: fast_peak_norm_map_segments,
            peak_parser_open_bytes: 0,
            peak_pending_bytes: 0,
            peak_parser_gap_bytes: 0,
            peak_input_line_bytes: 0,
        })
    }

    pub(crate) fn peak_norm_map_segments(&self) -> usize {
        self.peak_norm_map_segments
    }
    pub(crate) fn current_norm_map_segments(&self) -> usize {
        self.map.segs.len()
    }
    pub(crate) fn peak_parser_open_bytes(&self) -> usize {
        self.peak_parser_open_bytes
    }
    pub(crate) fn peak_pending_bytes(&self) -> usize {
        self.peak_pending_bytes
    }
    pub(crate) fn peak_parser_gap_bytes(&self) -> usize {
        self.peak_parser_gap_bytes
    }
    pub(crate) fn peak_input_line_bytes(&self) -> usize {
        self.peak_input_line_bytes
    }

    /// Next chunk and its span in the original text, `Ok(None)` at the end of the file.
    /// After an error the iterator is finished.
    pub fn next_chunk(&mut self) -> Result<Option<(Chunk, OrigSpan)>, ChipError> {
        if let Some(fast) = self.fast_queue.as_mut() {
            return Ok(fast.pop_front());
        }
        loop {
            if let Some(c) = self.queue.pop_front() {
                // Resolve the original span before pruning. Only prune after every chunk
                // already produced by this parser event has been consumed; then the
                // chunker's watermark covers overlap, pending merges and parser gaps.
                let span = self.map.span(&c);
                if self.queue.is_empty() {
                    let (safe_byte, safe_char) = if self.done {
                        (self.base.nb, self.base.nc)
                    } else {
                        self.chunker.earliest_future_start()
                    };
                    self.map.prune(safe_byte, safe_char);
                }
                return Ok(Some((c, span)));
            }
            if self.done {
                return Ok(None);
            }
            self.buf.clear();
            let n = match read_line_bounded(&mut self.reader, &mut self.buf, MAX_INPUT_LINE_BYTES) {
                Ok(n) => n,
                Err(error) => {
                    self.peak_input_line_bytes = self.peak_input_line_bytes.max(self.buf.len());
                    self.done = true;
                    return Err(error.into());
                }
            };
            self.peak_input_line_bytes = self.peak_input_line_bytes.max(n);
            if n == 0 {
                self.done = true;
                if let Err(error) = self.chunker.finish(&mut self.scratch) {
                    self.done = true;
                    return Err(error);
                }
                self.peak_parser_open_bytes = self
                    .peak_parser_open_bytes
                    .max(self.chunker.parser.open_bytes());
                self.peak_pending_bytes = self
                    .peak_pending_bytes
                    .max(self.chunker.peak_pending_bytes());
                self.peak_parser_gap_bytes = self
                    .peak_parser_gap_bytes
                    .max(self.chunker.parser.gap_bytes());
            } else {
                // Valid UTF-8 input stays borrowed from the reusable read buffer; a String is
                // allocated only when lossy decoding or normalization changes the text.
                let decoded: Cow<'_, str> = if self.lossy {
                    Cow::Owned(String::from_utf8_lossy(&self.buf).into_owned())
                } else {
                    match std::str::from_utf8(&self.buf) {
                        Ok(text) => Cow::Borrowed(text),
                        Err(error) => {
                            self.done = true;
                            return Err(invalid_utf8(&self.path, error));
                        }
                    }
                };
                let line = if self.first_line {
                    self.first_line = false;
                    match decoded {
                        Cow::Borrowed(s) => Cow::Borrowed(s.strip_prefix(BOM).unwrap_or(s)),
                        Cow::Owned(mut s) => {
                            if s.starts_with(BOM) {
                                s.drain(..BOM.len_utf8());
                            }
                            Cow::Owned(s)
                        }
                    }
                } else {
                    decoded
                };
                let owned = match self
                    .norm
                    .apply_mapped(line.as_ref(), self.base, &mut self.map)
                {
                    Cow::Owned(s) => Some(s),
                    Cow::Borrowed(_) => None,
                };
                self.peak_norm_map_segments = self.peak_norm_map_segments.max(self.map.segs.len());
                let normalized: &str = owned.as_deref().unwrap_or(line.as_ref());
                self.base.ob += line.len();
                self.base.oc += char_len(line.as_ref());
                self.base.nb += normalized.len();
                self.base.nc += char_len(normalized);
                if let Err(error) = self.chunker.push_line(normalized, &mut self.scratch) {
                    self.done = true;
                    return Err(error);
                }
                self.peak_parser_open_bytes = self
                    .peak_parser_open_bytes
                    .max(self.chunker.parser.open_bytes());
                self.peak_pending_bytes = self
                    .peak_pending_bytes
                    .max(self.chunker.peak_pending_bytes());
                self.peak_parser_gap_bytes = self
                    .peak_parser_gap_bytes
                    .max(self.chunker.parser.gap_bytes());
            }
            self.queue.extend(self.scratch.drain(..));
        }
    }
}
