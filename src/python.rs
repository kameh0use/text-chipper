use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use super::{
    context_prefix, smart_chunk_mapped, Backend, ChipError, Chunk, ChunkConfig, FileChunker,
    Normalization, OrigSpan,
};

// PyO3 bindings: the Python-facing API.

/// Raw per-file result: the chunks plus their spans in the original input.
type RawFileChunks = (Vec<Chunk>, Vec<OrigSpan>);

/// Parallel `items.iter().map(f).collect::<Result<Vec<_>, _>>()` with the sequential error
/// semantics: when several items fail, the error of the FIRST failing item (in input order)
/// is returned. rayon's own `collect::<Result<_, _>>()` returns an arbitrary one of them,
/// which would make the reported error depend on thread scheduling.
pub(crate) fn par_map_ordered<T, R, F>(items: &[T], f: F) -> Result<Vec<R>, ChipError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, ChipError> + Sync + Send,
{
    let results: Vec<Result<R, ChipError>> = items.par_iter().map(f).collect();
    results.into_iter().collect()
}

fn to_pyerr(e: ChipError) -> PyErr {
    match e {
        ChipError::Io(e) => PyErr::from(e),
        ChipError::Backend(m) => PyRuntimeError::new_err(m),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn parse_errors(errors: &str) -> PyResult<bool> {
    match errors {
        "strict" => Ok(false),
        "replace" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "errors must be 'strict' or 'replace' (got '{other}')"
        ))),
    }
}

/// One chunk of text with its position in the normalized input and in the original input.
///
/// `start_index`/`end_index` are character offsets and `start_byte`/`end_byte` are UTF-8 byte
/// offsets into the text *after* normalization; the `orig_*` fields map the same span back to
/// the string you passed in. `contextualized_text` is `text` prefixed with the header context.
#[pyclass(module = "text_chipper", frozen, get_all)]
#[derive(Clone, Debug)]
pub struct TextChunk {
    pub text: String,
    pub start_index: usize,
    pub end_index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    /// Offsets in the original input before NFC/NFKC normalization.
    pub orig_start_index: usize,
    pub orig_end_index: usize,
    pub orig_start_byte: usize,
    pub orig_end_byte: usize,
    /// Token count of the delivered text (header context included). Exact when `strict=True`
    /// (the default); in non-strict mode it may be an estimate.
    pub token_count: usize,
    /// Section headings that contain this chunk (outermost first). Always empty when
    /// `include_context=False`.
    pub headers: Vec<String>,
}

impl From<(Chunk, OrigSpan)> for TextChunk {
    fn from((c, o): (Chunk, OrigSpan)) -> Self {
        TextChunk {
            text: c.text,
            start_index: c.start_char,
            end_index: c.end_char,
            start_byte: c.start_byte,
            end_byte: c.end_byte,
            orig_start_index: o.start_char,
            orig_end_index: o.end_char,
            orig_start_byte: o.start_byte,
            orig_end_byte: o.end_byte,
            token_count: c.token_count,
            headers: c.headers.as_ref().clone(),
        }
    }
}

fn convert(chunks: Vec<Chunk>, spans: Vec<OrigSpan>) -> Vec<TextChunk> {
    chunks.into_iter().zip(spans).map(TextChunk::from).collect()
}

#[pymethods]
impl TextChunk {
    #[getter]
    fn context(&self) -> String {
        self.headers.join(" > ")
    }

    #[getter]
    fn contextualized_text(&self) -> String {
        if self.headers.is_empty() {
            self.text.clone()
        } else {
            format!("{}{}", context_prefix(&self.headers), self.text)
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("text", &self.text)?;
        d.set_item("start_index", self.start_index)?;
        d.set_item("end_index", self.end_index)?;
        d.set_item("start_byte", self.start_byte)?;
        d.set_item("end_byte", self.end_byte)?;
        d.set_item("orig_start_index", self.orig_start_index)?;
        d.set_item("orig_end_index", self.orig_end_index)?;
        d.set_item("orig_start_byte", self.orig_start_byte)?;
        d.set_item("orig_end_byte", self.orig_end_byte)?;
        d.set_item("token_count", self.token_count)?;
        d.set_item("headers", &self.headers)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(60).collect();
        let ellipsis = if self.text.chars().count() > 60 {
            "…"
        } else {
            ""
        };
        format!(
            "TextChunk(text={:?}, start={}, end={}, tokens={})",
            format!("{preview}{ellipsis}"),
            self.start_index,
            self.end_index,
            self.token_count
        )
    }
}

#[pyclass(module = "text_chipper")]
pub struct TextStreamIter {
    inner: Mutex<FileChunker>,
}

#[pymethods]
impl TextStreamIter {
    /// Peak number of normalization mapping segments retained during this stream.
    #[getter]
    fn peak_norm_map_segments(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_norm_map_segments()
    }

    /// Current number of normalization mapping segments retained in memory.
    #[getter]
    fn current_norm_map_segments(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .current_norm_map_segments()
    }

    /// High-water bytes held by the parser's currently open block.
    #[getter]
    fn peak_parser_open_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_parser_open_bytes()
    }

    /// High-water bytes held by pending body text plus deferred heading prefix.
    #[getter]
    fn peak_pending_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_pending_bytes()
    }

    /// High-water bytes held by parser separator/heading gaps.
    #[getter]
    fn peak_parser_gap_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_parser_gap_bytes()
    }

    /// Largest raw input line read by the streaming reader.
    #[getter]
    fn peak_input_line_bytes(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .peak_input_line_bytes()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<TextChunk>> {
        let next = py
            .detach(|| {
                let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                guard.next_chunk()
            })
            .map_err(to_pyerr)?;
        Ok(next.map(TextChunk::from))
    }
}

/// Token-budgeted chunker for Markdown and plain text.
///
/// Args:
///     max_length: token budget per chunk, header context included (0 = no limit).
///     overlap: tokens repeated between consecutive chunks when text must be split; must be
///         smaller than `max_length`.
///     encoding: `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, `chars` or `words`.
///         Ignored when `tokenizer_file` is given.
///     normalize: `"nfc"` (default), `"nfkc"` or `"none"`. Offsets are mapped back to the input.
///     snap_words: prefer word/sentence boundaries when a hard split is needed.
///     include_context: `True` (default) parses Markdown: headings become `headers` and a
///         `[ctx:...] ` prefix that counts against the budget; fenced code blocks and lists
///         stay intact when they fit (tables, block quotes and HTML blocks get no special
///         handling). `False` is a flat mode: Markdown is NOT parsed, chunks
///         are plain token windows (fences/lists/tables can be cut) and `headers` is empty.
///         `from_file*` gives the same chunks as `chunk()` on the file's text; in this mode
///         the whole file is read, and files larger than 32 MiB raise `OSError`.
///     merge_blocks: pack adjacent Markdown blocks into one chunk while they fit.
///     strict: `True` (default) re-counts every delivered chunk and guarantees
///         `token_count <= max_length`; a single grapheme that does not fit raises
///         `ValueError`. `False` skips that check (faster, no guarantee, `token_count` may be
///         an estimate; an oversized grapheme is returned as its own chunk).
///     tokenizer_file: path to a Hugging Face `tokenizer.json` (needs a build with the `hf`
///         feature).
///
/// All errors are raised as exceptions; no method returns an empty or partial result for
/// invalid input.
#[pyclass(module = "text_chipper")]
pub struct TextChipper {
    cfg: ChunkConfig,
    backend: Arc<Backend>,
    encoding: String,
}

impl TextChipper {
    fn process_file(
        &self,
        path: &str,
        lossy: bool,
    ) -> Result<(Vec<Chunk>, Vec<OrigSpan>), ChipError> {
        // Collects every chunk of the file. With `include_context` the source is streamed,
        // otherwise it is read whole (see `FileChunker`). Callers that also need
        // bounded *output* memory should use `from_file_stream()`.
        let mut stream = FileChunker::open(path, &self.cfg, &self.backend, lossy)?;
        let mut chunks = Vec::new();
        let mut spans = Vec::new();
        while let Some((chunk, span)) = stream.next_chunk()? {
            chunks.push(chunk);
            spans.push(span);
        }
        Ok((chunks, spans))
    }
}

#[pymethods]
impl TextChipper {
    #[new]
    #[pyo3(signature = (
        max_length=512,
        overlap=50,
        encoding="cl100k_base",
        *,
        normalize="nfc",
        snap_words=true,
        include_context=true,
        merge_blocks=true,
        strict=true,
        tokenizer_file=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        max_length: usize,
        overlap: usize,
        encoding: &str,
        normalize: &str,
        snap_words: bool,
        include_context: bool,
        merge_blocks: bool,
        strict: bool,
        tokenizer_file: Option<&str>,
    ) -> PyResult<Self> {
        let cfg = ChunkConfig {
            max_tokens: max_length,
            overlap_tokens: overlap,
            normalization: Normalization::parse(normalize).map_err(to_pyerr)?,
            snap_words,
            include_context,
            merge_blocks,
            strict,
        };
        cfg.validate().map_err(to_pyerr)?;

        // BPE tables take a while to build: do it without holding the GIL.
        let (backend, encoding) = py
            .detach(|| -> Result<(Arc<Backend>, String), ChipError> {
                match tokenizer_file {
                    Some(path) => Ok((Backend::from_tokenizer_file(path)?, format!("hf:{path}"))),
                    None => Ok((Backend::from_name(encoding)?, encoding.to_string())),
                }
            })
            .map_err(to_pyerr)?;

        Ok(TextChipper {
            cfg,
            backend,
            encoding,
        })
    }

    #[getter]
    fn max_length(&self) -> usize {
        self.cfg.max_tokens
    }

    #[getter]
    fn overlap(&self) -> usize {
        self.cfg.overlap_tokens
    }

    #[getter]
    fn encoding(&self) -> String {
        self.encoding.clone()
    }

    #[getter]
    fn strict(&self) -> bool {
        self.cfg.strict
    }

    /// Apply the configured Unicode normalization to `text`.
    fn normalize(&self, text: &str) -> String {
        self.cfg.normalization.apply(text).into_owned()
    }

    /// Number of tokens in `text` under the configured tokenizer.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        py.detach(|| self.backend.count(text))
    }

    /// Split `text` into chunks. Raises `ValueError` if no valid chunking exists (strict mode,
    /// a single grapheme exceeds `max_length`).
    fn chunk(&self, py: Python<'_>, text: &str) -> PyResult<Vec<TextChunk>> {
        let (chunks, spans) = py
            .detach(|| smart_chunk_mapped(text, &self.cfg, &self.backend))
            .map_err(to_pyerr)?;
        Ok(convert(chunks, spans))
    }

    /// Chunk several texts (in parallel when the batch is large enough). The result has one
    /// list per input, in input order. If any text fails, the exception for the FIRST failing
    /// text (in input order) is raised and no partial result is returned.
    fn chunk_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<TextChunk>>> {
        let total_bytes: usize = texts.iter().map(String::len).sum();
        let raw: Vec<(Vec<Chunk>, Vec<OrigSpan>)> = py
            .detach(|| {
                // Rayon scheduling is counterproductive for tiny batches. Use it only once
                // there are enough independent bytes to amortize the parallel work-stealing cost.
                if texts.len() < 4 || total_bytes < 64 * 1024 {
                    texts
                        .iter()
                        .map(|t| smart_chunk_mapped(t, &self.cfg, &self.backend))
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    par_map_ordered(&texts, |t| smart_chunk_mapped(t, &self.cfg, &self.backend))
                }
            })
            .map_err(to_pyerr)?;
        Ok(raw
            .into_iter()
            .map(|(c, spans)| convert(c, spans))
            .collect())
    }

    /// Chunk a UTF-8 text file. `errors="strict"` raises `OSError` on invalid UTF-8;
    /// `errors="replace"` substitutes U+FFFD.
    #[pyo3(signature = (filepath, errors="strict"))]
    // `from_*` is the public Python API name, so the Rust `from_*`-takes-no-self convention does not apply.
    #[allow(clippy::wrong_self_convention)]
    fn from_file(&self, py: Python<'_>, filepath: &str, errors: &str) -> PyResult<Vec<TextChunk>> {
        let lossy = parse_errors(errors)?;
        let (chunks, spans) = py
            .detach(|| self.process_file(filepath, lossy))
            .map_err(to_pyerr)?;
        Ok(convert(chunks, spans))
    }

    /// Chunk several files in parallel. Returns one list per file, in input order. If any file
    /// fails, the error of the FIRST failing file (in input order) is raised.
    #[pyo3(signature = (filepaths, errors="strict"))]
    // `from_*` is the public Python API name, so the Rust `from_*`-takes-no-self convention does not apply.
    #[allow(clippy::wrong_self_convention)]
    fn from_files(
        &self,
        py: Python<'_>,
        filepaths: Vec<String>,
        errors: &str,
    ) -> PyResult<Vec<Vec<TextChunk>>> {
        let lossy = parse_errors(errors)?;
        let raw: Result<Vec<RawFileChunks>, ChipError> =
            py.detach(|| par_map_ordered(&filepaths, |p| self.process_file(p, lossy)));
        Ok(raw
            .map_err(to_pyerr)?
            .into_iter()
            .map(|(c, spans)| convert(c, spans))
            .collect())
    }

    /// Iterate over the chunks of a file without collecting them. `batch_size_lines` is
    /// accepted for backwards compatibility and ignored. Errors (I/O, invalid UTF-8 with
    /// `errors="strict"`, over-long line) are raised from `__next__`.
    #[pyo3(signature = (filepath, batch_size_lines=1000, errors="strict"))]
    // `from_*` is the public Python API name, so the Rust `from_*`-takes-no-self convention does not apply.
    #[allow(clippy::wrong_self_convention)]
    fn from_file_stream(
        &self,
        filepath: &str,
        batch_size_lines: usize,
        errors: &str,
    ) -> PyResult<TextStreamIter> {
        let _ = batch_size_lines; // accepted for backwards compatibility; has no effect
        let lossy = parse_errors(errors)?;
        let inner =
            FileChunker::open(filepath, &self.cfg, &self.backend, lossy).map_err(to_pyerr)?;
        Ok(TextStreamIter {
            inner: Mutex::new(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TextChipper(max_length={}, overlap={}, encoding='{}')",
            self.cfg.max_tokens, self.cfg.overlap_tokens, self.encoding
        )
    }
}

#[cfg(feature = "stats")]
#[pyfunction]
fn _reset_tokenizer_stats() {
    super::backend::reset_count_stats();
}

#[cfg(feature = "stats")]
#[pyfunction]
fn _tokenizer_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    let sites = PyDict::new(py);
    let mut total_calls = 0u64;
    let mut total_bytes = 0u64;
    let mut total_nanos = 0u64;
    for (name, calls, bytes, nanos) in super::backend::count_stats() {
        let item = PyDict::new(py);
        item.set_item("calls", calls)?;
        item.set_item("input_bytes", bytes)?;
        item.set_item("elapsed_ms", nanos as f64 / 1_000_000.0)?;
        sites.set_item(name, item)?;
        total_calls += calls;
        total_bytes += bytes;
        total_nanos += nanos;
    }
    result.set_item("calls", total_calls)?;
    result.set_item("input_bytes", total_bytes)?;
    result.set_item("elapsed_ms", total_nanos as f64 / 1_000_000.0)?;
    result.set_item("sites", sites)?;
    Ok(result)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TextChunk>()?;
    m.add_class::<TextStreamIter>()?;
    m.add_class::<TextChipper>()?;
    #[cfg(feature = "stats")]
    {
        m.add_function(wrap_pyfunction!(_reset_tokenizer_stats, m)?)?;
        m.add_function(wrap_pyfunction!(_tokenizer_stats, m)?)?;
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
