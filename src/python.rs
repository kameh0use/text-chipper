use std::path::Path;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyType};

use crate::parallel::par_map_ordered;

use super::{
    context_prefix, smart_chunk_mapped, Backend, ChipError, Chunk, ChunkConfig, FileChunker,
    Normalization, OrigSpan, Tokenizer,
};

// PyO3 bindings: the Python-facing API.

/// Raw per-file result: the chunks plus their spans in the original input.
type RawFileChunks = (Vec<Chunk>, Vec<OrigSpan>);

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

#[derive(Clone, Copy)]
enum PyTokMode {
    Callable,
    EncodeNoSpecial,
    EncodePlain,
}

/// Thread-safe wrapper around a Python callable `fn(str) -> int` or an object with `.encode(str)`.
struct PyTokenizer {
    obj: Py<PyAny>,
    mode: PyTokMode,
}

impl PyTokenizer {
    fn call_py(&self, py: Python<'_>, text: &str) -> PyResult<usize> {
        let bound = self.obj.bind(py);
        match self.mode {
            PyTokMode::Callable => bound.call1((text,))?.extract::<usize>(),
            PyTokMode::EncodeNoSpecial => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("add_special_tokens", false)?;
                let ids = bound.call_method("encode", (text,), Some(&kwargs))?;
                ids.len()
            }
            PyTokMode::EncodePlain => {
                let ids = bound.call_method1("encode", (text,))?;
                ids.len()
            }
        }
    }
}

impl Tokenizer for PyTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        Python::attach(|py| {
            self.call_py(py, text)
                .unwrap_or_else(|_| text.chars().count())
        })
    }
}

fn is_builtin_encoding(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "cl100k_base" | "o200k_base" | "p50k_base" | "r50k_base" | "chars" | "words"
    )
}

/// Resolve a Python `tokenizer` argument into a native Rust `Backend` whenever possible
/// (extracting the JSON representation from Hugging Face `tokenizers.Tokenizer` or
/// `transformers.PreTrainedTokenizerFast` for zero-GIL native speed), or falling back to
/// `PyTokenizer` for custom Python callables and `.encode()` objects.
fn resolve_py_tokenizer(obj: &Bound<'_, PyAny>) -> PyResult<(Arc<Backend>, String)> {
    // 1. String or os.PathLike (built-in encoding name, tokenizer.json file, or model directory).
    let maybe_str: Option<String> = if obj.is_instance_of::<PyString>() {
        Some(obj.extract::<String>()?)
    } else if obj.hasattr("__fspath__")? {
        Some(obj.call_method0("__fspath__")?.extract::<String>()?)
    } else {
        None
    };
    if let Some(s) = maybe_str {
        if is_builtin_encoding(&s) {
            let b = Backend::from_name(&s).map_err(to_pyerr)?;
            return Ok((b, s));
        }
        let path = Path::new(&s);
        if path.is_dir() {
            let candidate = path.join("tokenizer.json");
            if candidate.is_file() {
                let p_str = candidate.to_string_lossy().into_owned();
                let b = Backend::from_tokenizer_file(&p_str).map_err(to_pyerr)?;
                return Ok((b, format!("hf:{p_str}")));
            }
        }
        let b = Backend::from_tokenizer_file(&s).map_err(to_pyerr)?;
        return Ok((b, format!("hf:{s}")));
    }

    // 2. Hugging Face `transformers.PreTrainedTokenizerFast` (has `.backend_tokenizer.to_str()`).
    if obj.hasattr("backend_tokenizer")? {
        let bt = obj.getattr("backend_tokenizer")?;
        if bt.hasattr("to_str")? {
            let json_str: String = bt.call_method0("to_str")?.extract()?;
            let label = obj
                .getattr("name_or_path")
                .and_then(|v| v.extract::<String>())
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "PreTrainedTokenizerFast".to_string());
            let b = Backend::from_tokenizer_bytes(json_str.as_bytes()).map_err(to_pyerr)?;
            return Ok((b, format!("hf:{label}")));
        }
    }

    // 3. Hugging Face `tokenizers.Tokenizer` (has `.to_str()`).
    if obj.hasattr("to_str")? {
        if let Ok(json_str) = obj
            .call_method0("to_str")
            .and_then(|v| v.extract::<String>())
        {
            let b = Backend::from_tokenizer_bytes(json_str.as_bytes()).map_err(to_pyerr)?;
            return Ok((b, "hf:Tokenizer".to_string()));
        }
    }

    // 4. An object with `.encode(text)` (e.g. tiktoken.Encoding, SentencePiece, slow HF tokenizer).
    if obj.hasattr("encode")? {
        let py = obj.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("add_special_tokens", false)?;
        let mode = if obj
            .call_method("encode", ("hello",), Some(&kwargs))
            .and_then(|ids| ids.len())
            .is_ok()
        {
            Some(PyTokMode::EncodeNoSpecial)
        } else if obj
            .call_method1("encode", ("hello",))
            .and_then(|ids| ids.len())
            .is_ok()
        {
            Some(PyTokMode::EncodePlain)
        } else {
            None
        };
        if let Some(mode) = mode {
            let ty_name = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "encode".to_string());
            let tok = PyTokenizer {
                obj: obj.clone().unbind(),
                mode,
            };
            return Ok((
                Backend::from_tokenizer(Arc::new(tok)),
                format!("custom:{ty_name}"),
            ));
        }
    }

    // 5. A Python callable `fn(text: str) -> int`.
    if obj.is_callable() {
        let test_res = obj.call1(("hello",))?;
        let _ = test_res.extract::<usize>().map_err(|_| {
            PyTypeError::new_err(
                "custom tokenizer callable must return a non-negative integer token count",
            )
        })?;
        let tok = PyTokenizer {
            obj: obj.clone().unbind(),
            mode: PyTokMode::Callable,
        };
        return Ok((
            Backend::from_tokenizer(Arc::new(tok)),
            "custom:callable".to_string(),
        ));
    }

    Err(PyTypeError::new_err(
        "tokenizer must be a Hugging Face Tokenizer, PreTrainedTokenizer, path to tokenizer.json,          built-in encoding name, object with .encode(text), or callable(text) -> int",
    ))
}

/// Resolve a pretrained model name, local directory, `tokenizer.json` path, or tokenizer object.
fn resolve_pretrained_backend(
    py: Python<'_>,
    model_name_or_path: &Bound<'_, PyAny>,
) -> PyResult<(Arc<Backend>, String)> {
    let maybe_str: Option<String> = if model_name_or_path.is_instance_of::<PyString>() {
        Some(model_name_or_path.extract::<String>()?)
    } else if model_name_or_path.hasattr("__fspath__")? {
        Some(
            model_name_or_path
                .call_method0("__fspath__")?
                .extract::<String>()?,
        )
    } else {
        None
    };

    let Some(s) = maybe_str else {
        return resolve_py_tokenizer(model_name_or_path);
    };

    if is_builtin_encoding(&s) {
        let b = Backend::from_name(&s).map_err(to_pyerr)?;
        return Ok((b, s));
    }

    let path = Path::new(&s);
    if path.is_file() {
        let b = Backend::from_tokenizer_file(&s).map_err(to_pyerr)?;
        return Ok((b, format!("hf:{s}")));
    }
    if path.is_dir() {
        let candidate = path.join("tokenizer.json");
        if candidate.is_file() {
            let p_str = candidate.to_string_lossy().into_owned();
            let b = Backend::from_tokenizer_file(&p_str).map_err(to_pyerr)?;
            return Ok((b, format!("hf:{s}")));
        }
    }

    // Try `huggingface_hub.hf_hub_download(repo_id=s, filename="tokenizer.json")`.
    if let Ok(hf_hub) = py.import("huggingface_hub") {
        if let Ok(downloaded) = hf_hub
            .call_method1("hf_hub_download", (s.as_str(), "tokenizer.json"))
            .and_then(|v| v.extract::<String>())
        {
            let b = Backend::from_tokenizer_file(&downloaded).map_err(to_pyerr)?;
            return Ok((b, format!("hf:{s}")));
        }
    }

    // Try `tokenizers.Tokenizer.from_pretrained(s)`.
    if let Ok(tok_mod) = py.import("tokenizers") {
        if let Ok(tok_cls) = tok_mod.getattr("Tokenizer") {
            if let Ok(tok_obj) = tok_cls.call_method1("from_pretrained", (s.as_str(),)) {
                let (b, _) = resolve_py_tokenizer(&tok_obj)?;
                return Ok((b, format!("hf:{s}")));
            }
        }
    }

    // Try `transformers.AutoTokenizer.from_pretrained(s)`.
    if let Ok(tr_mod) = py.import("transformers") {
        if let Ok(auto_cls) = tr_mod.getattr("AutoTokenizer") {
            if let Ok(tok_obj) = auto_cls.call_method1("from_pretrained", (s.as_str(),)) {
                let (b, _) = resolve_py_tokenizer(&tok_obj)?;
                return Ok((b, format!("hf:{s}")));
            }
        }
    }

    Err(PyRuntimeError::new_err(format!(
        "could not load pretrained tokenizer '{s}': pass a local path to tokenizer.json (or a          directory containing it), or install 'huggingface_hub', 'tokenizers', or 'transformers'"
    )))
}

/// One chunk of text with its position in the normalized input and in the original input.
///
/// `start_index`/`end_index` are character offsets and `start_byte`/`end_byte` are UTF-8 byte
/// offsets into the text *after* normalization; the `orig_*` fields map the same span back to
/// the string you passed in. `contextualized_text` is `text` prefixed with the header context
/// and, for continuation chunks of a split Markdown table, the repeated table header row.
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
    /// Token count of the delivered text (header context and repeated table header included).
    /// Exact when `strict=True` (the default); in non-strict mode it may be an estimate.
    pub token_count: usize,
    /// Section headings that contain this chunk (outermost first). Always empty when
    /// `include_context=False`.
    pub headers: Vec<String>,
    /// Repeated Markdown table header + delimiter lines when this chunk is a continuation slice
    /// of a split table (`None` for non-table chunks or the opening chunk of a table).
    pub table_header: Option<String>,
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
            table_header: c.table_header.as_deref().map(str::to_owned),
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
        match (self.headers.is_empty(), self.table_header.as_deref()) {
            (true, None) => self.text.clone(),
            (false, None) => format!("{}{}", context_prefix(&self.headers), self.text),
            (true, Some(th)) => format!("{th}\n{}", self.text),
            (false, Some(th)) => format!("{}{th}\n{}", context_prefix(&self.headers), self.text),
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
        d.set_item("table_header", &self.table_header)?;
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
        tokenizer=None,
        tokenizer_file=None,
        guard_encodings=None,
        guard_tokenizer_files=None,
        guard_tokenizers=None
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
        tokenizer: Option<&Bound<'_, PyAny>>,
        tokenizer_file: Option<&str>,
        guard_encodings: Option<Vec<String>>,
        guard_tokenizer_files: Option<Vec<String>>,
        guard_tokenizers: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        if tokenizer.is_some() && tokenizer_file.is_some() {
            return Err(PyValueError::new_err(
                "pass at most one of 'tokenizer' or 'tokenizer_file'",
            ));
        }
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

        let pre_primary = match tokenizer {
            Some(obj) => Some(resolve_py_tokenizer(obj)?),
            None => None,
        };
        let mut pre_guards = Vec::new();
        if let Some(objs) = guard_tokenizers.as_ref() {
            for obj in objs {
                pre_guards.push(resolve_py_tokenizer(obj)?);
            }
        }

        // BPE tables take a while to build: do it without holding the GIL.
        let (backend, encoding) = py
            .detach(|| -> Result<(Arc<Backend>, String), ChipError> {
                let (primary, mut label) = match (pre_primary, tokenizer_file) {
                    (Some(resolved), _) => resolved,
                    (None, Some(path)) => {
                        (Backend::from_tokenizer_file(path)?, format!("hf:{path}"))
                    }
                    (None, None) => (Backend::from_name(encoding)?, encoding.to_string()),
                };
                let mut parts = vec![primary];
                for name in guard_encodings.as_deref().unwrap_or(&[]) {
                    parts.push(Backend::from_name(name)?);
                    label.push_str(&format!("+{name}"));
                }
                for path in guard_tokenizer_files.as_deref().unwrap_or(&[]) {
                    parts.push(Backend::from_tokenizer_file(path)?);
                    label.push_str(&format!("+hf:{path}"));
                }
                for (gb, glabel) in pre_guards {
                    parts.push(gb);
                    label.push_str(&format!("+{glabel}"));
                }
                Ok((Backend::max_of(parts)?, label))
            })
            .map_err(to_pyerr)?;

        Ok(TextChipper {
            cfg,
            backend,
            encoding,
        })
    }

    /// Construct a `TextChipper` from a Hugging Face model ID (e.g. `"BAAI/bge-large-en-v1.5"`),
    /// a local directory or `tokenizer.json` file, or a pre-loaded tokenizer object.
    #[classmethod]
    #[pyo3(signature = (
        model_name_or_path,
        *,
        max_length=512,
        overlap=50,
        normalize="nfc",
        snap_words=true,
        include_context=true,
        merge_blocks=true,
        strict=true,
        guard_encodings=None,
        guard_tokenizer_files=None,
        guard_tokenizers=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_pretrained(
        cls: &Bound<'_, PyType>,
        model_name_or_path: &Bound<'_, PyAny>,
        max_length: usize,
        overlap: usize,
        normalize: &str,
        snap_words: bool,
        include_context: bool,
        merge_blocks: bool,
        strict: bool,
        guard_encodings: Option<Vec<String>>,
        guard_tokenizer_files: Option<Vec<String>>,
        guard_tokenizers: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let py = cls.py();
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

        let (primary, mut label) = resolve_pretrained_backend(py, model_name_or_path)?;
        let mut pre_guards = Vec::new();
        if let Some(objs) = guard_tokenizers.as_ref() {
            for obj in objs {
                pre_guards.push(resolve_py_tokenizer(obj)?);
            }
        }

        let (backend, encoding) = py
            .detach(|| -> Result<(Arc<Backend>, String), ChipError> {
                let mut parts = vec![primary];
                for name in guard_encodings.as_deref().unwrap_or(&[]) {
                    parts.push(Backend::from_name(name)?);
                    label.push_str(&format!("+{name}"));
                }
                for path in guard_tokenizer_files.as_deref().unwrap_or(&[]) {
                    parts.push(Backend::from_tokenizer_file(path)?);
                    label.push_str(&format!("+hf:{path}"));
                }
                for (gb, glabel) in pre_guards {
                    parts.push(gb);
                    label.push_str(&format!("+{glabel}"));
                }
                Ok((Backend::max_of(parts)?, label))
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
    #[allow(clippy::wrong_self_convention)]
    fn from_file_stream(
        &self,
        filepath: &str,
        batch_size_lines: usize,
        errors: &str,
    ) -> PyResult<TextStreamIter> {
        let _ = batch_size_lines;
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
