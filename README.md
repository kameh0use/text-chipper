# text_chipper

<img align="left" hspace="20" vspace="20" src="https://raw.githubusercontent.com/devel0mer/text-chipper/main/docs/logo.png" width="140" alt="text_chipper logo">

**High-performance, Markdown- and table-aware, strictly budget-enforced text chunking for RAG and LLM pipelines.
Built in Rust with zero-dependency Python bindings.**

[![Rust Version](https://img.shields.io/badge/rust-1.88.0-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/text_chipper.svg)](https://crates.io/crates/text_chipper)
[![docs.rs](https://img.shields.io/docsrs/text_chipper)](https://docs.rs/text_chipper)
[![PyPI](https://img.shields.io/pypi/v/text-chipper.svg)](https://pypi.org/project/text-chipper/)

<br clear="left">

- **Token-exact budget (`token_count <= max_length` guaranteed).** In the default strict mode, every chunk is re-tokenized in the exact form it will be delivered (section heading prefix + repeated table header + body). No silent 513–514 token overshoots that get truncated by 512-token embedding models or cross-encoder rerankers.
- **Markdown & GFM table-aware.** Fenced code blocks, list items, and Markdown tables stay intact when they fit. When a long table must be split, it cuts strictly at row boundaries and repeats the `| Header |` + `| --- |` rows on every continuation chunk (`chunk.table_header` and `chunk.contextualized_text`), counted against your token budget.
- **Zero-GIL Hugging Face & multi-tokenizer guards.** Load any model via `TextChipper.from_pretrained("BAAI/bge-large-en-v1.5")`, pass an existing `AutoTokenizer` / `tokenizers.Tokenizer` (extracted straight into the native Rust `tokenizers` backend for zero-GIL speed), pass a custom Python callable (`tokenizer=`), or enforce a single budget across multiple models at once (`guard_tokenizers=[...]`).
- **Grapheme-safe & exact source offsets.** Never splits inside a Unicode grapheme cluster (emojis, combining marks, CJK/Indic scripts). Every `chunk.text` is an exact slice of the normalized input (`doc[c.start_index:c.end_index] == c.text`), and `orig_start_*` / `orig_end_*` map back to the pre-normalized source text.
- **Fast & bounded-memory streaming.** Reaches **37.1 MiB/s** single-core throughput (2.4×–11.6× faster than Python splitters), has **0 runtime dependencies** (~4 MB installed, ~2 ms import), and grows memory by only **~56 MB** when streaming a 514.3 MiB file via `from_file_stream()`.

---

## ⚡ Quick Start

### Python

Install from PyPI (pre-built `abi3` wheels for Python ≥ 3.9 include the Rust core and Hugging Face `tokenizers` support with zero external dependencies):

```bash
pip install text-chipper
```

#### 1. Basic Markdown & Table-Aware Chunking

```python
from text_chipper import TextChipper

chipper = TextChipper(max_length=512, overlap=50, encoding="cl100k_base")

with open("README.md", encoding="utf-8") as file:
    for chunk in chipper.chunk(file.read()):
        print(chunk.token_count, chunk.headers)   # e.g. 487 ['Install', 'From source']
        print(chunk.contextualized_text)          # "[ctx:Install > From source] ..." + text
```

#### 2. Local LLM / Embedding Models (Hugging Face & Custom Tokenizers)

```python
from text_chipper import TextChipper

# Load from a local directory/file, or from Hugging Face Hub (if huggingface_hub/tokenizers/transformers is installed):
chipper = TextChipper.from_pretrained("BAAI/bge-large-en-v1.5", max_length=510, overlap=0)

# Or pass an existing transformers AutoTokenizer / tokenizers.Tokenizer (runs in native Rust without holding the GIL),
# or any custom Python callable(text) -> int / .encode(text) object:
# chipper = TextChipper(max_length=510, tokenizer=my_hf_tokenizer, guard_tokenizers=[reranker_tokenizer])
```

#### 3. Python API Reference at a Glance

| Entry Point / Property | Description |
| --- | --- |
| `TextChipper(max_length=512, overlap=50, encoding="cl100k_base", *, tokenizer=None, tokenizer_file=None, guard_encodings=None, guard_tokenizer_files=None, guard_tokenizers=None, ...)` | Configure a chunker with a built-in BPE/char/word backend, a `tokenizer.json` path, a Hugging Face tokenizer object, or a custom counter callable |
| `TextChipper.from_pretrained(model_name_or_path, **kwargs)` | Create a chunker from a Hugging Face model ID, local model directory, or `tokenizer.json` file |
| `chipper.chunk(text)` / `chipper.chunk_batch(texts)` | Split one string or a batch of strings (parallelized with Rayon on large batches) |
| `chipper.from_file(path)` / `chipper.from_files(paths)` | Split one or more UTF-8 files |
| `chipper.from_file_stream(path)` | Bounded-memory streaming iterator over a large UTF-8 Markdown/text file |
| `chunk.text` | Exact contiguous substring of the normalized input (`doc[chunk.start_index:chunk.end_index] == chunk.text`) |
| `chunk.contextualized_text` | Delivered text ready for embedding: `[ctx:H1 > H2] ` + repeated `table_header` (on split table continuation chunks) + `chunk.text` |
| `chunk.token_count` | Exact token count of `chunk.contextualized_text` (guaranteed `<= max_length` in `strict=True` mode) |
| `chunk.headers` / `chunk.table_header` | List of enclosing section headings (`list[str]`) and repeated table header (`str \| None`) |
| `chunk.start_byte`, `chunk.end_byte`, `chunk.orig_start_byte`, `chunk.orig_end_byte` | Exact UTF-8 byte offsets (plus `*_index` character offsets) in normalized and original input |

Adapters for LangChain and LlamaIndex are in [`integrations/`](https://github.com/devel0mer/text-chipper/tree/main/integrations).

### Rust

Add the crate from crates.io:

```bash
cargo add text_chipper
```

```rust
use text_chipper::{smart_chunk, Backend, ChunkConfig};

fn main() -> Result<(), text_chipper::ChipError> {
    let backend = Backend::from_name("cl100k_base")?;
    let cfg = ChunkConfig {
        max_tokens: 512,
        overlap_tokens: 50,
        ..Default::default()
    };
    let text = "## Retry configuration

The client retries failed requests up to three times.";

    for chunk in smart_chunk(text, &cfg, &backend)? {
        println!("{} {:?}", chunk.token_count, chunk.headers);
        println!("{}", chunk.contextualized_text());
    }
    Ok(())
}
```

The Rust crate (`text_chipper`) has no runtime dependency on Python or PyO3 and installs no global allocator by default. Optional Cargo features:
- `hf`: Hugging Face `tokenizer.json` support (`Backend::from_tokenizer_file`, `Backend::from_tokenizer_bytes`)
- `mimalloc`: opt-in global allocator
- `python`: PyO3 extension module used for the PyPI wheel (PyPI wheels enable `python` and `hf`)

See the [API documentation](https://docs.rs/text_chipper) for offsets, streaming (`FileChunker`), and custom tokenizers.

---

## Why not just split every N tokens?

A plain token window cuts wherever the budget ends. On Markdown, that can split a code block in the middle of a line or orphan table rows from their column headers.

Same input, same budget (`max_length = 62`, kept small so it fits on a screen):

![A plain tiktoken window splits a code block inside "timeout=10)", while text-chipper keeps the whole block in one chunk](https://raw.githubusercontent.com/devel0mer/text-chipper/main/docs/before_after.svg)

Reproduce it with [`examples/before_after.py`](https://github.com/devel0mer/text-chipper/blob/main/examples/before_after.py).

The same comparison on a real README excerpt: the fixed token window on the left, text-chipper on the right. [`benchmarks/before_after_readme.py`](https://github.com/devel0mer/text-chipper/blob/main/benchmarks/before_after_readme.py) runs it on any Markdown file.

![The same README excerpt split two ways: the fixed token window cuts a code comment and an emoji heading; text-chipper keeps both whole](https://raw.githubusercontent.com/devel0mer/text-chipper/main/docs/readme_before_after.svg)

### Table-aware splitting without breaking source offsets

When a 200-row GFM table exceeds `max_length`, naive splitters either cut in the middle of a row or emit continuation chunks of raw numbers (`| 046 | 57,500 |`) without the column header.

In `text_chipper` (`include_context=True`):
1. If a table fits within the token budget, it is kept intact in a single chunk.
2. If a table exceeds the budget, it is split strictly at row boundaries.
3. **Chunk 1** of the table starts at the beginning of the table in the source document, so `chunk.text` already begins with the header and `| --- |` delimiter rows (`chunk.table_header` is `None`).
4. **Chunks 2..N** of the table keep `chunk.text` as the exact contiguous slice of data rows in the source document (so `doc[chunk.start_index:chunk.end_index] == chunk.text` always holds for citation/highlighting), store the header + delimiter rows in `chunk.table_header`, and automatically prepend them in `chunk.contextualized_text`. The repeated table header is counted against `max_length`.

---

## 🛠️ Scope & Installation

### Scope

| Supported | Not supported |
| --- | --- |
| Markdown and plain text, passed as `str` or read from UTF-8 files | HTML, PDF, DOCX loaders (convert to Markdown/text first) |
| Tokenizers: `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, `chars`, `words`; Hugging Face `TextChipper.from_pretrained(...)`, `tokenizer.json`, `AutoTokenizer` / `tokenizers.Tokenizer` objects; custom counters from Python (`tokenizer=`) or Rust (`Tokenizer` trait) | Code-aware AST splitting (no function/class AST boundaries outside Markdown fences) |

### Installation from source

Requires a Rust toolchain. `maturin` enables the `python` and `hf` features automatically (configured in `pyproject.toml`):

```bash
pip install maturin
maturin develop --release        # into the active virtualenv
# or: maturin build --release    # build a wheel into target/wheels/
```

### Install size and startup

`text_chipper` has no runtime dependencies. The macOS arm64 wheel is 2.4 MiB (3.9 MB installed), and one `abi3` wheel per platform covers Python 3.9 and newer. Compared with other libraries installed into empty virtual environments on the same machine:

| Library | Dependencies | Installed size | `import` | Import + tokenizer + first chunk |
| --- | ---: | ---: | ---: | ---: |
| **text_chipper** | **0** | **~4 MB** | **~2 ms** | **26 ms** |
| semantic-text-splitter | 0 | ~16 MB | 2 ms | 49 ms |
| tiktoken (plain window) | 6 | ~8 MB | 21 ms | 130 ms |
| semchunk | 1 | ≤1 MB | 52 ms | 165 ms |
| chonkie | 11 | ~47 MB | 152 ms | 260 ms |
| langchain-text-splitters | 33 | ~45 MB | 207 ms | 307 ms |

Measured on macOS arm64 with Python 3.15, one measurement per library, warm start. Installed size is the size of the virtual environment minus an empty one (about ±1 MB). The last column uses the benchmark adapters (`import`, loading `cl100k_base`, chunking a tiny text), so it includes each library's tokenizer; semchunk's own size does not include the tokenizer it needs. How to reproduce: [`benchmarks/README.md`](https://github.com/devel0mer/text-chipper/blob/main/benchmarks/README.md#install-footprint).

---

## 📖 Chunking modes

![Four text-chipper modes compared by how to call them, Markdown handling, memory, speed and use case](https://raw.githubusercontent.com/devel0mer/text-chipper/main/docs/modes.svg)

### `include_context=True` (default): Markdown-aware

The Markdown block parser runs. Headings are tracked and attached to chunks as `headers`; the label `[ctx:A > B] ` counts against `max_length` (at most about a quarter of the budget is reserved for it). Fenced code blocks, list items, and GFM tables stay intact when they fit, and adjacent blocks are packed together while they fit (`merge_blocks=True`). Oversized GFM tables are split at row boundaries and repeat their header + delimiter row (`chunk.table_header` and `chunk.contextualized_text`), counted against `max_length`. Block quotes, HTML blocks, and indented code get no special handling; see [Markdown support](#markdown-support-and-commonmark-conformance).

### `include_context=False`: flat mode

This is **not** "the same thing without the header prefix". Flat mode does not parse Markdown at all: the text is cut into token windows (honouring `overlap`) that prefer paragraph, line, and word boundaries.

- Code fences, lists, and tables are **not** protected and can be cut.
- Headings are ordinary text, `headers` is always empty, `table_header` is always `None`, and `merge_blocks` has no effect.
- `from_file`, `from_files`, and `from_file_stream` give exactly the chunks `chunk()` gives for the file's text, at any size. Flat mode needs the whole file in memory, so files larger than 32 MiB raise `OSError` instead of being chunked differently. To process bigger files with bounded memory, use `include_context=True` (streamed), or split the file first.

---

## 📈 Benchmarks

The main benchmark compares `text_chipper` with seven other chunking approaches on 12 workloads: synthetic prose, Markdown and code; real texts such as PubMed, finance, Wikipedia and chat logs; and stress cases. Every library gets a 512-token budget, `cl100k_base` and no overlap. The charts below show the **single-core** lane from the 17 November 2026 run.

The run used a GitHub Actions runner (AMD EPYC 9V45, 4 logical CPUs, 15.6 GB RAM) and Python 3.12.16. Timings are medians of five cold runs in fresh processes. Compared versions: chonkie 1.7.0, langchain-text-splitters 1.1.3, semchunk 4.1.1, semantic-text-splitter 0.33.3 and tiktoken 0.14.0. `text_chipper` is the default Markdown-aware strict mode; `text_chipper (flat)` uses `include_context=False`. Raw results, metric definitions, and harness are in [`benchmarks/`](https://github.com/devel0mer/text-chipper/blob/main/benchmarks/) and [`benchmarks/results/20261117_133546/`](https://github.com/devel0mer/text-chipper/blob/main/benchmarks/results/20261117_133546/).

### Speed

Throughput averaged geometrically over the 12 workloads: `text_chipper` reaches **37.1 MiB/s** in the single-core lane. That is 2.4 times the throughput of a plain tiktoken window (15.5 MiB/s), and 3.9 to 11.6 times the throughput of the other chunking libraries.

![Bar chart: geometric-mean throughput of each library over all 12 workloads](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/speed_overall_latest.png)

`text_chipper` is the fastest approach on 10 of 12 workloads. The exceptions are `stress_cjk_unbroken`, where the tiktoken window reaches 19.7 MiB/s against `text_chipper`'s 16.0 MiB/s, and `real_chatlogs`, where `text_chipper` flat mode reaches 18.4 MiB/s against 16.1 MiB/s in the default mode.

![Heat map: MiB/s of each library on each workload](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/speed_per_every_workload.png)

### Memory

Peak memory of the whole Python process (including the interpreter and imported library), single core, averaged over the workloads. Mean process peaks range from 94 to 102 MB; the largest single-workload peak was 119 MB. `text_chipper` adds 1.7 MB on average while chunking; its largest measured increase is about 11.7 MB.

![Bar chart: peak memory of the Python process per library, single core](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/memory_single_core.png)

### Chunk quality

![Scorecard: token limit, words, source text and code fences per library, with speed](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/quality_scorecard.png)

- **Token limit**: no `text_chipper` chunk exceeded 512 tokens on any workload. Across all workloads, LangChain had 15 over-limit chunks; the tiktoken window and chonkie TokenChunker had 18 each. Their largest overshoot was two tokens for LangChain (514); the other two reached 513. Counts were checked independently with `cl100k_base` on delivered chunk text.
- **Words and code**: token windows cut words at up to 35.5 % of cut points and broke about 23 % of the measured small fences. Chonkie's RecursiveChunker broke about 22 %. `text_chipper` kept all 392 measured fences in `markdown_doc` and all 3,434 in `code_heavy` intact. Its only workload with a measured mid-word cut was `stress_giant_list` (under 0.1 % of cut points).
- **Coverage** is reported two ways. **Body coverage** counts source text found verbatim in chunk bodies; it is 99.57 % for `text_chipper` on `markdown_doc`. **Content + heading metadata** also counts recognized heading text carried in `headers`, giving 100 % content + heading metadata coverage. In the unsegmented CJK stress input, tiktoken window and both Chonkie chunkers produced replacement characters (U+FFFD) in 245 chunks; `text_chipper` produced none.

To regenerate the charts after a new benchmark run:

```bash
pip install matplotlib
python benchmarks/make_charts.py benchmarks/results/20261117_133546/results.json benchmarks/charts --svg
```

### Large files and streaming

For a larger input, we chunked a 514.3 MiB WikiText-103 file with 512-token chunks, `cl100k_base` and no overlap on a GitHub Actions runner (4 logical CPUs, Python 3.12, single-core lane). Raw results are in [`benchmarks/results/streaming_20261117/`](https://github.com/devel0mer/text-chipper/blob/main/benchmarks/results/streaming_20261117/).

![Bar chart: throughput of each library on a 514 MiB file, single-core lane](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/streaming_speed_single.png)

| Library | File given as | Time | MiB/s | Memory growth | Chunks over 512 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| text_chipper | `from_file_stream` | 25 s | 20.5 | 56 MB | 0 |
| text_chipper | 8 MiB blocks | 28 s | 18.3 | 140 MB | 0 |
| tiktoken window | 8 MiB blocks | 40 s | 12.8 | 149 MB | 5 of 234,790 |
| chonkie RecursiveChunker | 8 MiB blocks | 41 s | 12.5 | 110 MB | 0 |
| chonkie TokenChunker | 8 MiB blocks | 46 s | 11.1 | 208 MB | 5 of 234,790 |
| LangChain | 8 MiB blocks | 106 s | 4.9 | 98 MB | 10 of 292,611 |
| semchunk | 8 MiB blocks | 111 s | 4.6 | 3.3 GB | 0 |
| semantic-text-splitter | 8 MiB blocks | 152 s | 3.4 | 87 MB | 0 |

![Bar chart: memory growth of each library on a 514 MiB file, single-core lane](https://raw.githubusercontent.com/devel0mer/text-chipper/main/benchmarks/charts/streaming_memory_single.png)

- **Memory stays bounded while streaming.** The process grew by 56 MB on the 514 MiB file. Reading it into one string and calling `chunk()` grew by 4.6 GB. Verification hashes for `from_file_stream` and whole-file `chunk()` match.
- **Speed.** Streaming reached 20.5 MiB/s, 1.6 to 6.0 times the other libraries fed in blocks.
- **Token limit at scale.** The verification run found five one-token overshoots in each of the tiktoken window and Chonkie TokenChunker block paths, and ten in LangChain. `text_chipper` streaming stayed strictly within budget.

---

## 🔌 Pluggable tokenizers

The chunker only needs one thing from a tokenizer: how many tokens a piece of text has. Built-in backends cover the OpenAI encodings, `chars`, `words`, and Hugging Face `tokenizer.json` files (covering Llama, Qwen, DeepSeek, Mistral, BGE, and most open models).

### Python

From Python, you can load a Hugging Face model directly with `from_pretrained`, pass a pre-loaded `transformers` / `tokenizers` object (whose JSON is extracted into the native Rust backend for zero-GIL speed), or supply any custom `callable(text) -> int` or `.encode(text)` object via `tokenizer=` and `guard_tokenizers=`:

```python
from text_chipper import TextChipper

# 1. Hugging Face model ID, local model directory, or tokenizer.json path:
chipper = TextChipper.from_pretrained("BAAI/bge-large-en-v1.5", max_length=512)

# 2. Existing transformers AutoTokenizer / tokenizers.Tokenizer (zero-GIL Rust execution):
chipper = TextChipper(max_length=512, tokenizer=my_hf_tokenizer)

# 3. Custom Python callable or object with .encode(text):
chipper = TextChipper(max_length=512, tokenizer=lambda text: len(text.split()))
```

### Rust

Rust users can supply their own counter by implementing the `Tokenizer` trait:

```rust
use std::sync::Arc;
use text_chipper::{Backend, Tokenizer};

struct MyTokenizer; // wrap SentencePiece, a model-specific BPE, a lookup table, ...

impl Tokenizer for MyTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count() // your counting logic
    }
    // Optional, with safe defaults:
    // fn joins_are_additive(&self) -> bool { false }
    // fn initial_ratio(&self) -> f64 { 4.0 }
}

let backend = Backend::from_tokenizer(Arc::new(MyTokenizer));
// or, for a quick closure:
let backend = Backend::from_fn(|text| text.chars().count() / 4);
```

Your counter must be:
- **Deterministic**: the same text always gives the same count.
- **Monotone on prefixes**: `count(prefix) <= count(prefix + more)`. The window search relies on it; strict mode still re-counts every delivered chunk but cannot repair a non-monotone counter.
- **Thread-safe and cheap** (`Send + Sync`): it is called many times per chunk and possibly from several threads, so avoid blocking I/O per call.

---

## Markdown support and CommonMark conformance

`text_chipper` is a chunker, not a full CommonMark AST parser. Its block parser decides where a chunk may start or be cut; it reads one line at a time and keeps bounded state so it can stream files of any size with exact offsets.

| Construct | Handling |
| --- | --- |
| Text separated by blank lines | One **block**, the unit that is packed into chunks and split last. A paragraph, a table, or a block quote with no blank line inside is one block, until a fence or heading line ends it |
| Fenced code blocks (`` ``` `` and `~~~`, opening fence indented at most three spaces) | One block, kept intact when it fits; never scanned for headings, lists, or tables |
| ATX headings (`# Title`) and short setext headings (at most two lines and 160 characters) | Become `headers` metadata (`[ctx:...]`), not body text |
| YAML (`---`) and TOML (`+++`) front matter at the start of a document | Kept verbatim as one metadata block; an unterminated one is abandoned after 200 lines or 32 KiB |
| List items (nested, ordered, lazy continuation, inside block quotes) | Located by a dedicated scanner and used as cut points: when a block must be split, item boundaries are preferred over line boundaries, and an item that fits is kept whole, even when blank lines inside it make it several blocks |
| GFM tables (header row + delimiter row `\| --- \|` + body rows) | Kept intact when they fit; oversized tables are split at row boundaries and repeat their header + delimiter row on continuation chunks (`table_header` and `contextualized_text`) within the token budget while keeping `text` as the exact source slice |
| Block quotes, HTML blocks, indented code, link reference definitions | **No special handling.** They are ordinary text |

### Measured agreement with `pulldown-cmark`

`cargo test --features diagnostics commonmark_block_boundary -- --nocapture` (after `python scripts/fetch_commonmark.py`) compares the lines on which our parser starts blocks with the lines on which `pulldown-cmark` starts top-level blocks, over the 652 examples of the CommonMark 0.31.2 spec.

| Measurement | Result |
| --- | --- |
| Examples where every block start agrees | 496 / 652 (76.1 %) |
| Same, ignoring link reference definitions | 567 / 652 (87.0 %) |
| Our block starts with no reference counterpart | 145 / 844 (17.2 %); 70 / 769 (9.1 %) ignoring link reference definitions |
| Reference block starts we also start (recall) | paragraph 96.2 %, code block 96.9 %, HTML block 96.4 %, block quote 92.5 %, list 90.2 %, thematic break 40.6 % |

### Where we differ from CommonMark, and why

1. **Link reference definitions stay text.** CommonMark consumes them without producing a block, which accounts for 75 of the 145 extra starts. Chunks must cover the source text, so a definition is ordinary content here.
2. **A blank line always ends a block**, even inside lists, indented code, and HTML blocks that CommonMark treats as one block. For lists, the chunker compensates: an item that fits is kept whole across blank lines.
3. **Lists, block quotes, thematic breaks, and HTML blocks do not interrupt a paragraph.** They start a new block only after a blank line. A line that introduces a list ("Steps:") therefore stays in the same block as the list, and the two are never separated while the block fits.

---

## ⚖️ Budget guarantee: `strict` and Multi-Tokenizer Guards

| | `strict=True` (default) | `strict=False` |
| --- | --- | --- |
| Final exact re-count of every delivered chunk | yes | no |
| `token_count <= max_length` guaranteed | yes | **no** |
| `token_count` | exact | may be an estimate |
| A single grapheme cluster larger than the budget | raises `ValueError` | returned as its own oversized chunk |
| Speed | slightly slower | faster |

### Other tokenizers and guards

If chunks go to more than one model (for example, an embedding model and a cross-encoder reranker, or an LLM with a different vocabulary), add **tokenizer guards**. The budget is enforced against the *highest* token count across the primary tokenizer and every listed guard:

```python
chipper = TextChipper(
    max_length=510,  # reserve two positions for BGE's [CLS] and [SEP] under a 512 limit
    encoding="cl100k_base",
    guard_encodings=["o200k_base"],
    guard_tokenizer_files=["bge-large-en-v1.5-tokenizer.json"],
    # or pass pre-loaded tokenizer objects / callables via guard_tokenizers=[...]:
)
```

See the [tokenizer guard guide](https://github.com/devel0mer/text-chipper/blob/main/tokenizer-guard/README.md) for supported options, overhead handling, and the acceptance benchmark.

---

## ⚠️ Errors & Known Limitations

Every failure is raised as an exception and nothing returns a partial or empty result for bad input:
- `ValueError`: invalid configuration, unknown encoding, or (strict mode) a single grapheme that does not fit the budget.
- `RuntimeError`: the tokenizer backend failed to load (e.g. missing or invalid `tokenizer_file` or unresolvable `from_pretrained` model).
- `OSError`: unreadable file, invalid UTF-8 (unless `errors="replace"`), a physical line longer than 1 MiB in the streaming reader (`include_context=True`), or a file larger than 32 MiB in flat mode (`include_context=False`).
- `chunk_batch` and `from_files` raise the error of the **first** failing input in input order and return no partial results.

Known limitations:
- Non-strict mode gives no budget guarantee.
- The strict guarantee holds for the configured tokenizer and its guards only; text counted by an unlisted tokenizer can exceed the budget.
- Flat mode reads whole files (32 MiB limit); streaming with bounded memory is only available with `include_context=True`.
- Block recognition is limited to what [Markdown support](#markdown-support-and-commonmark-conformance) lists. In particular there is no special block quote or HTML block handling, and multi-line cells or merged cells outside standard GFM pipe tables are treated as ordinary text.
- A custom token counter (`Tokenizer` in Rust or `tokenizer=` in Python) that is not deterministic or not monotone on prefixes can produce suboptimal chunk boundaries.
- `from_file_stream` is single-threaded; on several cores, `chunk()` is faster but needs more memory.

---

## 🧪 Development & Testing

```bash
cargo test --release                    # unit, oracle, property and robustness tests
pytest tests -v                         # Python binding tests (after `maturin develop`)
python scripts/fetch_commonmark.py      # refresh the pinned CommonMark 0.31.2 corpus
cargo test --features diagnostics commonmark_block_boundary -- --nocapture   # block-boundary agreement report
scripts/run_release_suite.sh            # all tests + diagnostics, with summary
scripts/run_validation.sh               # full validation matrix
pytest benchmarks/tests -q              # self-tests of the benchmark harness
```

Repository layout: `src/` library code; `src/tests/` Rust test modules (unit, oracle, parallel, robustness, release suite); `tests/` Python binding tests and fixtures; `benches/` criterion microbenchmarks; `benchmarks/` cross-library benchmark; `integrations/` LangChain and LlamaIndex adapters; `scripts/` validation and profiling tools.

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
