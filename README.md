# text_chipper

<table>
<tr>
<td valign="middle">
<img src="docs/logo.svg" width="120" alt="text_chipper logo">
</td>
<td valign="middle">

**High-performance, Markdown-aware, strictly budget-enforced text chunking for RAG and LLM pipelines.
Built in Rust with Python bindings.**

[![Rust Version](https://img.shields.io/badge/rust-1.88.0-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/text_chipper.svg)](https://crates.io/crates/text_chipper)
[![docs.rs](https://img.shields.io/docsrs/text_chipper)](https://docs.rs/text_chipper)
[![PyPI](https://img.shields.io/pypi/v/text-chipper.svg)](https://pypi.org/project/text-chipper/)

</td>
</tr>
</table>

- **Token-exact budget.** In the default strict mode every chunk is re-tokenized exactly as it
  will be delivered (header context included), so `token_count <= max_length` always holds.
- **Structure-aware.** Text is divided at blank lines into blocks that are packed into chunks and
  split internally only when a block does not fit. Fenced code blocks and list items are kept
  intact when they fit, and headings become `headers` metadata for each chunk. See
  [Markdown support](#markdown-support-and-commonmark-conformance) for exactly what is and is
  not recognized.
- **Never cuts inside a grapheme cluster** (emoji sequences, combining marks).
- **Offsets map back to your input**, even when Unicode normalization (NFC/NFKC) is applied.

- Block boundaries are compared against `pulldown-cmark` on the CommonMark 0.31.2 spec corpus
  (results and limits below); there are also property tests and a robustness suite. This is a
  chunker, not a complete CommonMark parser.

---

## ⚡ Quick Start

### Python

```bash
pip install text-chipper
```


```python
from text_chipper import TextChipper

chipper = TextChipper(max_length=512, overlap=50, encoding="cl100k_base")

for chunk in chipper.chunk(open("README.md", encoding="utf-8").read()):
    print(chunk.token_count, chunk.headers)   # e.g. 487 ['Install', 'From source']
    print(chunk.contextualized_text)          # "[ctx:Install > From source] ..." + text
```
### Why not just split every N tokens?

A plain token window cuts wherever the budget ends. On Markdown, that can 
mean a code block split in the middle of a line. Same input, same budget 
(`max_length = 62`, kept small so it fits on a screen):

![A plain tiktoken window splits a code block inside "timeout=10)", while text-chipper keeps the whole block in one chunk](docs/before_after.svg)

Reproduce it with [`examples/before_after.py`](examples/before_after.py).

The same comparison on a real README excerpt: the fixed token window on the left, text-chipper on the right. [`benchmarks/before_after_readme.py`](benchmarks/before_after_readme.py) runs it on any Markdown file.

![The same README excerpt split two ways: the fixed token window cuts a code comment and an emoji heading; text-chipper keeps both whole](docs/readme_before_after.svg)


Other entry points: `chunk_batch(texts)`, `from_file(path)`, `from_files(paths)`,
`from_file_stream(path)` (an iterator), `count_tokens(text)`.

Each `TextChunk` has `text`, `headers`, `contextualized_text`, `token_count`, character/byte
offsets in the normalized text (`start_index`, `end_index`, `start_byte`, `end_byte`) and in
your original input (`orig_start_index`, `orig_end_index`, `orig_start_byte`, `orig_end_byte`).

Adapters for LangChain and LlamaIndex are in [`integrations/`](integrations/).

### Rust

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
    let text = std::fs::read_to_string("README.md").expect("readable UTF-8 file");

    for chunk in smart_chunk(&text, &cfg, &backend)? {
        println!("{} {:?}", chunk.token_count, chunk.headers);
        println!("{}", chunk.contextualized_text());
    }
    Ok(())
}
```

The Rust crate is plain Rust: no Python or PyO3 is compiled and no global allocator is
installed by default. Optional Cargo features: `mimalloc` (use mimalloc as the global
allocator) and `hf` (Hugging Face `tokenizer.json` support). The `python` feature builds the
Python extension module and is only meant for the PyPI wheel. See the
[API documentation](https://docs.rs/text_chipper) for offsets, streaming (`FileChunker`) and
custom tokenizers.

---



## 🛠️ Scope & Installation

### Scope

| Supported | Not supported |
| --- | --- |
| Markdown and plain text, passed as `str` or read from UTF-8 files | HTML, PDF, DOCX loaders |
| Tokenizers: `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, `chars`, `words`; Hugging Face `tokenizer.json` (build with the `hf` feature); any counter you supply from Rust through the `Tokenizer` trait (see [Pluggable tokenizers](#pluggable-tokenizers)) | Code-aware splitting (no AST, no function/class boundaries) |

Convert other formats to Markdown or text before chunking.

### Installation from source

Requires a Rust toolchain. `maturin` enables the `python` feature automatically
(configured in `pyproject.toml`):

```bash
pip install maturin
maturin develop --release        # into the active virtualenv
# or: maturin build --release    # build a wheel into target/wheels/
```

### Install size and startup

text_chipper has no runtime dependencies. The macOS arm64 wheel is 2.4 MiB (3.9 MB installed), and one `abi3`
wheel per platform covers Python 3.9 and newer. Compared with the other libraries, installed into empty
virtual environments on the same machine:

| Library | Dependencies | Installed size | `import` | Import + tokenizer + first chunk |
| --- | ---: | ---: | ---: | ---: |
| **text_chipper** | **0** | **~4 MB** | **~2 ms** | **26 ms** |
| semantic-text-splitter | 0 | ~16 MB | 2 ms | 49 ms |
| tiktoken (plain window) | 6 | ~8 MB | 21 ms | 130 ms |
| semchunk | 1 | ≤1 MB | 52 ms | 165 ms |
| chonkie | 11 | ~47 MB | 152 ms | 260 ms |
| langchain-text-splitters | 33 | ~45 MB | 207 ms | 307 ms |

Measured on macOS arm64 with Python 3.15, one measurement per library, warm start. Installed size is the size of
the virtual environment minus an empty one (about ±1 MB). The last column uses the benchmark adapters
(`import`, loading `cl100k_base`, chunking a tiny text), so it includes each library's tokenizer; semchunk's
own size does not include the tokenizer it needs. The first import after a fresh install took about 0.5 s on
macOS once (probably the system checking the new binary); later imports took about 2 ms. Linux and Windows were
not measured. How to reproduce: [`benchmarks/README.md`](benchmarks/README.md#install-footprint).

---

## 📖 Chunking modes

![Four text-chipper modes compared by how to call them, Markdown handling, memory, speed and use case](docs/modes.svg)

### `include_context=True` (default): Markdown-aware

The Markdown block parser runs. Headings are tracked and attached to chunks as `headers`; the
label `[ctx:A > B] ` counts against `max_length` (at most about a quarter of the budget is
reserved for it). Fenced code blocks and list items stay intact when they fit, and adjacent
blocks are packed together while they fit (`merge_blocks=True`). Tables, block quotes, HTML
blocks and indented code get no special handling; see
[Markdown support](#markdown-support-and-commonmark-conformance).

### `include_context=False`: flat mode

This is **not** "the same thing without the header prefix". Flat mode does not parse Markdown
at all: the text is cut into token windows (honouring `overlap`) that prefer paragraph, line and
word boundaries.

- Code fences, lists and tables are **not** protected and can be cut.
- Headings are ordinary text and `headers` is always empty; `merge_blocks` has no effect.
- `chunk()` and `chunk_batch()` follow the mode you set: with `include_context=False` they behave this way.
- `from_file`, `from_files` and `from_file_stream` give exactly the chunks `chunk()` gives for
  the file's text, at any size. Flat mode needs the whole file in memory (peak memory is roughly
  an order of magnitude larger than the file for ASCII text; this is estimated from the data
  structures, not measured), so files larger than 32 MiB raise `OSError` instead of being
  chunked differently. To process bigger files with bounded memory, use `include_context=True`
  (streamed), or split the file first.

Use flat mode when you want plain overlapping token windows over arbitrary text; use the default
mode for Markdown documents.


## 📈 Benchmarks

text_chipper is compared with seven other ways of chunking text on the same inputs: 12 workloads (synthetic
prose, Markdown and code, real-world texts such as PubMed, finance, Wikipedia and chat logs, and stress
cases), 512-token chunks, `cl100k_base`, no overlap. All charts below show the **single-core** lane.

Setup: GitHub Actions runner (AMD EPYC 9V74, 4 logical CPUs, 15.6 GB RAM), Python 3.12.16, cold start (every
timing is the first and only call in a fresh process), median of 5 rounds. Compared libraries: chonkie 1.7.0,
langchain-text-splitters 1.1.3, semchunk 4.1.1, semantic-text-splitter 0.33.2, tiktoken 0.14.0 (plain token
window). `text_chipper` is the default mode (Markdown-aware, strict) and `text_chipper (flat)` is
`include_context=False`. Results come from one run on a shared virtual machine, so treat the ratios as
indicative, especially on the small files (under 120 KiB, where a call takes only a few milliseconds).
Method, metric definitions, raw data and the harness are in [`benchmarks/`](benchmarks/).

### Speed

Throughput averaged over all 12 workloads (geometric mean): text_chipper reaches 34.0 MiB/s, which is 2.8
times the speed of a plain tiktoken token window and 4.4 to 15 times the speed of the other chunking
libraries.

![Bar chart: geometric-mean throughput of each library over all 12 workloads](benchmarks/charts/speed_overall.png)

text_chipper is the fastest library on 11 of 12 workloads. The exception is `stress_cjk_unbroken` (a megabyte
of CJK text without separators), where it reaches 7.6 MiB/s while tiktoken window (15.8 MiB/s) and chonkie
TokenChunker (11.2 MiB/s) are faster. On `real_chatlogs` (39 KiB, 2 to 3 ms per call) text_chipper reaches 20.4 MiB/s against 12.3 for
tiktoken window, but at this call length the spread between processes is large (over 50 % for
text_chipper), so treat that ratio as rough.

![Heat map: MiB/s of each library on each workload](benchmarks/charts/speed_per_workload.png)

### Memory

Peak memory of the whole Python process (interpreter and imported library included), single core, mean over
all workloads. Every library stays between 96 and 104 MB, most of it the interpreter and the tokenizer.
text_chipper adds 1.4 MB on average while chunking and about 10 MB on its worst workload.

![Bar chart: peak memory of the Python process per library, single core](benchmarks/charts/memory_single_core.png)

### Chunk quality

![Scorecard: token limit, words, source text and code fences per library, with speed](benchmarks/charts/quality_scorecard.png)

- **Token limit**: no text_chipper chunk exceeded 512 tokens on any workload. LangChain went over the limit
  in up to 16.7 % of chunks (3 of the 18 chunks of `real_chatlogs`; its largest chunk had 514 tokens), and
  tiktoken window and chonkie TokenChunker in 2.6 % of the chunks of `multilingual` (513 tokens). Token
  counts are measured independently with `cl100k_base` on the text each library delivers.
- **Words and code**: token windows cut words in half (up to 35.5 % of their cut points) and break 21 to 23 %
  of the small code fences (those that fit in half the budget); chonkie RecursiveChunker breaks about 22 %.
  text_chipper made one cut inside a word in the whole run (1 of 1,259 cut points in `stress_giant_list`,
  none on the other 11 workloads) and kept every small fence whole (392 in `markdown_doc`, 3,434 in
  `code_heavy`).
- **Coverage** means the share of the input's non-whitespace text that appears in some chunk. text_chipper
  (default mode) covers 99.57 % of `markdown_doc` and 100 % of the other 11 workloads. The 3,613
  non-whitespace characters missing from the chunk text of `markdown_doc` are all heading text, which
  text_chipper returns in each chunk's `headers` (and in `contextualized_text`) instead of the chunk body;
  counting them, coverage is 100 % and no character is unexplained. text_chipper (flat) covers 100 %
  everywhere. tiktoken window, chonkie TokenChunker and chonkie RecursiveChunker lose a third (32.6 %) of the
  unsegmented CJK input, and tiktoken window and chonkie TokenChunker lose 27.6 % of `multilingual`: their
  chunks contain replacement characters (U+FFFD) instead of the original text.
- **Chunk ends**: semchunk and chonkie RecursiveChunker end slightly more chunks at a sentence or line break
  on average (91.7 % against 85.9 % for text_chipper). The gap comes mostly from `real_wikitexts` (43.1 %
  against 100 %) and `real_pubmed` (88.2 % against 100 %).

To regenerate the charts after a new benchmark run:

```bash
pip install matplotlib
python benchmarks/make_charts.py benchmarks/results/20261113_213722/results.json benchmarks/charts
```

---


### Large files and streaming

The comparison above uses files of up to about 1 MB. To see what happens on a big input, we chunked one
514.3 MiB text file (WikiText-103 raw) with 512-token chunks, `cl100k_base` and no overlap. Each run is a fresh
process on a GitHub Actions runner (4 logical CPUs, Python 3.12), and every library is configured exactly as in the
benchmark above. Time includes reading the file. Memory is the growth of the process's peak memory after warm-up.
Numbers are medians of 3 runs from one workflow run per lane; this section shows the single-core lane, the all-cores
lane and the raw data are in [`benchmarks/results/streaming_20261114/`](benchmarks/results/streaming_20261114/).

text_chipper reads the file with `from_file_stream`. The other libraries take a string, so they were given the file in
blocks of about 8 MiB (text_chipper also ran in this block mode). Every library also ran on the whole file read into
one string, see the notes below.

![Bar chart: throughput of each library on a 514 MiB file, single-core lane](benchmarks/charts/streaming_speed_single.png)

| Library | File given as | Time | MiB/s | Memory growth | Chunks over 512 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| text_chipper | `from_file_stream` | 24 s | 21.3 | 71 MB | 0 |
| text_chipper | 8 MiB blocks | 28 s | 18.0 | 130 MB | 0 |
| tiktoken window | 8 MiB blocks | 39 s | 13.1 | 149 MB | 5 of 234,790 |
| chonkie RecursiveChunker | 8 MiB blocks | 41 s | 12.4 | 117 MB | 0 |
| chonkie TokenChunker | 8 MiB blocks | 47 s | 11.0 | 210 MB | 5 of 234,790 |
| LangChain | 8 MiB blocks | 108 s | 4.8 | 83 MB | 10 of 292,611 |
| semchunk | 8 MiB blocks | 112 s | 4.6 | 3.3 GB | 0 |
| semantic-text-splitter | 8 MiB blocks | 158 s | 3.2 | 87 MB | 0 |

![Bar chart: memory growth of each library on a 514 MiB file, single-core lane](benchmarks/charts/streaming_memory_single.png)

- **Memory stays flat when streaming.** The process grew by 71 MB
  on a 514 MiB file (125 MB peak for the whole process). Reading the file into one string and calling `chunk()` grew by
  4.6 GB. The output is the same: the SHA-256 of the delivered
  chunk text is identical for `from_file_stream` and for `chunk()` on the whole file, in both lanes.
- **Speed.** Streaming on one core reaches 21.3 MiB/s,
  1.6 to 6.7 times faster than the other libraries fed in blocks. The gap is smaller than on the small files above
  (2.8 times against the tiktoken window there, 1.6 times here).
- **All-cores lane.** `from_file_stream` is single-threaded (it used one core), so it reached
  19.8 MiB/s there. text_chipper in block mode used about 2.4 cores
  and reached 30.9 MiB/s; the other libraries stayed on one core.
  If you have the cores and memory, block mode is faster than streaming; streaming is the choice when memory is the limit.

<details open>
<summary>All-cores lane charts (speed and memory)</summary>

![Bar chart: throughput of each library on a 514 MiB file, all-cores lane](benchmarks/charts/streaming_speed_all.png)

![Bar chart: memory growth of each library on a 514 MiB file, all-cores lane](benchmarks/charts/streaming_memory_all.png)

</details>

- **Token limit at scale.** The overshoot is rare on this text: one-token overshoots (513) in 5 of 234,790 chunks
  for the tiktoken window and chonkie TokenChunker, and in 10 of 292,611 for LangChain. text_chipper, chonkie
  RecursiveChunker, semchunk and semantic-text-splitter had none. The 16.7 % above is for a small chat-log file, so the
  rate depends on the text.

Notes and limits:

- **Whole-file mode.** All libraries peak at 4.6 to 9.0 GB when the file is read into one string (full table in
  [`benchmarks/results/streaming_20261114/summary.md`](benchmarks/results/streaming_20261114/summary.md)).
  text_chipper, LangChain, chonkie RecursiveChunker and semantic-text-splitter are within 2 MB of each other, which
  suggests that the peak is set by reading the file into a Python string rather than by the library; we have not
  verified this. The tiktoken window and chonkie TokenChunker also hold the full token list (7.7 and 9.0 GB).
  semchunk grows to 3.3 GB even in block mode (probably its default token-count cache, not verified).
- **Noise.** This is one workflow run per lane. The two lanes ran on different virtual machines, so do not compare
  lanes for single-threaded modes, and the same streaming mode took between 22 and 29 s in different workflow runs.
  Treat ratios as indicative.
- **Blocks are our choice.** Feeding a string-only library in 8 MiB blocks is how one would process a large file with it;
  chunks at block edges can differ from a one-call run.
- **Not measured:** retrieval quality.


## 🔌 Pluggable tokenizers

The chunker only needs one thing from a tokenizer: how many tokens a piece of text has. Built-in
backends cover the OpenAI encodings, `chars`, `words` and Hugging Face `tokenizer.json` files
(which covers Llama, DeepSeek, Gemini-style and most open models). For anything else, Rust users
can supply their own counter by implementing the `Tokenizer` trait:

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

Pass the resulting `Arc<Backend>` to `SmartChunker`, `smart_chunk` and the other entry points
exactly as you would a built-in backend.

Your counter must be:

- **Deterministic**: the same text always gives the same count.
- **Monotone on prefixes**: `count(prefix) <= count(prefix + more)`. The window search relies on
  it; strict mode still re-counts every delivered chunk but cannot repair a non-monotone counter.
- **Thread-safe and cheap** (`Send + Sync`): it is called many times per chunk and possibly from
  several threads, so avoid blocking I/O per call.

`joins_are_additive` should stay `false` unless you have verified that
`count(A + sep + B) <= count(A) + count(sep) + count(B)` for newline-terminated separators; with
`false` the chunker keeps a safety slack when merging blocks. `initial_ratio` (bytes per token) only
seeds the window search and affects speed, never correctness.

The Python API does not accept a custom counter yet: use `encoding=` or `tokenizer_file=`.

## Markdown support and CommonMark conformance

text_chipper is a chunker, not a CommonMark parser. Its block parser decides where a chunk may
start or be cut; it does not build a document tree. It reads one line at a time and keeps only
bounded state, which is what lets it stream files of any size with exact offsets. It recognizes
only what that job needs:

| Construct | Handling |
| --- | --- |
| Text separated by blank lines | One **block**, the unit that is packed into chunks and split last. A paragraph, a table or a block quote with no blank line inside is one block, until a fence or heading line ends it |
| Fenced code blocks (`` ``` `` and `~~~`, opening fence indented at most three spaces) | One block, kept intact when it fits; never scanned for headings |
| ATX headings (`# Title`) and short setext headings (at most two lines and 160 characters) | Become `headers`, not body text |
| YAML (`---`) and TOML (`+++`) front matter at the start of a document | Kept verbatim as one metadata block; an unterminated one is abandoned after 200 lines or 32 KiB |
| List items (nested, ordered, lazy continuation, inside block quotes) | Located by a dedicated scanner and used as cut points: when a block must be split, item boundaries are preferred over line boundaries, and an item that fits is kept whole, even when blank lines inside it make it several blocks |
| Tables, block quotes, HTML blocks, indented code, link reference definitions | **No special handling.** They are ordinary text. A long table is cut preferably at line boundaries and its header row is not repeated |

### Measured agreement with `pulldown-cmark`

`cargo test --features diagnostics commonmark_block_boundary -- --nocapture` (after
`python scripts/fetch_commonmark.py`) compares the lines on which our parser starts blocks with the
lines on which `pulldown-cmark` starts top-level blocks, over the 652 examples of the CommonMark
0.31.2 spec. It compares block **start lines only**: not block contents, end positions or nested
structure. ATX headings are left out (they become `headers`, not blocks) and GFM tables are not
part of the corpus.

| Measurement | Result |
| --- | --- |
| Examples where every block start agrees | 496 / 652 (76.1 %) |
| Same, ignoring link reference definitions | 567 / 652 (87.0 %) |
| Our block starts with no reference counterpart | 145 / 844 (17.2 %); 70 / 769 (9.1 %) ignoring link reference definitions |
| Reference block starts we also start (recall) | paragraph 96.2 %, code block 96.9 %, HTML block 96.4 %, block quote 92.5 %, list 90.2 %, thematic break 40.6 % |

### Where we differ from CommonMark, and why

Three differences are deliberate. When choosing them, usefulness of the resulting chunks mattered
more than agreement with CommonMark. We have **not** benchmarked retrieval quality for these
choices, so the reasons below are design rationale, not measured results.

1. **Link reference definitions stay text.** CommonMark consumes them without producing a block,
   which accounts for 75 of the 145 extra starts. Chunks must cover the source text, so a
   definition is ordinary content here.
2. **A blank line always ends a block**, even inside lists, indented code and HTML blocks that
   CommonMark treats as one block. This accounts for 69 of the remaining 70 extra starts (the
   last one is an edge case involving a link reference definition, example 216). A blank line is
   the cheapest reliable boundary signal in prose, and it keeps the parser line-oriented without
   tracking containers. For lists the chunker compensates: an item that fits is kept whole across
   blank lines (covered by tests). Inside indented code and HTML blocks there is no such
   compensation, so a blank line there can become a cut point when the block does not fit.
3. **Lists, block quotes, thematic breaks and HTML blocks do not interrupt a paragraph.** They
   start a new block only after a blank line. A line that introduces a list ("Steps:") therefore
   stays in the same block as the list, and the two are never separated while the block fits.
   The cost is that a quote or thematic break written directly under a paragraph line belongs to
   that paragraph's block. All 52 block starts we miss begin directly after a non-blank line,
   where our parser starts a new block only at a fence or a heading; 32 of them are themselves
   lists, block quotes, thematic breaks or HTML blocks.

Do not use text_chipper as a CommonMark implementation. If you need an AST or the exact
CommonMark block structure, use a Markdown parser and chunk its output.

## ⚖️ Budget guarantee: `strict`

| | `strict=True` (default) | `strict=False` |
| --- | --- | --- |
| Final exact re-count of every delivered chunk | yes | no |
| `token_count <= max_length` guaranteed | yes | **no** |
| `token_count` | exact | may be an estimate |
| A single grapheme cluster larger than the budget | raises `ValueError` | returned as its own oversized chunk |
| Speed | slightly slower | faster |

Use strict mode whenever chunks are fed to an embedding model or a context window with a hard
limit. Non-strict mode is a best-effort mode for cases where the budget is only a target.


## ⚠️ Errors

Every failure is raised as an exception and nothing returns a partial or empty result for bad
input:

- `ValueError`: invalid configuration, unknown encoding, or (strict mode) a grapheme that does
  not fit the budget.
- `RuntimeError`: the tokenizer backend failed to load (e.g. missing or invalid `tokenizer_file`).
- `OSError`: unreadable file, invalid UTF-8 (unless `errors="replace"`), a physical line
  longer than 1 MiB in the streaming reader (`include_context=True`), or a file larger than
  32 MiB in flat mode (`include_context=False`).
- `chunk_batch` and `from_files` raise the error of the **first** failing input in input order
  and return no partial results.

## Known limitations

- Non-strict mode gives no budget guarantee (see above).
- Flat mode reads whole files (32 MiB limit); streaming with bounded memory is only available
  with `include_context=True`.
- Block recognition is limited to what [Markdown support](#markdown-support-and-commonmark-conformance)
  lists. In particular there is no table, block quote or HTML block handling, and a long table
  is split without repeating its header row.
- Custom Rust tokenizers are not available from Python yet, and a counter that is not
  deterministic or not monotone on prefixes can produce suboptimal chunk boundaries.
- With a Hugging Face tokenizer, a tokenizer failure while counting falls back to the character
  count (counting has no error channel). This should not happen for valid text; treat counts as
  best effort in that unlikely case.
- `from_file_stream(..., batch_size_lines=...)` accepts but ignores `batch_size_lines`.
- `from_file_stream` is single-threaded; on several cores, `chunk()` (or chunking in blocks) is faster
  but needs much more memory (see [Large files and streaming](#large-files-and-streaming)).

## 🧪 Development & Testing

```bash
cargo test --release                    # unit, oracle, property and robustness tests
pytest tests -v                         # Python binding tests (after `maturin develop`)
python scripts/fetch_commonmark.py      # refresh the pinned CommonMark 0.31.2 corpus
cargo test --features diagnostics commonmark_block_boundary -- --nocapture   # block-boundary agreement report
scripts/run_release_suite.sh          # all tests + diagnostics, with summary
scripts/run_validation.sh             # full validation matrix
pytest benchmarks/tests -q              # self-tests of the benchmark harness
```

Repository layout: `src/` library code; `src/tests/` Rust test modules (unit, oracle, parallel,
robustness, release suite); `tests/` Python binding tests and fixtures; `benches/` criterion
microbenchmarks; `benchmarks/` cross-library benchmark; `integrations/` LangChain and LlamaIndex adapters; `scripts/` validation and profiling tools.

The long-running reports, stress tests and the release suite are compiled only with the
`diagnostics` feature, so a plain `cargo test` lists just the regular tests. Run them with
`cargo test --release --features diagnostics -- --nocapture --test-threads=1`, or one at a time
by name.

Cargo features (none are on by default, so Rust users get a pure-Rust library): `python` (PyO3
bindings; used by the PyPI wheel, implies `mimalloc`), `mimalloc` (opt-in global allocator), `hf` (Hugging Face tokenizers), `stats` (tokenizer-call
profiling, see `scripts/profile_tokenizer.py`), `diagnostics` (long-running diagnostic tests and the
release suite).

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
