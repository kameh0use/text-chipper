# text_chipper

[![Rust Version](https://img.shields.io/badge/rust-1.88.0-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/kameh0use/text-chipper)

High-performance, Markdown-aware, strictly budget-enforced text chunking for RAG and LLM pipelines.
Built in Rust with Python bindings. 

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

![A plain tiktoken window splits a code block inside "timeout=10)", while text-chipper keeps the whole block in one chunk](examples/docs/before_after.svg)

Reproduce it with [`examples/before_after.py`](examples/before_after.py).


Other entry points: `chunk_batch(texts)`, `from_file(path)`, `from_files(paths)`,
`from_file_stream(path)` (an iterator), `count_tokens(text)`.

Each `TextChunk` has `text`, `headers`, `contextualized_text`, `token_count`, character/byte
offsets in the normalized text (`start_index`, `end_index`, `start_byte`, `end_byte`) and in
your original input (`orig_start_index`, `orig_end_index`, `orig_start_byte`, `orig_end_byte`).

Adapters for LangChain and LlamaIndex are in [`integrations/`](integrations/).

---



## 🛠️ Scope & Installation

### Scope

| Supported | Not supported |
| --- | --- |
| Markdown and plain text, passed as `str` or read from UTF-8 files | HTML, PDF, DOCX loaders |
| Tokenizers: `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, `chars`, `words`; Hugging Face `tokenizer.json` (build with the `hf` feature); any counter you supply from Rust through the `Tokenizer` trait (see [Pluggable tokenizers](#pluggable-tokenizers)) | Code-aware splitting (no AST, no function/class boundaries) |

Convert other formats to Markdown or text before chunking.

### Installation from source

Requires a Rust toolchain:

```bash
pip install maturin
maturin develop --release        # into the active virtualenv
# or: maturin build --release    # build a wheel into target/wheels/
```

---

## 📖 Chunking modes

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
- `chunk()` and `chunk_batch()` always behave this way.
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

Cargo features: `python` (default marker), `mimalloc` (default; disable if your Rust application
sets its own global allocator), `hf` (Hugging Face tokenizers), `stats` (tokenizer-call
profiling, see `scripts/profile_tokenizer.py`), `diagnostics` (long-running diagnostic tests and the
release suite).

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
