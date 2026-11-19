# Tokenizer guard

Tokenizer guard is an opt-in mode for chunks that will be consumed by more than one tokenizer. Add
the extra tokenizers to `guard_encodings` and/or `guard_tokenizer_files`; the primary tokenizer still
comes from `encoding` or `tokenizer_file`.

```python
from text_chipper import TextChipper

chipper = TextChipper(
    max_length=510,  # reserve two positions for BGE's [CLS] and [SEP] under a 512-token limit
    encoding="o200k_base",
    guard_tokenizer_files=["bge-large-en-v1.5-tokenizer.json"],
)
chunks = chipper.chunk(text)
```

Every chunk is sized against the highest token count returned by the primary tokenizer and all
listed guards. In this example the configured set is `o200k_base` + BGE, so those chunks fit both.
An unlisted tokenizer is not covered. To cover several consumers, list each one as a guard. For
Hugging Face tokenizers, build with the `hf` feature and provide local `tokenizer.json` files.

## Behavior

- Guard mode is enabled by providing at least one guard. With no guards, the ordinary
  single-tokenizer path keeps its existing behavior.
- Each guard adds a tokenizer count pass. Chunks may be shorter because the strict budget follows
  whichever configured tokenizer produces the most tokens.
- Clean-boundary residual fill is also opt-in. It scans sentence and line boundaries in the next
  text block, then recounts the exact candidate before accepting it.
- The Python API counts tokenizer output. If a downstream model adds special tokens outside that
  count, reserve those positions in `max_length`.

Rust callers can combine counters with `Backend::max_of(...)`. Use
`Backend::max_of_with_overhead(...)` to add per-tokenizer overhead such as BOS, CLS, or SEP tokens.

## Verify the guard

The repository's [acceptance benchmark](../tokenizer_benchmarks/README.md) recounts delivered chunks
with every configured consumer tokenizer and fails if any guarded chunk exceeds the budget. Its
CI workflow runs the pinned real-world corpus and uploads result files as an artifact.
