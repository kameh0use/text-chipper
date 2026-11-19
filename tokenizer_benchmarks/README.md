# Tokenizer guard acceptance benchmark

The guard is opt-in: pass one or more `guard_encodings` or `guard_tokenizer_files` to `TextChipper`.
The chunker then budgets against the maximum token count across the primary tokenizer and every
listed guard. Every delivered chunk fits each listed tokenizer; unlisted consumer tokenizers are
not covered. The clean-boundary residual fill runs only in this multi-tokenizer mode, so the
ordinary single-tokenizer path keeps its previous behavior.

The acceptance workflow uses a 512-token consumer limit and a 510-token chunk budget to reserve
two tokens for BGE's `[CLS]` and `[SEP]`. It runs 12 generated and pinned real-world workloads with
`cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, Qwen3, DeepSeek-V3, and BGE as guards. Every
guarded chunk is recounted with every configured tokenizer; the workflow fails unless every
guarded combination has zero overflow. Result files are uploaded as CI artifacts rather than
committed to the branch.

See the [benchmark workflow](../.github/workflows/tokenizer-guard-benchmark.yml) for the exact
acceptance command.

## Run locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install "tiktoken==0.14.0" "tokenizers==0.23.3" "numpy>=1.26" "huggingface_hub>=0.24" "pytest>=8"
maturin develop --release --features python,hf
python -m pytest benchmarks/tests -q
cd benchmarks && python -m chunkbench fetch && cd ..
python benchmarks/cross_tokenizer.py \
  --size-mib 0.05 \
  --libs text_chipper \
  --tokenizers cl100k_base o200k_base p50k_base r50k_base qwen deepseek BGE \
  --no-claude --guard --strict --require-real --assert-guard-zero \
  --out tokenizer_benchmarks/results/local
```

Tokenizer files and real corpora are pinned by [`benchmarks/tokenizers/tokenizers.lock.json`](../benchmarks/tokenizers/tokenizers.lock.json) and [`benchmarks/corpora.lock.json`](../benchmarks/corpora.lock.json). The acceptance run fails if any listed tokenizer is unavailable or any guarded chunk exceeds the budget.

## Results

Each result directory contains:

- `report.md`: readable summary and cross-tokenizer overflow tables.
- `results.json`: complete metrics, tokenizer revisions, hashes, and environment versions.
- `rows.csv`: per-workload rows for analysis.

GitHub Actions result directories also contain `commit.txt` and `pip-freeze.txt` for provenance.
