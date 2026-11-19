# Tokenizer budget guarantee benchmark

This benchmark measures the fill of chunks made under a hard token budget and recounts each delivered chunk with every configured tokenizer. It keeps the tokenizer guard experiment separate from the main benchmark suite.

## Current result

The reference run used a 512-token consumer limit and a 510-token chunk budget, leaving two tokens for BGE's `[CLS]` and `[SEP]`. It covered 12 generated and pinned real-world workloads. The guard included `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`, Qwen3, DeepSeek-V3, and BGE.

| Measure | Previous fill strategy | Clean-boundary fill |
| --- | ---: | ---: |
| Natural workloads, mean fill | 80.3% | 87.1% |
| Natural workloads, median fill | 88.4% | 94.3% |
| All workloads, mean fill | 82.0% | 88.1% |
| All workloads, median fill | 90.4% | 94.9% |
| Cross-tokenizer overflow | 0 | 0 |
| Content coverage | 100% | 100% |

The updated strategy fills a pending chunk from the next text block when the added prefix reaches a sentence or line boundary. It recounts the exact delivered candidate before accepting it. Code blocks remain intact. Markdown fence integrity remained 100% in both runs.

The baseline is the local reference run; the updated result is from a GitHub-hosted Linux runner. The workflow uploads later runs as artifacts without committing generated files automatically.

- [Previous fill strategy](results/fill-baseline/report.md)
- [Clean-boundary fill](results/fill-clean-boundary/report.md)
- [GitHub Actions run 44538546894](results/github-actions-44538546894/report.md)
- [Benchmark workflow](../.github/workflows/tokenizer-guard-benchmark.yml)

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
