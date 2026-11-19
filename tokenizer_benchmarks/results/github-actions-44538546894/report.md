# Cross-tokenizer matrix benchmark

Consumer limit **512** tokens, chunk budget **510** (headroom 2). Each returned `delivered` string is recounted with every tokenizer below. Rows of the matrices = the tokenizer the chunker was configured with, columns = the tokenizer that consumes the text. Diagonal cells (bold) are the chunker judged by its own tokenizer.

**Reading the numbers.** *Overflow* = share of chunks above the limit. *Excess* = tokens a truncating consumer would lose, as % of all tokens it saw (so 2 lost tokens and 292 lost tokens are not the same). *Fill* = chunk size / budget; a chunker can avoid overflow by under-filling, so check fill. Aggregates use the **natural** workloads (synthetic + real); the `stress_*` workloads are shown separately because they would dominate any pooled rate.

## Tokenizers

| Key | Source | Commit | SHA-256 | Vocab | Note |
| --- | --- | --- | --- | --- | --- |
| `cl100k_base` | tiktoken:cl100k_base | - | - | 100277 | GPT-3.5 / GPT-4 |
| `o200k_base` | tiktoken:o200k_base | - | - | 200019 | GPT-4o and later OpenAI models |
| `p50k_base` | tiktoken:p50k_base | - | - | 50281 | Codex / older OpenAI models |
| `r50k_base` | tiktoken:r50k_base | - | - | 50257 | GPT-2 / older OpenAI models |
| `qwen` | hf:Qwen/Qwen3-8B | `b968826d9c46` | `aeb13307a71a` | 151669 | Qwen2.5 / Qwen3 family tokenizer |
| `deepseek` | hf:deepseek-ai/DeepSeek-V3 | `e815299b0bcb` | `621ac2e32d0d` | 128815 | DeepSeek-V3 / R1 tokenizer |
| `BGE` | hf:BAAI/bge-large-en-v1.5 | `69e45f6c5f5c` | `d241a60d5e8f` | 30522 | embedding model; [CLS]/[SEP] are counted |

## 1. Does the chunker respect its own budget?

Diagonal (A = A) pairs, natural workloads. Cell = `budget violations · median fill · chunks under 50% fill`. A violation is a chunk that is over budget when recounted with the chunker's own tokenizer; it should be 0. For BGE the recount excludes [CLS]/[SEP], the overflow tables below include them.

| Library | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 0 · 0.95 · 8.3% | 0 · 0.95 · 7.5% | 0 · 0.95 · 9.8% | 0 · 0.96 · 9.9% | 0 · 0.95 · 7.3% | 0 · 0.96 · 7.4% | 0 · 0.96 · 7.5% |

## 2. Cross-tokenizer matrix: text_chipper

### Overflow rate (% of chunks over the limit)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | 0.1 | 17.5 | 18.4 | 47.4 | 1.5 | 14.6 |
| `o200k_base` | 9.2 | **0.0** | 21.1 | 21.8 | 56.0 | 7.5 | 19.4 |
| `p50k_base` | 21.9 | 18.2 | **0.0** | 1.5 | 42.2 | 18.4 | 10.5 |
| `r50k_base` | 20.7 | 17.5 | 0.0 | **0.0** | 41.5 | 18.3 | 10.6 |
| `qwen` | 2.7 | 0.0 | 11.7 | 12.1 | **0.0** | 1.3 | 10.5 |
| `deepseek` | 15.9 | 9.3 | 25.6 | 27.1 | 59.2 | **0.0** | 23.1 |
| `BGE` | 30.0 | 26.1 | 23.8 | 24.5 | 46.8 | 27.5 | **0.0** |
| `guard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Excess (% of tokens a truncating consumer loses)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.00** | 0.00 | 2.47 | 2.61 | 4.12 | 0.01 | 0.77 |
| `o200k_base` | 1.73 | **0.00** | 4.71 | 4.86 | 5.32 | 0.45 | 2.50 |
| `p50k_base` | 0.86 | 0.72 | **0.00** | 0.03 | 5.98 | 0.81 | 0.30 |
| `r50k_base` | 0.81 | 0.69 | 0.00 | **0.00** | 5.84 | 0.81 | 0.30 |
| `qwen` | 0.73 | 0.00 | 3.30 | 3.30 | **0.00** | 0.02 | 1.18 |
| `deepseek` | 1.54 | 0.23 | 4.37 | 4.53 | 5.23 | **0.00** | 2.29 |
| `BGE` | 1.94 | 1.51 | 2.81 | 3.06 | 7.14 | 1.59 | **0.00** |
| `guard` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Worst chunk (max token count under the consumer)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **510** | 513 | 980 | 980 | 657 | 521 | 714 |
| `o200k_base` | 1007 | **510** | 1733 | 1733 | 902 | 705 | 1102 |
| `p50k_base` | 564 | 564 | **510** | 531 | 710 | 567 | 556 |
| `r50k_base` | 564 | 564 | 510 | **510** | 710 | 567 | 556 |
| `qwen` | 785 | 507 | 1284 | 1284 | **510** | 530 | 943 |
| `deepseek` | 842 | 561 | 1395 | 1395 | 750 | **510** | 996 |
| `BGE` | 960 | 647 | 1200 | 1200 | 759 | 715 | **512** |
| `guard` | 503 | 498 | 510 | 510 | 510 | 497 | 510 |

### Median count difference (consumer − chunker tokens)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | -4.0 | -3.0 | -3.0 | 26.0 | -2.0 | -6.0 |
| `o200k_base` | 4.0 | **0.0** | -1.0 | -0.5 | 36.0 | 1.0 | -2.0 |
| `p50k_base` | 1.0 | -1.0 | **0.0** | 0.0 | 20.0 | -2.5 | -4.0 |
| `r50k_base` | 1.0 | -1.0 | 0.0 | **0.0** | 16.0 | -4.0 | -4.0 |
| `qwen` | -27.0 | -36.0 | -27.0 | -26.0 | **0.0** | -38.0 | -33.5 |
| `deepseek` | 2.0 | -2.0 | 1.0 | 2.0 | 39.0 | **0.0** | -4.0 |
| `BGE` | 6.0 | 2.0 | 6.0 | 6.0 | 27.0 | 3.0 | **2.0** |
| `guard` | -39.0 | -44.0 | -28.0 | -26.0 | 0.0 | -44.0 | -45.0 |

### p95 count difference (consumer − chunker tokens)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | 1.0 | 56.0 | 65.0 | 103.0 | 8.0 | 40.0 |
| `o200k_base` | 17.0 | **0.0** | 53.0 | 64.0 | 116.0 | 18.0 | 47.0 |
| `p50k_base` | 39.0 | 37.0 | **0.0** | 14.0 | 143.0 | 39.0 | 25.0 |
| `r50k_base` | 38.0 | 37.0 | 0.0 | **0.0** | 137.0 | 39.0 | 25.0 |
| `qwen` | 0.0 | -3.0 | 40.0 | 40.0 | **0.0** | 2.0 | 33.0 |
| `deepseek` | 37.0 | 22.0 | 53.0 | 63.0 | 125.0 | **0.0** | 61.0 |
| `BGE` | 63.0 | 56.0 | 78.0 | 103.0 | 164.0 | 58.0 | **2.0** |
| `guard` | -7.0 | -10.0 | 0.0 | 0.0 | 0.0 | -8.0 | 2.0 |

## 3. Library comparison

Natural workloads, pooled. *Self* = A = B. *Cross* = every A ≠ B pair (Claude excluded). The worst pair is the (chunker → consumer) combination with the highest overflow rate.

| Library | Self overflow % | Cross overflow % | Cross excess % | Worst pair |
| --- | ---: | ---: | ---: | ---: |
| text_chipper | 0.0 | 19.6 | 2.23 | `deepseek`→`qwen` (59.2%) |

## 3b. Guard tokenizers

The `guard` row chunks with the **maximum** count over every tokenizer above (Python: `guard_encodings` / `guard_tokenizer_files`), with the same budget. Overflow is pooled over all consumer tokenizers above (natural workloads); Claude cannot be a guard member, so it is not part of the guarantee. Median fill shows what the guard costs: a chunker can avoid overflow by under-filling.

### text_chipper

| Chunked with | Chunks | Over limit | Overflow % | Worst chunk | Median fill | Under 50% fill % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | 6265 | 891 | 14.2 | 980 | 0.95 | 8.3 |
| `o200k_base` | 6062 | 1170 | 19.3 | 1733 | 0.95 | 7.5 |
| `p50k_base` | 6454 | 1040 | 16.1 | 710 | 0.95 | 9.8 |
| `r50k_base` | 6489 | 1007 | 15.5 | 710 | 0.96 | 9.9 |
| `qwen` | 6706 | 367 | 5.5 | 1284 | 0.95 | 7.3 |
| `deepseek` | 6034 | 1381 | 22.9 | 1395 | 0.96 | 7.4 |
| `BGE` | 6083 | 1554 | 25.5 | 1200 | 0.96 | 7.5 |
| `guard` | 7028 | 0 | 0.0 | 510 | 0.94 | 7.1 |

### Guard chunk quality

These source-fidelity and boundary metrics are computed on the guarded chunks themselves. Token overflow remains a hard acceptance gate; quality metrics are optimization targets.

| Workload | Chunks | Content coverage % | Located % | Mid-word cuts % | Clean ends % | Fences intact % | Mean fill % | Tiny chunks % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prose_en` | 32 | 100.0 | 100.0 | 0.0 | 100.0 | – | 96.2 | 0.0 |
| `markdown_doc` | 39 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 90.3 | 0.0 |
| `multilingual` | 58 | 100.0 | 100.0 | 0.0 | 100.0 | – | 92.3 | 0.0 |
| `code_heavy` | 55 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 84.1 | 0.0 |
| `real_chatlogs` | 18 | 100.0 | 100.0 | 0.0 | 94.1 | – | 90.1 | 0.0 |
| `real_finance` | 428 | 100.0 | 100.0 | 0.0 | 98.4 | – | 85.8 | 1.2 |
| `real_pubmed` | 281 | 100.0 | 100.0 | 0.0 | 87.5 | – | 88.2 | 1.4 |
| `real_state_of_the_union` | 24 | 100.0 | 100.0 | 0.0 | 100.0 | – | 94.7 | 0.0 |
| `real_wikitexts` | 69 | 100.0 | 100.0 | 0.0 | 35.3 | – | 78.4 | 0.0 |

## 4. Per workload (overflow %)

### `cl100k_base` → `BGE`

| Workload | text_chipper |
| --- | ---: |
| prose_en | 41.9 |
| markdown_doc | 27.8 |
| multilingual | 34.2 |
| code_heavy | 10.6 |
| stress_single_line *(stress)* | 56.7 |
| stress_giant_list *(stress)* | 67.7 |
| stress_cjk_unbroken *(stress)* | 0.0 |
| real_chatlogs | 66.7 |
| real_finance | 1.9 |
| real_pubmed | 25.8 |
| real_state_of_the_union | 18.2 |
| real_wikitexts | 0.0 |
| **natural, pooled** | 14.6 |
| **natural, macro-average** | 25.2 |

### `BGE` → `BGE`

| Workload | text_chipper |
| --- | ---: |
| prose_en | 0.0 |
| markdown_doc | 0.0 |
| multilingual | 0.0 |
| code_heavy | 0.0 |
| stress_single_line *(stress)* | 0.0 |
| stress_giant_list *(stress)* | 0.0 |
| stress_cjk_unbroken *(stress)* | 0.0 |
| real_chatlogs | 0.0 |
| real_finance | 0.0 |
| real_pubmed | 0.0 |
| real_state_of_the_union | 0.0 |
| real_wikitexts | 0.0 |
| **natural, pooled** | 0.0 |
| **natural, macro-average** | 0.0 |

## 5. Claude (token-counting API, sampled)

Not run (`--no-claude`).

## Limits

This measures token counts and hard-limit overflow, not embedding or retrieval quality, and not the model's behaviour past the limit (most embedders truncate; most LLM APIs reject). Chunks are counted without BOS/EOS for LLM tokenizers; add 1 token per prompt if you need it. One seed and 0.25 MiB per synthetic workload: rates on the small real corpora (tens of chunks) are noisy. Raw per-workload rows, pooled statistics and package versions are in `results.json` / `rows.csv`.
