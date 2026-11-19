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
| text_chipper | 0 · 0.89 · 11.8% | 0 · 0.90 · 11.9% | 0 · 0.89 · 13.0% | 0 · 0.89 · 13.3% | 0 · 0.90 · 10.4% | 0 · 0.89 · 11.8% | 0 · 0.90 · 11.5% |

## 2. Cross-tokenizer matrix: text_chipper

### Overflow rate (% of chunks over the limit)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | 0.1 | 11.7 | 12.3 | 31.3 | 1.0 | 7.2 |
| `o200k_base` | 6.4 | **0.0** | 12.6 | 13.1 | 37.3 | 5.0 | 10.8 |
| `p50k_base` | 14.5 | 12.5 | **0.0** | 0.7 | 30.5 | 12.3 | 5.5 |
| `r50k_base` | 13.8 | 12.1 | 0.0 | **0.0** | 30.2 | 12.0 | 5.6 |
| `qwen` | 1.9 | 0.0 | 8.1 | 8.3 | **0.0** | 1.0 | 6.3 |
| `deepseek` | 8.9 | 4.2 | 13.2 | 14.8 | 35.7 | **0.0** | 11.2 |
| `BGE` | 20.4 | 17.7 | 16.8 | 17.2 | 35.4 | 17.7 | **0.0** |
| `guard` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Excess (% of tokens a truncating consumer loses)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.00** | 0.00 | 1.92 | 2.06 | 3.01 | 0.01 | 0.42 |
| `o200k_base` | 1.53 | **0.00** | 4.23 | 4.36 | 3.90 | 0.29 | 1.99 |
| `p50k_base` | 0.66 | 0.56 | **0.00** | 0.02 | 4.66 | 0.60 | 0.15 |
| `r50k_base` | 0.62 | 0.54 | 0.00 | **0.00** | 4.59 | 0.60 | 0.16 |
| `qwen` | 0.44 | 0.00 | 2.95 | 2.95 | **0.00** | 0.01 | 0.91 |
| `deepseek` | 1.08 | 0.12 | 3.56 | 3.71 | 3.76 | **0.00** | 1.49 |
| `BGE` | 1.51 | 1.12 | 2.36 | 2.61 | 5.71 | 1.16 | **0.00** |
| `guard` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Worst chunk (max token count under the consumer)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **510** | 514 | 1011 | 1011 | 659 | 521 | 676 |
| `o200k_base` | 997 | **510** | 1722 | 1722 | 851 | 684 | 1153 |
| `p50k_base` | 560 | 560 | **510** | 531 | 705 | 562 | 549 |
| `r50k_base` | 560 | 560 | 510 | **510** | 705 | 562 | 549 |
| `qwen` | 791 | 507 | 1184 | 1184 | **510** | 525 | 872 |
| `deepseek` | 865 | 551 | 1377 | 1377 | 750 | **510** | 1026 |
| `BGE` | 930 | 647 | 1119 | 1119 | 759 | 692 | **512** |
| `guard` | 508 | 497 | 510 | 510 | 510 | 504 | 511 |

### Median count difference (consumer − chunker tokens)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | -3.0 | -4.0 | -4.0 | 25.0 | -2.0 | -6.0 |
| `o200k_base` | 3.0 | **0.0** | -3.0 | -2.0 | 32.0 | 1.0 | -3.0 |
| `p50k_base` | 1.0 | -1.0 | **0.0** | 0.0 | 19.5 | -2.0 | -2.0 |
| `r50k_base` | 1.0 | -1.0 | 0.0 | **0.0** | 15.5 | -3.0 | -2.0 |
| `qwen` | -25.0 | -32.0 | -26.0 | -25.0 | **0.0** | -33.0 | -31.0 |
| `deepseek` | 2.0 | -1.0 | 0.0 | 0.0 | 34.0 | **0.0** | -4.0 |
| `BGE` | 6.0 | 2.0 | 4.0 | 4.0 | 26.0 | 2.0 | **2.0** |
| `guard` | -35.0 | -40.0 | -26.5 | -25.0 | 0.0 | -40.5 | -40.0 |

### p95 count difference (consumer − chunker tokens)

| chunked with ↓ / counted with → | `cl100k_base` | `o200k_base` | `p50k_base` | `r50k_base` | `qwen` | `deepseek` | `BGE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | **0.0** | 1.0 | 56.0 | 65.0 | 98.0 | 7.0 | 36.0 |
| `o200k_base` | 16.0 | **0.0** | 52.0 | 62.0 | 112.0 | 16.0 | 46.0 |
| `p50k_base` | 39.0 | 36.0 | **0.0** | 14.0 | 140.0 | 38.0 | 24.0 |
| `r50k_base` | 38.0 | 36.0 | 0.0 | **0.0** | 133.0 | 38.0 | 24.0 |
| `qwen` | 0.0 | -3.0 | 35.0 | 35.0 | **0.0** | 2.0 | 29.0 |
| `deepseek` | 37.0 | 20.0 | 50.0 | 63.0 | 115.0 | **0.0** | 57.0 |
| `BGE` | 61.0 | 52.0 | 78.0 | 97.0 | 160.0 | 51.0 | **2.0** |
| `guard` | -6.0 | -8.0 | 0.0 | 0.0 | 0.0 | -8.0 | 2.0 |

## 3. Library comparison

Natural workloads, pooled. *Self* = A = B. *Cross* = every A ≠ B pair (Claude excluded). The worst pair is the (chunker → consumer) combination with the highest overflow rate.

| Library | Self overflow % | Cross overflow % | Cross excess % | Worst pair |
| --- | ---: | ---: | ---: | ---: |
| text_chipper | 0.0 | 12.7 | 1.76 | `o200k_base`→`qwen` (37.3%) |

## 3b. Guard tokenizers

The `guard` row chunks with the **maximum** count over every tokenizer above (Python: `guard_encodings` / `guard_tokenizer_files`), with the same budget. Overflow is pooled over all consumer tokenizers above (natural workloads); Claude cannot be a guard member, so it is not part of the guarantee. Median fill shows what the guard costs: a chunker can avoid overflow by under-filling.

### text_chipper

| Chunked with | Chunks | Over limit | Overflow % | Worst chunk | Median fill | Under 50% fill % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cl100k_base` | 6804 | 619 | 9.1 | 1011 | 0.89 | 11.8 |
| `o200k_base` | 6601 | 804 | 12.2 | 1722 | 0.90 | 11.9 |
| `p50k_base` | 7028 | 763 | 10.9 | 705 | 0.89 | 13.0 |
| `r50k_base` | 7056 | 742 | 10.5 | 705 | 0.89 | 13.3 |
| `qwen` | 7217 | 264 | 3.7 | 1184 | 0.90 | 10.4 |
| `deepseek` | 6622 | 833 | 12.6 | 1377 | 0.89 | 11.8 |
| `BGE` | 6622 | 1184 | 17.9 | 1119 | 0.90 | 11.5 |
| `guard` | 7616 | 0 | 0.0 | 511 | 0.88 | 9.8 |

### Guard chunk quality

These source-fidelity and boundary metrics are computed on the guarded chunks themselves. Token overflow remains a hard acceptance gate; quality metrics are optimization targets.

| Workload | Chunks | Content coverage % | Located % | Mid-word cuts % | Clean ends % | Fences intact % | Mean fill % | Tiny chunks % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prose_en` | 35 | 100.0 | 100.0 | 0.0 | 100.0 | – | 87.7 | 0.0 |
| `markdown_doc` | 40 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 88.0 | 0.0 |
| `multilingual` | 70 | 100.0 | 100.0 | 0.0 | 100.0 | – | 76.2 | 0.0 |
| `code_heavy` | 55 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 84.1 | 0.0 |
| `real_chatlogs` | 19 | 100.0 | 100.0 | 0.0 | 83.3 | – | 85.4 | 0.0 |
| `real_finance` | 464 | 100.0 | 100.0 | 0.0 | 97.4 | – | 79.2 | 1.3 |
| `real_pubmed` | 312 | 100.0 | 100.0 | 0.0 | 86.8 | – | 79.4 | 0.0 |
| `real_state_of_the_union` | 24 | 100.0 | 100.0 | 0.0 | 100.0 | – | 94.6 | 0.0 |
| `real_wikitexts` | 69 | 100.0 | 100.0 | 0.0 | 35.3 | – | 78.4 | 0.0 |

## 4. Per workload (overflow %)

### `cl100k_base` → `BGE`

| Workload | text_chipper |
| --- | ---: |
| prose_en | 21.2 |
| markdown_doc | 10.5 |
| multilingual | 29.3 |
| code_heavy | 10.6 |
| stress_single_line *(stress)* | 56.7 |
| stress_giant_list *(stress)* | 67.7 |
| stress_cjk_unbroken *(stress)* | 0.0 |
| real_chatlogs | 66.7 |
| real_finance | 1.7 |
| real_pubmed | 7.8 |
| real_state_of_the_union | 0.0 |
| real_wikitexts | 0.0 |
| **natural, pooled** | 7.2 |
| **natural, macro-average** | 16.4 |

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
