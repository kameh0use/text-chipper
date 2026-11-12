# Benchmark report

## Environment

- CPU: Apple M5 (10 logical CPUs), RAM: 16.0 GB
- OS: macOS-26.5.1-arm64-arm-64bit-Mach-O; Python 3.15.2 (CPython)
- Reference tokenizer: `cl100k_base`; budget: 512 tokens; overlaps: [0]; rounds: 1
- Versions: numpy 2.5.4, semchunk 4.1.1, text-chipper 0.2.0, tiktoken 0.14.0
- Project commit: c8046fa; run at 2026-11-12T21:06:36+00:00; load average before run: 1.67919921875

## How to read the speed numbers

- **MiB/s**: median over independent processes of the time of the *first and only* chunking call on the unmodified text. A fresh process cannot reuse a result or token-count cache, so these numbers are safe to quote. Import and tokenizer loading are excluded (`init_ms` shows them).
- **spread %**: (max - min) / median over the processes; a quick noise check.

## Speed summary (geometric mean of throughput over all workloads)

**lane `single`, overlap 0**

| library | geomean MiB/s | workloads |
| --- | ---: | ---: |
| text_chipper | 32.5 | 7/7 |
| semchunk | 4.4 | 7/7 |

## Results per workload

Columns are defined in `benchmarks/README.md`. `–` means not applicable or not measured. Failures are listed at the end.

### `multilingual` (synthetic, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 33.3 | 30.1 | 0 | 30 | 1.00 | 14 | 818 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.145 | 92.0 |
| semchunk | 4.0 | 252.7 | 0 | 253 | 1.00 | 14 | 814 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.148 | 92.3 |

### `stress_single_line` (stress, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semchunk | 5.1 | 197.0 | 0 | 197 | 1.00 | 4 | 609 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.087 | 95.7 |
| text_chipper | 38.0 | 26.2 | 0 | 26 | 1.00 | 10 | 594 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.085 | 95.7 |

### `code_heavy` (synthetic, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 38.2 | 26.2 | 0 | 26 | 1.00 | 15 | 920 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.253 | 88.0 |
| semchunk | 3.0 | 336.2 | 0 | 336 | 1.00 | 5 | 920 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.252 | 87.7 |

### `prose_en` (synthetic, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semchunk | 5.5 | 182.5 | 0 | 183 | 1.00 | 5 | 649 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.097 | 95.0 |
| text_chipper | 55.9 | 17.9 | 0 | 18 | 1.00 | 10 | 654 | 511 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.096 | 93.7 |

### `markdown_doc` (synthetic, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semchunk | 4.3 | 230.5 | 0 | 230 | 1.00 | 5 | 692 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 16.2 | 0.101 | 96.3 |
| text_chipper | 45.6 | 21.9 | 0 | 22 | 1.00 | 11 | 734 | 512 | 0.00 | 99.568 | 100.000 | 100.0 | 0.0 | 99.7 | 100.0 | 1.9 | 0.107 | 94.3 |

### `stress_giant_list` (stress, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 42.2 | 23.7 | 0 | 24 | 1.00 | 15 | 631 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.1 | 99.8 | – | 0.0 | 0.060 | 96.0 |
| semchunk | 5.0 | 200.1 | 0 | 200 | 1.00 | 4 | 630 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.060 | 96.0 |

### `stress_cjk_unbroken` (stress, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | spread % | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | body cov % | content cov % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 7.4 | 136.0 | 0 | 136 | 1.00 | 9 | 772 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| semchunk | 4.5 | 221.0 | 0 | 221 | 1.00 | 42 | 752 | 512 | 0.00 | 100.000 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
