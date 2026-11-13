# Benchmark report

## Environment

- CPU: Apple M5 (10 logical CPUs), RAM: 16.0 GB
- OS: macOS-26.5.1-arm64-arm-64bit-Mach-O; Python 3.15.2 (CPython)
- Reference tokenizer: `cl100k_base`; budget: 512 tokens; overlaps: [0]; rounds: 3
- Versions: chonkie 1.7.0, langchain-text-splitters 1.1.3, numpy 2.5.4, semantic-text-splitter 0.33.2, text-chipper 0.2.0, tiktoken 0.14.0
- Project commit: n/a; run at 2026-11-11T11:04:41+00:00; load average before run: 2.58544921875

## Speed summary (geometric mean of throughput over all workloads)

**lane `all`, overlap 0**

| library | geomean MiB/s | workloads |
| --- | ---: | ---: |
| LangChain | 2.8 | 7/7 |
| text_chipper | 98.5 | 7/7 |
| tiktoken window | 12.2 | 7/7 |
| semantic-text-splitter | 1.8 | 7/7 |
| chonkie RecursiveChunker | 12.7 | 7/7 |
| chonkie TokenChunker | 11.0 | 7/7 |
| text_chipper (flat) | 33.8 | 7/7 |

**lane `single`, overlap 0**

| library | geomean MiB/s | workloads |
| --- | ---: | ---: |
| tiktoken window | 12.2 | 7/7 |
| chonkie RecursiveChunker | 12.7 | 7/7 |
| chonkie TokenChunker | 11.0 | 7/7 |
| LangChain | 2.8 | 7/7 |
| text_chipper (flat) | 25.6 | 7/7 |
| semantic-text-splitter | 1.8 | 7/7 |
| text_chipper | 38.7 | 7/7 |

## Results per workload

Columns are defined in `benchmarks/README.md`. `–` means not applicable or not measured. Failures are listed at the end.

### `markdown_doc` (synthetic, markdown) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiktoken window | 11.3 | 266.7 | 267 | 1.00 | 24 | 1,869 | 512 | 0.00 | 100.000 | 100.0 | 25.5 | 7.1 | 77.8 | 3.3 | 0.092 | 86.0 |
| semantic-text-splitter | 1.1 | 2,730.7 | 2,731 | 1.00 | 11 | 2,573 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.2 | 100.0 | 7.4 | 0.122 | 94.3 |
| chonkie TokenChunker | 10.3 | 289.9 | 315 | 1.09 | 40 | 1,869 | 512 | 0.00 | 100.000 | 100.0 | 25.5 | 7.1 | 77.8 | 3.3 | 0.092 | 86.0 |
| LangChain | 3.5 | 866.9 | 867 | 1.00 | 10 | 2,524 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.0 | 9.1 | 9.0 | 0.109 | 89.3 |
| text_chipper | 166.5 | 18.0 | 67 | 3.73 | 305 | 2,216 | 512 | 0.00 | 99.459 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.116 | 95.3 |
| chonkie RecursiveChunker | 9.8 | 306.5 | 307 | 1.00 | 26 | 1,981 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 76.7 | 8.9 | 0.104 | 96.3 |
| text_chipper (flat) | 47.8 | 62.7 | 89 | 1.42 | 64 | 2,077 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 98.3 | 100.0 | 16.6 | 0.107 | 94.7 |

### `markdown_doc` (synthetic, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiktoken window | 11.2 | 267.4 | 267 | 1.00 | 24 | 1,869 | 512 | 0.00 | 100.000 | 100.0 | 25.5 | 7.1 | 77.8 | 3.3 | 0.092 | 86.0 |
| chonkie TokenChunker | 10.2 | 293.7 | 318 | 1.08 | 41 | 1,869 | 512 | 0.00 | 100.000 | 100.0 | 25.5 | 7.1 | 77.8 | 3.3 | 0.092 | 86.0 |
| chonkie RecursiveChunker | 9.8 | 305.8 | 306 | 1.00 | 26 | 1,981 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 76.7 | 8.9 | 0.104 | 96.3 |
| LangChain | 3.6 | 825.3 | 825 | 1.00 | 10 | 2,524 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.0 | 9.1 | 9.0 | 0.109 | 89.3 |
| text_chipper (flat) | 34.5 | 87.0 | 87 | 1.00 | 53 | 2,077 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 98.3 | 100.0 | 16.6 | 0.107 | 94.7 |
| semantic-text-splitter | 1.1 | 2,706.2 | 2,706 | 1.00 | 11 | 2,573 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.2 | 100.0 | 7.4 | 0.122 | 94.3 |
| text_chipper | 52.4 | 57.3 | 57 | 1.00 | 36 | 2,216 | 512 | 0.00 | 99.459 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.116 | 95.3 |

### `stress_giant_list` (stress, markdown) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangChain | 3.9 | 767.2 | 767 | 1.00 | 9 | 1,956 | 496 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.064 | 97.3 |
| text_chipper | 131.7 | 22.8 | 60 | 2.65 | 112 | 1,883 | 512 | 0.00 | 100.000 | 100.0 | 0.1 | 99.8 | – | 0.0 | 0.062 | 97.7 |
| chonkie RecursiveChunker | 10.0 | 298.6 | 299 | 1.00 | 26 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| chonkie TokenChunker | 10.8 | 277.6 | 299 | 1.08 | 41 | 1,831 | 512 | 0.00 | 100.000 | 100.0 | 31.0 | 7.9 | – | 0.0 | 0.058 | 93.0 |
| semantic-text-splitter | 2.2 | 1,350.9 | 1,351 | 1.00 | 11 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| text_chipper (flat) | 52.1 | 57.6 | 80 | 1.39 | 56 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| tiktoken window | 12.0 | 250.9 | 251 | 1.00 | 24 | 1,831 | 512 | 0.00 | 100.000 | 100.0 | 31.0 | 7.9 | – | 0.0 | 0.058 | 93.0 |

### `stress_giant_list` (stress, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chonkie RecursiveChunker | 9.9 | 304.1 | 304 | 1.00 | 26 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| text_chipper | 48.3 | 62.1 | 62 | 1.00 | 53 | 1,883 | 512 | 0.00 | 100.000 | 100.0 | 0.1 | 99.8 | – | 0.0 | 0.062 | 97.7 |
| tiktoken window | 12.0 | 250.1 | 250 | 1.00 | 24 | 1,831 | 512 | 0.00 | 100.000 | 100.0 | 31.0 | 7.9 | – | 0.0 | 0.058 | 93.0 |
| semantic-text-splitter | 2.2 | 1,356.0 | 1,356 | 1.00 | 11 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| LangChain | 3.9 | 762.7 | 763 | 1.00 | 9 | 1,956 | 496 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.064 | 97.3 |
| text_chipper (flat) | 38.0 | 79.0 | 79 | 1.00 | 55 | 1,882 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | 0.0 | 0.062 | 97.7 |
| chonkie TokenChunker | 10.7 | 279.3 | 301 | 1.08 | 41 | 1,831 | 512 | 0.00 | 100.000 | 100.0 | 31.0 | 7.9 | – | 0.0 | 0.058 | 93.0 |

### `stress_cjk_unbroken` (stress, plain) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangChain | 0.9 | 2,119.5 | 2,118 | 1.00 | 64 | 1,626 | 492 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| semantic-text-splitter | 0.4 | 5,420.0 | 5,420 | 1.00 | 6 | 1,504 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| tiktoken window | 16.0 | 124.6 | 125 | 1.00 | 76 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| text_chipper (flat) | 7.5 | 267.9 | 329 | 1.23 | 50 | 1,542 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| chonkie RecursiveChunker | 12.5 | 159.9 | 165 | 1.03 | 91 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| chonkie TokenChunker | 14.0 | 142.8 | 148 | 1.04 | 82 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| text_chipper | 15.0 | 133.4 | 260 | 1.95 | 56 | 1,543 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |

### `stress_cjk_unbroken` (stress, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chonkie RecursiveChunker | 12.6 | 158.5 | 164 | 1.03 | 91 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| LangChain | 0.9 | 2,121.7 | 2,122 | 1.00 | 64 | 1,626 | 492 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| chonkie TokenChunker | 13.9 | 143.4 | 149 | 1.04 | 82 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| text_chipper | 7.8 | 257.7 | 258 | 1.00 | 28 | 1,543 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| text_chipper (flat) | 6.1 | 325.4 | 325 | 1.00 | 31 | 1,542 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |
| tiktoken window | 16.1 | 124.5 | 125 | 1.00 | 79 | 1,502 | 512 | 0.00 | 67.225 | 67.2 | 0.0 | 0.0 | – | – | – | – |
| semantic-text-splitter | 0.4 | 5,424.8 | 5,425 | 1.00 | 6 | 1,504 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 0.0 | – | – | – | – |

### `stress_single_line` (stress, plain) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper | 141.5 | 21.1 | 62 | 2.93 | 68 | 1,782 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.9 | – | – | 0.094 | 97.0 |
| tiktoken window | 13.1 | 227.5 | 228 | 1.00 | 22 | 1,738 | 512 | 0.00 | 100.000 | 100.0 | 28.6 | 5.1 | – | – | 0.089 | 91.0 |
| chonkie TokenChunker | 12.0 | 248.5 | 275 | 1.11 | 39 | 1,738 | 512 | 0.00 | 100.000 | 100.0 | 28.6 | 5.1 | – | – | 0.089 | 91.0 |
| chonkie RecursiveChunker | 6.0 | 501.3 | 501 | 1.00 | 39 | 1,866 | 493 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.099 | 96.0 |
| text_chipper (flat) | 46.6 | 64.1 | 87 | 1.36 | 82 | 1,781 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.094 | 96.7 |
| LangChain | 1.5 | 2,019.5 | 2,018 | 1.00 | 181 | 1,740 | 514 | 2.59 | 100.000 | 100.0 | 0.0 | 7.2 | – | – | 0.089 | 90.7 |
| semantic-text-splitter | 3.3 | 909.2 | 909 | 1.00 | 4 | 1,781 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.094 | 96.7 |

### `stress_single_line` (stress, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangChain | 1.5 | 2,013.4 | 2,013 | 1.00 | 181 | 1,740 | 514 | 2.59 | 100.000 | 100.0 | 0.0 | 7.2 | – | – | 0.089 | 90.7 |
| tiktoken window | 12.9 | 232.7 | 233 | 1.00 | 23 | 1,738 | 512 | 0.00 | 100.000 | 100.0 | 28.6 | 5.1 | – | – | 0.089 | 91.0 |
| chonkie TokenChunker | 12.1 | 247.7 | 274 | 1.10 | 39 | 1,738 | 512 | 0.00 | 100.000 | 100.0 | 28.6 | 5.1 | – | – | 0.089 | 91.0 |
| chonkie RecursiveChunker | 6.0 | 496.1 | 496 | 1.00 | 39 | 1,866 | 493 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.099 | 96.0 |
| semantic-text-splitter | 3.3 | 902.4 | 902 | 1.00 | 4 | 1,781 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.094 | 96.7 |
| text_chipper | 47.9 | 62.4 | 62 | 1.00 | 32 | 1,782 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 99.9 | – | – | 0.094 | 97.0 |
| text_chipper (flat) | 34.8 | 85.9 | 86 | 1.00 | 78 | 1,781 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.094 | 96.7 |

### `prose_en` (synthetic, plain) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-text-splitter | 3.8 | 792.4 | 792 | 1.00 | 4 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| chonkie TokenChunker | 12.0 | 250.8 | 277 | 1.11 | 39 | 1,751 | 512 | 0.00 | 100.000 | 100.0 | 30.0 | 4.8 | – | – | 0.085 | 93.0 |
| LangChain | 4.7 | 637.1 | 637 | 1.00 | 8 | 1,966 | 508 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.096 | 96.0 |
| chonkie RecursiveChunker | 13.2 | 226.5 | 227 | 1.00 | 23 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| tiktoken window | 12.9 | 232.2 | 232 | 1.00 | 23 | 1,751 | 512 | 0.00 | 100.000 | 100.0 | 30.0 | 4.8 | – | – | 0.085 | 93.0 |
| text_chipper (flat) | 52.3 | 57.3 | 80 | 1.40 | 50 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| text_chipper | 189.6 | 15.8 | 54 | 3.44 | 298 | 1,958 | 511 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.096 | 95.3 |

### `prose_en` (synthetic, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_chipper (flat) | 37.6 | 79.7 | 80 | 1.00 | 53 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| tiktoken window | 12.9 | 232.0 | 232 | 1.00 | 23 | 1,751 | 512 | 0.00 | 100.000 | 100.0 | 30.0 | 4.8 | – | – | 0.085 | 93.0 |
| LangChain | 5.0 | 597.0 | 597 | 1.00 | 7 | 1,966 | 508 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.096 | 96.0 |
| semantic-text-splitter | 3.8 | 799.8 | 800 | 1.00 | 4 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| chonkie TokenChunker | 12.0 | 251.0 | 277 | 1.10 | 39 | 1,751 | 512 | 0.00 | 100.000 | 100.0 | 30.0 | 4.8 | – | – | 0.085 | 93.0 |
| chonkie RecursiveChunker | 13.3 | 225.8 | 226 | 1.00 | 22 | 1,943 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.095 | 96.0 |
| text_chipper | 66.4 | 45.2 | 45 | 1.00 | 36 | 1,958 | 511 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.096 | 95.3 |

### `multilingual` (synthetic, plain) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiktoken window | 14.2 | 211.9 | 212 | 1.00 | 84 | 2,080 | 513 | 1.39 | 72.531 | 72.2 | 36.8 | 4.9 | – | – | 0.095 | 62.0 |
| semantic-text-splitter | 3.5 | 861.9 | 862 | 1.00 | 10 | 2,426 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| LangChain | 5.2 | 578.6 | 579 | 1.00 | 11 | 2,449 | 510 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.156 | 90.7 |
| text_chipper (flat) | 27.1 | 110.8 | 151 | 1.36 | 84 | 2,428 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| chonkie RecursiveChunker | 98.9 | 30.3 | 30 | 1.00 | 179 | 2,428 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| chonkie TokenChunker | 12.5 | 239.6 | 247 | 1.03 | 123 | 2,080 | 513 | 1.39 | 72.531 | 72.2 | 36.8 | 4.9 | – | – | 0.095 | 62.0 |
| text_chipper | 95.9 | 31.3 | 158 | 5.04 | 264 | 2,437 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.156 | 90.7 |

### `multilingual` (synthetic, plain) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-text-splitter | 3.5 | 861.8 | 862 | 1.00 | 11 | 2,426 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| text_chipper (flat) | 22.6 | 132.8 | 133 | 1.00 | 60 | 2,428 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| chonkie TokenChunker | 12.6 | 238.0 | 245 | 1.03 | 123 | 2,080 | 513 | 1.39 | 72.531 | 72.2 | 36.8 | 4.9 | – | – | 0.095 | 62.0 |
| chonkie RecursiveChunker | 98.3 | 30.5 | 31 | 1.00 | 177 | 2,428 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.155 | 90.0 |
| LangChain | 5.2 | 580.6 | 581 | 1.00 | 11 | 2,449 | 510 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.156 | 90.7 |
| tiktoken window | 14.1 | 212.3 | 212 | 1.00 | 84 | 2,080 | 513 | 1.39 | 72.531 | 72.2 | 36.8 | 4.9 | – | – | 0.095 | 62.0 |
| text_chipper | 45.0 | 66.6 | 67 | 1.00 | 45 | 2,437 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | – | – | 0.156 | 90.7 |

### `code_heavy` (synthetic, markdown) — lane `all`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-text-splitter | 1.8 | 1,707.8 | 1,708 | 1.00 | 6 | 2,764 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.255 | 89.0 |
| tiktoken window | 7.8 | 385.6 | 386 | 1.00 | 55 | 2,430 | 512 | 0.00 | 100.000 | 100.0 | 16.8 | 12.6 | 79.2 | 0.0 | 0.202 | 72.3 |
| LangChain | 2.8 | 1,065.8 | 1,066 | 1.00 | 8 | 2,831 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 72.3 | 0.0 | 0.231 | 69.7 |
| text_chipper (flat) | 41.4 | 72.5 | 113 | 1.56 | 70 | 2,767 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.257 | 89.0 |
| text_chipper | 106.1 | 28.3 | 143 | 5.07 | 364 | 2,765 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.256 | 89.0 |
| chonkie TokenChunker | 7.0 | 426.8 | 444 | 1.04 | 71 | 2,430 | 512 | 0.00 | 100.000 | 100.0 | 16.8 | 12.6 | 79.2 | 0.0 | 0.202 | 72.3 |
| chonkie RecursiveChunker | 5.6 | 534.8 | 535 | 1.00 | 35 | 2,475 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 78.3 | 0.0 | 0.211 | 76.0 |

### `code_heavy` (synthetic, markdown) — lane `single`, overlap 0

| library | MiB/s | median ms | CPU ms | CPU/wall | RSS Δ MB | chunks | max tok | over % | coverage % | located % | mid-word % | clean end % | fence ok % | orphan h % | IoU | 1-chunk % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic-text-splitter | 1.8 | 1,707.5 | 1,708 | 1.00 | 6 | 2,764 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.255 | 89.0 |
| text_chipper (flat) | 30.6 | 98.1 | 98 | 1.00 | 53 | 2,767 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.257 | 89.0 |
| tiktoken window | 7.8 | 387.0 | 387 | 1.00 | 55 | 2,430 | 512 | 0.00 | 100.000 | 100.0 | 16.8 | 12.6 | 79.2 | 0.0 | 0.202 | 72.3 |
| LangChain | 2.8 | 1,072.9 | 1,073 | 1.00 | 7 | 2,831 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 72.3 | 0.0 | 0.231 | 69.7 |
| chonkie RecursiveChunker | 5.6 | 537.6 | 538 | 1.00 | 35 | 2,475 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 78.3 | 0.0 | 0.211 | 76.0 |
| chonkie TokenChunker | 7.0 | 429.1 | 446 | 1.04 | 71 | 2,430 | 512 | 0.00 | 100.000 | 100.0 | 16.8 | 12.6 | 79.2 | 0.0 | 0.202 | 72.3 |
| text_chipper | 46.1 | 65.1 | 65 | 1.00 | 52 | 2,765 | 512 | 0.00 | 100.000 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.256 | 89.0 |
