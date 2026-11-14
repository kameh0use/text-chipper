# Large-file benchmark: WikiText-103 raw, 514.3 MiB

512-token chunks, `cl100k_base`, no overlap, `fmt=plain`. One fresh process per run, median of 3 timed runs per mode
(plus one verify run per mode for the chunk checks). Time includes reading the file. Memory is the growth of the
peak process memory after warm-up. The `single` and `all` lanes ran as separate jobs on different virtual machines.
Raw data: `results_single.jsonl`, `results_all.jsonl`. Method: `benchmarks/streaming_bench.py`.

## Lane `single` (one CPU, thread variables set to 1)

| Library | File given as | Time | MiB/s | Memory growth | Chunks over 512 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| text_chipper | `from_file_stream` | 24 s | 21.3 | 71 MB | 0 |
| text_chipper | 8 MiB blocks | 28 s | 18.0 | 130 MB | 0 |
| text_chipper | whole file as one string | 35 s | 14.6 | 4.6 GB | 0 |
| tiktoken window | 8 MiB blocks | 39 s | 13.1 | 149 MB | 5 of 234,790 |
| chonkie RecursiveChunker | 8 MiB blocks | 41 s | 12.4 | 117 MB | 0 |
| tiktoken window | whole file as one string | 44 s | 11.7 | 7.7 GB | 6 of 234,759 |
| chonkie TokenChunker | 8 MiB blocks | 47 s | 11.0 | 210 MB | 5 of 234,790 |
| chonkie RecursiveChunker | whole file as one string | 48 s | 10.8 | 4.6 GB | 0 |
| chonkie TokenChunker | whole file as one string | 74 s | 7.0 | 9.0 GB | 6 of 234,759 |
| LangChain | 8 MiB blocks | 108 s | 4.8 | 83 MB | 10 of 292,611 |
| semchunk | 8 MiB blocks | 112 s | 4.6 | 3.3 GB | 0 |
| LangChain | whole file as one string | 118 s | 4.4 | 4.6 GB | 10 of 292,590 |
| semchunk | whole file as one string | 118 s | 4.4 | 6.6 GB | 0 |
| semantic-text-splitter | 8 MiB blocks | 158 s | 3.2 | 87 MB | 0 |
| semantic-text-splitter | whole file as one string | 491 s | 1.0 | 4.6 GB | 0 |

## Lane `all` (no restriction)

| Library | File given as | Time | MiB/s | Memory growth | Chunks over 512 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| text_chipper | 8 MiB blocks | 17 s | 30.9 | 274 MB | 0 |
| text_chipper | whole file as one string | 21 s | 24.2 | 4.6 GB | 0 |
| text_chipper | `from_file_stream` | 26 s | 19.8 | 71 MB | 0 |
| tiktoken window | 8 MiB blocks | 42 s | 12.3 | 143 MB | 5 of 234,790 |
| chonkie RecursiveChunker | 8 MiB blocks | 46 s | 11.3 | 110 MB | 0 |
| tiktoken window | whole file as one string | 48 s | 10.7 | 7.7 GB | 6 of 234,759 |
| chonkie TokenChunker | 8 MiB blocks | 50 s | 10.2 | 195 MB | 5 of 234,790 |
| chonkie RecursiveChunker | whole file as one string | 52 s | 9.9 | 4.6 GB | 0 |
| chonkie TokenChunker | whole file as one string | 76 s | 6.7 | 8.9 GB | 6 of 234,759 |
| LangChain | 8 MiB blocks | 111 s | 4.6 | 83 MB | 10 of 292,611 |
| semchunk | 8 MiB blocks | 114 s | 4.5 | 3.3 GB | 0 |
| LangChain | whole file as one string | 120 s | 4.3 | 4.6 GB | 10 of 292,590 |
| semchunk | whole file as one string | 122 s | 4.2 | 6.6 GB | 0 |
| semantic-text-splitter | 8 MiB blocks | 153 s | 3.4 | 80 MB | 0 |
| semantic-text-splitter | whole file as one string | 530 s | 1.0 | 4.6 GB | 0 |
