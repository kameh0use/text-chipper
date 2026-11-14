# Benchmarks

Two benchmarks live here. Both compare text_chipper with the same libraries (chonkie, langchain-text-splitters,
semchunk, semantic-text-splitter and a plain tiktoken token window), configured by the same adapters in
[`chunkbench/adapters.py`](chunkbench/adapters.py), at 512 tokens with `cl100k_base`.

| Benchmark | Question | Entry point | Results |
| --- | --- | --- | --- |
| Main harness | Speed, memory and chunk quality over 12 workloads of up to about 1 MB | `python -m chunkbench` | [`results/20261113_213722/`](results/20261113_213722/) |
| Large file | What happens on one 514 MiB file, and what does streaming change? | [`streaming_bench.py`](streaming_bench.py) | [`results/streaming_20261114/`](results/streaming_20261114/) |

The charts used in the main README are in [`charts/`](charts/).

## Main harness

```bash
pip install -r requirements.txt
python -m chunkbench check     # are all libraries importable?
python -m chunkbench fetch     # download the real corpora once
python -m chunkbench run       # run the workloads (see --help for lanes, rounds and workloads)
python -m chunkbench report results/<run>
python -m chunkbench plot results/<run>
```

Each run writes a folder under `results/` with `results.json`, `report.md`, `pip_freeze.txt` and `commit.txt`.
`report.md` explains how to read the numbers. Lanes: `single` pins the process to one CPU and sets the thread
variables to 1, `all` applies no restriction.

## Large-file benchmark

[`streaming_bench.py`](streaming_bench.py) chunks one big file. One run is one mode in one fresh process:

| Mode | What it does |
| --- | --- |
| `text_chipper:stream` | `TextChipper.from_file_stream(path)`, chunks consumed one at a time |
| `<adapter>:whole` | read the whole file into one string, call the library once |
| `<adapter>:blocks` | give the library blocks of about 8 MiB cut at line boundaries (one call per block) |

`<adapter>` is a name from `chunkbench.adapters.ADAPTERS` (`text_chipper`, `tiktoken_window`, `chonkie_token`,
`chonkie_recursive`, `semchunk`, `semantic_text_splitter`, `langchain`). `text_chipper:stream` exists only for
`text_chipper`.

What is measured:

- **Time:** end to end for the chunking work, including reading the file. Verification is not timed.
- **Memory:** growth of the process's peak memory after warm-up (library imported, tokenizer loaded). Memory comes
  from runs without `--verify`, because verification builds extra objects.
- **Verification (`--verify`):** counts the tokens of every delivered chunk with the reference tokenizer and hashes the
  delivered text. Identical hashes mean identical chunks.
- **Lanes:** `--lane single` (default: one CPU, thread variables 1) or `--lane all`.

Run it on your machine:

```bash
python benchmarks/prepare_corpus.py corpus.txt           # WikiText-103 raw, prints size and SHA-256
python benchmarks/streaming_bench.py corpus.txt --mode text_chipper:stream --lane single
python benchmarks/streaming_bench.py corpus.txt --mode langchain:blocks --lane single --verify
```

The GitHub Actions workflow [`streaming-bench.yml`](../.github/workflows/streaming-bench.yml) runs every mode in both
lanes on separate runners. Inputs: `modes` (`chipper`, `blocks` or `all`), `max_mib` (use only the first N MiB of the
corpus, 0 = full), `rounds` (timed runs per mode, default 3, plus one verify run) and `timeout_s`. Then:

```bash
python benchmarks/summarize_streaming.py results.jsonl          # Markdown table
python benchmarks/plot_streaming.py results/streaming_<date>/*.jsonl   # charts into benchmarks/charts/
```

### Latest results (514.3 MiB)

See [`results/streaming_20261114/summary.md`](results/streaming_20261114/summary.md) for both lanes and every mode,
and the "Large files and streaming" section of the main README for the discussion.

Single-core lane:

![Throughput on a 514 MiB file, single-core lane](charts/streaming_speed_single.png)

![Memory growth on a 514 MiB file, single-core lane](charts/streaming_memory_single.png)

All-cores lane:

![Throughput on a 514 MiB file, all-cores lane](charts/streaming_speed_all.png)

![Memory growth on a 514 MiB file, all-cores lane](charts/streaming_memory_all.png)

### Limits

- One workflow run per lane on shared virtual machines. The two lanes ran on different machines, so do not compare lanes
  for single-threaded modes. Treat ratios as indicative.
- Block mode is our choice of how to feed a string-only library; chunks at block edges can differ from a one-call run.
- In whole-file mode the memory peak is dominated by reading the file into a Python string, which every library pays.
- Retrieval quality is not measured.
