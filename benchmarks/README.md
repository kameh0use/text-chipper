# Benchmarks

The benchmark suite compares text_chipper with other chunking libraries on speed, memory use and chunk quality.
The main run uses 12 workloads of up to about 1 MiB; the large-file run checks streaming on a 514.3 MiB text file.
Both use 512-token chunks, `cl100k_base` and no overlap unless noted.

| Benchmark | Question | Entry point | Current results |
| --- | --- | --- | --- |
| Main harness | Speed, memory and chunk quality over 12 workloads | `python -m chunkbench` | [`results/20261117_133546/`](results/20261117_133546/) |
| Large file | What happens on one 514.3 MiB file, and what does streaming change? | [`streaming_bench.py`](streaming_bench.py) | [`results/streaming_20261117/`](results/streaming_20261117/) |

The main README charts are in [`charts/`](charts/). The latest large-file result includes only the single-core lane; all-cores charts from earlier runs are not current comparisons.

## Main harness

```bash
pip install -r requirements.txt
python -m chunkbench check     # are all libraries importable?
python -m chunkbench fetch     # download the real corpora once
python -m chunkbench run       # run workloads (see --help for lanes, rounds and workloads)
python -m chunkbench report results/<run>
python -m chunkbench plot results/<run>
```

Each run writes `results.json`, `report.md`, `summary.csv`, `pip_freeze.txt` and `commit.txt`. The 17 November run used
five cold runs on a GitHub Actions runner (AMD EPYC 9V45, 4 logical CPUs, Python 3.12.16). Its report explains the
metrics and limitations. Lanes: `single` pins the process to one CPU and sets thread variables to 1; `all` applies no
restriction. The README summary charts use the `single` lane.

### Latest 12-workload results

The [per-workload report](results/20261117_133546/report.md) and [raw results](results/20261117_133546/results.json) include every library and metric. The run covers:

- **Synthetic:** `prose_en`, `markdown_doc`, `code_heavy`, `multilingual`
- **Real-world:** `real_pubmed`, `real_wikitexts`, `real_finance`, `real_chatlogs`, `real_state_of_the_union`
- **Stress:** `stress_single_line`, `stress_cjk_unbroken`, `stress_giant_list`

| Measure | 17 November result |
| --- | --- |
| Throughput | 37.1 MiB/s geometric mean; fastest on 10 of 12 workloads |
| Chunks over 512 tokens | text_chipper: 0; LangChain: 15 (maximum 514); tiktoken window and Chonkie TokenChunker: 18 each (maximum 513) |
| Mid-word cuts | text_chipper: 1 in 1,259 measured cut points (`stress_giant_list`); none in the other 11 workloads |



The reports separate two coverage measures. **Body coverage** is the share of non-whitespace source characters found
verbatim in chunk bodies. **Content + heading metadata coverage** additionally counts recognized Markdown heading
lines when their titles are carried in a chunk's `headers`. The second measure evaluates the text content the chunker
preserves; it does not imply that Markdown heading markers are copied into the body. For example, `markdown_doc`
reports 99.568% body coverage and 100% content + heading metadata coverage for text_chipper.

Regenerate the README charts from that run:

```bash
pip install matplotlib
python benchmarks/make_charts.py benchmarks/results/20261117_133546/results.json benchmarks/charts --svg
```

## Large-file benchmark

[`streaming_bench.py`](streaming_bench.py) chunks one big file. Each mode runs in a fresh process:

| Mode | What it does |
| --- | --- |
| `text_chipper:stream` | `TextChipper.from_file_stream(path)`, chunks consumed one at a time |
| `<adapter>:whole` | Read the whole file into one string, then call the library once |
| `<adapter>:blocks` | Give the library blocks of about 8 MiB cut at line boundaries, one call per block |

`<adapter>` is a name from `chunkbench.adapters.ADAPTERS` (`text_chipper`, `tiktoken_window`, `chonkie_token`,
`chonkie_recursive`, `semchunk`, `semantic_text_splitter`, `langchain`). Only text_chipper has the streaming mode.

What is measured:

- **Time:** end to end for chunking, including reading the file. Verification is not timed.
- **Memory:** increase in peak process memory after warm-up. Use runs without `--verify`, because verification builds extra per-chunk objects.
- **Verification (`--verify`):** count every delivered chunk with the reference tokenizer and hash the delivered text. Matching hashes mean the chunk text is identical.
- **Lanes:** `--lane single` (default) pins to one CPU; `--lane all` applies no restriction.

The current attached result has three timed repetitions and a separate verification run per mode in the single-core lane. The data
is stored as JSON Lines and the summary is in `summary.md`:

```bash
python benchmarks/summarize_streaming.py benchmarks/results/streaming_20261117/streaming-results.jsonl
python benchmarks/plot_streaming.py benchmarks/results/streaming_20261117/streaming-results.jsonl
```

The plotting script renders only lanes present in its input. For a fresh run on your machine:

```bash
python benchmarks/prepare_corpus.py corpus.txt           # WikiText-103 raw, prints size and SHA-256
python benchmarks/streaming_bench.py corpus.txt --mode text_chipper:stream --lane single
python benchmarks/streaming_bench.py corpus.txt --mode langchain:blocks --lane single --verify
```

The GitHub Actions workflow [`streaming-bench.yml`](../.github/workflows/streaming-bench.yml) runs modes in separate
runners. Inputs include `modes` (`chipper`, `blocks` or `all`), `max_mib` (0 = full corpus), `rounds` (timed runs per
mode, plus one verification run) and `timeout_s`.

### Latest results (514.3 MiB)

The single-core run reached 20.5 MiB/s with `from_file_stream` and grew peak memory by 56 MB. The same file read into a
string for whole-file `chunk()` grew by about 4.6 GB. The source data and detailed summary are in
[`results/streaming_20261117/`](results/streaming_20261117/).

![Throughput on a 514 MiB file, single-core lane](charts/streaming_speed_single.png)

![Memory growth on a 514 MiB file, single-core lane](charts/streaming_memory_single.png)

## Replacement characters (U+FFFD)

[`cjk_loss_check.py`](cjk_loss_check.py) counts chunks that contain U+FFFD and the characters that actually disappear.
`loss_report(source, chunks)` removes U+FFFD from the joined chunks and compares the remaining non-whitespace characters
with the source.

- The check is valid for splitters whose chunks are non-overlapping and contain the source text. It is not valid for
  text_chipper's default mode because headings are returned in `headers`, not in the chunk body.
- The main README coverage metric is stricter: a chunk with U+FFFD cannot be matched verbatim, so it leaves coverage.
  This is why the CJK benchmark reports 32.6% coverage loss while far fewer characters actually disappear.
- A cut point can destroy at most one character, so the number of damaged chunks bounds actual character loss.

| Input | Splitter | Chunks with U+FFFD | Characters lost |
| --- | --- | ---: | ---: |
| CJK workload in the 17 November main run | tiktoken window | 245 of 751 | bounded by 245, not individually recounted in this run |
| Synthetic CJK file, about 1.7 M characters | tiktoken window | 546 of 2,833 | 286 (0.017%) |
| Same synthetic file | text_chipper flat mode | 0 | 0 |

## Before and after on a real README

[`before_after_readme.py`](before_after_readme.py) chunks one Markdown file with a fixed token window and with text_chipper
(default mode), then writes both outputs to a report and counts chunks that split a code fence or contain U+FFFD.

```bash
python benchmarks/before_after_readme.py README.md --max-tokens 256 --out report.md
```

Fence detection checks the chunk body. A chunk entirely inside a code block with no fence marker is not flagged.

## Install footprint

Dependencies, installed size and startup time were measured separately (macOS arm64, Python 3.15, one measurement per
library). The figures are in the main README under "Install size and startup".

```bash
# Create one empty virtual environment per library, then compare installed packages and size.
python3 -m venv /tmp/empty && du -sk /tmp/empty/lib/python*/site-packages | cut -f1
for p in chonkie langchain-text-splitters semchunk semantic-text-splitter tiktoken; do
  python3 -m venv /tmp/c_$p && /tmp/c_$p/bin/pip install -q $p
  echo "$p: $(/tmp/c_$p/bin/pip list --format=freeze | grep -vc '^pip==') packages (including itself)," \
       "$(du -sk /tmp/c_$p/lib/python*/site-packages | cut -f1) KB"
done
```

### Limits

- The main and large-file results are each one workflow run on a shared virtual machine; treat ratios as indicative.
- The latest large-file JSONL contains only the single-core lane. Do not infer all-cores performance from it.
- Block mode is our choice of how to feed a string-only library; chunk boundaries can differ at block edges.
- Whole-file memory includes reading the full input into a Python string.
- Retrieval quality is not measured.
