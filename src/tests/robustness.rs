//! Robust Document Processing validation pack.
//!
//! These tests intentionally separate hard guarantees from diagnostic comparisons.
//! Diagnostic tests are compiled only with `--features diagnostics` (run with `--nocapture`).
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use pulldown_cmark::{Event, Options, Parser};
use serde_json::Value;

use crate::backend::Backend;
use crate::block::Block;
use crate::chunk_types::ChunkConfig;
use crate::file::FileChunker;
use crate::markdown::{list_parser::parse_list_items, BlockParser};
use crate::normalization::Normalization;
use crate::{smart_chunk, ChipError};

fn cfg(max_tokens: usize, context: bool) -> ChunkConfig {
    ChunkConfig {
        max_tokens,
        overlap_tokens: 0,
        normalization: Normalization::None,
        snap_words: false,
        include_context: context,
        merge_blocks: true,
        strict: true,
    }
}

fn temp_file(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "text_chipper_{label}_{}_{}.md",
        std::process::id(),
        nonce
    ))
}

fn oracle_code_block_count(src: &str) -> usize {
    Parser::new_ext(src, Options::empty())
        .filter(|event| matches!(event, Event::Start(tag) if format!("{tag:?}").starts_with("CodeBlock")))
        .count()
}

fn oracle_item_lines(src: &str) -> Vec<usize> {
    Parser::new_ext(src, Options::empty())
        .into_offset_iter()
        .filter_map(|(event, range)| match event {
            Event::Start(tag) if format!("{tag:?}").starts_with("Item") => {
                Some(src[..range.start].bytes().filter(|b| *b == b'\n').count())
            }
            _ => None,
        })
        .collect()
}

fn assert_list_ranges_valid(source: &str) {
    for item in parse_list_items(source) {
        assert!(
            item.start_byte <= item.end_byte,
            "inverted list range: {item:?}"
        );
        assert!(
            item.end_byte <= source.len(),
            "list range exceeds source: {item:?}"
        );
        assert!(source.is_char_boundary(item.start_byte));
        assert!(source.is_char_boundary(item.end_byte));
        assert!(item.start_line <= item.end_line);
    }
}

// -----------------------------------------------------------------------------
// 1. CommonMark corpus integration
// -----------------------------------------------------------------------------

/// Fetch tests/fixtures/commonmark_spec.json before running this test. The entire
/// official corpus is passed through our incremental block parser. This is a
/// robustness/offset smoke test, NOT a claim that BlockParser implements all of
/// CommonMark's AST semantics.
#[test]
// Diagnostic (cargo feature `diagnostics`): needs `python scripts/fetch_commonmark.py`; run with `cargo test --features diagnostics <name> -- --nocapture`
#[cfg(feature = "diagnostics")]
fn commonmark_official_corpus_is_safe_to_scan() {
    let path = Path::new("tests/fixtures/commonmark_spec.json");
    let raw = fs::read_to_string(path).expect("fetch the CommonMark corpus first");
    let root: Value = serde_json::from_str(&raw).expect("valid CommonMark spec JSON");
    let examples = root.as_array().expect("CommonMark spec is an array");
    let mut parsed = 0usize;
    let mut max_example_bytes = 0usize;
    let mut max_blocks = 0usize;

    for example in examples {
        let markdown = example["markdown"].as_str().unwrap_or("");
        max_example_bytes = max_example_bytes.max(markdown.len());
        let mut parser = BlockParser::new();
        let mut blocks: Vec<Block> = Vec::new();
        for line in markdown.split_inclusive('\n') {
            parser.push_line(line, &mut blocks).unwrap();
        }
        parser.finish(&mut blocks);
        for block in &blocks {
            assert!(block.start_byte <= markdown.len());
            assert!(markdown.is_char_boundary(block.start_byte));
        }
        assert_list_ranges_valid(markdown);
        max_blocks = max_blocks.max(blocks.len());
        parsed += 1;
    }
    println!("COMMONMARK_CORPUS examples={parsed} max_example_bytes={max_example_bytes} max_blocks={max_blocks}");
    assert!(
        parsed > 500,
        "unexpectedly small CommonMark corpus ({parsed} examples)"
    );
}

/// Reports where our list-item start-line scanner differs from pulldown-cmark.
/// Differences are diagnostic because this parser deliberately supports a subset.
#[test]
// Diagnostic (cargo feature `diagnostics`): diagnostic; run with --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn commonmark_list_differential_report() {
    let raw = fs::read_to_string("tests/fixtures/commonmark_spec.json")
        .expect("fetch CommonMark corpus first");
    let examples: Value = serde_json::from_str(&raw).expect("valid CommonMark JSON");
    let mut compared = 0usize;
    let mut differing = 0usize;
    let mut examples_shown = 0usize;
    for example in examples.as_array().expect("array") {
        let src = example["markdown"].as_str().unwrap_or("");
        let ours: Vec<_> = parse_list_items(src).iter().map(|i| i.start_line).collect();
        let reference = oracle_item_lines(src);
        compared += 1;
        if ours != reference {
            differing += 1;
            if examples_shown < 20 {
                println!("DIFF section={:?} example={:?}\n  ours={ours:?}\n  oracle={reference:?}\n  markdown={src:?}",
                    example["section"], example["example"]);
                examples_shown += 1;
            }
        }
    }
    println!(
        "LIST_DIFFERENTIAL compared={compared} differing={differing} rate={:.2}%",
        100.0 * differing as f64 / compared.max(1) as f64
    );
}

// -----------------------------------------------------------------------------
// 2. pulldown-cmark differential tests on the supported contract
// -----------------------------------------------------------------------------

#[test]
fn list_scanner_matches_reference_on_supported_core_constructs() {
    let cases = [
        "- a\n- b\n- c\n",
        "* a\n* b\n  * nested\n",
        "1. one\n2. two\n",
        "10) ten\n11) eleven\n",
        "- parent\n  - child\n    - grandchild\n- sibling\n",
        "- item\n  ```\n  - not a list item\n  ```\n- next\n",
        "- parent\n  lazy continuation\ncontinued\n- next\n",
    ];
    for src in cases {
        assert_eq!(
            parse_list_items(src)
                .iter()
                .map(|i| i.start_line)
                .collect::<Vec<_>>(),
            oracle_item_lines(src),
            "supported list case diverged: {src:?}"
        );
        assert_list_ranges_valid(src);
    }
}

#[test]
fn fenced_code_block_scanner_matches_reference_on_supported_cases() {
    let cases = [
        "```rust\nfn main() {}\n```\n",
        "~~~text\npayload\n~~~~\n",
        "before\n\n```\n- not a list\n```\nafter\n",
        "```python\nprint('unterminated')\n",
    ];
    for src in cases {
        let mut parser = BlockParser::new();
        let mut blocks = Vec::new();
        for line in src.split_inclusive('\n') {
            parser.push_line(line, &mut blocks).unwrap();
        }
        parser.finish(&mut blocks);
        let ours = blocks.iter().filter(|block| block.is_code).count();
        assert_eq!(
            ours,
            oracle_code_block_count(src),
            "fence mismatch: {src:?}"
        );
    }
}

#[test]
fn parser_block_stream_is_stable_across_line_boundaries() {
    let src =
        "# Heading\r\n\r\nparagraph one\r\nparagraph two\r\n```rust\r\nfn main() {}\r\n```\r\n";
    let mut parser = BlockParser::new();
    let mut out = Vec::new();
    for line in src.split_inclusive('\n') {
        parser.push_line(line, &mut out).unwrap();
    }
    parser.finish(&mut out);
    assert!(!out.is_empty());
    assert!(out.iter().all(|b| b.start_byte <= src.len()));
    assert!(out.iter().all(|b| src.is_char_boundary(b.start_byte)));
}

// -----------------------------------------------------------------------------
// 3. Markdown recovery / malformed input
// -----------------------------------------------------------------------------

#[test]
fn malformed_markdown_recovers_without_invalid_ranges() {
    let cases = [
        "```rust\nfn main() {\n- not a list\n", // unterminated fence
        "~~~\n```\n~~~\n",                      // mismatched fence marker
        "-\n- item\n    - nested\n",            // empty list item
        "> > > quote\n> - list\n",              // nested quote-like input
        "<div>\n- html-looking text\n</div>\n",
        "# heading\n\n[broken](\n\ntext\n",
        "***\n---\n___\n",
        "1. ordered\n999999999999999999999999999. overflow\n",
        "\t- tab item\n\t\t- deeper\n",
        "a\u{0308} e\u{301} 👩‍👩‍👧‍👦\n",
    ];
    for src in cases {
        assert_list_ranges_valid(src);
        let mut parser = BlockParser::new();
        let mut out = Vec::new();
        for line in src.split_inclusive('\n') {
            parser.push_line(line, &mut out).unwrap();
        }
        parser.finish(&mut out);
        assert!(out.iter().all(|b| b.start_byte <= src.len()));
        assert!(out.iter().all(|b| src.is_char_boundary(b.start_byte)));
    }
}

#[test]
fn unterminated_fence_is_flushed_at_true_eof() {
    let mut parser = BlockParser::new();
    let mut out = Vec::new();
    parser.push_line("before\n", &mut out).unwrap();
    parser.push_line("```rust\n", &mut out).unwrap();
    parser.push_line("fn main() {}\n", &mut out).unwrap();
    parser.finish(&mut out);
    assert!(out.iter().any(|b| b.is_code));
    assert!(out.iter().any(|b| b.text.contains("fn main")));
}

// -----------------------------------------------------------------------------
// 4. Memory / streaming stress
// -----------------------------------------------------------------------------

#[test]
fn parser_splits_oversized_single_line_without_losing_utf8_or_offsets() {
    let src = "Grüßen 👩‍💻 東京".repeat(100_000);
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    parser.push_line(&src, &mut blocks).unwrap();
    parser.finish(&mut blocks);
    assert!(blocks.len() > 1);
    let mut reconstructed = String::new();
    let mut next_byte = 0usize;
    let mut next_char = 0usize;
    for block in &blocks {
        assert!(block.text.len() <= crate::markdown::MAX_OPEN_BLOCK_BYTES);
        assert_eq!(block.start_byte, next_byte);
        assert_eq!(block.start_char, next_char);
        assert!(block.text.is_char_boundary(block.text.len()));
        next_byte += block.text.len();
        next_char += block.text.chars().count();
        reconstructed.push_str(&block.text);
    }
    assert_eq!(reconstructed, src);
    assert_eq!(next_byte, src.len());
    assert_eq!(next_char, src.chars().count());
}

#[test]
fn chunker_rejects_oversized_external_block_without_copying_it() {
    let backend = Backend::from_name("chars").unwrap();
    let mut chunker = crate::SmartChunker::new(cfg(0, false), backend);
    let block = Block {
        text: "x".repeat(crate::markdown::MAX_OPEN_BLOCK_BYTES + 1),
        start_byte: 0,
        start_char: 0,
        sep_before: String::new(),
        headers: Arc::new(Vec::new()),
        is_code: false,
    };
    let error = chunker
        .handle_block(block, &mut Vec::new())
        .expect_err("oversized external block must be rejected");
    assert!(matches!(error, ChipError::Io(ref e) if e.kind() == std::io::ErrorKind::InvalidData));
}

#[test]
fn parser_emits_pre_fence_blocks_without_retaining_them_until_fence_close() {
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    parser.push_line("before\n", &mut blocks).unwrap();
    parser.push_line("```text\n", &mut blocks).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "before");
    parser.push_line("payload\n", &mut blocks).unwrap();
    assert_eq!(blocks.len(), 1, "only the open code block remains buffered");
    parser.finish(&mut blocks);
    assert_eq!(blocks.len(), 2);
    assert!(blocks[1].is_code);
}

#[test]
fn streaming_reader_rejects_oversized_physical_line_with_bounded_buffer() {
    let backend = Backend::from_name("chars").unwrap();
    let path = temp_file("oversized_line");
    fs::write(&path, vec![b'x'; crate::file::MAX_INPUT_LINE_BYTES + 4096]).unwrap();
    let mut stream =
        FileChunker::open(path.to_str().unwrap(), &cfg(128, true), &backend, false).unwrap();
    let error = stream
        .next_chunk()
        .expect_err("line should exceed streaming cap");
    assert!(matches!(error, ChipError::Io(ref e) if e.kind() == std::io::ErrorKind::InvalidData));
    assert!(stream.peak_input_line_bytes() <= crate::file::MAX_INPUT_LINE_BYTES + 1);
    assert!(
        stream.next_chunk().unwrap().is_none(),
        "a failed stream is terminal"
    );
    let _ = fs::remove_file(path);
}

/// Flat mode must not depend on file size: the file API returns exactly what `smart_chunk`
/// returns for the same text, including on a file larger than the old 1 MiB threshold, with
/// Markdown constructs that a block-aware path would treat differently.
#[test]
fn flat_mode_file_result_equals_chunk_beyond_one_mib() {
    let backend = Backend::from_name("chars").unwrap();
    let unit = "# Title\n\nSome paragraph text here.\n\n- item one\n- item two\n\n```rust\nfn main() {}\n```\n\n";
    let text = unit.repeat(1_300_000 / unit.len() + 1);
    assert!(text.len() > 1024 * 1024);
    let path = temp_file("flat_equals_chunk");
    fs::write(&path, text.as_bytes()).unwrap();
    let c = cfg(256, false);
    let (want, want_spans) = crate::smart_chunk_mapped(&text, &c, &backend).unwrap();
    let mut stream = FileChunker::open(path.to_str().unwrap(), &c, &backend, false).unwrap();
    let (mut got, mut got_spans) = (Vec::new(), Vec::new());
    while let Some((chunk, span)) = stream.next_chunk().unwrap() {
        got.push(chunk);
        got_spans.push(span);
    }
    assert_eq!(want, got);
    assert_eq!(want_spans, got_spans);
    let _ = fs::remove_file(path);
}

/// Over the cap, flat mode fails explicitly instead of switching algorithm; the streaming
/// (context) mode still opens the same file.
#[test]
fn flat_mode_rejects_files_over_the_cap_explicitly() {
    let backend = Backend::from_name("chars").unwrap();
    let path = temp_file("flat_over_cap");
    let f = File::create(&path).unwrap();
    f.set_len(crate::file::FLAT_FILE_MAX_BYTES + 1).unwrap(); // sparse; never read
    drop(f);
    match FileChunker::open(path.to_str().unwrap(), &cfg(128, false), &backend, false) {
        Err(ChipError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("flat mode must reject files over the cap"),
    }
    assert!(
        FileChunker::open(path.to_str().unwrap(), &cfg(128, true), &backend, false).is_ok(),
        "streaming mode has no whole-file cap"
    );
    let _ = fs::remove_file(path);
}

#[test]
fn pending_memory_metric_includes_deferred_heading_prefix() {
    let backend = Backend::from_name("chars").unwrap();
    let path = temp_file("pending_heading_prefix");
    let heading = "H".repeat(700_000);
    let body = "B".repeat(700_000);
    fs::write(&path, format!("# {heading}\n\n{body}\n")).unwrap();
    let mut config = cfg(0, true);
    config.strict = false;
    let mut stream = FileChunker::open(path.to_str().unwrap(), &config, &backend, false).unwrap();
    while stream.next_chunk().unwrap().is_some() {}
    assert!(stream.peak_pending_bytes() <= 2 * 1024 * 1024);
    assert!(stream.peak_parser_open_bytes() <= crate::markdown::MAX_OPEN_BLOCK_BYTES);
    assert!(stream.peak_parser_gap_bytes() <= crate::markdown::MAX_PARSER_GAP_BYTES);
    assert!(
        stream.peak_pending_bytes() >= heading.len(),
        "metric must include deferred heading prefix"
    );
    let _ = fs::remove_file(path);
}

#[test]
fn parser_rejects_unbounded_separator_gap() {
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    let oversized_gap = "\n".repeat(crate::markdown::MAX_PARSER_GAP_BYTES + 1);
    let error = parser
        .push_line(&oversized_gap, &mut blocks)
        .expect_err("gap must be capped");
    assert!(matches!(error, ChipError::Io(ref e) if e.kind() == std::io::ErrorKind::InvalidData));
}

/// Resource diagnostic: oversized physical lines must be rejected at the configured
/// input cap, while many ordinary lines continue through the streaming parser.
#[test]
// Diagnostic (cargo feature `diagnostics`): resource diagnostic; run with --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn memory_stress_giant_single_line_and_many_short_lines() {
    let backend = Backend::from_name("chars").unwrap();
    let cfg = cfg(1024, true);
    let giant_path = temp_file("giant_line");
    let short_path = temp_file("short_lines");
    let giant = "x".repeat(8 * 1024 * 1024);
    fs::write(&giant_path, giant.as_bytes()).unwrap();
    {
        let mut f = File::create(&short_path).unwrap();
        for _ in 0..250_000 {
            f.write_all(b"A short paragraph with a little text.\n\n")
                .unwrap();
        }
    }

    {
        let mut stream =
            FileChunker::open(giant_path.to_str().unwrap(), &cfg, &backend, false).unwrap();
        let error = stream
            .next_chunk()
            .expect_err("oversized physical lines must fail explicitly");
        assert!(
            matches!(error, ChipError::Io(ref e) if e.kind() == std::io::ErrorKind::InvalidData)
        );
        assert!(stream.peak_input_line_bytes() <= crate::file::MAX_INPUT_LINE_BYTES + 1);
        println!(
            "MEMORY_CASE name=giant_single_line rejected=true peak_input_line_bytes={}",
            stream.peak_input_line_bytes()
        );
    }
    {
        let mut stream =
            FileChunker::open(short_path.to_str().unwrap(), &cfg, &backend, false).unwrap();
        let mut chunks = 0usize;
        while stream.next_chunk().unwrap().is_some() {
            chunks += 1;
        }
        assert!(stream.peak_input_line_bytes() <= crate::file::MAX_INPUT_LINE_BYTES);
        assert!(stream.peak_parser_open_bytes() <= crate::markdown::MAX_OPEN_BLOCK_BYTES);
        assert!(stream.peak_pending_bytes() <= 2 * 1024 * 1024);
        assert!(stream.peak_parser_gap_bytes() <= crate::markdown::MAX_PARSER_GAP_BYTES);
        println!("MEMORY_CASE name=many_short_lines chunks={chunks} peak_input_line_bytes={} peak_parser_open_bytes={} peak_pending_bytes={} peak_parser_gap_bytes={} peak_norm_map_segments={}",
            stream.peak_input_line_bytes(), stream.peak_parser_open_bytes(), stream.peak_pending_bytes(),
            stream.peak_parser_gap_bytes(), stream.peak_norm_map_segments());
    }
    let _ = fs::remove_file(giant_path);
    let _ = fs::remove_file(short_path);
}

#[test]
// Diagnostic (cargo feature `diagnostics`): large allocation diagnostic; run with --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn memory_stress_nested_depth_and_large_fence() {
    let mut src = String::new();
    for depth in 0..20_000 {
        src.push_str(&format!("{}- level {depth}\n", " ".repeat(depth.min(80))));
    }
    src.push_str("```text\n");
    src.push_str(&"code payload 0123456789\n".repeat(200_000));
    // Deliberately leave the fence open to exercise EOF recovery.
    let mut parser = BlockParser::new();
    let mut out = Vec::new();
    let started = Instant::now();
    for line in src.split_inclusive('\n') {
        parser.push_line(line, &mut out).unwrap();
    }
    parser.finish(&mut out);
    println!(
        "MEMORY_NESTED bytes={} lines={} blocks={} elapsed_ms={}",
        src.len(),
        src.lines().count(),
        out.len(),
        started.elapsed().as_millis()
    );
    assert!(out.iter().all(|b| b.start_byte <= src.len()));
}

// -----------------------------------------------------------------------------
// 5. Chunk quality / AI-readiness invariants
// -----------------------------------------------------------------------------

#[test]
fn strict_chunks_respect_budget_and_source_coordinates() {
    let backend = Backend::from_name("chars").unwrap();
    let src = (0..80)
        .map(|i| format!("Paragraph {i}: multilingual text Grüßen 日本語 emoji 👩‍💻.\n\n"))
        .collect::<String>();
    let chunks = smart_chunk(&src, &cfg(96, false), &backend).unwrap();
    assert!(chunks.len() > 1);
    let mut previous_start = 0usize;
    for chunk in &chunks {
        assert!(
            chunk.token_count <= 96,
            "token overflow: {}",
            chunk.token_count
        );
        assert!(chunk.start_byte <= chunk.end_byte && chunk.end_byte <= src.len());
        assert!(src.is_char_boundary(chunk.start_byte));
        assert!(src.is_char_boundary(chunk.end_byte));
        assert_eq!(&src[chunk.start_byte..chunk.end_byte], chunk.text);
        assert!(chunk.start_char <= chunk.end_char);
        assert!(
            chunk.start_byte >= previous_start,
            "chunk source order regressed"
        );
        previous_start = chunk.start_byte;
        assert_eq!(
            chunk.token_count,
            backend.count(&chunk.contextualized_text())
        );
    }
}

#[test]
fn strict_chunks_never_split_grapheme_clusters() {
    let backend = Backend::from_name("chars").unwrap();
    let grapheme = "👩‍👩‍👧‍👦";
    let src = format!(
        "{}\n{}\n{}\n",
        grapheme.repeat(20),
        grapheme.repeat(20),
        grapheme.repeat(20)
    );
    let chunks = smart_chunk(&src, &cfg(18, false), &backend).unwrap();
    for chunk in chunks {
        assert!(chunk.token_count <= 18);
        assert!(!chunk.text.contains("\u{200d}\n"));
        assert!(!chunk.text.contains("\n\u{200d}"));
    }
}

#[test]
fn source_coverage_and_chunking_are_deterministic_for_plain_text() {
    let backend = Backend::from_name("chars").unwrap();
    let src = "alpha beta gamma\n\nsecond paragraph\n\nthird paragraph\n";
    let a = smart_chunk(src, &cfg(12, false), &backend).unwrap();
    let b = smart_chunk(src, &cfg(12, false), &backend).unwrap();
    assert_eq!(a, b);
    assert!(!a.is_empty());
    assert!(a.iter().all(|c| !c.text.is_empty()));
}

// -----------------------------------------------------------------------------
// 6. Performance regression diagnostics
// -----------------------------------------------------------------------------

#[test]
// Diagnostic (cargo feature `diagnostics`): benchmark diagnostic; run with --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn performance_regression_smoke_report() {
    let backend = Backend::from_name("chars").unwrap();
    let cfg = cfg(512, false);
    let paragraph = "Text Chipper processes large and malformed documents for AI retrieval.\n\n";
    let src = paragraph.repeat(30_000); // ~2 MiB
    let mut samples = Vec::new();
    for round in 0..5 {
        let start = Instant::now();
        let chunks = smart_chunk(&src, &cfg, &backend).unwrap();
        let elapsed = start.elapsed();
        let mib_s = src.len() as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64().max(0.000_001);
        println!("PERF_CASE name=whole_document round={round} bytes={} chunks={} elapsed_ms={} mib_per_s={:.2}", src.len(), chunks.len(), elapsed.as_millis(), mib_s);
        samples.push(elapsed.as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "PERF_SUMMARY median_ms={:.2} p95_ms={:.2}",
        samples[2] * 1000.0,
        samples[4] * 1000.0
    );
    if let Ok(limit) = std::env::var("TEXT_CHIPPER_PERF_MAX_MS") {
        let limit: f64 = limit
            .parse()
            .expect("TEXT_CHIPPER_PERF_MAX_MS must be numeric");
        assert!(
            samples[2] * 1000.0 <= limit,
            "median runtime exceeded configured limit"
        );
    }
}

#[test]
fn invalid_utf8_file_policy_is_explicit() {
    let path = temp_file("invalid_utf8");
    fs::write(&path, [0x66, 0x80, 0x6f]).unwrap();
    let strict = crate::read_text_file(path.to_str().unwrap(), false);
    assert!(matches!(strict, Err(ChipError::Io(_))));
    let lossy = crate::read_text_file(path.to_str().unwrap(), true).unwrap();
    assert!(lossy.contains('\u{fffd}'));
    let _ = fs::remove_file(path);
}

/// Diagnostic: where does the time go on a whitespace flood?
/// Run: cargo test --release whitespace_flood_diagnostic --features diagnostics -- --nocapture
#[test]
// Diagnostic (cargo feature `diagnostics`): diagnostic: whitespace flood timing breakdown; run with --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn whitespace_flood_diagnostic() {
    let be = Backend::from_name("cl100k_base").unwrap();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    for unit in [" ", "\t"] {
        let label = if unit == " " { "space" } else { "tab" };
        for kib in [16usize, 32, 64, 128] {
            let n = kib * 1024;
            let doc = format!("a{}b", unit.repeat(n));
            let flood = unit.repeat(n);

            let t = Instant::now();
            let tokens_doc = be.count(&doc);
            let count_doc = ms(t);

            let t = Instant::now();
            let tokens_flood = be.count(&flood);
            let count_flood = ms(t);

            let mut line = format!(
                "WSD {label} {kib:>3}KiB count(doc)={count_doc:.2}ms tokens={tokens_doc} \
                 count(flood)={count_flood:.2}ms tokens={tokens_flood}"
            );
            for (name, norm, ctx) in [
                ("nfc+ctx", Normalization::Nfc, true),
                ("none+ctx", Normalization::None, true),
                ("none+noctx", Normalization::None, false),
            ] {
                let c = ChunkConfig {
                    max_tokens: 512,
                    overlap_tokens: 0,
                    normalization: norm,
                    snap_words: true,
                    include_context: ctx,
                    merge_blocks: true,
                    strict: true,
                };
                let t = Instant::now();
                let r = crate::smart_chunk_mapped(&doc, &c, &be);
                let dt = ms(t);
                let chunks = r.map(|v| v.0.len()).unwrap_or(0);
                line.push_str(&format!(" | {name}: {dt:.2}ms chunks={chunks}"));
            }
            println!("{line}");
        }
    }
}
