use super::*;
use crate::{normalization::NormSeg, split_engine::char_len};
use std::{io, sync::Arc};

fn cfg(max: usize, ov: usize, ctx: bool) -> ChunkConfig {
    ChunkConfig {
        max_tokens: max,
        overlap_tokens: ov,
        normalization: Normalization::None,
        snap_words: true,
        include_context: ctx,
        merge_blocks: true,
        strict: true,
    }
}

#[test]
fn char_len_ascii_fast_path_matches_unicode_count() {
    assert_eq!(char_len("plain ASCII 123"), 15);
    assert_eq!(char_len("Grüßen 東京 👩🏽‍💻"), "Grüßen 東京 👩🏽‍💻".chars().count());
    assert_eq!(char_len(""), 0);
}

#[test]
fn norm_map_prune_preserves_all_future_lookups() {
    let mut map = NormMap::default();
    for i in 0..5000usize {
        let n = i * 10;
        map.segs.push(NormSeg {
            b: [n, n + 2, n, n + 3],
            c: [n, n + 2, n, n + 3],
        });
    }
    let watermark = 3000 * 10 + 2;
    let expected: Vec<(usize, usize, usize, usize)> = (watermark..50000)
        .step_by(7)
        .map(|x| {
            (
                map.map_byte(x, false),
                map.map_byte(x, true),
                map.map_char(x, false),
                map.map_char(x, true),
            )
        })
        .collect();
    map.prune(watermark, watermark);
    let actual: Vec<(usize, usize, usize, usize)> = (watermark..50000)
        .step_by(7)
        .map(|x| {
            (
                map.map_byte(x, false),
                map.map_byte(x, true),
                map.map_char(x, false),
                map.map_char(x, true),
            )
        })
        .collect();
    assert_eq!(actual, expected);
    assert!(map.segs.len() < 2500);
}

#[test]
fn offsets_and_budget_chars_accented_latin() {
    let text = "Schöne Welt, große Straße, süßer Tag. ".repeat(300);
    let backend = Backend::from_name("chars").unwrap();
    let chunks = smart_chunk(&text, &cfg(100, 10, false), &backend).unwrap();
    assert!(chunks.len() > 10);
    for c in &chunks {
        assert!(c.token_count <= 100, "budget exceeded: {}", c.token_count);
        let slice: String = text
            .chars()
            .skip(c.start_char)
            .take(c.end_char - c.start_char)
            .collect();
        assert_eq!(slice, c.text);
        assert_eq!(&text[c.start_byte..c.end_byte], c.text);
    }
    assert_eq!(
        chunks.last().unwrap().end_char,
        text.trim_end().chars().count()
    );
}

#[test]
fn words_backend_does_not_drop_text() {
    let body = "Wort ".repeat(2000);
    let text = format!("# Größen\n\n{body}\n");
    let backend = Backend::from_name("words").unwrap();
    let chunks = smart_chunk(&text, &cfg(64, 8, true), &backend).unwrap();
    assert!(!chunks.is_empty());
    let total: usize = chunks.iter().map(|c| c.token_count).sum();
    assert!(total >= 2000);
    assert!(chunks
        .iter()
        .all(|c| c.headers.as_slice() == ["Größen".to_string()]));
}

#[test]
fn tiny_budget_terminates() {
    let backend = Backend::from_name("chars").unwrap();
    let chunks = smart_chunk("abcdefghij", &cfg(1, 0, false), &backend).unwrap();
    assert_eq!(chunks.len(), 10);
}

#[test]
fn strict_budget_returns_error_when_one_grapheme_cannot_fit() {
    let backend = Backend::from_name("chars").unwrap();
    let err = smart_chunk("👩🏽‍💻", &cfg(1, 0, false), &backend)
        .expect_err("a multi-scalar grapheme cannot be split under a one-token budget");
    assert!(matches!(
        err,
        ChipError::ChunkTooLarge {
            required_tokens: 4,
            max_tokens: 1,
            start_byte: 0,
        }
    ));
}

#[test]
fn non_strict_budget_does_not_error_for_large_grapheme() {
    let backend = Backend::from_name("chars").unwrap();
    let mut config = cfg(1, 0, false);
    config.strict = false;
    let chunks = smart_chunk("👩🏽‍💻", &config, &backend).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "👩🏽‍💻");
}

#[test]
fn unspaced_text_is_windowed_and_covered() {
    let text = "abcdefghijklmnopqrstuvwxyz".repeat(200); // no separators at all
    let backend = Backend::from_name("chars").unwrap();
    let chunks = smart_chunk(&text, &cfg(100, 0, false), &backend).unwrap();
    let total: usize = chunks.iter().map(|c| c.text.chars().count()).sum();
    assert_eq!(total, text.chars().count());
    assert!(chunks.iter().all(|c| c.token_count <= 100));
}

#[test]
fn overlap_terminates_and_repeats() {
    let text = "Ein Satz steht hier. ".repeat(400);
    let backend = Backend::from_name("chars").unwrap();
    let chunks = smart_chunk(&text, &cfg(120, 40, false), &backend).unwrap();
    assert!(chunks.len() > 5);
    for w in chunks.windows(2) {
        assert!(w[1].end_byte > w[0].end_byte);
    }
    assert!(chunks.windows(2).any(|w| w[1].start_byte < w[0].end_byte));
}

#[test]
fn invalid_utf8_file_is_rejected_or_replaced_by_policy() {
    let path = std::env::temp_dir().join(format!("chipper_invalid_utf8_{}.md", std::process::id()));
    std::fs::write(&path, b"valid\xfftext").unwrap();
    let path_str = path.to_str().unwrap();

    let strict = read_text_file(path_str, false);
    assert!(matches!(strict, Err(ChipError::Io(ref e)) if e.kind() == io::ErrorKind::InvalidData));

    let replaced = read_text_file(path_str, true).unwrap();
    assert!(replaced.contains('\u{fffd}'));
    assert!(replaced.starts_with("valid"));
    assert!(replaced.ends_with("text"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn bom_file_stream_and_batch_agree() {
    let body = format!("# Größen\n\n{}", "Grüßen 👩‍💻 東京 text. ".repeat(120));
    let with_bom = format!("\u{feff}{body}");
    let path = std::env::temp_dir().join(format!("chipper_bom_{}.md", std::process::id()));
    std::fs::write(&path, with_bom.as_bytes()).unwrap();
    let backend = Backend::from_name("cl100k_base").unwrap();
    let config = cfg(64, 0, true);
    let batch_text = read_text_file(path.to_str().unwrap(), false).unwrap();
    let batch = smart_chunk(&batch_text, &config, &backend).unwrap();
    let mut stream = FileChunker::open(path.to_str().unwrap(), &config, &backend, false).unwrap();
    let mut streamed = Vec::new();
    while let Some((c, _span)) = stream.next_chunk().unwrap() {
        streamed.push(c);
    }
    assert_eq!(batch, streamed);
    for c in &streamed {
        assert!(backend.count(&c.contextualized_text()) <= 64);
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn nfkc_latin_cjk_and_emoji_chunks_stay_within_limit() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    let mut c = cfg(40, 0, true);
    c.normalization = Normalization::Nfkc;
    let text = format!(
        "# Größen ﬃ\n\n{}",
        "Ångström ﬁ 東京 👩🏽‍💻 e\u{301} Καλημέρα. ".repeat(200)
    );
    let chunks = smart_chunk(&text, &c, &backend).unwrap();
    assert!(!chunks.is_empty());
    for ch in &chunks {
        let n = backend.count(&ch.contextualized_text());
        assert!(n <= 40, "strict limit exceeded: {n}");
        assert_eq!(ch.token_count, n);
    }
}

#[test]
fn strict_bpe_contextualized_chunks_obey_limit() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    let text = format!("# Heading\n\n{}", "Hello 👩‍👩‍👧‍👦 café 東京 é. ".repeat(300));
    let chunks = smart_chunk(&text, &cfg(48, 0, true), &backend).unwrap();
    assert!(!chunks.is_empty());
    for c in &chunks {
        let delivered = c.contextualized_text();
        let n = backend.count(&delivered);
        assert!(n <= 48, "strict limit exceeded: {n}: {:?}", c.text);
        assert_eq!(c.token_count, n);
    }
}

#[test]
fn strict_bpe_without_context_obeys_limit() {
    let backend = Backend::from_name("o200k_base").unwrap();
    let text = "👨‍👩‍👧‍👦🇩🇰 中文かな 한글 e\u{301} café ".repeat(250);
    let chunks = smart_chunk(&text, &cfg(32, 0, false), &backend).unwrap();
    assert!(!chunks.is_empty());
    for c in &chunks {
        let n = backend.count(&c.text);
        assert!(n <= 32, "strict limit exceeded: {n}");
        assert_eq!(c.token_count, n);
    }
}

#[test]
fn prefers_paragraph_boundaries() {
    let para = "Ein kurzer Absatz steht hier. ".repeat(3);
    let text = format!("{0}\n\n{0}\n\n{0}\n\n{0}", para.trim());
    let backend = Backend::from_name("chars").unwrap();
    let chunks = smart_chunk(&text, &cfg(230, 0, false), &backend).unwrap();
    // every chunk is a whole number of paragraphs
    for c in &chunks {
        assert!(
            c.text.ends_with("hier."),
            "cut inside a paragraph: {:?}",
            c.text
        );
    }
}

#[test]
fn long_context_heading_cannot_collapse_body_budget_to_one_token() {
    let backend = Arc::new(Backend::from_name("cl100k_base").unwrap());
    let cfg = ChunkConfig {
        max_tokens: 64,
        overlap_tokens: 0,
        snap_words: true,
        include_context: true,
        merge_blocks: true,
        strict: true,
        normalization: Normalization::None,
    };
    let heading_doc = format!(
        "# {}\n\n## {}\n\n{}",
        "Long top-level heading ".repeat(5),
        "Long nested heading with symbols ﬃ ＡＢＣ ".repeat(3),
        "Body paragraph contains normal prose and enough words. ".repeat(160),
    );
    let chunks = smart_chunk(&heading_doc, &cfg, &backend).unwrap();
    assert!(!chunks.is_empty());
    assert!(
        chunks.len() < 500,
        "context caused excessive fragmentation: {}",
        chunks.len()
    );
    for chunk in &chunks {
        assert!(backend.count(&chunk.contextualized_text()) <= 64);
        assert!(backend.count(&chunk.text) > 1 || chunk.text.chars().count() <= 1);
    }
    assert!(
        chunks.iter().any(|c| !c.headers.is_empty()),
        "all context was discarded"
    );
}

#[test]
fn adjacent_sections_merge_with_common_context_when_they_fit() {
    let text = "# Root\n\n## One\n\nalpha\n\n## Two\n\nbeta\n";
    let backend = Backend::from_name("chars").unwrap();
    let mut c = cfg(128, 0, true);
    c.normalization = Normalization::None;
    let chunks = smart_chunk(text, &c, &backend).unwrap();
    assert_eq!(chunks.len(), 1, "short sibling sections should coalesce");
    assert_eq!(chunks[0].headers.as_slice(), &["Root".to_string()]);
    assert!(chunks[0].text.contains("## One"));
    assert!(chunks[0].text.contains("## Two"));
    assert_eq!(
        &text[chunks[0].start_byte..chunks[0].end_byte],
        chunks[0].text
    );
    assert!(chunks[0].token_count <= 128);
}

#[test]
fn block_parser_keeps_fence_state_across_incremental_pushes() {
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    parser.push_line("before\n", &mut blocks).unwrap();
    parser.push_line("```rust\n", &mut blocks).unwrap();
    parser.push_line("# not a heading\n", &mut blocks).unwrap();
    assert_eq!(
        blocks.len(),
        1,
        "the completed pre-fence block is emitted immediately"
    );
    assert_eq!(blocks[0].text, "before");
    parser.push_line("```\n", &mut blocks).unwrap();
    parser.push_line("after\n", &mut blocks).unwrap();
    parser.finish(&mut blocks);

    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].text, "before");
    assert!(blocks[1].is_code);
    assert!(blocks[1].text.contains("# not a heading"));
    assert_eq!(blocks[2].text, "after");
}

#[test]
fn block_parser_finish_closes_unterminated_fence_at_true_eof() {
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    parser.push_line("```python\n", &mut blocks).unwrap();
    parser.push_line("print('ok')\n", &mut blocks).unwrap();
    assert!(
        blocks.is_empty(),
        "only the unfinished code block is still open"
    );
    parser.finish(&mut blocks);
    assert_eq!(blocks.len(), 1);
    assert!(blocks[0].is_code);
    assert_eq!(blocks[0].text, "```python\nprint('ok')");
}

#[test]
fn block_parser_handles_crlf_and_multibyte_heading_offsets() {
    let text = "# Größen\r\n\r\nÄrger\r\n";
    let mut parser = BlockParser::new();
    let mut blocks = Vec::new();
    for line in text.split_inclusive('\n') {
        parser.push_line(line, &mut blocks).unwrap();
    }
    parser.finish(&mut blocks);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "Ärger");
    assert_eq!(blocks[0].start_byte, text.find("Ärger").unwrap());
    assert_eq!(
        blocks[0].start_char,
        text.chars().position(|c| c == 'Ä').unwrap()
    );
    assert_eq!(blocks[0].headers.as_slice(), &["Größen".to_string()]);
}

#[test]
fn nfkc_offsets_map_to_original_compatibility_text() {
    let original = "# Ｆｕｌｌｗｉｄｔｈ\n\nThe ﬁ ligature and ① circled digit.\n";
    let backend = Backend::from_name("chars").unwrap();
    let mut c = cfg(100, 0, true);
    c.normalization = Normalization::Nfkc;
    let (chunks, spans) = smart_chunk_mapped(original, &c, &backend).unwrap();
    assert!(!chunks.is_empty());
    for (chunk, span) in chunks.iter().zip(spans.iter()) {
        let raw = &original[span.start_byte..span.end_byte];
        assert_eq!(Normalization::Nfkc.apply(raw), chunk.text);
        let chars: String = original
            .chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect();
        assert_eq!(chars, raw);
    }
}

#[test]
fn non_strict_token_count_matches_delivered_text() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    let mut c = cfg(64, 0, true);
    c.strict = false;
    let text = format!(
        "# Heading\n\nShort one.\n\nShort two.\n\n{}",
        "Hello café 東京. ".repeat(200)
    );
    let chunks = smart_chunk(&text, &c, &backend).unwrap();
    assert!(!chunks.is_empty());
    for ch in &chunks {
        assert_eq!(ch.token_count, backend.count(&ch.contextualized_text()));
    }
}

#[test]
fn context_merge_estimates_keep_exact_counts_and_limits() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    // Many tiny blocks across several sections stress the join-drift margin.
    let mut text = String::new();
    for sec in 0..40 {
        text.push_str(&format!("# Section {sec}\n\n"));
        for para in 0..25 {
            text.push_str(&format!("Paragraph {para} café 東京, short.\n\n"));
        }
        text.push_str("## Sub\n\n- item one\n- item two\n\n");
    }
    for max in [24usize, 64, 256] {
        let chunks = smart_chunk(&text, &cfg(max, 0, true), &backend).unwrap();
        assert!(!chunks.is_empty());
        for ch in &chunks {
            let delivered = ch.contextualized_text();
            assert_eq!(ch.token_count, backend.count(&delivered), "max={max}");
            assert!(ch.token_count <= max, "max={max}");
        }
    }
}

// --- tokenizer backend parity -------------------------------------------------------------
// `Backend::count` must equal the ordinary (special-token-free) BPE length for every
// supported encoding. `tiktoken-rs` is the reference implementation.

fn parity_corpus() -> Vec<String> {
    let mut v: Vec<String> = [
        "",
        " ",
        "\n",
        "hello",
        "Hello, world! Don't panic; it's fine.",
        "   leading and trailing   ",
        "tabs\tand\r\nCRLF\r\nlines\n\n\n",
        "1234567890 3.14159 0xDEADBEEF 2026-11-06",
        "Grüßen, café, e\u{0301}, ﬃ, 東京、日本語、中文、한국어",
        "emoji 👩🏽\u{200d}💻 🇩🇰 👨\u{200d}👩\u{200d}👧\u{200d}👦 ✌️",
        "fn main() { println!(\"{}\", 1 + 2); } // code",
        "https://example.com/a/b?c=d&e=f#frag",
        // Special-token strings must be tokenized as ordinary text, never as one special id.
        "<|endoftext|>",
        "a <|endoftext|> b",
        "<|fim_prefix|>x<|fim_middle|>y<|fim_suffix|>z",
        "<|im_start|>user\nhi<|im_end|>",
        "<|endofprompt|>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.push("alpha beta gamma; ".repeat(2_000));
    v.push("東京、日本語。".repeat(2_000));
    v.push("x".repeat(10_000));
    v.push(" ".repeat(5_000));
    v.push("Paragraph café 😀.\n\n".repeat(1_000));
    v
}

#[test]
fn backend_count_matches_tiktoken_rs_ordinary_encoding() {
    let references = [
        ("cl100k_base", tiktoken_rs::cl100k_base().unwrap()),
        ("o200k_base", tiktoken_rs::o200k_base().unwrap()),
        ("p50k_base", tiktoken_rs::p50k_base().unwrap()),
        ("r50k_base", tiktoken_rs::r50k_base().unwrap()),
    ];
    let corpus = parity_corpus();
    for (name, reference) in &references {
        let backend = Backend::from_name(name).unwrap();
        for text in &corpus {
            assert_eq!(
                backend.count(text),
                reference.encode_ordinary(text).len(),
                "encoding={name} text starts with {:?}",
                text.chars().take(24).collect::<String>()
            );
        }
    }
}

#[test]
fn backend_count_treats_special_token_text_as_ordinary() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    // As ordinary text this is several tokens; as a special token it would be exactly one.
    assert!(backend.count("<|endoftext|>") > 1);
    assert_eq!(backend.count(""), 0);
}

// --- list-aware splitting ------------------------------------------------------------------

#[test]
fn oversized_list_block_splits_at_item_boundaries() {
    let backend = Backend::from_name("chars").unwrap();
    // 31 chars per top-level item (parent + two nested items); budget fits two items.
    let text = "- item one\n  - sub a\n  - sub b\n".repeat(10);
    let chunks = smart_chunk(&text, &cfg(75, 0, true), &backend).unwrap();
    assert!(chunks.len() > 1);
    for ch in &chunks {
        assert!(
            ch.text.starts_with("- item one"),
            "chunk starts inside an item: {:?}",
            ch.text
        );
        assert!(ch.token_count <= 75);
    }
}

#[test]
fn multi_paragraph_list_item_that_fits_is_never_cut_between_its_paragraphs() {
    let backend = Backend::from_name("chars").unwrap();
    // Loose list: the parser closes a block at each blank line, so every item is two blocks.
    // Each item is 26 chars; the budget fits two items plus separators but not three.
    let text: String = (1..=9)
        .map(|i| format!("- item {i}\n\n  continuation {i}\n\n"))
        .collect();
    let chunks = smart_chunk(&text, &cfg(70, 0, true), &backend).unwrap();
    assert!(chunks.len() > 1);
    for ch in &chunks {
        assert!(
            ch.text.starts_with("- item"),
            "chunk starts mid-item: {:?}",
            ch.text
        );
        assert_eq!(
            ch.text.matches("- item").count(),
            ch.text.matches("continuation").count(),
            "item cut between its paragraphs: {:?}",
            ch.text
        );
        assert!(ch.token_count <= 70);
    }
}

#[test]
fn list_item_stays_whole_when_continuation_block_also_carries_the_next_item() {
    let backend = Backend::from_name("chars").unwrap();
    // Only blank lines end a block, so "  more {i}\n- item {i+1}" is ONE block.
    let mut text = String::from("- item 0\n");
    for i in 1..=9 {
        text.push_str(&format!("- item {i}\n\n  more {i}\n"));
    }
    let chunks = smart_chunk(&text, &cfg(40, 0, true), &backend).unwrap();
    assert!(chunks.len() > 1);
    for ch in &chunks {
        for i in 1..=9 {
            let has_item = ch.text.contains(&format!("- item {i}\n"))
                || ch.text.ends_with(&format!("- item {i}"));
            let has_more = ch.text.contains(&format!("more {i}"));
            assert!(
                !has_item || has_more,
                "item {i} cut before its continuation: {:?}",
                ch.text
            );
            assert!(
                !has_more || has_item,
                "chunk starts mid-item {i}: {:?}",
                ch.text
            );
        }
        assert!(ch.token_count <= 40);
    }
}

#[test]
fn oversized_single_item_splits_at_nested_items_not_continuation_lines() {
    let backend = Backend::from_name("chars").unwrap();
    let text = format!(
        "- parent\n{}",
        "  - child text one\n    continuation words\n".repeat(8)
    );
    let chunks = smart_chunk(&text, &cfg(100, 0, true), &backend).unwrap();
    assert!(chunks.len() > 1);
    for ch in &chunks {
        assert!(
            ch.text.starts_with("- parent") || ch.text.starts_with("- child text one"),
            "chunk starts mid-item: {:?}",
            ch.text
        );
    }
}

#[test]
fn flat_mode_never_attaches_headers() {
    // include_context=false does not parse Markdown: headings are ordinary text and no chunk
    // carries header metadata (see the ChunkConfig documentation).
    let backend = Backend::from_name("chars").unwrap();
    let text = "# Title\n\nsome body text that needs several chunks to hold. ".repeat(12);
    let chunks = smart_chunk(&text, &cfg(40, 0, false), &backend).unwrap();
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert!(c.headers.is_empty());
        assert!(c.token_count <= 40);
        assert_eq!(c.contextualized_text(), c.text);
    }
}
