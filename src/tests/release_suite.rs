//! Release validation suite: ONE run, ONE report.
//!
//! Sections
//!   A  Unicode + character/token-limit edge cases (matrix: backend x budget x norm x ctx x overlap)
//!   B  Markdown structure: known-structure documents (atomic blocks, heading context)
//!   C  Robustness + fuzzing (random markdown soup, mutations, pathological inputs, stream==batch)
//!   D  Oracle checks against pulldown-cmark (reported, not required to be 100% CommonMark)
//!   E  Performance regression (scaling ratios, throughput floors, optional baseline file)
//!   F  Writing systems (Indic, Arabic-script, Hebrew, Cyrillic, Greek, Armenian, Georgian,
//!      Thai/Lao/Khmer/Burmese, Tibetan, Ethiopic, CJK, Vietnamese NFD ...)
//!
//! Rules are HARD (a failure fails the test) or SOFT (reported only). Everything runs first and
//! the report is printed at the end, so one run shows every problem at once.
//!
//! Run:  cargo test --release release_suite --features diagnostics -- --nocapture --test-threads=1
//! Env:  RS_FUZZ_ITERS (3000)  RS_SEED  RS_ONLY=A|B|C|D|E|F  RS_QUICK=1
//!       RS_PERF_SAVE=<file> (write timings)  RS_PERF_BASELINE=<file> (compare)  RS_PERF_TOL (0.35)
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pulldown_cmark::{Event, Options, Parser};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    smart_chunk_mapped, Backend, BlockParser, ChipError, Chunk, ChunkConfig, FileChunker,
    Normalization, OrigSpan,
};

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------

struct RuleStat {
    hard: bool,
    checks: usize,
    fails: usize,
    samples: Vec<String>,
}

#[derive(Default)]
struct Report {
    rules: BTreeMap<String, RuleStat>,
    info: Vec<String>,
}

impl Report {
    fn rec(&mut self, hard: bool, rule: &str, ok: bool, detail: &dyn Fn() -> String) {
        let st = self
            .rules
            .entry(rule.to_string())
            .or_insert_with(|| RuleStat {
                hard,
                checks: 0,
                fails: 0,
                samples: Vec::new(),
            });
        st.checks += 1;
        if !ok {
            st.fails += 1;
            if st.samples.len() < 3 {
                st.samples.push(detail());
            }
        }
    }
    fn hard(&mut self, rule: &str, ok: bool, detail: &dyn Fn() -> String) {
        self.rec(true, rule, ok, detail);
    }
    fn soft(&mut self, rule: &str, ok: bool, detail: &dyn Fn() -> String) {
        self.rec(false, rule, ok, detail);
    }
    fn info(&mut self, line: String) {
        self.info.push(line);
    }
}

fn trunc_dbg(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    let mut d = format!("{:?}", t);
    if s.chars().count() > n {
        d.push_str("...");
    }
    d
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u() % (n.max(1) as u64)) as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        let i = self.below(xs.len());
        &xs[i]
    }
    fn chance(&mut self, num: usize, den: usize) -> bool {
        self.below(den) < num
    }
}

fn be(name: &str) -> Arc<Backend> {
    Backend::from_name(name).unwrap()
}

fn cfg(max: usize, ov: usize, norm: Normalization, snap: bool, ctx: bool) -> ChunkConfig {
    ChunkConfig {
        max_tokens: max,
        overlap_tokens: ov,
        normalization: norm,
        snap_words: snap,
        include_context: ctx,
        merge_blocks: true,
        strict: true,
    }
}

// ---------------------------------------------------------------------------------------------
// Running + invariant checking
// ---------------------------------------------------------------------------------------------

struct Outcome {
    chunks: Vec<Chunk>,
    spans: Vec<OrigSpan>,
    err: Option<String>,
    too_large: bool,
    panicked: bool,
    elapsed: Duration,
}

fn run(src: &str, c: &ChunkConfig, b: &Arc<Backend>) -> Outcome {
    let t0 = Instant::now();
    let r = catch_unwind(AssertUnwindSafe(|| smart_chunk_mapped(src, c, b)));
    let elapsed = t0.elapsed();
    match r {
        Err(_) => Outcome {
            chunks: Vec::new(),
            spans: Vec::new(),
            err: Some("panic".to_string()),
            too_large: false,
            panicked: true,
            elapsed,
        },
        Ok(Err(e)) => {
            let tl = matches!(e, ChipError::ChunkTooLarge { .. });
            Outcome {
                chunks: Vec::new(),
                spans: Vec::new(),
                err: Some(e.to_string()),
                too_large: tl,
                panicked: false,
                elapsed,
            }
        }
        Ok(Ok((chunks, spans))) => Outcome {
            chunks,
            spans,
            err: None,
            too_large: false,
            panicked: false,
            elapsed,
        },
    }
}

fn same_outcome(a: &Outcome, b: &Outcome) -> bool {
    a.chunks == b.chunks && a.spans == b.spans && a.err == b.err
}

fn grapheme_bounds(s: &str) -> Vec<usize> {
    let mut v: Vec<usize> = s.grapheme_indices(true).map(|(i, _)| i).collect();
    v.push(s.len());
    v
}

/// Core invariants every successful chunking must satisfy.
fn check_outcome(
    rep: &mut Report,
    label: &str,
    src: &str,
    c: &ChunkConfig,
    bname: &str,
    b: &Arc<Backend>,
    o: &Outcome,
) {
    let desc = || {
        format!(
            "[{label}] be={bname} max={} ov={} norm={:?} snap={} ctx={} src={}",
            c.max_tokens,
            c.overlap_tokens,
            c.normalization,
            c.snap_words,
            c.include_context,
            trunc_dbg(src, 120)
        )
    };
    rep.hard("no_panic", !o.panicked, &|| desc());
    if o.panicked {
        return;
    }
    let nt_cow = c.normalization.apply(src);
    let nt: &str = &nt_cow;
    let small = nt.len() <= 300_000;

    if let Some(e) = &o.err {
        rep.hard("error_is_only_chunk_too_large", o.too_large, &|| {
            format!("{} err={e}", desc())
        });
        if o.too_large && small && (!c.include_context || !src.contains('#')) {
            let justified = nt.graphemes(true).any(|g| b.count(g) > c.max_tokens);
            rep.hard("chunk_too_large_error_is_justified", justified, &|| {
                format!("{} err={e}", desc())
            });
        }
        return;
    }

    rep.hard(
        "spans_len_matches_chunks_len",
        o.chunks.len() == o.spans.len(),
        &|| desc(),
    );
    if o.chunks.len() != o.spans.len() {
        return;
    }
    if nt.trim().is_empty() {
        rep.soft("blank_input_yields_no_chunks", o.chunks.is_empty(), &|| {
            desc()
        });
        return;
    }
    // Documents that contain only headings legitimately yield no chunks (headings become context).
    rep.soft(
        "nonblank_input_yields_chunks_except_heading_only_docs",
        !o.chunks.is_empty() || heading_only_doc(nt),
        &|| desc(),
    );

    let bounds = grapheme_bounds(nt);
    let mut cov: Vec<u8> = if small {
        vec![0u8; nt.len()]
    } else {
        Vec::new()
    };
    let mut prev_start = 0usize;
    let mut prev_end = 0usize;
    let mut prev_ospan = 0usize;

    for (i, ch) in o.chunks.iter().enumerate() {
        let sp = &o.spans[i];
        rep.hard("chunk_text_nonempty", !ch.text.is_empty(), &|| {
            format!("{} chunk#{i}", desc())
        });
        if c.strict && c.max_tokens > 0 {
            rep.hard(
                "budget_token_count_le_max",
                ch.token_count <= c.max_tokens,
                &|| {
                    format!(
                        "{} chunk#{i} token_count={} text={}",
                        desc(),
                        ch.token_count,
                        trunc_dbg(&ch.text, 80)
                    )
                },
            );
            let real = b.count(&ch.contextualized_text());
            rep.hard(
                "token_count_equals_recount",
                real == ch.token_count,
                &|| {
                    format!(
                        "{} chunk#{i} reported={} recount={real}",
                        desc(),
                        ch.token_count
                    )
                },
            );
        }
        let in_bounds = ch.start_byte <= ch.end_byte
            && ch.end_byte <= nt.len()
            && nt.is_char_boundary(ch.start_byte)
            && nt.is_char_boundary(ch.end_byte);
        rep.hard("offsets_valid_char_boundaries", in_bounds, &|| {
            format!("{} chunk#{i} {}..{}", desc(), ch.start_byte, ch.end_byte)
        });
        if !in_bounds {
            continue;
        }
        rep.hard(
            "text_equals_source_slice",
            nt[ch.start_byte..ch.end_byte] == ch.text,
            &|| {
                format!(
                    "{} chunk#{i} text={} slice={}",
                    desc(),
                    trunc_dbg(&ch.text, 60),
                    trunc_dbg(&nt[ch.start_byte..ch.end_byte], 60)
                )
            },
        );
        if nt.len() <= 20_000 {
            let sc = nt[..ch.start_byte].chars().count();
            let ec = sc + nt[ch.start_byte..ch.end_byte].chars().count();
            rep.hard(
                "char_offsets_consistent_with_byte_offsets",
                sc == ch.start_char && ec == ch.end_char,
                &|| {
                    format!(
                        "{} chunk#{i} chars {}..{} expected {sc}..{ec}",
                        desc(),
                        ch.start_char,
                        ch.end_char
                    )
                },
            );
        }
        rep.hard(
            "chunk_start_on_grapheme_boundary",
            bounds.binary_search(&ch.start_byte).is_ok(),
            &|| format!("{} chunk#{i} start={}", desc(), ch.start_byte),
        );
        rep.hard(
            "chunk_end_on_grapheme_boundary",
            bounds.binary_search(&ch.end_byte).is_ok(),
            &|| format!("{} chunk#{i} end={}", desc(), ch.end_byte),
        );
        rep.hard(
            "chunk_starts_nondecreasing",
            ch.start_byte >= prev_start,
            &|| format!("{} chunk#{i}", desc()),
        );
        rep.soft("chunk_ends_nondecreasing", ch.end_byte >= prev_end, &|| {
            format!("{} chunk#{i}", desc())
        });
        prev_start = ch.start_byte;
        prev_end = ch.end_byte;

        let ok_span = sp.start_byte <= sp.end_byte
            && sp.end_byte <= src.len()
            && src.is_char_boundary(sp.start_byte)
            && src.is_char_boundary(sp.end_byte);
        rep.hard("original_span_valid", ok_span, &|| {
            format!(
                "{} chunk#{i} span={}..{}",
                desc(),
                sp.start_byte,
                sp.end_byte
            )
        });
        if c.normalization == Normalization::None {
            rep.hard(
                "original_span_equals_chunk_span_without_normalization",
                sp.start_byte == ch.start_byte && sp.end_byte == ch.end_byte,
                &|| format!("{} chunk#{i}", desc()),
            );
        } else if ok_span {
            let cov_text = c.normalization.apply(&src[sp.start_byte..sp.end_byte]);
            rep.soft(
                "original_span_renormalizes_to_contain_chunk_text",
                cov_text.contains(ch.text.as_str()),
                &|| format!("{} chunk#{i} text={}", desc(), trunc_dbg(&ch.text, 60)),
            );
        }
        rep.soft(
            "original_span_starts_nondecreasing",
            sp.start_byte >= prev_ospan,
            &|| format!("{} chunk#{i}", desc()),
        );
        prev_ospan = sp.start_byte;
        if small {
            for x in &mut cov[ch.start_byte..ch.end_byte] {
                *x = x.saturating_add(1);
            }
        }
    }

    // With context on, heading text is carried in `headers`, not in chunk bodies, so the
    // body-coverage check only applies to inputs without headings. Setext headings
    // ("Title\n=====") count as headings just like ATX ones.
    let has_heading = has_any_heading(src);
    if small && (!c.include_context || !has_heading) {
        let mut missing: Option<usize> = None;
        let mut dup = false;
        for (i, chr) in nt.char_indices() {
            if chr.is_whitespace() {
                continue;
            }
            if cov[i] == 0 && missing.is_none() {
                missing = Some(i);
            }
            if cov[i] > 1 {
                dup = true;
            }
        }
        rep.hard("coverage_no_text_loss", missing.is_none(), &|| {
            let at = missing.unwrap_or(0);
            let around: String = nt.get(at..).unwrap_or("").chars().take(30).collect();
            format!("{} first_missing_byte={at} around={:?}", desc(), around)
        });
        if c.overlap_tokens == 0 {
            rep.soft("no_duplicated_text_when_overlap_zero", !dup, &|| desc());
        }
    }
}

/// Quality heuristics about WHERE chunks are cut (soft: reported with rates).
fn quality_checks(
    rep: &mut Report,
    label: &str,
    src: &str,
    c: &ChunkConfig,
    bname: &str,
    b: &Arc<Backend>,
    o: &Outcome,
) {
    if o.err.is_some() || o.panicked || c.normalization != Normalization::None {
        return;
    }
    let bytes = src.as_bytes();
    let desc = |what: &str, at: usize| {
        let lo = at.saturating_sub(25);
        let hi = (at + 25).min(src.len());
        let mut lo2 = lo;
        while lo2 > 0 && !src.is_char_boundary(lo2) {
            lo2 -= 1;
        }
        let mut hi2 = hi;
        while hi2 < src.len() && !src.is_char_boundary(hi2) {
            hi2 += 1;
        }
        format!(
            "[{label}] be={bname} max={} ov={} {what} at byte {at}: ...{:?}...",
            c.max_tokens,
            c.overlap_tokens,
            &src[lo2..hi2]
        )
    };
    const ABBR: [&str; 11] = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "e.g.", "i.e.", "vs.", "Fig.", "No.", "St.",
    ];
    for ch in &o.chunks {
        let mut cuts: Vec<usize> = vec![ch.end_byte];
        if c.overlap_tokens == 0 {
            cuts.push(ch.start_byte);
        }
        for &p in &cuts {
            if p == 0 || p >= bytes.len() {
                continue;
            }
            if c.snap_words
                && bytes[p - 1].is_ascii_alphanumeric()
                && bytes[p].is_ascii_alphanumeric()
            {
                let mut l = p;
                while l > 0 && bytes[l - 1].is_ascii_alphanumeric() {
                    l -= 1;
                }
                let mut r = p;
                while r < bytes.len() && bytes[r].is_ascii_alphanumeric() {
                    r += 1;
                }
                let short = b.count(&src[l..r]) * 2 <= c.max_tokens;
                if short {
                    rep.soft("mid_word_split_of_short_ascii_word", false, &|| {
                        desc("cut inside word", p)
                    });
                } else {
                    rep.soft("mid_word_split_of_short_ascii_word", true, &|| {
                        String::new()
                    });
                }
            }
            let digit_dot_digit = (bytes[p - 1] == b'.'
                && p >= 2
                && bytes[p - 2].is_ascii_digit()
                && bytes[p].is_ascii_digit())
                || (bytes[p] == b'.'
                    && bytes[p - 1].is_ascii_digit()
                    && p + 1 < bytes.len()
                    && bytes[p + 1].is_ascii_digit());
            rep.soft("split_inside_decimal_number", !digit_dot_digit, &|| {
                desc("cut inside number", p)
            });
        }
        // chunk END ends with an abbreviation like "Dr." (sentence is not really over)
        let head = &src[..ch.end_byte.min(src.len())];
        let mut ends_abbr = false;
        for a in ABBR {
            if head.ends_with(a) {
                let before = head.len() - a.len();
                let prev_ok = before == 0
                    || head[..before]
                        .chars()
                        .next_back()
                        .map(|x| x.is_whitespace())
                        .unwrap_or(true);
                if prev_ok && ch.end_byte < src.len() {
                    ends_abbr = true;
                }
            }
        }
        rep.soft("chunk_ends_right_after_abbreviation", !ends_abbr, &|| {
            desc("chunk ends after abbreviation", ch.end_byte)
        });
        if let Some(first) = ch.text.chars().next() {
            let orphan = matches!(
                first,
                ')' | ']'
                    | '}'
                    | '.'
                    | ','
                    | ';'
                    | ':'
                    | '!'
                    | '?'
                    | '\u{201d}'
                    | '\u{2019}'
                    | '\u{3002}'
                    | '\u{3001}'
            );
            rep.soft("chunk_starts_with_closing_punctuation", !orphan, &|| {
                desc("chunk starts with closing punctuation", ch.start_byte)
            });
        }
        if let Some(last) = ch.text.chars().next_back() {
            let dangling = matches!(last, '(' | '[' | '{' | '\u{201c}' | '\u{2018}' | '\u{300c}');
            rep.soft("chunk_ends_with_opening_punctuation", !dangling, &|| {
                desc("chunk ends with opening punctuation", ch.end_byte)
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A. Unicode + character/token-limit edge cases
// ---------------------------------------------------------------------------------------------

fn add(v: &mut Vec<(String, String, bool)>, label: &str, s: String, prose: bool) {
    v.push((label.to_string(), s, prose));
}

fn unicode_inputs() -> Vec<(String, String, bool)> {
    let mut v: Vec<(String, String, bool)> = Vec::new();
    // trivial / degenerate
    add(&mut v, "empty", String::new(), false);
    add(
        &mut v,
        "whitespace_only",
        " \n\t\n  \u{a0}\u{3000}\n".to_string(),
        false,
    );
    add(&mut v, "newlines_only", "\n\n\n\n".to_string(), false);
    add(&mut v, "single_ascii", "a".to_string(), false);
    add(&mut v, "single_astral", "\u{1F600}".to_string(), false);
    add(
        &mut v,
        "nul_and_controls",
        "a\u{0}b\u{0}c \u{1b}[31mred\u{1b}[0m \u{7f}del".to_string(),
        false,
    );
    add(
        &mut v,
        "bom_in_middle",
        "alpha \u{feff}beta \u{feff}gamma".to_string(),
        false,
    );
    add(
        &mut v,
        "noncharacters_private_use",
        "x\u{fdd0}y\u{fffe}z\u{ffff}w\u{e000}\u{f8ff}v".to_string(),
        false,
    );
    // combining / clusters
    add(
        &mut v,
        "combining_run_e_acute",
        "e\u{301}".repeat(200),
        false,
    );
    add(
        &mut v,
        "stacked_marks_then_text",
        format!("a{} tail words follow here", "\u{301}".repeat(60)),
        false,
    );
    add(&mut v, "zalgo", "Z\u{35b}\u{35c}\u{35d}\u{35e}\u{35f}a\u{360}\u{361}\u{362}l\u{363}\u{364}\u{365}g\u{366}o\u{367} text".repeat(8), false);
    add(
        &mut v,
        "hangul_jamo_sequences",
        "\u{1100}\u{1161}\u{11a8}".repeat(100),
        false,
    );
    add(&mut v, "hangul_syllables_prose", "\u{d55c}\u{ad6d}\u{c5b4} \u{d14d}\u{c2a4}\u{d2b8} \u{bb38}\u{c7a5}\u{c785}\u{b2c8}\u{b2e4}. ".repeat(60), true);
    add(
        &mut v,
        "zwj_family_emoji",
        "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}".repeat(60),
        false,
    );
    add(
        &mut v,
        "zwj_emoji_with_spaces",
        "\u{1F469}\u{200D}\u{1F4BB} works \u{1F468}\u{200D}\u{1F680} flies ".repeat(40),
        true,
    );
    add(
        &mut v,
        "skin_tone_modifiers",
        "\u{1F44B}\u{1F3FD} hello \u{1F44D}\u{1F3FF} ok ".repeat(50),
        true,
    );
    add(
        &mut v,
        "flags_even",
        "\u{1F1E9}\u{1F1EA}\u{1F1F9}\u{1F1F7}".repeat(60),
        false,
    );
    add(
        &mut v,
        "flags_odd_regional_indicators",
        format!("{}x", "\u{1F1E9}".repeat(7)),
        false,
    );
    add(
        &mut v,
        "keycaps_and_vs16",
        "1\u{fe0f}\u{20e3} \u{2764}\u{fe0f} ".repeat(60),
        true,
    );
    add(
        &mut v,
        "tag_sequence_flag",
        "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F} ".repeat(40),
        true,
    );
    add(
        &mut v,
        "single_huge_grapheme",
        format!("a{}", "\u{301}".repeat(500)),
        false,
    );
    // scripts
    add(&mut v, "devanagari_conjuncts", "\u{915}\u{94d}\u{937}\u{924}\u{94d}\u{930}\u{93f}\u{92f} \u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940} \u{92d}\u{93e}\u{937}\u{93e} ".repeat(60), true);
    add(&mut v, "thai_no_spaces", "\u{e20}\u{e32}\u{e29}\u{e32}\u{e44}\u{e17}\u{e22}\u{e44}\u{e21}\u{e48}\u{e21}\u{e35}\u{e0a}\u{e48}\u{e2d}\u{e07}\u{e27}\u{e48}\u{e32}\u{e07}".repeat(60), false);
    add(&mut v, "arabic_with_harakat", "\u{627}\u{64e}\u{644}\u{652}\u{639}\u{64e}\u{631}\u{64e}\u{628}\u{650}\u{64a}\u{64e}\u{651}\u{629}\u{64f} \u{644}\u{64f}\u{63a}\u{64e}\u{629}\u{64c} ".repeat(60), true);
    add(&mut v, "hebrew_niqqud", "\u{5e9}\u{5b8}\u{5c1}\u{5dc}\u{5d5}\u{5b9}\u{5dd} \u{5e2}\u{5d5}\u{5b9}\u{5dc}\u{5b8}\u{5dd} ".repeat(60), true);
    add(
        &mut v,
        "bidi_controls",
        "abc \u{202e}def\u{202c} ghi \u{200f}jkl\u{200e} ".repeat(40),
        true,
    );
    add(&mut v, "cjk_with_fullwidth_punct", "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3067}\u{3059}\u{3002}\u{ff01}\u{ff1f}\u{ff0c}".repeat(80), true);
    add(
        &mut v,
        "chinese_no_punct",
        "\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}\u{6211}\u{4eec}\u{53bb}\u{516c}\u{56ed}"
            .repeat(80),
        false,
    );
    add(
        &mut v,
        "latin_extended_accents",
        "\u{141}\u{f3}d\u{17a} \u{c5}ngstr\u{f6}m \u{152}uvre \u{d1}and\u{fa} \u{10c}e\u{161}tina \u{150}rs\u{e9}g. ".repeat(60),
        true,
    );
    add(
        &mut v,
        "latin_nfd_accents",
        "Cafe\u{301} cre\u{300}me bru\u{302}le\u{301}e a\u{308}o\u{308}u\u{308} ".repeat(60),
        true,
    );
    add(&mut v, "nfkc_ligatures_fullwidth", "\u{fb01}nal \u{fb02}ow \u{fb00} \u{ff21}\u{ff22}\u{ff23}\u{ff11}\u{ff12}\u{ff13} \u{2460}\u{2461} \u{338f} x\u{b2} ".repeat(50), true);
    // whitespace flavours
    add(
        &mut v,
        "exotic_spaces",
        "a\u{a0}b\u{2009}c\u{3000}d\u{2003}e\u{200b}f\u{2060}g ".repeat(60),
        true,
    );
    add(
        &mut v,
        "line_separators",
        "line\u{2028}sep\u{2029}para\u{85}nel\u{b}vt\u{c}ff tail ".repeat(40),
        true,
    );
    add(
        &mut v,
        "crlf_document",
        "First line\r\nsecond line\r\n\r\nNew paragraph here.\r\n\r\n".repeat(40),
        true,
    );
    add(&mut v, "lone_cr", "a\rb\rc\r\rd ".repeat(60), true);
    add(
        &mut v,
        "tabs_and_mixed_indent",
        "\tindented\n  \tmixed\n    four\n\t\tdeep\n".repeat(40),
        true,
    );
    // words / limits
    add(&mut v, "one_very_long_word", "a".repeat(5000), false);
    add(
        &mut v,
        "long_url",
        format!(
            "see https://example.com/{}?q=1&r=2#frag end",
            "segment/".repeat(200)
        ),
        true,
    );
    add(
        &mut v,
        "long_hyphenated_and_snake",
        "state-of-the-art_snake_case_word ".repeat(100),
        true,
    );
    add(
        &mut v,
        "special_token_text",
        "<|endoftext|> <|fim_prefix|> <|im_start|> text ".repeat(40),
        true,
    );
    add(
        &mut v,
        "numbers_and_decimals",
        "Pi is 3.14159, cost 1,000.50 USD, 1e-9, v2.0.1, 12:30:45, 192.168.0.1. ".repeat(50),
        true,
    );
    add(&mut v, "abbreviations_prose", "Dr. Smith met Mr. Jones e.g. at St. Mary's on Jan. 5th. Prof. Lee said: \"No. 5 is vs. No. 6.\" Then everyone left. ".repeat(40), true);
    add(&mut v, "quotes_and_brackets", "He said (loudly) \u{201c}hello there\u{201d} and [left] {quietly}; \u{2018}really\u{2019}? Yes! ".repeat(50), true);
    add(
        &mut v,
        "punctuation_storm",
        "...!!!??? --- *** ___ ~~~ ``` ### >>> ".repeat(60),
        false,
    );
    add(&mut v, "english_prose", "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs! How vexingly quick daft zebras jump? ".repeat(60), true);
    // markdown + unicode
    add(&mut v, "markdown_unicode_headings", "# Ba\u{15f}l\u{131}k \u{65e5}\u{672c}\u{8a9e} \u{1F469}\u{200D}\u{1F4BB}\n\nmetin metin metin.\n\n## Alt ba\u{15f}l\u{131}k\n\n- madde bir\n- madde iki\n\n```rust\nlet x = \"\u{e9}\u{301}\";\n```\n\nSon paragraf.\n".repeat(12), false);
    v
}

fn section_unicode(rep: &mut Report, quick: bool) {
    let inputs = unicode_inputs();
    let backends = ["chars", "words", "cl100k_base"];
    let norms = [Normalization::None, Normalization::Nfc, Normalization::Nfkc];
    let maxes: Vec<usize> = if quick {
        vec![1, 4, 16, 64]
    } else {
        vec![1, 2, 3, 5, 8, 16, 64]
    };
    let mut runs = 0usize;
    for (label, src, prose) in &inputs {
        for bn in backends {
            let b = be(bn);
            for &max in &maxes {
                for &norm in &norms {
                    for &ctx in &[false, true] {
                        for &ovk in &[0usize, 1] {
                            let ov = if ovk == 0 { 0 } else { max / 4 };
                            let c = cfg(max, ov, norm, true, ctx);
                            let o = run(src, &c, &b);
                            runs += 1;
                            check_outcome(rep, &format!("A:{label}"), src, &c, bn, &b, &o);
                            if *prose && !ctx && max >= 64 {
                                quality_checks(rep, &format!("A:{label}"), src, &c, bn, &b, &o);
                            }
                            rep.hard(
                                "A_case_time_under_5s",
                                o.elapsed < Duration::from_secs(5),
                                &|| {
                                    format!(
                                        "[A:{label}] be={bn} max={max} took {:.0} ms",
                                        ms(o.elapsed)
                                    )
                                },
                            );
                        }
                    }
                }
            }
            // snap_words = false variant
            for &max in &[8usize, 64] {
                let c = cfg(max, 0, Normalization::Nfc, false, false);
                let o = run(src, &c, &b);
                runs += 1;
                check_outcome(rep, &format!("A:{label}:nosnap"), src, &c, bn, &b, &o);
            }
        }
    }
    // Where chunks are cut, measured at realistic budgets (forced cuts at tiny budgets are not signal).
    for (label, src, prose) in &inputs {
        if !*prose {
            continue;
        }
        for bn in backends {
            let b = be(bn);
            for &max in &[64usize, 128, 256] {
                let c = cfg(max, 0, Normalization::None, true, false);
                let o = run(src, &c, &b);
                quality_checks(rep, &format!("A3:{label}"), src, &c, bn, &b, &o);
            }
        }
    }
    rep.info(format!(
        "A unicode matrix: {} inputs, {} chunking runs",
        inputs.len(),
        runs
    ));
}

/// Exact-fit boundaries: max == size must give one chunk; max == size-1 must split.
fn section_exact_fit(rep: &mut Report) {
    let texts: Vec<(&str, String)> = vec![
        ("ascii_run", "a".repeat(100)),
        ("words", vec!["word"; 40].join(" ")),
        ("latin_extended", vec!["\u{142}\u{f3}d\u{17a}"; 30].join(" ")),
        (
            "zwj_families",
            "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}".repeat(10),
        ),
        ("cjk", "\u{65e5}\u{672c}\u{8a9e}".repeat(33)),
        ("combining", "e\u{301}".repeat(50)),
        (
            "mixed",
            ["mixed \u{c7}\u{d6}\u{15e} \u{1F600} text"; 10].join(" "),
        ),
    ];
    for (label, t) in &texts {
        for bn in ["chars", "words", "cl100k_base"] {
            let b = be(bn);
            let n = b.count(t);
            if n < 2 {
                continue;
            }
            for &ctx in &[false, true] {
                let c_fit = cfg(n, 0, Normalization::None, true, ctx);
                let o = run(t, &c_fit, &b);
                check_outcome(rep, &format!("A2:{label}:fit"), t, &c_fit, bn, &b, &o);
                if o.err.is_none() && !o.panicked {
                    rep.hard("exact_fit_text_is_one_chunk", o.chunks.len() == 1, &|| {
                        format!(
                            "[A2:{label}] be={bn} ctx={ctx} max=count={n} got {} chunks",
                            o.chunks.len()
                        )
                    });
                }
                let c_tight = cfg(n - 1, 0, Normalization::None, true, ctx);
                let o2 = run(t, &c_tight, &b);
                check_outcome(rep, &format!("A2:{label}:tight"), t, &c_tight, bn, &b, &o2);
                if o2.err.is_none() && !o2.panicked {
                    rep.hard(
                        "one_under_exact_fit_splits_into_multiple_chunks",
                        o2.chunks.len() >= 2,
                        &|| {
                            format!(
                                "[A2:{label}] be={bn} ctx={ctx} max={} got {} chunks",
                                n - 1,
                                o2.chunks.len()
                            )
                        },
                    );
                }
            }
        }
    }
    // Unspaced ASCII: minimal chunk count and exact overlap (chars backend).
    let b = be("chars");
    let n = 1000usize;
    let t = "a".repeat(n);
    for &(max, ov) in &[(64usize, 0usize), (64, 8), (50, 10), (10, 3), (1, 0)] {
        let c = cfg(max, ov, Normalization::None, true, false);
        let o = run(&t, &c, &b);
        check_outcome(rep, "A2:unspaced", &t, &c, "chars", &b, &o);
        if o.err.is_some() || o.panicked {
            continue;
        }
        let expected = if ov == 0 {
            n.div_ceil(max)
        } else {
            1 + (n - max).div_ceil(max - ov)
        };
        rep.soft(
            "unspaced_ascii_chunk_count_is_minimal",
            o.chunks.len() <= expected,
            &|| {
                format!(
                    "max={max} ov={ov}: got {} chunks, expected <= {expected}",
                    o.chunks.len()
                )
            },
        );
        if ov > 0 {
            let mut ok = true;
            for w in o.chunks.windows(2) {
                if w[0].end_byte < w[1].start_byte || w[0].end_byte - w[1].start_byte > ov {
                    ok = false;
                }
            }
            rep.soft("overlap_never_exceeds_configured_value", ok, &|| {
                format!("max={max} ov={ov}")
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// B. Markdown structure with KNOWN ground truth
// ---------------------------------------------------------------------------------------------

struct KnownBlock {
    kind: &'static str,
    start: usize,
    end: usize,
    atomic: bool,
}

const WORDS: [&str; 24] = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "model",
    "token",
    "chunk",
    "learning",
    "data",
    "\u{15f}ehir",
    "\u{131}\u{15f}\u{131}k",
    "g\u{fc}zel",
    "caf\u{e9}",
    "\u{65e5}\u{672c}\u{8a9e}",
    "naive",
    "resume",
    "graph",
    "vector",
    "index",
];

fn words_line(rng: &mut Rng, n: usize) -> String {
    let mut out = String::new();
    for i in 0..n {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(rng.pick(&WORDS));
    }
    out
}

/// Document of well-formed blocks separated by blank lines; returns the byte range of every block.
fn gen_core_doc(rng: &mut Rng, max: usize, b: &Arc<Backend>) -> (String, Vec<KnownBlock>) {
    let mut s = String::new();
    let mut blocks: Vec<KnownBlock> = Vec::new();
    let n = 6 + rng.below(14);
    for _ in 0..n {
        if !s.is_empty() {
            s.push_str("\n\n");
        }
        let start = s.len();
        let mut items: Vec<(usize, usize)> = Vec::new();
        let kind: &'static str;
        match rng.below(7) {
            0 | 1 => {
                kind = "paragraph";
                let lines = 1 + rng.below(3);
                for l in 0..lines {
                    if l > 0 {
                        s.push('\n');
                    }
                    let w = 3 + rng.below(6);
                    s.push_str(&words_line(rng, w));
                    s.push('.');
                }
            }
            2 => {
                kind = "fence";
                let fch = if rng.chance(1, 3) { '~' } else { '`' };
                let flen = 3 + rng.below(2);
                let fence: String = std::iter::repeat_n(fch, flen).collect();
                s.push_str(&fence);
                if rng.chance(1, 2) {
                    s.push_str("rust");
                }
                s.push('\n');
                let nl = 2 + rng.below(6);
                for _ in 0..nl {
                    match rng.below(6) {
                        0 => s.push('\n'),
                        1 => s.push_str("# fake heading FAKEHDR\n"),
                        2 => s.push_str("- not a list item\n"),
                        3 => {
                            let short: String = std::iter::repeat_n(fch, flen - 1).collect();
                            s.push_str(&short);
                            s.push('\n');
                        }
                        _ => s.push_str("let x = foo(bar);\n"),
                    }
                }
                s.push_str(&fence);
            }
            3 => {
                kind = "list";
                let count = 2 + rng.below(5);
                let ordered = rng.chance(1, 2);
                for k in 0..count {
                    if k > 0 {
                        s.push('\n');
                    }
                    let a = s.len();
                    if ordered {
                        s.push_str(&format!("{}. ", k + 1));
                    } else {
                        s.push_str("- ");
                    }
                    let w = 2 + rng.below(5);
                    s.push_str(&words_line(rng, w));
                    items.push((a, s.len()));
                }
            }
            4 => {
                kind = "table";
                s.push_str("| name | value |\n|------|-------|\n");
                let rows = 1 + rng.below(4);
                for r in 0..rows {
                    if r > 0 {
                        s.push('\n');
                    }
                    s.push_str(&format!("| {} | {} |", rng.pick(&WORDS), r));
                }
            }
            5 => {
                kind = "quote";
                let lines = 1 + rng.below(3);
                for l in 0..lines {
                    if l > 0 {
                        s.push('\n');
                    }
                    s.push_str("> ");
                    let w = 3 + rng.below(5);
                    s.push_str(&words_line(rng, w));
                }
            }
            _ => {
                kind = "paragraph";
                s.push_str("Unicode \u{1F469}\u{200D}\u{1F4BB} paragraph with e\u{301} marks and \u{65e5}\u{672c}\u{8a9e} text, ok.");
            }
        }
        let end = s.len();
        let atomic = b.count(&s[start..end]) <= max;
        blocks.push(KnownBlock {
            kind,
            start,
            end,
            atomic,
        });
        for (a, e) in items {
            let item_atomic = b.count(&s[a..e]) <= max;
            blocks.push(KnownBlock {
                kind: "list_item",
                start: a,
                end: e,
                atomic: item_atomic,
            });
        }
    }
    s.push('\n');
    (s, blocks)
}

fn atomic_rule(kind: &str) -> &'static str {
    match kind {
        "paragraph" => "atomic_paragraph_not_cut_when_it_fits",
        "fence" => "atomic_fenced_code_not_cut_when_it_fits",
        "list" => "atomic_list_not_cut_when_it_fits",
        "list_item" => "atomic_list_item_not_cut_when_it_fits",
        "table" => "atomic_table_not_cut_when_it_fits",
        _ => "atomic_quote_not_cut_when_it_fits",
    }
}

fn section_core_structure(rep: &mut Report, quick: bool) {
    let docs = if quick { 60 } else { 250 };
    let mut rng = Rng::new(env_u64("RS_SEED", 0xC0FFEE) ^ 0xB1);
    let setups: [(&str, [usize; 3]); 3] = [
        ("chars", [40, 80, 160]),
        ("words", [8, 16, 32]),
        ("cl100k_base", [16, 32, 64]),
    ];
    for (bn, maxes) in setups {
        let b = be(bn);
        for &max in &maxes {
            for d in 0..docs {
                let (src, blocks) = gen_core_doc(&mut rng, max, &b);
                let c = cfg(max, 0, Normalization::None, true, true);
                let o = run(&src, &c, &b);
                let label = format!("B1#{d}");
                check_outcome(rep, &label, &src, &c, bn, &b, &o);
                if o.err.is_some() || o.panicked {
                    continue;
                }
                rep.hard(
                    "no_fake_header_from_fenced_code",
                    o.chunks
                        .iter()
                        .all(|ch| !ch.headers.iter().any(|h| h.contains("FAKEHDR"))),
                    &|| format!("[{label}] be={bn} max={max} src={}", trunc_dbg(&src, 200)),
                );
                let mut cuts: Vec<usize> = Vec::new();
                for ch in &o.chunks {
                    cuts.push(ch.start_byte);
                    cuts.push(ch.end_byte);
                }
                for blk in &blocks {
                    let inside: Vec<usize> = cuts
                        .iter()
                        .copied()
                        .filter(|&x| blk.start < x && x < blk.end)
                        .collect();
                    if blk.atomic {
                        rep.hard(atomic_rule(blk.kind), inside.is_empty(), &|| {
                            format!(
                                "[{label}] be={bn} max={max} cut at {:?} inside {} {}..{}: {}",
                                inside,
                                blk.kind,
                                blk.start,
                                blk.end,
                                trunc_dbg(&src[blk.start..blk.end], 160)
                            )
                        });
                    } else if blk.kind == "fence" {
                        // oversized fence: cuts should fall on line boundaries
                        let bytes = src.as_bytes();
                        let ok = inside
                            .iter()
                            .all(|&x| bytes[x - 1] == b'\n' || bytes[x] == b'\n');
                        rep.soft("oversized_fence_is_cut_on_line_boundaries", ok, &|| {
                            format!(
                                "[{label}] be={bn} max={max} cuts={:?} fence={}",
                                inside,
                                trunc_dbg(&src[blk.start..blk.end], 160)
                            )
                        });
                    } else if blk.kind == "list" {
                        // oversized list: cuts should fall at item starts
                        // A cut may only fall inside a line when that line alone exceeds the
                        // budget (it has to be split somewhere); otherwise it must sit on a
                        // line boundary.
                        let bytes = src.as_bytes();
                        let line_is_oversized = |x: usize| {
                            let ls = src[..x].rfind('\n').map_or(0, |p| p + 1);
                            let le = src[x..].find('\n').map_or(src.len(), |p| x + p);
                            b.count(src[ls..le].trim()) > max
                        };
                        let ok = inside.iter().all(|&x| {
                            bytes[x - 1] == b'\n' || bytes[x] == b'\n' || line_is_oversized(x)
                        });
                        rep.soft("oversized_list_is_cut_on_line_boundaries", ok, &|| {
                            format!(
                                "[{label}] be={bn} max={max} cuts={:?} list={}",
                                inside,
                                trunc_dbg(&src[blk.start..blk.end], 160)
                            )
                        });
                    }
                }
            }
        }
    }
}

// ---- heading context ----

struct HeadingDoc {
    src: String,
    marks: Vec<(String, Vec<String>)>,
    titles: Vec<String>,
}

fn gen_heading_doc(rng: &mut Rng) -> HeadingDoc {
    let mut s = String::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut marks: Vec<(String, Vec<String>)> = Vec::new();
    let mut titles: Vec<String> = Vec::new();
    let mut n = 0usize;
    let flavours = [
        "alpha",
        "beta gamma",
        "\u{dc}n\u{ef}code",
        "\u{65e5}\u{672c}\u{8a9e}",
        "C# lang",
    ];
    let count = 8 + rng.below(16);
    for _ in 0..count {
        let k = rng.below(9);
        if k <= 2 {
            let level = 1 + rng.below(4);
            while let Some((l, _)) = stack.last() {
                if *l >= level {
                    stack.pop();
                } else {
                    break;
                }
            }
            let title = format!("Title{} {}", n, rng.pick(&flavours));
            n += 1;
            stack.push((level, title.clone()));
            titles.push(title.clone());
            s.push_str(&"#".repeat(level));
            s.push(' ');
            s.push_str(&title);
            if rng.chance(1, 5) {
                s.push_str(" ##");
            }
            s.push_str("\n\n");
        } else {
            let marker = format!("MARK{}X", n);
            n += 1;
            let path: Vec<String> = stack.iter().map(|(_, t)| t.clone()).collect();
            match k {
                3 => s.push_str(&format!("#hashtag {marker} text\n\n")),
                4 => s.push_str(&format!("####### seven hashes {marker}\n\n")),
                5 => s.push_str(&format!("```\n# FAKEHDR{n}\n{marker}\n```\n\n")),
                6 => s.push_str(&format!("{marker} first line\n# Interrupting heading\n\n")),
                _ => s.push_str(&format!("{marker} {}.\n\n", words_line(rng, 6))),
            }
            marks.push((marker, path));
            if k == 6 {
                // an ATX heading line directly after a paragraph line IS a heading in CommonMark
                stack.clear();
                stack.push((1, "Interrupting heading".to_string()));
                titles.push("Interrupting heading".to_string());
            }
        }
    }
    HeadingDoc {
        src: s,
        marks,
        titles,
    }
}

fn is_title_prefix(h: &str, titles: &[String]) -> bool {
    titles.iter().any(|t| t.starts_with(h))
}

fn section_heading_context(rep: &mut Report, quick: bool) {
    let docs = if quick { 60 } else { 250 };
    let mut rng = Rng::new(env_u64("RS_SEED", 0xC0FFEE) ^ 0xB2);
    let setups: [(&str, usize, usize); 3] = [
        ("chars", 400, 70),
        ("words", 150, 14),
        ("cl100k_base", 200, 30),
    ];
    for (bn, big, tight) in setups {
        let b = be(bn);
        for d in 0..docs {
            let hd = gen_heading_doc(&mut rng);
            for (budget, is_big) in [(big, true), (tight, false)] {
                let c = cfg(budget, 0, Normalization::None, true, true);
                let o = run(&hd.src, &c, &b);
                let label = format!("B2#{d}:{}", if is_big { "big" } else { "tight" });
                check_outcome(rep, &label, &hd.src, &c, bn, &b, &o);
                if o.err.is_some() || o.panicked {
                    continue;
                }
                for ch in &o.chunks {
                    for h in ch.headers.iter() {
                        let ok = is_title_prefix(h, &hd.titles);
                        rep.hard("header_text_is_never_invented", ok, &|| {
                            format!(
                                "[{label}] be={bn} header={:?} not a known title; src={}",
                                h,
                                trunc_dbg(&hd.src, 240)
                            )
                        });
                    }
                    for (m, path) in &hd.marks {
                        if !ch.text.contains(m.as_str()) {
                            continue;
                        }
                        if is_big {
                            let hs: Vec<String> = ch.headers.iter().cloned().collect();
                            let is_prefix = hs.len() <= path.len() && hs[..] == path[..hs.len()];
                            rep.hard("chunk_headers_are_ancestor_path_of_its_content", is_prefix, &|| {
                                format!("[{label}] be={bn} marker={m} headers={:?} expected_prefix_of={:?}", hs, path)
                            });
                        }
                    }
                }
                if is_big {
                    // precision: a chunk holding a single marker should carry the full path
                    for ch in &o.chunks {
                        let inside: Vec<&(String, Vec<String>)> = hd
                            .marks
                            .iter()
                            .filter(|(m, _)| ch.text.contains(m.as_str()))
                            .collect();
                        if inside.len() == 1 {
                            let hs: Vec<String> = ch.headers.iter().cloned().collect();
                            rep.soft(
                                "single_section_chunk_carries_full_heading_path",
                                hs == inside[0].1,
                                &|| {
                                    format!(
                                        "[{label}] be={bn} marker={} headers={:?} expected={:?}",
                                        inside[0].0, hs, inside[0].1
                                    )
                                },
                            );
                        }
                    }
                }
                for (m, _) in &hd.marks {
                    let present = o.chunks.iter().any(|ch| ch.text.contains(m.as_str()));
                    rep.hard("section_content_never_dropped", present, &|| {
                        format!(
                            "[{label}] be={bn} marker {m} missing; src={}",
                            trunc_dbg(&hd.src, 240)
                        )
                    });
                }
            }
        }
    }
}

/// (name, source, expected headers of the chunk containing "MARK", hard?)
fn header_cases() -> Vec<(&'static str, String, Vec<&'static str>, bool)> {
    vec![
        (
            "basic_two_levels",
            "# A\n\n## B\n\nMARK".to_string(),
            vec!["A", "B"],
            true,
        ),
        (
            "skipped_level",
            "# A\n\n### C\n\nMARK".to_string(),
            vec!["A", "C"],
            true,
        ),
        (
            "level_reset",
            "# A\n\n## B\n\n# D\n\nMARK".to_string(),
            vec!["D"],
            true,
        ),
        (
            "sibling_replaces_sibling",
            "# A\n\n## B\n\n## C\n\nMARK".to_string(),
            vec!["A", "C"],
            true,
        ),
        (
            "closing_hashes_removed",
            "## Title ##\n\nMARK".to_string(),
            vec!["Title"],
            true,
        ),
        (
            "trailing_spaces_trimmed",
            "## Title   \n\nMARK".to_string(),
            vec!["Title"],
            true,
        ),
        (
            "three_space_indent_ok",
            "   ## Title\n\nMARK".to_string(),
            vec!["Title"],
            true,
        ),
        (
            "four_space_indent_is_code",
            "    # code\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "hash_without_space",
            "#hashtag\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "seven_hashes",
            "####### x\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "heading_inside_backtick_fence",
            "```\n# fake\n```\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "heading_inside_tilde_fence",
            "~~~\n# fake\n~~~\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "nested_fence_lengths",
            "````\n```\n# fake\n```\n````\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "unicode_title",
            "# Ba\u{15f}l\u{131}k \u{65e5}\u{672c}\u{8a9e} \u{1F469}\u{200D}\u{1F4BB}\n\nMARK"
                .to_string(),
            vec!["Ba\u{15f}l\u{131}k \u{65e5}\u{672c}\u{8a9e} \u{1F469}\u{200D}\u{1F4BB}"],
            true,
        ),
        (
            "crlf_heading",
            "# A\r\n\r\nMARK".to_string(),
            vec!["A"],
            true,
        ),
        (
            "text_right_after_heading",
            "# A\nMARK".to_string(),
            vec!["A"],
            true,
        ),
        (
            "hash_inside_title_kept",
            "# C# language\n\nMARK".to_string(),
            vec!["C# language"],
            true,
        ),
        (
            "space_hash_word_inside_title",
            "# A #B\n\nMARK".to_string(),
            vec!["A #B"],
            true,
        ),
        (
            "thematic_break_is_not_heading",
            "# A\n\n---\n\nMARK".to_string(),
            vec!["A"],
            true,
        ),
        (
            "escaped_hash",
            "\\# not heading\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "six_level_chain",
            "# 1\n\n## 2\n\n### 3\n\n#### 4\n\n##### 5\n\n###### 6\n\nMARK".to_string(),
            vec!["1", "2", "3", "4", "5", "6"],
            true,
        ),
        (
            "heading_after_paragraph_without_blank",
            "text\n# A\nMARK".to_string(),
            vec!["A"],
            true,
        ),
        (
            "tab_after_hash",
            "#\tTitle\n\nMARK".to_string(),
            vec!["Title"],
            false,
        ),
        (
            "inline_markup_in_title_raw",
            "## *Em* `code` [l](u)\n\nMARK".to_string(),
            vec!["*Em* `code` [l](u)"],
            false,
        ),
        (
            "setext_h1",
            "Setext Title\n============\n\nMARK".to_string(),
            vec!["Setext Title"],
            false,
        ),
        (
            "setext_h2",
            "Setext Title\n------------\n\nMARK".to_string(),
            vec!["Setext Title"],
            false,
        ),
        (
            "heading_in_blockquote",
            "> # Quoted\n\nMARK".to_string(),
            vec![],
            false,
        ),
        (
            "heading_in_list_item",
            "- # In list\n\nMARK".to_string(),
            vec![],
            false,
        ),
        ("empty_heading", "#\n\nMARK".to_string(), vec![], false),
        (
            "front_matter_yaml_with_blank_line",
            "---\ntitle: x\n\ntags: [a]\n---\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "front_matter_yaml_no_blank_line",
            "---\ntitle: x\ntags: [a]\n---\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "front_matter_dots_terminator",
            "---\ntitle: x\n...\n\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "front_matter_toml",
            "+++\ntitle = \"x\"\n+++\n\n# H\n\nMARK".to_string(),
            vec!["H"],
            true,
        ),
        (
            "front_matter_then_heading",
            "---\ntitle: x\n---\n\n# Real\n\nMARK".to_string(),
            vec!["Real"],
            true,
        ),
        (
            "thematic_break_then_heading",
            "---\n\n# Real\n\nMARK".to_string(),
            vec!["Real"],
            true,
        ),
        (
            "thematic_break_directly_before_heading",
            "---\n# Real\nMARK".to_string(),
            vec!["Real"],
            true,
        ),
        (
            "front_matter_only_at_document_start",
            "intro\n\n---\ntitle: x\n---\n\n# Real\n\nMARK".to_string(),
            vec!["Real"],
            false,
        ),
        (
            "long_paragraph_plus_underline_is_text",
            format!("{}\n---\nMARK", "word ".repeat(100)),
            vec![],
            true,
        ),
        (
            "three_line_paragraph_plus_underline_is_text",
            "one\ntwo\nthree\n---\nMARK".to_string(),
            vec![],
            true,
        ),
        (
            "two_line_setext_title",
            "Line one\nline two\n=====\n\nMARK".to_string(),
            vec!["Line one line two"],
            false,
        ),
        (
            "unterminated_front_matter_like_block",
            "---\nkey: v\n\n# Real\n\nMARK".to_string(),
            vec!["Real"],
            false,
        ),
    ]
}

fn section_header_cases(rep: &mut Report) {
    for (name, src, expected, hard) in header_cases() {
        for (bn, max) in [("chars", 400usize), ("words", 150), ("cl100k_base", 200)] {
            let b = be(bn);
            let mut c = cfg(max, 0, Normalization::None, true, true);
            c.merge_blocks = false; // test heading parsing, not cross-section merging
            let o = run(&src, &c, &b);
            check_outcome(rep, &format!("B3:{name}"), &src, &c, bn, &b, &o);
            if o.err.is_some() || o.panicked {
                continue;
            }
            let found = o.chunks.iter().find(|ch| ch.text.contains("MARK"));
            let rule = if hard {
                "header_case_matches_expectation"
            } else {
                "header_case_edge_reported"
            };
            match found {
                None => rep.rec(hard, rule, false, &|| {
                    format!(
                        "[B3:{name}] be={bn} MARK chunk missing; src={}",
                        trunc_dbg(&src, 120)
                    )
                }),
                Some(ch) => {
                    let got: Vec<String> = ch.headers.iter().cloned().collect();
                    let exp: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
                    rep.rec(hard, rule, got == exp, &|| {
                        format!(
                            "[B3:{name}] be={bn} got={:?} expected={:?} src={}",
                            got,
                            exp,
                            trunc_dbg(&src, 120)
                        )
                    });
                }
            }
        }
    }
}

/// A list marker ("2.", "-") must never end up alone in a chunk.
fn section_orphan_markers(rep: &mut Report) {
    let docs: [&str; 4] = [
        "1. gamma gamma caf\u{e9} alpha model resume\n2. sit learning resume caf\u{e9} caf\u{e9} learning\n3. lorem learning chunk delta\n",
        "- aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk\n- llll mmmm nnnn oooo pppp qqqq rrrr ssss tttt uuuu vvvv\n",
        "10. one two three four five six seven eight nine ten eleven twelve\n11. thirteen fourteen fifteen sixteen seventeen eighteen\n",
        "* star item with quite a few words in it to overflow the budget here\n* another star item with quite a few words in it as well\n",
    ];
    for (di, d) in docs.iter().enumerate() {
        for (bn, maxes) in [
            ("chars", [20usize, 30, 40, 50]),
            ("words", [3, 4, 5, 6]),
            ("cl100k_base", [6, 8, 10, 12]),
        ] {
            let b = be(bn);
            for &max in &maxes {
                for &ctx in &[false, true] {
                    let c = cfg(max, 0, Normalization::None, true, ctx);
                    let o = run(d, &c, &b);
                    check_outcome(rep, &format!("B4:doc{di}"), d, &c, bn, &b, &o);
                    if o.err.is_some() || o.panicked {
                        continue;
                    }
                    for ch in &o.chunks {
                        let t = ch.text.trim();
                        let marker_only = matches!(t, "-" | "*" | "+" | ">")
                            || (t.len() >= 2
                                && t.len() <= 4
                                && t.ends_with('.')
                                && t[..t.len() - 1].chars().all(|x| x.is_ascii_digit()));
                        rep.hard(
                            "no_chunk_consists_only_of_a_list_marker",
                            !marker_only,
                            &|| {
                                format!(
                                    "[B4:doc{di}] be={bn} max={max} ctx={ctx} chunk={:?}",
                                    ch.text
                                )
                            },
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// F. Writing systems
// ---------------------------------------------------------------------------------------------

struct Script {
    name: &'static str,
    sentences: &'static [&'static str],
    /// Words are separated by spaces (false for Thai, Lao, Khmer, Burmese, Tibetan, CJK ...).
    spaced: bool,
    /// Characters that legitimately end a sentence in this script (empty = none defined).
    ends: &'static str,
}

const SCRIPTS: &[Script] = &[
    Script {
        name: "hindi_devanagari",
        spaced: true,
        ends: "\u{964}\u{965}.?!",
        sentences: &[
            "भारत एक विशाल और विविधताओं से भरा देश है।",
            "यहाँ अनेक भाषाएँ और बोलियाँ बोली जाती हैं।",
            "क्षत्रिय, ज्ञान, श्रद्धा और द्वार जैसे संयुक्त अक्षर वाले शब्द आम हैं।",
        ],
    },
    Script {
        name: "bengali",
        spaced: true,
        ends: "\u{964}.?!",
        sentences: &[
            "বাংলা ভাষা বিশ্বের অন্যতম সমৃদ্ধ ভাষা।",
            "এটি কোটি কোটি মানুষের মাতৃভাষা।",
            "যুক্তাক্ষর যেমন ক্ষ ও জ্ঞ প্রায়ই দেখা যায়।",
        ],
    },
    Script {
        name: "gurmukhi_punjabi",
        spaced: true,
        ends: "\u{964}.?!",
        sentences: &["ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਬਹੁਤ ਮਿੱਠੀ ਹੈ।", "ਇਹ ਗੁਰਮੁਖੀ ਲਿਪੀ ਵਿੱਚ ਲਿਖੀ ਜਾਂਦੀ ਹੈ।"],
    },
    Script {
        name: "gujarati",
        spaced: true,
        ends: "\u{964}.?!",
        sentences: &[
            "ગુજરાતી ભાષા ગુજરાત રાજ્યની મુખ્ય ભાષા છે.",
            "તે લાખો લોકો બોલે છે અને લખે છે.",
        ],
    },
    Script {
        name: "tamil",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "தமிழ் உலகின் மிகப் பழமையான மொழிகளில் ஒன்று.",
            "இது பல நாடுகளில் பேசப்படுகிறது.",
            "புள்ளி வைத்த எழுத்துகள் மெய்யெழுத்துகள் எனப்படும்.",
        ],
    },
    Script {
        name: "telugu",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "తెలుగు భారతదేశంలోని ప్రధాన భాషలలో ఒకటి.",
            "సంయుక్తాక్షరాలు ఇక్కడ చాలా సాధారణం.",
        ],
    },
    Script {
        name: "kannada",
        spaced: true,
        ends: ".?!",
        sentences: &["ಕನ್ನಡ ಕರ್ನಾಟಕದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.", "ಇದು ಬಹಳ ಪ್ರಾಚೀನ ಭಾಷೆ."],
    },
    Script {
        name: "malayalam",
        spaced: true,
        ends: ".?!",
        sentences: &["മലയാളം കേരളത്തിലെ പ്രധാന ഭാഷയാണ്.", "ഇതിന് ചില്ലക്ഷരങ്ങൾ ഉണ്ട്."],
    },
    Script {
        name: "sinhala",
        spaced: true,
        ends: ".?!",
        sentences: &["සිංහල ශ්‍රී ලංකාවේ ප්‍රධාන භාෂාවකි.", "එය ලක්ෂ ගණනක් අය කතා කරති."],
    },
    Script {
        name: "arabic_with_harakat",
        spaced: true,
        ends: ".\u{61f}?!:;\u{61b}",
        sentences: &[
            "اللُّغَةُ العَرَبِيَّةُ مِنْ أَقْدَمِ اللُّغَاتِ الحَيَّةِ فِي العَالَمِ.",
            "هل تعلم أنها تُكتب من اليمين إلى اليسار؟",
            "نعم، وهذا يجعلها فريدة بين اللغات.",
        ],
    },
    Script {
        name: "arabic_plain",
        spaced: true,
        ends: ".\u{61f}?!",
        sentences: &[
            "اللغة العربية من أقدم اللغات الحية في العالم.",
            "هل تعلم أنها تكتب من اليمين إلى اليسار؟",
        ],
    },
    Script {
        name: "persian_zwnj",
        spaced: true,
        ends: ".\u{61f}?!",
        sentences: &[
            "زبان فارسی یکی از زبان‌های هندواروپایی است.",
            "می‌خواهم کتاب‌های بیشتری بخوانم.",
            "آیا شما هم همین‌طور هستید؟",
        ],
    },
    Script {
        name: "urdu",
        spaced: true,
        ends: "\u{6d4}\u{61f}?!",
        sentences: &[
            "اردو پاکستان کی قومی زبان ہے۔",
            "یہ نستعلیق رسم الخط میں لکھی جاتی ہے۔",
        ],
    },
    Script {
        name: "hebrew_niqqud",
        spaced: true,
        ends: ".?!\u{5c3}",
        sentences: &[
            "עִבְרִית הִיא שָׂפָה שֵׁמִית עַתִּיקָה.",
            "הִיא נִכְתֶּבֶת מִיָּמִין לִשְׂמֹאל.",
            "שָׁלוֹם עוֹלָם!",
        ],
    },
    Script {
        name: "hebrew_plain",
        spaced: true,
        ends: ".?!\u{5c3}",
        sentences: &["עברית היא שפה שמית עתיקה.", "היא נכתבת מימין לשמאל."],
    },
    Script {
        name: "russian_cyrillic",
        spaced: true,
        ends: ".?!\u{2026}",
        sentences: &[
            "Русский язык является одним из самых распространённых языков мира.",
            "Он относится к восточнославянской группе языков.",
            "Съешь же ещё этих мягких французских булок, да выпей чаю.",
        ],
    },
    Script {
        name: "ukrainian_cyrillic",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "Українська мова є державною мовою України.",
            "Їжак, ґанок та європейський — слова з особливими літерами.",
        ],
    },
    Script {
        name: "greek_question_mark",
        spaced: true,
        ends: ".!\u{37e};",
        sentences: &[
            "Η ελληνική γλώσσα έχει μακρά ιστορία.",
            "Ποια είναι η πρωτεύουσα της Ελλάδας\u{37e}",
            "Η Αθήνα είναι η πρωτεύουσα.",
        ],
    },
    Script {
        name: "armenian",
        spaced: true,
        ends: "\u{589}.",
        sentences: &[
            "Հայոց լեզուն հնդեվրոպական լեզվաընտանիքի առանձին ճյուղ է։",
            "Այն գրվում է հայկական այբուբենով։",
            "Հայերենը շատ հին լեզու է։",
        ],
    },
    Script {
        name: "georgian",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "ქართული ენა ქართველური ენების ოჯახს მიეკუთვნება.",
            "ქართული ანბანი უნიკალურია.",
        ],
    },
    Script {
        name: "thai",
        spaced: false,
        ends: "",
        sentences: &[
            "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ไม่มีช่องว่างระหว่างคำ",
            "ประโยคจึงต่อเนื่องกันไปการตัดคำจึงเป็นเรื่องยาก",
        ],
    },
    Script {
        name: "lao",
        spaced: false,
        ends: "",
        sentences: &[
            "ພາສາລາວເປັນພາສາທີ່ມີສຽງວັນນະຍຸດບໍ່ມີຊ່ອງວ່າງລະຫວ່າງຄຳ",
            "ປະໂຫຍກຈຶ່ງຕໍ່ເນື່ອງກັນໄປ",
        ],
    },
    Script {
        name: "khmer",
        spaced: false,
        ends: "\u{17d4}\u{17d5}?!",
        sentences: &[
            "ភាសាខ្មែរគឺជាភាសាផ្លូវការរបស់ប្រទេសកម្ពុជា។",
            "វាមានអក្សរកាត់ជាច្រើន។",
            "ពាក្យមិនមានដកឃ្លា។",
        ],
    },
    Script {
        name: "burmese",
        spaced: false,
        ends: "\u{104b}",
        sentences: &[
            "မြန်မာဘာသာစကားသည် မြန်မာနိုင်ငံ၏ တရားဝင်ဘာသာစကားဖြစ်သည်။",
            "၎င်းတွင် ဗျည်းထပ်များ ပါဝင်သည်။",
        ],
    },
    Script {
        name: "tibetan",
        spaced: false,
        ends: "\u{f0d}\u{f0e}",
        sentences: &["བོད་སྐད་ནི་བོད་ཀྱི་སྐད་ཡིག་ཡིན།", "དེ་ནི་ཡིག་གཟུགས་ཁྱད་ཆོས་ཅན་ཡིན།"],
    },
    Script {
        name: "amharic_ethiopic",
        spaced: true,
        ends: "\u{1362}\u{1367}.",
        sentences: &[
            "አማርኛ የኢትዮጵያ ኦፊሴላዊ ቋንቋ ነው።",
            "ፊደሉ ፊደል ገበታ ይባላል።",
            "ይህ በጣም ጥንታዊ ነው።",
        ],
    },
    Script {
        name: "chinese",
        spaced: false,
        ends: "\u{3002}\u{ff01}\u{ff1f}",
        sentences: &[
            "汉语是世界上使用人数最多的语言。",
            "它没有空格分词。",
            "我们今天去公园玩，好吗？",
        ],
    },
    Script {
        name: "japanese",
        spaced: false,
        ends: "\u{3002}\u{ff01}\u{ff1f}\u{300d}",
        sentences: &[
            "日本語は漢字、ひらがな、カタカナを使います。",
            "これは世界でも珍しい書記体系です。",
            "「こんにちは。」と彼は言いました。",
        ],
    },
    Script {
        name: "korean_hangul",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "한국어는 한글이라는 고유한 문자를 사용합니다.",
            "세종대왕이 창제했습니다.",
            "자음과 모음을 조합해서 글자를 만듭니다.",
        ],
    },
    Script {
        name: "vietnamese",
        spaced: true,
        ends: ".?!",
        sentences: &[
            "Tiếng Việt có nhiều dấu thanh và dấu mũ.",
            "Chữ quốc ngữ dùng bảng chữ cái Latinh.",
            "Xin chào các bạn!",
        ],
    },
    Script {
        name: "cherokee",
        spaced: true,
        ends: ".",
        sentences: &["ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ᎠᏍᎦᏯ ᎤᎾᏙᏢᏒ.", "ᎾᏍᎩ ᎤᏬᏚᎯ ᎠᏂᏴᏫᏯ."],
    },
    Script {
        name: "thaana_dhivehi",
        spaced: true,
        ends: ".",
        sentences: &["ދިވެހި ބަސް ދިވެހިރާއްޖޭގެ ރަސްމީ ބަހެވެ.", "އެއީ ވަރަށް ރީތި ބަހެކެވެ."],
    },
    Script {
        name: "mixed_bidi_multi_script",
        spaced: true,
        ends: "",
        sentences: &[
            "Hello שלום مرحبا नमस्ते 你好 здравствуйте สวัสดี 안녕하세요.",
            "RTL \u{202b}שלום\u{202c} and LTR mixed, 123 456.",
        ],
    },
];

const CLOSERS: &str = "\"')]}\u{bb}\u{201d}\u{2019}\u{300d}\u{300f}\u{ff09}\u{3011}\u{300b}";

fn script_doc(sc: &Script, reps: usize) -> String {
    let mut out = String::new();
    for _ in 0..reps {
        for sent in sc.sentences {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(sent);
        }
    }
    out
}

fn section_scripts(rep: &mut Report, quick: bool) {
    let backends = ["chars", "words", "cl100k_base"];
    let norms = [Normalization::None, Normalization::Nfc, Normalization::Nfkc];
    let maxes: Vec<usize> = if quick {
        vec![4, 64]
    } else {
        vec![4, 16, 64, 256]
    };
    let mut runs = 0usize;
    for sc in SCRIPTS {
        let doc = script_doc(sc, 12);
        let variants: Vec<(String, String)> = vec![
            (sc.name.to_string(), doc.clone()),
            (format!("{}_nfd", sc.name), doc.nfd().collect::<String>()),
        ];
        for (vname, text) in &variants {
            for bn in backends {
                let b = be(bn);
                rep.hard(
                    "script_text_counts_as_nonzero_tokens",
                    b.count(text) > 0,
                    &|| format!("[F:{vname}] be={bn}"),
                );
                for &max in &maxes {
                    for &norm in &norms {
                        for &ctx in &[false, true] {
                            for &ovk in &[0usize, 1] {
                                let ov = if ovk == 0 { 0 } else { max / 4 };
                                let c = cfg(max, ov, norm, true, ctx);
                                let o = run(text, &c, &b);
                                runs += 1;
                                check_outcome(rep, &format!("F:{vname}"), text, &c, bn, &b, &o);
                                rep.hard(
                                    "F_case_time_under_5s",
                                    o.elapsed < Duration::from_secs(5),
                                    &|| {
                                        format!(
                                            "[F:{vname}] be={bn} max={max} took {:.0} ms",
                                            ms(o.elapsed)
                                        )
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        // Exact fit in this script: max == size is one chunk, one less must split.
        let b = be("chars");
        let n = b.count(&doc);
        if n >= 2 {
            let one = script_doc(sc, 1);
            let n1 = b.count(&one);
            if n1 >= 2 {
                let c_fit = cfg(n1, 0, Normalization::None, true, false);
                let o = run(&one, &c_fit, &b);
                if o.err.is_none() && !o.panicked {
                    rep.hard(
                        "script_exact_fit_is_one_chunk",
                        o.chunks.len() == 1,
                        &|| {
                            format!(
                                "[F:{}] max=count={n1} got {} chunks",
                                sc.name,
                                o.chunks.len()
                            )
                        },
                    );
                }
            }
        }

        // Chunks of a text made of short sentences must end at sentence ends in this script.
        if !sc.ends.is_empty() {
            let longest = sc
                .sentences
                .iter()
                .map(|x| x.chars().count())
                .max()
                .unwrap_or(1);
            for &ctx in &[false, true] {
                let max = longest * 2 + 4;
                let c = cfg(max, 0, Normalization::None, true, ctx);
                let o = run(&doc, &c, &b);
                check_outcome(
                    rep,
                    &format!("F:{}:sentences", sc.name),
                    &doc,
                    &c,
                    "chars",
                    &b,
                    &o,
                );
                if o.err.is_none() && !o.panicked && o.chunks.len() >= 4 {
                    let total = o.chunks.len() - 1;
                    let mut good = 0usize;
                    let mut bad_example = String::new();
                    for ch in &o.chunks[..total] {
                        let t = ch.text.trim_end();
                        let term = t.chars().rev().find(|c| !CLOSERS.contains(*c));
                        if term.is_some_and(|c| sc.ends.contains(c)) {
                            good += 1;
                        } else if bad_example.is_empty() {
                            bad_example = trunc_dbg(
                                &t.chars()
                                    .rev()
                                    .take(30)
                                    .collect::<String>()
                                    .chars()
                                    .rev()
                                    .collect::<String>(),
                                40,
                            );
                        }
                    }
                    let rate = good as f64 / total as f64;
                    rep.hard("script_chunks_end_at_sentence_boundaries", rate >= 0.9, &|| {
                        format!(
                            "[F:{}] ctx={ctx} max={max}: {good}/{total} chunk ends at a sentence end; example end: {bad_example}",
                            sc.name
                        )
                    });
                }
            }
        }

        // Space-delimited scripts: cuts should not fall inside short words.
        if sc.spaced {
            for bn in ["chars", "cl100k_base"] {
                let bk = be(bn);
                for &max in &[64usize, 256] {
                    let c = cfg(max, 0, Normalization::None, true, false);
                    let o = run(&doc, &c, &bk);
                    if o.err.is_some() || o.panicked {
                        continue;
                    }
                    for ch in &o.chunks {
                        for &p in &[ch.start_byte, ch.end_byte] {
                            if p == 0 || p >= doc.len() {
                                continue;
                            }
                            let l = doc[..p].chars().next_back();
                            let r = doc[p..].chars().next();
                            let (Some(l), Some(r)) = (l, r) else { continue };
                            if l.is_whitespace() || r.is_whitespace() {
                                continue;
                            }
                            let ws = doc[..p]
                                .char_indices()
                                .rev()
                                .find(|(_, c)| c.is_whitespace())
                                .map(|(i, c)| i + c.len_utf8())
                                .unwrap_or(0);
                            let we = doc[p..]
                                .find(char::is_whitespace)
                                .map_or(doc.len(), |i| p + i);
                            let word = &doc[ws..we];
                            let short = bk.count(word) * 2 <= max;
                            rep.soft("script_cut_inside_short_word", !short, &|| {
                                format!(
                                    "[F:{}] be={bn} max={max} cut inside word {:?}",
                                    sc.name, word
                                )
                            });
                        }
                    }
                }
            }
        }
    }
    rep.info(format!(
        "F scripts: {} writing systems (+NFD variants), {} chunking runs",
        SCRIPTS.len(),
        runs
    ));
}

// ---------------------------------------------------------------------------------------------
// C. Robustness + fuzzing
// ---------------------------------------------------------------------------------------------

const FRAGS: &[&str] = &[
    "# Heading", "## Sub heading ##", "###", "#hashtag", "####### seven", "    # indented",
    "Setext\n=====", "Setext2\n-----", "```", "```rust", "~~~", "````", "``` ```",
    "```\ncode\n```", "~~~\n# not heading\n~~~", "- item", "* item", "+ item", "1. one", "2) two",
    "- [ ] task", "  - nested", "    - deeper", "-", "1.", "10000000000. big", "> quote",
    "> > nested", ">", "| a | b |\n|---|---|\n| 1 | 2 |", "---", "***", "___",
    "<div>html</div>", "<!-- comment -->", "<br/>", "[link](http://example.com/a_b)", "![img](x.png)",
    "`inline`", "**bold**", "*em*", "~~strike~~", "[^1]: footnote",
    "https://example.com/very/long/path?query=1&b=2", "Hello world.", "Dr. Smith e.g. this i.e. that.",
    "3.14", "1,000.50", "It's", "don't", "state-of-the-art", "snake_case_word",
    "T\u{fc}rk\u{e7}e: \u{131}\u{11f}\u{fc}\u{15f}\u{f6}\u{e7}\u{130}\u{11e}\u{dc}\u{15e}\u{d6}\u{c7}",
    "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3002}", "\u{d55c}\u{ad6d}\u{c5b4}",
    "\u{e44}\u{e17}\u{e22}", "\u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629}", "\u{5e2}\u{5d1}\u{5e8}\u{5d9}\u{5ea}",
    "\u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940}", "\u{1F600}", "\u{1F469}\u{200D}\u{1F4BB}",
    "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}", "\u{1F1E9}\u{1F1EA}", "e\u{301}",
    "\u{200d}", "\u{feff}", "\u{0}", "\u{7f}", "\t", "   ", "\u{a0}", "\u{2028}", "\u{3000}", "\r",
    "\u{fb01}", "\u{ff21}", "\u{301}", "\\#", "\\`", "&amp;", "<|endoftext|>",
];

const SEPS: &[&str] = &[
    "", " ", "\n", "\n", "\n\n", "\n\n", "\r\n", "\n\n\n", "\t", "  \n",
];

fn mutate(rng: &mut Rng, s: &str) -> String {
    let mut v: Vec<char> = s.chars().collect();
    if v.is_empty() {
        return String::new();
    }
    let extras = [
        '\n',
        ' ',
        '`',
        '#',
        '>',
        '-',
        '\u{301}',
        '\u{200d}',
        '\u{1F600}',
        '\u{e9}',
    ];
    for _ in 0..(1 + rng.below(4)) {
        if v.is_empty() {
            break;
        }
        let i = rng.below(v.len());
        match rng.below(4) {
            0 => {
                v.remove(i);
            }
            1 => {
                let ch = v[i];
                v.insert(i, ch);
            }
            2 => {
                let j = rng.below(v.len());
                v.swap(i, j);
            }
            _ => {
                let extra = *rng.pick(&extras);
                v.insert(i, extra);
            }
        }
    }
    v.into_iter().collect()
}

fn gen_soup(rng: &mut Rng) -> String {
    let n = 1 + rng.below(40);
    let mut s = String::new();
    for _ in 0..n {
        s.push_str(rng.pick(FRAGS));
        s.push_str(rng.pick(SEPS));
    }
    if rng.chance(1, 3) {
        s = mutate(rng, &s);
    }
    s
}

fn rand_cfg(rng: &mut Rng) -> (ChunkConfig, &'static str) {
    let names: [&'static str; 4] = ["chars", "chars", "words", "cl100k_base"];
    let bn: &'static str = rng.pick(&names);
    let maxes: [usize; 14] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 512];
    let max = *rng.pick(&maxes);
    let ov = if rng.chance(1, 2) { 0 } else { rng.below(max) };
    let norms: [Normalization; 3] = [Normalization::None, Normalization::Nfc, Normalization::Nfkc];
    let norm = *rng.pick(&norms);
    let snap = rng.chance(3, 4);
    let ctx = rng.chance(2, 3);
    (cfg(max, ov, norm, snap, ctx), bn)
}

fn stream_inner(
    path: &str,
    c: &ChunkConfig,
    b: &Arc<Backend>,
) -> Result<(Vec<Chunk>, Vec<OrigSpan>), ChipError> {
    let mut st = FileChunker::open(path, c, b, false)?;
    let mut cs: Vec<Chunk> = Vec::new();
    let mut ss: Vec<OrigSpan> = Vec::new();
    while let Some((ch, sp)) = st.next_chunk()? {
        cs.push(ch);
        ss.push(sp);
    }
    Ok((cs, ss))
}

fn stream_chunks(
    src: &str,
    c: &ChunkConfig,
    b: &Arc<Backend>,
    tag: &str,
) -> Result<(Vec<Chunk>, Vec<OrigSpan>), String> {
    let path = std::env::temp_dir().join(format!("rs_suite_{}_{}.md", std::process::id(), tag));
    if let Err(e) = std::fs::write(&path, src.as_bytes()) {
        return Err(e.to_string());
    }
    let p = path.to_string_lossy().to_string();
    let r = catch_unwind(AssertUnwindSafe(|| stream_inner(&p, c, b)));
    let _ = std::fs::remove_file(&path);
    match r {
        Err(_) => Err("panic".to_string()),
        Ok(Err(e)) => Err(e.to_string()),
        Ok(Ok(v)) => Ok(v),
    }
}

fn total_fails(rep: &Report) -> usize {
    rep.rules.values().map(|s| s.fails).sum()
}

fn failing_rules(src: &str, c: &ChunkConfig, bn: &str, b: &Arc<Backend>) -> Vec<String> {
    let o = run(src, c, b);
    let mut r = Report::default();
    check_outcome(&mut r, "shrink", src, c, bn, b, &o);
    r.rules
        .iter()
        .filter(|(_, st)| st.fails > 0)
        .map(|(k, _)| k.clone())
        .collect()
}

/// Delta-debugging: remove pieces of `src` while `rule` keeps failing.
fn shrink_failure(rule: &str, src: &str, c: &ChunkConfig, bn: &str, b: &Arc<Backend>) -> String {
    let fails = |s: &str| -> bool { failing_rules(s, c, bn, b).iter().any(|r| r == rule) };
    let mut cur: Vec<char> = src.chars().collect();
    let mut n = 2usize;
    let mut iters = 0usize;
    while cur.len() >= 2 && iters < 600 {
        let chunk = cur.len().div_ceil(n);
        let mut reduced = false;
        let mut start = 0usize;
        while start < cur.len() {
            let end = (start + chunk).min(cur.len());
            let cand: String = cur[..start].iter().chain(cur[end..].iter()).collect();
            iters += 1;
            if !cand.is_empty() && fails(&cand) {
                cur = cand.chars().collect();
                n = (n - 1).max(2);
                reduced = true;
                break;
            }
            start = end;
        }
        if !reduced {
            if n >= cur.len() {
                break;
            }
            n = (n * 2).min(cur.len());
        }
    }
    cur.into_iter().collect()
}

fn section_fuzz(rep: &mut Report, quick: bool) {
    let iters = env_usize("RS_FUZZ_ITERS", if quick { 400 } else { 3000 });
    let mut rng = Rng::new(env_u64("RS_SEED", 0xC0FFEE));
    let mut slowest = 0.0f64;
    let mut streams = 0usize;
    let mut shrunk_by_rule: BTreeMap<String, usize> = BTreeMap::new();
    for it in 0..iters {
        let src = gen_soup(&mut rng);
        let (c, bn) = rand_cfg(&mut rng);
        let b = be(bn);
        let o = run(&src, &c, &b);
        let label = format!("C:fuzz#{it}");
        let before = total_fails(rep);
        check_outcome(rep, &label, &src, &c, bn, &b, &o);
        if total_fails(rep) > before {
            for rule in failing_rules(&src, &c, bn, &b) {
                let is_hard = rep.rules.get(&rule).is_some_and(|st| st.hard);
                let seen = shrunk_by_rule.get(&rule).copied().unwrap_or(0);
                if !is_hard || seen >= 2 {
                    continue;
                }
                shrunk_by_rule.insert(rule.clone(), seen + 1);
                let min = shrink_failure(&rule, &src, &c, bn, &b);
                let o2 = run(&min, &c, &b);
                let listing: Vec<String> = o2
                    .chunks
                    .iter()
                    .take(8)
                    .map(|ch| {
                        format!(
                            "{}..{}:{}",
                            ch.start_byte,
                            ch.end_byte,
                            trunc_dbg(&ch.text, 30)
                        )
                    })
                    .collect();
                rep.info(format!(
                    "SHRUNK\t{rule}\tbe={bn} max={} ov={} norm={:?} snap={} ctx={}\tsrc={}\tchunks=[{}]",
                    c.max_tokens,
                    c.overlap_tokens,
                    c.normalization,
                    c.snap_words,
                    c.include_context,
                    trunc_dbg(&min, 200),
                    listing.join(" | ")
                ));
            }
        }
        slowest = slowest.max(ms(o.elapsed));
        rep.hard(
            "fuzz_case_time_under_2s",
            o.elapsed < Duration::from_secs(2),
            &|| {
                format!(
                    "[{label}] be={bn} took {:.0} ms src={}",
                    ms(o.elapsed),
                    trunc_dbg(&src, 120)
                )
            },
        );
        if it % 5 == 0 {
            let o2 = run(&src, &c, &b);
            rep.hard(
                "deterministic_same_input_same_output",
                same_outcome(&o, &o2),
                &|| {
                    format!(
                        "[{label}] be={bn} max={} src={}",
                        c.max_tokens,
                        trunc_dbg(&src, 160)
                    )
                },
            );
        }
        if it % 25 == 0 && c.include_context && !src.starts_with('\u{feff}') && c.validate().is_ok()
        {
            streams += 1;
            let sres = stream_chunks(&src, &c, &b, &format!("{it}"));
            match &sres {
                Err(_) => {
                    rep.hard("file_stream_equals_batch", o.err.is_some(), &|| {
                        format!(
                            "[{label}] be={bn} stream errored but batch succeeded; src={}",
                            trunc_dbg(&src, 160)
                        )
                    });
                }
                Ok((sc, ss)) => {
                    let eq = o.err.is_none() && *sc == o.chunks && *ss == o.spans;
                    rep.hard("file_stream_equals_batch", eq, &|| {
                        let first = sc
                            .iter()
                            .zip(o.chunks.iter())
                            .position(|(x, y)| x != y);
                        format!(
                            "[{label}] be={bn} max={} ov={} norm={:?} stream_chunks={} batch_chunks={} first_diff={:?} src={}",
                            c.max_tokens,
                            c.overlap_tokens,
                            c.normalization,
                            sc.len(),
                            o.chunks.len(),
                            first,
                            trunc_dbg(&src, 200)
                        )
                    });
                }
            }
        }
    }
    rep.info(format!(
        "C fuzz: {iters} cases, {streams} stream-vs-batch comparisons, slowest {slowest:.1} ms"
    ));
}

// ---- heading-only documents (they legitimately yield no chunks: headings become context) ----

fn is_atx_line(line: &str) -> bool {
    let l = line.trim_end_matches('\r');
    let spaces = l.bytes().take_while(|&b| b == b' ').count();
    if spaces > 3 {
        return false;
    }
    let rest = &l[spaces..];
    let hashes = rest.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return false;
    }
    matches!(rest.as_bytes().get(hashes), None | Some(b' ') | Some(b'\t'))
}

fn is_setext_underline_line(line: &str) -> bool {
    let l = line.trim_end_matches('\r');
    let spaces = l.bytes().take_while(|&b| b == b' ').count();
    if spaces > 3 {
        return false;
    }
    let rest = l[spaces..].trim_end_matches([' ', '\t']);
    match rest.as_bytes().first() {
        Some(&c) if c == b'=' || c == b'-' => rest.bytes().all(|b| b == c),
        _ => false,
    }
}

fn can_be_setext_title_line(line: &str) -> bool {
    let spaces = line.bytes().take_while(|&b| b == b' ').count();
    if spaces > 3 {
        return false;
    }
    let t = &line[spaces..];
    let Some(c) = t.chars().next() else {
        return false;
    };
    if matches!(c, '-' | '*' | '+' | '>' | '|' | '#' | '\t') {
        return false;
    }
    let digits = t.bytes().take_while(u8::is_ascii_digit).count();
    !(digits > 0 && matches!(t.as_bytes().get(digits), Some(b'.') | Some(b')')))
}

/// True when the document contains at least one ATX heading line or a setext heading
/// (a paragraph line directly followed by a `===`/`---` underline).
fn has_any_heading(s: &str) -> bool {
    // Lenient on purpose (a false "has heading" only skips the strict body-coverage check):
    // the library decides on the FIRST line of the open paragraph, and may see the
    // underline after normalization (e.g. trailing U+3000 becomes a plain space).
    fn lenient_underline(line: &str) -> bool {
        let l = line.trim_end();
        let spaces = l.bytes().take_while(|&b| b == b' ').count();
        if spaces > 3 {
            return false;
        }
        let rest = &l[spaces..];
        match rest.as_bytes().first() {
            Some(&c) if c == b'=' || c == b'-' => rest.bytes().all(|b| b == c),
            _ => false,
        }
    }
    let lines: Vec<&str> = s.lines().collect();
    lines.iter().enumerate().any(|(j, l)| {
        if l.trim_start().starts_with('#') {
            return true;
        }
        if !lenient_underline(l) {
            return false;
        }
        // Walk back to the first line of the paragraph this underline would close.
        let mut k = j;
        // A fence line (opening or closing) ends the previous block in the library's parser.
        // Use the library's own fence rule: "``` `" (backtick in the info string) is NOT a
        // fence, so a hand-rolled starts_with("```") check misclassifies it.
        let is_fence = |l: &str| crate::markdown::scanners::opening_fence(l).is_some();
        while k > 0
            && !lines[k - 1].trim().is_empty()
            && !is_atx_line(lines[k - 1])
            && !is_fence(lines[k - 1])
        {
            k -= 1;
        }
        k < j && can_be_setext_title_line(lines[k])
    })
}

/// True when every non-blank line is an ATX heading, or a paragraph line directly followed
/// by a setext underline. Independent of the library's parser on purpose.
fn heading_only_doc(s: &str) -> bool {
    let lines: Vec<&str> = s.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.trim().is_empty() || is_atx_line(l) {
            i += 1;
            continue;
        }
        if can_be_setext_title_line(l)
            && i + 1 < lines.len()
            && is_setext_underline_line(lines[i + 1])
        {
            i += 2;
            continue;
        }
        // CommonMark setext titles may span two lines (the parser accepts up to two).
        if can_be_setext_title_line(l)
            && !l.trim().is_empty()
            && i + 2 < lines.len()
            && !lines[i + 1].trim().is_empty()
            && !is_atx_line(lines[i + 1])
            && can_be_setext_title_line(lines[i + 1])
            && is_setext_underline_line(lines[i + 2])
        {
            i += 3;
            continue;
        }
        return false;
    }
    true
}

fn pathological_inputs() -> Vec<(&'static str, String)> {
    vec![
        ("giant_single_line_words_2MB", "word ".repeat(400_000)),
        ("no_whitespace_1MiB", "x".repeat(1 << 20)),
        ("backticks_200k", "`".repeat(200_000)),
        (
            "nested_quotes_100k",
            format!("{} text", ">".repeat(100_000)),
        ),
        (
            "deep_indented_list",
            (0..3000)
                .map(|i| format!("{}- item {}\n", " ".repeat(i % 400), i))
                .collect::<String>(),
        ),
        (
            "many_headings_50k",
            (0..50_000)
                .map(|i| format!("# H{i}\n\ntext {i}\n\n"))
                .collect::<String>(),
        ),
        ("million_blank_lines", "\n".repeat(1_000_000)),
        ("many_tiny_paragraphs_200k", "p\n\n".repeat(200_000)),
        (
            "unterminated_fence_100k_lines",
            format!("```\n{}", "code line\n".repeat(100_000)),
        ),
        (
            "combining_flood_100k",
            format!("a{}", "\u{301}".repeat(100_000)),
        ),
        (
            "zwj_chain_50k",
            format!("\u{1F468}{}", "\u{200D}\u{1F469}".repeat(50_000)),
        ),
        ("regional_indicator_flood_100k", "\u{1F1E9}".repeat(100_001)),
        (
            "table_50k_rows",
            format!("| a | b |\n|---|---|\n{}", "| 1 | 2 |\n".repeat(50_000)),
        ),
        (
            "html_comment_unclosed_big",
            format!("<!-- {}", "x ".repeat(200_000)),
        ),
        ("bracket_flood_100k", "[".repeat(100_000)),
        ("crlf_big_doc", "line of text\r\n\r\n".repeat(100_000)),
        ("nul_flood_100k", "\u{0}".repeat(100_000)),
        ("space_flood_100k", format!("a{}b", " ".repeat(100_000))),
        ("tab_flood_100k", format!("a{}b", "\t".repeat(100_000))),
        (
            "mixed_whitespace_flood",
            format!("{}x", " \t".repeat(50_000)),
        ),
        ("nbsp_flood_100k", format!("a{}b", "\u{a0}".repeat(100_000))),
        ("quote_flood_500k", "\"".repeat(500_000)),
        ("closing_bracket_flood_500k", ")".repeat(500_000)),
        ("ascii_sentence_flood", ". ".repeat(300_000)),
        ("terminator_then_closer_flood", ".\") ".repeat(150_000)),
        ("danda_flood_200k", "\u{964}".repeat(200_000)),
        ("cjk_full_stop_flood_200k", "\u{3002}".repeat(200_000)),
        (
            "devanagari_virama_flood",
            format!("\u{915}{}", "\u{94d}".repeat(100_000)),
        ),
        (
            "thai_tone_mark_flood",
            format!("\u{e01}{}", "\u{e48}".repeat(100_000)),
        ),
        (
            "arabic_diacritic_flood",
            format!("\u{628}{}", "\u{64e}".repeat(100_000)),
        ),
        ("zwnj_flood", "\u{628}\u{200c}".repeat(100_000)),
    ]
}

fn section_pathological(rep: &mut Report, quick: bool) {
    let inputs = pathological_inputs();
    let mut worst = (String::new(), 0.0f64);
    for (name, src) in &inputs {
        for bn in ["chars", "words", "cl100k_base"] {
            let b = be(bn);
            let ctxs: Vec<bool> = if quick { vec![true] } else { vec![true, false] };
            for ctx in ctxs {
                let c = cfg(256, 16, Normalization::Nfc, true, ctx);
                let o = run(src, &c, &b);
                let label = format!("C2:{name}:ctx={ctx}");
                check_outcome(rep, &label, src, &c, bn, &b, &o);
                rep.hard(
                    "pathological_input_time_under_10s",
                    o.elapsed < Duration::from_secs(10),
                    &|| format!("[{label}] be={bn} took {:.0} ms", ms(o.elapsed)),
                );
                if ms(o.elapsed) > worst.1 {
                    worst = (format!("{label} be={bn}"), ms(o.elapsed));
                }
            }
        }
    }
    rep.info(format!(
        "C2 pathological: {} inputs, slowest {} = {:.0} ms",
        inputs.len(),
        worst.0,
        worst.1
    ));
}

// ---------------------------------------------------------------------------------------------
// D. Oracle checks against pulldown-cmark (CommonMark reference). Reported, not required.
// ---------------------------------------------------------------------------------------------

struct OracleNode {
    kind: &'static str,
    start: usize,
    end: usize,
    nested: bool,
}

fn oracle_nodes(src: &str) -> Vec<OracleNode> {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut nodes: Vec<OracleNode> = Vec::new();
    let mut depth = 0usize;
    for (ev, range) in Parser::new_ext(src, opts).into_offset_iter() {
        match ev {
            Event::Start(tag) => {
                let d = format!("{:?}", tag);
                let kind: &'static str = if d.starts_with("Paragraph") {
                    "paragraph"
                } else if d.starts_with("CodeBlock") {
                    if d.contains("Fenced") {
                        "fence"
                    } else {
                        "indented_code"
                    }
                } else if d.starts_with("Table(") {
                    "table"
                } else if d.starts_with("HtmlBlock") {
                    "html"
                } else if d.starts_with("Item") {
                    "item"
                } else if d.starts_with("List") {
                    "list"
                } else if d.starts_with("BlockQuote") {
                    "quote"
                } else {
                    ""
                };
                if !kind.is_empty() {
                    nodes.push(OracleNode {
                        kind,
                        start: range.start,
                        end: range.end,
                        nested: depth > 0,
                    });
                }
                if kind == "list" || kind == "item" || kind == "quote" {
                    depth += 1;
                }
            }
            Event::End(end) => {
                let d = format!("{:?}", end);
                if d.starts_with("List") || d.starts_with("Item") || d.starts_with("BlockQuote") {
                    depth = depth.saturating_sub(1);
                }
            }
            _ => {}
        }
    }
    nodes
}

fn gen_realistic(rng: &mut Rng) -> String {
    let mut s = String::new();
    let n = 5 + rng.below(12);
    for _ in 0..n {
        if !s.is_empty() {
            s.push_str("\n\n");
        }
        match rng.below(10) {
            0..=2 => {
                let lines = 1 + rng.below(4);
                for l in 0..lines {
                    if l > 0 {
                        s.push('\n');
                    }
                    let w = 4 + rng.below(8);
                    s.push_str(&words_line(rng, w));
                    s.push('.');
                }
            }
            3 => {
                let fch = if rng.chance(1, 2) { "```" } else { "~~~" };
                s.push_str(fch);
                s.push_str("\nfn main() {\n\n    println!(\"x\");\n}\n");
                s.push_str(fch);
            }
            4 => {
                s.push_str("- first item\n- second item\n  - nested a\n  - nested b\n    - deeper\n- third item");
            }
            5 => {
                s.push_str("3. start at three\n4. next\n\n   continuation paragraph\n5. last");
            }
            6 => {
                s.push_str("> quoted text line\n> continues here\n>\n> - quoted list\n> - second");
            }
            7 => {
                s.push_str("| h1 | h2 |\n|----|----|\n| a | b |\n| c | d |");
            }
            8 => {
                s.push_str("<div>\nraw html block\n</div>");
            }
            _ => {
                s.push_str("- loose item one\n\n- loose item two\n\n- loose item three");
            }
        }
    }
    s.push('\n');
    s
}

fn section_oracle(rep: &mut Report, quick: bool) {
    let docs = if quick { 80 } else { 400 };
    let mut rng = Rng::new(env_u64("RS_SEED", 0xC0FFEE) ^ 0xD1);
    let setups: [(&str, [usize; 3]); 2] =
        [("chars", [60, 120, 240]), ("cl100k_base", [16, 32, 64])];
    for d in 0..docs {
        let src = gen_realistic(&mut rng);
        let nodes = oracle_nodes(&src);

        // list scanner vs reference
        let ours: Vec<usize> = crate::parse_list_items(&src)
            .iter()
            .map(|i| i.start_line)
            .collect();
        let mut theirs: Vec<usize> = Vec::new();
        for n in nodes.iter().filter(|n| n.kind == "item") {
            theirs.push(src[..n.start].bytes().filter(|&x| x == b'\n').count());
        }
        rep.soft(
            "list_item_start_lines_match_pulldown",
            ours == theirs,
            &|| {
                format!(
                    "[D#{d}] ours={:?} pulldown={:?} src={}",
                    ours,
                    theirs,
                    trunc_dbg(&src, 200)
                )
            },
        );

        // fenced block count vs reference
        let mut parser = BlockParser::new();
        let mut blocks = Vec::new();
        let mut parse_ok = true;
        for line in src.split_inclusive('\n') {
            if parser.push_line(line, &mut blocks).is_err() {
                parse_ok = false;
            }
        }
        parser.finish(&mut blocks);
        let our_fences = blocks.iter().filter(|x| x.is_code).count();
        let their_fences = nodes.iter().filter(|n| n.kind == "fence").count();
        rep.soft(
            "fenced_block_count_matches_pulldown",
            parse_ok && our_fences == their_fences,
            &|| {
                format!(
                    "[D#{d}] ours={our_fences} pulldown={their_fences} src={}",
                    trunc_dbg(&src, 200)
                )
            },
        );

        // atomicity of CommonMark leaf blocks that fit the budget
        for (bn, maxes) in setups {
            let b = be(bn);
            for &max in &maxes {
                let c = cfg(max, 0, Normalization::None, true, true);
                let o = run(&src, &c, &b);
                check_outcome(rep, &format!("D#{d}"), &src, &c, bn, &b, &o);
                if o.err.is_some() || o.panicked {
                    continue;
                }
                let mut cuts: Vec<usize> = Vec::new();
                for ch in &o.chunks {
                    cuts.push(ch.start_byte);
                    cuts.push(ch.end_byte);
                }
                for n in &nodes {
                    if n.kind == "list" || n.kind == "quote" {
                        continue;
                    }
                    if n.kind == "item"
                        && nodes
                            .iter()
                            .any(|m| m.kind == "list" && m.start > n.start && m.end <= n.end)
                    {
                        continue; // item with nested list: may legitimately be split at nested items
                    }
                    let text = src[n.start..n.end].trim_end();
                    if text.is_empty() || b.count(text) > max {
                        continue;
                    }
                    let end = n.start + text.len();
                    let inside: Vec<usize> = cuts
                        .iter()
                        .copied()
                        .filter(|&x| n.start < x && x < end)
                        .collect();
                    let variant = if n.kind == "item" {
                        if text.contains("\n\n") {
                            "multi_paragraph"
                        } else {
                            "single_paragraph"
                        }
                    } else if n.nested {
                        "nested"
                    } else {
                        "toplevel"
                    };
                    let rule = format!("oracle_{}_not_cut_when_it_fits_{}", n.kind, variant);
                    rep.soft(&rule, inside.is_empty(), &|| {
                        format!(
                            "[D#{d}] be={bn} max={max} cut at {:?} inside {} {}..{}: {}",
                            inside,
                            n.kind,
                            n.start,
                            end,
                            trunc_dbg(text, 160)
                        )
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// E. Performance regression
// ---------------------------------------------------------------------------------------------

const ACCENTED_WORDS: [&str; 16] = [
    "Gr\u{f6}\u{df}e",
    "\u{c5}ngstr\u{f6}m",
    "fa\u{e7}ade",
    "na\u{ef}ve",
    "\u{141}\u{f3}d\u{17a}",
    "\u{10c}e\u{161}tina",
    "\u{fc}ber",
    "Stra\u{df}e",
    "caf\u{e9}",
    "se\u{f1}or",
    "\u{152}uvre",
    "\u{f8}ye",
    "\u{e4}hnlich",
    "co\u{f6}perate",
    "Z\u{fc}rich",
    "\u{c9}cole",
];

fn prose_from(rng: &mut Rng, words: &[&str], kib: usize) -> String {
    let mut s = String::new();
    while s.len() < kib * 1024 {
        let n = 20 + rng.below(40);
        for i in 0..n {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(rng.pick(words));
            if rng.below(12) == 0 {
                s.push('.');
            }
        }
        s.push_str(".\n\n");
    }
    s
}

fn perf_text(name: &str, kib: usize) -> String {
    let mut rng = Rng::new(0xFEED ^ kib as u64);
    let target = kib * 1024;
    match name {
        "english" => prose_from(&mut rng, &WORDS[..14], kib),
        "accented_latin" => prose_from(&mut rng, &ACCENTED_WORDS, kib),
        "markdown_blocks" => {
            let b = be("chars");
            let mut s = String::new();
            while s.len() < target {
                let (d, _) = gen_core_doc(&mut rng, 200, &b);
                s.push_str(&d);
                s.push('\n');
            }
            s
        }
        "headings_sections" => {
            let mut s = String::new();
            while s.len() < target {
                s.push_str(&gen_heading_doc(&mut rng).src);
            }
            s
        }
        "unicode_heavy" => {
            let mut s = String::new();
            while s.len() < target {
                s.push_str(rng.pick(FRAGS));
                s.push_str(if rng.chance(1, 3) { "\n\n" } else { " " });
            }
            s
        }
        "cjk_unbroken" => {
            let mut s = String::new();
            while s.len() < target {
                let cp = 0x4E00 + rng.below(0x5000) as u32;
                s.push(char::from_u32(cp).unwrap_or('x'));
            }
            s
        }
        "base64_oneline" => {
            let alpha = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut s = String::new();
            while s.len() < target {
                s.push(alpha[rng.below(alpha.len())] as char);
            }
            s
        }
        "url_run" => {
            let mut s = String::new();
            while s.len() < target {
                s.push_str(&format!(
                    "https://h{}.example.com/p{}/{}?q={}#f",
                    rng.below(1000),
                    rng.below(1000),
                    rng.below(100000),
                    rng.below(100000)
                ));
            }
            s
        }
        "many_tiny_paragraphs" => "p q r.\n\n".repeat(target / 8 + 1),
        "indented_lines" => {
            let mut s = String::new();
            let mut i = 0usize;
            while s.len() < target {
                s.push_str(&" ".repeat(i % 400));
                s.push_str(&format!("- item {i}\n"));
                i += 1;
            }
            s
        }
        "giant_list" => {
            let mut s = String::new();
            let mut i = 0usize;
            while s.len() < target {
                s.push_str(&format!("- list item number {i} with some words\n"));
                i += 1;
            }
            s
        }
        "giant_fence" => {
            let mut s = String::from("```rust\n");
            let mut i = 0usize;
            while s.len() < target {
                s.push_str(&format!("let value_{i} = compute(arg_{i}, {i});\n"));
                i += 1;
            }
            s.push_str("```\n");
            s
        }
        _ => {
            // long_line_words: one huge line with spaces, no newline
            let mut s = String::new();
            while s.len() < target {
                s.push_str(rng.pick(&WORDS));
                s.push(' ');
            }
            s
        }
    }
}

fn time_min(src: &str, c: &ChunkConfig, b: &Arc<Backend>, reps: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let r = smart_chunk_mapped(src, c, b);
        let d = ms(t.elapsed());
        std::hint::black_box(&r);
        if d < best {
            best = d;
        }
    }
    best
}

fn section_perf(rep: &mut Report, quick: bool) {
    if cfg!(debug_assertions) {
        rep.info("E performance: SKIPPED (debug build). Use --release.".to_string());
        return;
    }
    let names = [
        "english",
        "accented_latin",
        "markdown_blocks",
        "headings_sections",
        "unicode_heavy",
        "cjk_unbroken",
        "base64_oneline",
        "url_run",
        "many_tiny_paragraphs",
        "giant_list",
        "giant_fence",
        "long_line_words",
        "indented_lines",
    ];
    let (small_kib, big_kib) = if quick {
        (256usize, 1024usize)
    } else {
        (512usize, 2048usize)
    };
    let tol: f64 = std::env::var("RS_PERF_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35);
    let mut baseline: BTreeMap<String, f64> = BTreeMap::new();
    if let Ok(path) = std::env::var("RS_PERF_BASELINE") {
        if let Ok(txt) = std::fs::read_to_string(&path) {
            for line in txt.lines() {
                let mut it = line.split('\t');
                if let (Some(k), Some(v)) = (it.next(), it.next()) {
                    if let Ok(x) = v.parse::<f64>() {
                        baseline.insert(k.to_string(), x);
                    }
                }
            }
        } else {
            rep.info(format!(
                "E: RS_PERF_BASELINE '{path}' not readable; skipping baseline comparison"
            ));
        }
    }
    let mut save = String::new();
    rep.info(
        "PERF\tworkload\tbackend\tconfig\tms_small\tms_big\tratio(4x size)\tMiB/s(big)".to_string(),
    );
    for name in names {
        let t_small = perf_text(name, small_kib);
        let t_big = perf_text(name, big_kib);
        for bn in ["chars", "words", "cl100k_base"] {
            let b = be(bn);
            let default_cfg = ChunkConfig {
                max_tokens: 512,
                overlap_tokens: 50,
                ..Default::default()
            };
            let noctx = cfg(512, 0, Normalization::Nfc, true, false);
            for (cname, c) in [("default", default_cfg), ("noctx", noctx)] {
                let ms_small = time_min(&t_small, &c, &b, 3);
                let ms_big = time_min(&t_big, &c, &b, 2);
                let ratio = ms_big / ms_small.max(0.001);
                let mib_s = (t_big.len() as f64 / (1024.0 * 1024.0)) / (ms_big / 1000.0).max(1e-9);
                let size_ratio = t_big.len() as f64 / t_small.len().max(1) as f64;
                rep.hard("perf_scaling_not_superlinear", ratio <= size_ratio * 2.0 || ms_big < 25.0, &|| {
                    format!("{name}/{bn}/{cname}: {ms_small:.1} ms -> {ms_big:.1} ms for {size_ratio:.1}x input (ratio {ratio:.1})")
                });
                rep.hard(
                    "perf_throughput_at_least_1_MiB_per_s",
                    mib_s >= 1.0,
                    &|| format!("{name}/{bn}/{cname}: {mib_s:.2} MiB/s"),
                );
                let key = format!("{name}|{bn}|{cname}");
                if let Some(base) = baseline.get(&key) {
                    rep.soft(
                        "perf_vs_baseline_within_tolerance",
                        ms_big <= base * (1.0 + tol) || ms_big < 5.0,
                        &|| {
                            format!(
                                "{key}: {ms_big:.1} ms vs baseline {base:.1} ms (+{:.0}%)",
                                (ms_big / base - 1.0) * 100.0
                            )
                        },
                    );
                }
                save.push_str(&format!("{key}\t{ms_big:.3}\n"));
                rep.info(format!(
                    "PERF\t{name}\t{bn}\t{cname}\t{ms_small:.2}\t{ms_big:.2}\t{ratio:.2}\t{mib_s:.1}"
                ));
            }
        }
    }
    // Whitespace floods are the classic quadratic trap for regex/BPE tokenizers: check scaling
    // with small inputs (16 KiB -> 64 KiB) so a quadratic implementation is caught quickly.
    for (name, unit) in [
        ("space_flood", " "),
        ("tab_flood", "\t"),
        ("newline_flood_inline", "\u{b}"),
    ] {
        for bn in ["chars", "words", "cl100k_base"] {
            let b = be(bn);
            let c = cfg(512, 0, Normalization::Nfc, true, true);
            let small = format!("a{}b", unit.repeat(16 * 1024));
            let big = format!("a{}b", unit.repeat(64 * 1024));
            let t_small = time_min(&small, &c, &b, 2);
            let t_big = time_min(&big, &c, &b, 1);
            let ratio = t_big / t_small.max(0.001);
            rep.hard("whitespace_flood_scaling_not_superlinear", ratio <= 8.0 || t_big < 25.0, &|| {
                format!("{name}/{bn}: {t_small:.1} ms (16 KiB) -> {t_big:.1} ms (64 KiB), ratio {ratio:.1} (linear = 4)")
            });
            rep.info(format!(
                "PERF_WS\t{name}\t{bn}\t{t_small:.2}\t{t_big:.2}\t{ratio:.2}"
            ));
        }
    }
    // many small calls: fixed per-call overhead
    let b = be("cl100k_base");
    let small_doc = prose_from(&mut Rng::new(7), &WORDS[..14], 1);
    let c = ChunkConfig::default();
    let calls = if quick { 2000 } else { 10_000 };
    let t = Instant::now();
    for _ in 0..calls {
        let r = smart_chunk_mapped(&small_doc, &c, &b);
        std::hint::black_box(&r);
    }
    let per_call_us = ms(t.elapsed()) * 1000.0 / calls as f64;
    rep.info(format!(
        "PERF_SMALL_CALLS\t{calls} calls of {} bytes\t{per_call_us:.1} us/call",
        small_doc.len()
    ));
    rep.soft(
        "small_text_call_overhead_under_2ms",
        per_call_us < 2000.0,
        &|| format!("{per_call_us:.1} us/call"),
    );

    if let Ok(path) = std::env::var("RS_PERF_SAVE") {
        match std::fs::write(&path, save) {
            Ok(_) => rep.info(format!("E: wrote performance baseline to {path}")),
            Err(e) => rep.info(format!("E: could not write baseline {path}: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

#[test]
// Diagnostic (cargo feature `diagnostics`): full release validation: cargo test --release release_suite --features diagnostics -- --nocapture --test-threads=1
#[cfg(feature = "diagnostics")]
fn release_suite() {
    let quick = std::env::var("RS_QUICK").is_ok();
    let only = std::env::var("RS_ONLY").unwrap_or_default();
    let want = |s: &str| only.is_empty() || only.contains(s);
    let mut rep = Report::default();
    let t_all = Instant::now();
    let mut timings: Vec<(String, f64)> = Vec::new();

    if want("A") {
        let t = Instant::now();
        section_unicode(&mut rep, quick);
        section_exact_fit(&mut rep);
        timings.push((
            "A_unicode_and_limits".to_string(),
            t.elapsed().as_secs_f64(),
        ));
    }
    if want("B") {
        let t = Instant::now();
        section_core_structure(&mut rep, quick);
        section_heading_context(&mut rep, quick);
        section_header_cases(&mut rep);
        section_orphan_markers(&mut rep);
        timings.push((
            "B_markdown_structure".to_string(),
            t.elapsed().as_secs_f64(),
        ));
    }
    if want("C") {
        let t = Instant::now();
        section_fuzz(&mut rep, quick);
        section_pathological(&mut rep, quick);
        timings.push(("C_robustness_fuzz".to_string(), t.elapsed().as_secs_f64()));
    }
    if want("D") {
        let t = Instant::now();
        section_oracle(&mut rep, quick);
        timings.push(("D_oracle_pulldown".to_string(), t.elapsed().as_secs_f64()));
    }
    if want("F") {
        let t = Instant::now();
        section_scripts(&mut rep, quick);
        timings.push(("F_writing_systems".to_string(), t.elapsed().as_secs_f64()));
    }
    if want("E") {
        let t = Instant::now();
        section_perf(&mut rep, quick);
        timings.push(("E_performance".to_string(), t.elapsed().as_secs_f64()));
    }

    println!(
        "\nRS_BEGIN\tversion=1\tquick={quick}\tbuild={}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    for (name, secs) in &timings {
        println!("RS_SECTION\t{name}\t{secs:.1}s");
    }
    for (rule, st) in &rep.rules {
        println!(
            "RS_RULE\t{}\t{}\tchecks={}\tfails={}",
            if st.hard { "HARD" } else { "SOFT" },
            rule,
            st.checks,
            st.fails
        );
    }
    for (rule, st) in &rep.rules {
        for sample in &st.samples {
            println!(
                "RS_SAMPLE\t{}\t{}\t{}",
                if st.hard { "HARD" } else { "SOFT" },
                rule,
                sample
            );
        }
    }
    for line in &rep.info {
        println!("RS_INFO\t{line}");
    }
    let hard_rules_failed: Vec<&String> = rep
        .rules
        .iter()
        .filter(|(_, s)| s.hard && s.fails > 0)
        .map(|(k, _)| k)
        .collect();
    let soft_rules_failed = rep
        .rules
        .iter()
        .filter(|(_, s)| !s.hard && s.fails > 0)
        .count();
    let total_checks: usize = rep.rules.values().map(|s| s.checks).sum();
    println!(
        "RS_TOTAL\tchecks={total_checks}\thard_rules_failed={}\tsoft_rules_with_findings={soft_rules_failed}\telapsed={:.1}s",
        hard_rules_failed.len(),
        t_all.elapsed().as_secs_f64()
    );
    assert!(
        hard_rules_failed.is_empty(),
        "HARD rules failed: {:?} (see RS_SAMPLE lines above)",
        hard_rules_failed
    );
}
