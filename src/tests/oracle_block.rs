//! Block-boundary agreement with `pulldown-cmark` over the official CommonMark corpus.
//!
//! This is a *measurement*, not a conformance proof: it asks "does our block parser start a
//! block on the same line where the reference starts a top-level block?" and reports the
//! share per construct, plus a breakdown of WHY the two disagree.
//!
//! Run (after `python scripts/fetch_commonmark.py`):
//!
//!     cargo test commonmark_block_boundary --features diagnostics -- --nocapture
//!
//! Output lines are prefixed:
//!   BLOCK_BOUNDARY  summary numbers (publishable)
//!   SPURIOUS        our block starts that are not a reference top-level block start,
//!                   grouped by "where the reference thinks that line is | shape of the line"
//!   MISSED          reference block starts we do not start, grouped by kind and context
//!
//! NOTE: written without being able to compile it. If `BlockParser::new()` or the `Block`
//! fields differ from what the crate exports, adjust `our_block_start_lines` only.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use pulldown_cmark::{Event, Options, Parser};
use serde_json::Value;

use crate::{Block, BlockParser};

/// Ratchet this up to the measured value so a later parser change cannot silently lower the
/// number you publish in the README.
const MIN_EXACT_MATCH_PERCENT: f64 = 76.0;

/// Zero-based line of a byte offset (byte-wise, so any offset is safe).
fn line_of(src: &str, byte: usize) -> usize {
    let end = byte.min(src.len());
    src.as_bytes()[..end].iter().filter(|&&b| b == b'\n').count()
}

struct RefBlock {
    start: usize,
    end: usize,
    kind: String,
}

/// Every top-level block of the reference parse, headings included (headings are only used
/// to locate lines; they are not compared, because BlockParser consumes ATX/setext headings
/// as section context instead of emitting blocks).
fn oracle_blocks(src: &str) -> Vec<RefBlock> {
    let mut depth = 0usize;
    let mut out = Vec::new();
    for (event, range) in Parser::new_ext(src, Options::empty()).into_offset_iter() {
        let start = line_of(src, range.start);
        let end = line_of(src, range.end.saturating_sub(1).max(range.start));
        match event {
            Event::Start(tag) => {
                if depth == 0 {
                    // Debug keeps this independent of pulldown-cmark's Tag variant shapes.
                    let dbg = format!("{tag:?}");
                    let kind: String = dbg.chars().take_while(|c| c.is_alphanumeric()).collect();
                    out.push(RefBlock { start, end, kind });
                }
                depth += 1;
            }
            Event::End(_) => depth = depth.saturating_sub(1),
            Event::Rule if depth == 0 => out.push(RefBlock {
                start,
                end,
                kind: "Rule".to_string(),
            }),
            _ => {}
        }
    }
    out
}

fn our_block_start_lines(src: &str) -> BTreeSet<usize> {
    let mut parser = BlockParser::new();
    let mut blocks: Vec<Block> = Vec::new();
    for line in src.split_inclusive('\n') {
        parser.push_line(line, &mut blocks).unwrap();
    }
    parser.finish(&mut blocks);
    blocks.iter().map(|b| line_of(src, b.start_byte)).collect()
}

fn is_rule(t: &str) -> bool {
    let mut chars = t.chars().filter(|&c| c != ' ' && c != '\t');
    let Some(m) = chars.next() else { return false };
    matches!(m, '-' | '_' | '*')
        && chars.all(|c| c == m)
        && t.chars().filter(|&c| c == m).count() >= 3
}

fn is_list_marker(t: &str) -> bool {
    let b = t.as_bytes();
    match b.first() {
        Some(b'-') | Some(b'+') | Some(b'*') => b.get(1).is_none_or(|&c| c == b' ' || c == b'\t'),
        Some(c) if c.is_ascii_digit() => {
            let digits = b.iter().take_while(|c| c.is_ascii_digit()).count();
            digits <= 9
                && matches!(b.get(digits), Some(b'.') | Some(b')'))
                && b.get(digits + 1).is_none_or(|&c| c == b' ' || c == b'\t')
        }
        _ => false,
    }
}

/// Coarse shape of a source line, to see what kind of line we wrongly start a block on.
fn shape(line: &str) -> &'static str {
    let t = line.trim_start_matches(' ');
    let indent = line.len() - t.len();
    if line.starts_with('\t') || indent >= 4 {
        "indented (>=4 cols)"
    } else if t.trim().is_empty() {
        "blank"
    } else if is_rule(t) {
        "thematic break"
    } else if t.starts_with('>') {
        "block quote"
    } else if is_list_marker(t) {
        "list item"
    } else if t.starts_with('<') {
        "html-like"
    } else if t.starts_with('[') && t.contains("]:") {
        "link reference definition"
    } else if t.starts_with('#') {
        "heading-like"
    } else {
        "other text"
    }
}

#[derive(Default)]
struct Tally {
    total: usize,
    hit: usize,
}

type Groups = BTreeMap<String, (usize, Vec<String>)>;

fn add(g: &mut Groups, key: String, example: String) {
    let e = g.entry(key).or_default();
    e.0 += 1;
    if e.1.len() < 2 {
        e.1.push(example);
    }
}

fn print_groups(prefix: &str, g: &Groups) {
    let mut v: Vec<_> = g.iter().collect();
    v.sort_by(|a, b| (b.1).0.cmp(&(a.1).0).then_with(|| a.0.cmp(b.0)));
    for (key, (n, examples)) in v {
        println!("{prefix} {n:>4}  {key}");
        for e in examples {
            println!("{prefix}          e.g. {e}");
        }
    }
}

fn snippet(line: &str) -> String {
    let s: String = line.chars().take(40).collect();
    format!("{s:?}")
}

#[test]
// Diagnostic (cargo feature `diagnostics`): needs `python scripts/fetch_commonmark.py`; run with `cargo test --features diagnostics <name> -- --nocapture`
#[cfg(feature = "diagnostics")]
fn commonmark_block_boundary_compliance_report() {
    let raw = fs::read_to_string("tests/fixtures/commonmark_spec.json")
        .expect("fetch the CommonMark corpus first");
    let examples: Value = serde_json::from_str(&raw).expect("valid CommonMark JSON");
    let examples = examples.as_array().expect("array");

    let mut by_kind: BTreeMap<String, Tally> = BTreeMap::new();
    let mut spurious: Groups = BTreeMap::new();
    let mut missed: Groups = BTreeMap::new();
    let (mut n, mut exact) = (0usize, 0usize);
    let (mut ours_total, mut ours_spurious) = (0usize, 0usize);
    // Link reference definitions (and other text the reference consumes without emitting a
    // block) are content for chunking, so they are counted separately and reported both ways.
    let (mut spurious_defs, mut exact_ignoring_defs) = (0usize, 0usize);

    for ex in examples {
        let src = ex["markdown"].as_str().unwrap_or("");
        let id = ex["example"].as_u64().unwrap_or(0);
        let lines: Vec<&str> = src.split('\n').collect();
        let refs = oracle_blocks(src);
        let comparable: Vec<&RefBlock> = refs.iter().filter(|b| b.kind != "Heading").collect();
        let theirs_lines: BTreeSet<usize> = comparable.iter().map(|b| b.start).collect();
        let ours = our_block_start_lines(src);

        let mut ok = true;
        let mut non_def_issue = false;

        // Reference block starts we do not start (recall).
        for b in &comparable {
            let t = by_kind.entry(b.kind.clone()).or_default();
            t.total += 1;
            if ours.contains(&b.start) {
                t.hit += 1;
                continue;
            }
            ok = false;
            non_def_issue = true;
            let prev = if b.start == 0 {
                "first line of document"
            } else if lines.get(b.start - 1).is_none_or(|l| l.trim().is_empty()) {
                "after a blank line"
            } else {
                "directly after a non-blank line (interrupts a paragraph)"
            };
            let line = lines.get(b.start).copied().unwrap_or("");
            let detail = if b.kind == "Rule" {
                if line.trim_start().starts_with('-') {
                    " [dashes: setext-eligible]"
                } else {
                    " [stars/underscores]"
                }
            } else {
                ""
            };
            add(
                &mut missed,
                format!("{}: {prev}{detail}", b.kind),
                format!("#{id} line {} {}", b.start, snippet(line)),
            );
        }

        // Our block starts the reference does not have (precision).
        for &l in &ours {
            ours_total += 1;
            if theirs_lines.contains(&l) {
                continue;
            }
            ours_spurious += 1;
            ok = false;
            let line = lines.get(l).copied().unwrap_or("");
            let place_outside = !refs.iter().any(|b| b.start <= l && l <= b.end);
            if place_outside || shape(line) == "link reference definition" {
                spurious_defs += 1;
            } else {
                non_def_issue = true;
            }
            let place = match refs.iter().find(|b| b.start <= l && l <= b.end) {
                Some(b) if b.start == l => format!("at start of reference {}", b.kind),
                Some(b) => format!("inside reference {}", b.kind),
                None => "outside every reference block".to_string(),
            };
            add(
                &mut spurious,
                format!("{place} | {}", shape(line)),
                format!("#{id} line {l} {}", snippet(line)),
            );
        }

        n += 1;
        if ok {
            exact += 1;
        }
        if !non_def_issue {
            exact_ignoring_defs += 1;
        }
    }

    let pct = 100.0 * exact as f64 / n as f64;
    println!("BLOCK_BOUNDARY examples={n} exact_match={exact} ({pct:.1} %)");
    println!(
        "BLOCK_BOUNDARY spurious_starts={ours_spurious} of {ours_total} ({:.1} %)",
        100.0 * ours_spurious as f64 / ours_total.max(1) as f64
    );
    println!(
        "BLOCK_BOUNDARY ignoring link reference definitions: exact_match={exact_ignoring_defs} ({:.1} %), spurious_starts={} of {} ({:.1} %)",
        100.0 * exact_ignoring_defs as f64 / n as f64,
        ours_spurious - spurious_defs,
        ours_total - spurious_defs,
        100.0 * (ours_spurious - spurious_defs) as f64 / (ours_total - spurious_defs).max(1) as f64
    );
    for (kind, t) in &by_kind {
        println!(
            "BLOCK_BOUNDARY kind={kind:<12} recall={}/{} ({:.1} %)",
            t.hit,
            t.total,
            100.0 * t.hit as f64 / t.total.max(1) as f64
        );
    }
    println!("--- our block starts without a reference counterpart ---");
    print_groups("SPURIOUS", &spurious);
    println!("--- reference block starts we do not start ---");
    print_groups("MISSED", &missed);

    assert!(
        pct >= MIN_EXACT_MATCH_PERCENT,
        "block-boundary agreement fell to {pct:.1} %"
    );
}
