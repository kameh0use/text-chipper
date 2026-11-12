//! Differential tests of the list scanner against `pulldown-cmark`, a CommonMark reference.
//!
//! * `list_scanner_agrees_with_reference_on_core_cases` asserts agreement on constructs the
//!   scanner is meant to handle.
//! * `oracle_known_deviations_report` and `oracle_random_report` are diagnostics: they only
//!   print. Run them with
//!   `cargo test oracle_ --features diagnostics -- --nocapture`
//!   to see where the scanner disagrees with CommonMark and decide what is worth fixing.
use pulldown_cmark::{Event, Options, Parser};

use crate::markdown::list_parser::parse_list_items;

fn line_of(src: &str, byte: usize) -> usize {
    src[..byte].bytes().filter(|&b| b == b'\n').count()
}

/// Zero-based start line of every list item according to the reference parser.
fn oracle_item_lines(src: &str) -> Vec<usize> {
    let mut lines = Vec::new();
    for (event, range) in Parser::new_ext(src, Options::empty()).into_offset_iter() {
        if let Event::Start(tag) = event {
            // `Debug` keeps this independent of pulldown-cmark's Tag variant shapes.
            if format!("{tag:?}").starts_with("Item") {
                lines.push(line_of(src, range.start));
            }
        }
    }
    lines
}

fn our_item_lines(src: &str) -> Vec<usize> {
    parse_list_items(src).iter().map(|i| i.start_line).collect()
}

const AGREED: &[(&str, &str)] = &[
    ("flat bullets", "- a\n- b\n- c\n"),
    ("star bullets nested", "* a\n* b\n  * c\n"),
    ("ordered", "1. one\n2. two\n"),
    ("paren ordered", "10) ten\n11) eleven\n"),
    ("nested three deep", "- a\n  - b\n    - c\n- d\n"),
    ("lazy continuation", "- a\n  lazy\ncontinuation\n- b\n"),
    ("loose item", "- a\n\n  para\n\n- b\n"),
    (
        "fence inside item",
        "- item\n  ```\n  - not\n  ```\n- next\n",
    ),
    (
        "fence inside ordered item",
        "1. a\n\n   ```\n   - x\n   ```\n",
    ),
    ("mixed markers", "* x\n+ y\n"),
    ("list interrupts paragraph", "text\n- a\n- b\n"),
    ("wide marker, indented lookalike", "100. x\n    - y\n"),
];

#[test]
fn list_scanner_agrees_with_reference_on_core_cases() {
    for (name, src) in AGREED {
        assert_eq!(
            our_item_lines(src),
            oracle_item_lines(src),
            "case `{name}`: {src:?}"
        );
    }
}

const KNOWN_RISKS: &[(&str, &str)] = &[
    (
        "ordered not starting at 1 cannot interrupt a paragraph",
        "text\n2. b\n",
    ),
    ("empty items", "-\n- b\n"),
    ("thematic break then list", "* * *\n- a\n"),
    ("tab after marker and tab indent", "-\ta\n\t- b\n"),
    ("list inside block quote", "> - a\n> - b\n"),
    ("html block hides list lookalikes", "<div>\n- a\n</div>\n"),
    ("indented code", "    - not a list\n"),
    ("setext heading", "Foo\n---\n- a\n"),
];

#[test]
// Diagnostic (cargo feature `diagnostics`): diagnostic report: cargo test oracle_known --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn oracle_known_deviations_report() {
    println!("{:<58} {:<14} {:<14} status", "case", "ours", "reference");
    for (name, src) in KNOWN_RISKS {
        let (ours, theirs) = (our_item_lines(src), oracle_item_lines(src));
        let status = if ours == theirs { "agree" } else { "DIFFERS" };
        println!("{name:<58} {ours:<14?} {theirs:<14?} {status}");
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.next() % xs.len()]
    }
}

#[test]
// Diagnostic (cargo feature `diagnostics`): diagnostic report: cargo test oracle_random --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn oracle_random_report() {
    const LINES: &[&str] = &[
        "- a",
        "* b",
        "+ c",
        "1. d",
        "2) e",
        "  - nested",
        "    - deeper",
        "   1. o",
        "  continuation",
        "lazy text",
        "",
        "",
        "```",
        "  ```",
        "> quote",
        "para",
        "\t- tab",
        "10. x",
        "-",
        "---",
        "    code",
    ];
    let mut rng = Lcg(0x5EED);
    let (total, mut differing) = (3000usize, 0usize);
    let mut shown = 0;
    for _ in 0..total {
        let n = 3 + rng.next() % 8;
        let mut doc = String::new();
        for _ in 0..n {
            doc.push_str(rng.pick(LINES));
            doc.push('\n');
        }
        let (ours, theirs) = (our_item_lines(&doc), oracle_item_lines(&doc));
        if ours != theirs {
            differing += 1;
            if shown < 8 && doc.len() < 60 {
                shown += 1;
                println!("{doc:?}\n    ours {ours:?}  reference {theirs:?}");
            }
        }
    }
    println!(
        "{differing} of {total} random documents disagree ({:.1} %)",
        100.0 * differing as f64 / total as f64
    );
}
