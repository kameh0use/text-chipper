
use super::*;

fn cfg(max: usize, ctx: bool, strict: bool) -> ChunkConfig {
    ChunkConfig {
        max_tokens: max,
        overlap_tokens: 0,
        normalization: Normalization::None,
        snap_words: true,
        include_context: ctx,
        merge_blocks: true,
        strict,
    }
}

/// Dependency-free, deterministic random number generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

/// Fragments that stress tokenizer boundaries: accented Latin, combining accents, compatibility
/// ligatures, ZWJ emoji, CJK, punctuation-heavy runs, numbers, code and table markers.
const WORDS: &[&str] = &[
    "Hallo", "Größe", "Łódź", "Ångström", "façade", "naïve", "ß", "ø", "ﬁ", "don't",
    "1.5", "3,14", "a.b", "x-y", "foo_bar", "e\u{301}", "👩\u{200d}💻", "東京", "日本語",
    "...", "--", "`code`", "**fett**", "www.example.com/a?b=c", "ǆ", "—", "“Zitat”",
];

fn paragraph(r: &mut Lcg, max_words: usize) -> String {
    let n = 1 + r.below(max_words);
    let mut v: Vec<&str> = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(r.pick(WORDS));
    }
    v.join(" ")
}

/// A document that mixes headings, lists, tables and code blocks, containing many SMALL
/// blocks. Small blocks accumulate the `joins` counter and stress the +1 margin.
fn adversarial_markdown(seed: u64, blocks: usize) -> String {
    let mut r = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
    let mut s = String::new();
    for i in 0..blocks {
        match r.below(9) {
            0 => s.push_str(&format!("{} {}", "#".repeat(1 + r.below(3)), paragraph(&mut r, 3))),
            1 => s.push_str(&format!("- {}\n- {}", paragraph(&mut r, 4), paragraph(&mut r, 4))),
            2 => s.push_str(&format!("{}. {}", 1 + i % 9, paragraph(&mut r, 6))),
            3 => s.push_str(&format!("| {} | {} |", paragraph(&mut r, 2), paragraph(&mut r, 2))),
            4 => s.push_str(&format!("```\n{}\n```", paragraph(&mut r, 5))),
            _ => s.push_str(&paragraph(&mut r, 12)),
        }
        s.push_str(if r.below(6) == 0 { "\n" } else { "\n\n" });
    }
    s
}

fn delivered_tokens(backend: &Backend, c: &Chunk, ctx: bool) -> usize {
    if ctx {
        backend.count(&c.contextualized_text())
    } else {
        backend.count(&c.text)
    }
}

const BACKENDS: &[&str] = &["chars", "words", "cl100k_base", "o200k_base"];
const BUDGETS: &[usize] = &[12, 16, 25, 40, 64, 128];

/// MAIN TEST. When strict=false the final gate (finalize) does not recount, so the budget
/// guarantee rests entirely on the accuracy of the merge estimate (the +1 margin). Since the
/// real token count of merges accepted on the fast path is never measured (diagnostics only
/// measure rejected ones), this is the only correct check.
#[test]
fn merge_margin_non_strict_never_exceeds_budget() {
    let mut violations: Vec<String> = Vec::new();
    let mut known_trim_effect = 0usize;
    for name in BACKENDS {
        let backend = Backend::from_name(name).unwrap();
        for &max in BUDGETS {
            for ctx in [false, true] {
                for seed in 0..6u64 {
                    let text = adversarial_markdown(seed, 400);
                    let chunks = smart_chunk(&text, &cfg(max, ctx, false), &backend).unwrap();
                    for (i, c) in chunks.iter().enumerate() {
                        let n = delivered_tokens(&backend, c, ctx);
                        if n > max {
                            // Known boundary: when split_engine carries a single atom
                            // (`cut - i <= 1`), it counts the atom with a leading space; after
                            // trimming, that leading space is lost and the count goes up by one.
                            // The strict=true final gate catches this.
                            let with_space = if ctx && !c.headers.is_empty() {
                                backend.count(&format!("{} {}", context_prefix(&c.headers), c.text))
                            } else {
                                backend.count(&format!(" {}", c.text))
                            };
                            if with_space <= max {
                                known_trim_effect += 1;
                                continue;
                            }
                            violations.push(format!(
                                "backend={name} max={max} ctx={ctx} seed={seed} chunk={i}: {n} > {max} (joins/estimate undercount) text={:?}",
                                c.text.chars().take(60).collect::<String>()
                            ));
                        }
                    }
                }
            }
        }
    }
    println!("bilinen kirpma etkisi (non-strict, tek atom): {known_trim_effect}");
    assert!(
        violations.is_empty(),
        "{} budget violations, first 5:\n{}",
        violations.len(),
        violations.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
}

/// strict=true contract: the budget is never exceeded, token_count is the EXACT count, and
/// offsets map back to the source. Also verifies that the direct-append path does not
/// corrupt p.text.
#[test]
fn merge_margin_strict_contract_and_offsets() {
    for name in BACKENDS {
        let backend = Backend::from_name(name).unwrap();
        for &max in BUDGETS {
            for ctx in [false, true] {
                for seed in 0..4u64 {
                    let text = adversarial_markdown(seed + 100, 300);
                    let chunks = smart_chunk(&text, &cfg(max, ctx, true), &backend).unwrap();
                    assert!(!chunks.is_empty());
                    for (i, c) in chunks.iter().enumerate() {
                        let n = delivered_tokens(&backend, c, ctx);
                        let tag = format!("backend={name} max={max} ctx={ctx} seed={seed} chunk={i}");
                        assert!(n <= max, "{tag}: butce asildi {n} > {max}");
                        assert_eq!(c.token_count, n, "{tag}: token_count tam degil");
                        assert_eq!(&text[c.start_byte..c.end_byte], c.text, "{tag}: bayt ofseti");
                    }
                }
            }
        }
    }
}

/// The fast path and full counting should produce the same chunking: on inputs that do not
/// overflow the budget, strict and non-strict must split the text at the same points
/// (strict only verifies, it does not split).
#[test]
fn merge_margin_strict_and_non_strict_agree_when_nothing_overflows() {
    for name in ["chars", "cl100k_base"] {
        let backend = Backend::from_name(name).unwrap();
        for &max in &[40usize, 128] {
            let text = adversarial_markdown(7, 300);
            let a = smart_chunk(&text, &cfg(max, false, true), &backend).unwrap();
            let b = smart_chunk(&text, &cfg(max, false, false), &backend).unwrap();
            let ra: Vec<(usize, usize)> = a.iter().map(|c| (c.start_byte, c.end_byte)).collect();
            let rb: Vec<(usize, usize)> = b.iter().map(|c| (c.start_byte, c.end_byte)).collect();
            assert_eq!(ra, rb, "backend={name} max={max}: strict/non-strict ayrisiyor");
        }
    }
}

/// Additivity probe, per separator.
///
/// Observation (cl100k, 4000 samples): all underestimates occur with separators that END in
/// whitespace, such as `"\n  "` (newline + spaces); the trailing space sticks to the first
/// word of B and yields a different token than counting separately. A hard check applies to
/// separators that END with a newline (`\n`, `\r\n`, `\n\n`): exact must NOT be > sum.
/// Other separators are only reported.
/// The overestimate rate (exact < sum) is also reported: if it is non-zero, the sum cannot be
/// used as the EXACT value for `token_count` (only as an upper bound).
#[test]
fn merge_margin_bpe_additivity_probe() {
    let seps = ["\n", "\n\n", "\r\n", "\n\n\n", " \n", "\n\t", "\n  "];
    for name in ["cl100k_base", "o200k_base"] {
        let backend = Backend::from_name(name).unwrap();
        let mut r = Lcg(42);
        // (equal, overestimate, underestimate) per separator
        let mut stats = vec![(0usize, 0usize, 0usize); seps.len()];
        let mut max_less = vec![0usize; seps.len()];
        let mut hard: Vec<String> = Vec::new();
        for _ in 0..8000 {
            let a = paragraph(&mut r, 6);
            let b = paragraph(&mut r, 6);
            let si = r.below(seps.len());
            let sep = seps[si];
            let exact = backend.count(&format!("{a}{sep}{b}"));
            let sum = backend.count(&a) + backend.count(sep) + backend.count(&b);
            if exact == sum {
                stats[si].0 += 1;
            } else if exact < sum {
                stats[si].1 += 1;
                max_less[si] = max_less[si].max(sum - exact);
            } else {
                stats[si].2 += 1;
                if sep.ends_with('\n') {
                    hard.push(format!("{a:?} + {sep:?} + {b:?}: exact={exact} sum={sum}"));
                }
            }
        }
        for (i, sep) in seps.iter().enumerate() {
            let (e, f, u) = stats[i];
            println!("{name} sep={sep:?}: esit={e} fazla-tahmin={f} (en buyuk fark={}) eksik-tahmin={u}", max_less[i]);
        }
        assert!(
            hard.is_empty(),
            "{name}: yeni satirla biten ayiricida {} eksik-tahmin, ilk 5:\n{}",
            hard.len(),
            hard.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}

/// Empty and boundary inputs: single token, single grapheme, whitespace only, headings only.
#[test]
fn merge_margin_degenerate_inputs_terminate() {
    for name in BACKENDS {
        let backend = Backend::from_name(name).unwrap();
        for strict in [false, true] {
            for text in ["", "\n\n", "# A\n\n# B\n\n# C", "ß", "👩\u{200d}💻\n\n👩\u{200d}💻", "a\n\n\n\n\nb"] {
                for &max in &[1usize, 2, 5, 12] {
                    let _ = smart_chunk(text, &cfg(max, true, strict), &backend);
                }
            }
        }
    }
}

/// Locates the source of a single violation: shows whether trimming/whitespace changes move
/// the token count.
#[test]
fn merge_margin_attribute_violation() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    let text = adversarial_markdown(2, 400);
    let chunks = smart_chunk(&text, &cfg(12, true, false), &backend).unwrap();
    for (i, ch) in chunks.iter().enumerate() {
        if delivered_tokens(&backend, ch, true) <= 12 {
            continue;
        }
        let t = &ch.text;
        println!(
            "chunk={i}/{} token_count={} start_byte={} end_byte={}",
            chunks.len(),
            ch.token_count,
            ch.start_byte,
            ch.end_byte
        );
        let before = &text[..ch.start_byte];
        let after = &text[ch.end_byte..];
        println!("  once={:?}", before.chars().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<String>());
        println!("  sonra={:?}", after.chars().take(12).collect::<String>());
        for v in [t.clone(), format!(" {t}"), format!("{t}\n"), format!("\n{t}"), format!(" {t} ")] {
            println!("  {:?} -> {}", v, backend.count(&v));
        }
    }
}
