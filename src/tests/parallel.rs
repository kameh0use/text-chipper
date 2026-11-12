//! Differential tests for the intra-document parallel path: for every generated document,
//! configuration and segment count the parallel result must equal the sequential result
//! exactly (chunks, original-text spans, normalization segment count, and errors).

use crate::{
    backend::Backend,
    chunk_types::{Chunk, ChunkConfig},
    chunker::{
        parallel::{
            chunk_context_parallel, has_dense_tiny_paragraph_prefix, plan, ParallelOpts,
            DEFAULT_MIN_BYTES,
        },
        smart_chunk_mapped_profile_with,
    },
    normalization::Normalization,
};

// ---- deterministic corpus -----------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
}

const WORDS: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "retrieval",
    "chunk",
    "token",
    "processing",
    "documents",
    "header",
    "context",
    "x",
    "an",
    "the",
    "of",
    "a",
    "T\u{fc}rk\u{e7}e",
    "\u{130}stanbul",
    "na\u{ef}ve",
    "cafe\u{301}",
    "\u{6771}\u{4eac}",
    "\u{65e5}\u{672c}\u{8a9e}",
    "\u{d55c}\u{ad6d}\u{c5b4}",
    "\u{1f642}",
    "\u{1f469}\u{1f3fd}\u{200d}\u{1f4bb}",
    "\u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629}",
    "\u{5e9}\u{5dc}\u{5d5}\u{5dd}",
];

fn words(r: &mut Rng, n: usize) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(n);
    for _ in 0..n {
        let i = r.below(WORDS.len());
        parts.push(WORDS[i]);
    }
    parts.join(" ")
}

fn gen_doc(seed: u64, target_bytes: usize) -> String {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut s = String::new();
    if r.below(8) == 0 {
        s.push_str("---\ntitle: test\ntags: [a, b]\n---\n\n");
    }
    while s.len() < target_bytes {
        match r.below(14) {
            0 | 1 => {
                let level = 1 + r.below(4);
                let n = 1 + r.below(5);
                s.push_str(&"#".repeat(level));
                s.push(' ');
                let w = words(&mut r, n);
                s.push_str(&w);
                s.push_str("\n\n");
            }
            2..=5 => {
                let n = 3 + r.below(120);
                let w = words(&mut r, n);
                s.push_str(&w);
                s.push_str("\n\n");
            }
            6 | 7 => {
                let items = 2 + r.below(8);
                for i in 0..items {
                    let n = 2 + r.below(25);
                    let w = words(&mut r, n);
                    s.push_str(if r.below(2) == 0 { "- " } else { "* " });
                    let _ = i;
                    s.push_str(&w);
                    s.push('\n');
                    if r.below(4) == 0 {
                        let n2 = 2 + r.below(30);
                        let w2 = words(&mut r, n2);
                        s.push_str("\n  ");
                        s.push_str(&w2);
                        s.push('\n');
                    }
                }
                s.push('\n');
            }
            8 => {
                let items = 2 + r.below(6);
                for i in 0..items {
                    let n = 2 + r.below(20);
                    let w = words(&mut r, n);
                    s.push_str(&format!("{}. {}\n", i + 1, w));
                }
                s.push('\n');
            }
            9 => {
                s.push_str("```rust\n");
                let lines = 1 + r.below(30);
                for _ in 0..lines {
                    if r.below(6) == 0 {
                        s.push('\n');
                    } else {
                        let n = 1 + r.below(10);
                        let w = words(&mut r, n);
                        s.push_str("let v = \"");
                        s.push_str(&w);
                        s.push_str("\";\n");
                    }
                }
                s.push_str("```\n\n");
            }
            10 => {
                s.push_str("| a | b |\n|---|---|\n");
                let rows = 1 + r.below(8);
                for _ in 0..rows {
                    let w1 = words(&mut r, 2);
                    let w2 = words(&mut r, 3);
                    s.push_str(&format!("| {w1} | {w2} |\n"));
                }
                s.push('\n');
            }
            11 => {
                let lines = 1 + r.below(4);
                for _ in 0..lines {
                    let n = 3 + r.below(20);
                    let w = words(&mut r, n);
                    s.push_str("> ");
                    s.push_str(&w);
                    s.push('\n');
                }
                s.push('\n');
            }
            12 => {
                // Oversized paragraph: forces split_block / finalize splitting.
                let n = 400 + r.below(1200);
                let w = words(&mut r, n);
                s.push_str(&w);
                s.push_str("\n\n");
            }
            _ => {
                let n = 1 + r.below(6);
                let w = words(&mut r, n);
                s.push_str(&w);
                s.push('\n');
            }
        }
    }
    if r.below(6) == 0 {
        s = s.replace('\n', "\r\n");
    }
    if r.below(5) == 0 {
        while s.ends_with('\n') || s.ends_with('\r') {
            s.pop();
        }
    }
    s
}

fn cfg(max: usize, ctx: bool, merge: bool, strict: bool, norm: Normalization) -> ChunkConfig {
    ChunkConfig {
        max_tokens: max,
        overlap_tokens: if max >= 24 { max / 8 } else { 0 },
        normalization: norm,
        snap_words: true,
        include_context: ctx,
        merge_blocks: merge,
        strict,
    }
}

fn configs() -> Vec<ChunkConfig> {
    let mut v = Vec::new();
    for &max in &[0usize, 2, 6, 24, 128, 512] {
        for &(ctx, merge) in &[(true, true), (true, false), (false, true)] {
            for &strict in &[true, false] {
                for &norm in &[Normalization::None, Normalization::Nfc] {
                    v.push(cfg(max, ctx, merge, strict, norm));
                }
            }
        }
    }
    v
}

// ---- comparison helpers -----------------------------------------------------------------

type Run = Result<(Vec<Chunk>, String), String>;

fn run(text: &str, c: &ChunkConfig, backend: &std::sync::Arc<Backend>, opts: &ParallelOpts) -> Run {
    match smart_chunk_mapped_profile_with(text, c, backend, opts) {
        Ok((chunks, spans, segs)) => Ok((chunks, format!("{spans:?}|{segs}"))),
        Err(e) => Err(format!("{e:?}")),
    }
}

fn assert_same(reference: &Run, got: &Run, label: &str) {
    match (reference, got) {
        (Ok((ca, sa)), Ok((cb, sb))) => {
            assert_eq!(ca.len(), cb.len(), "chunk count differs: {label}");
            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert_eq!(x, y, "chunk {i} differs: {label}");
            }
            assert_eq!(sa, sb, "spans differ: {label}");
        }
        (Err(ea), Err(eb)) => assert_eq!(ea, eb, "errors differ: {label}"),
        _ => panic!(
            "ok/err mismatch: {label}: reference={:?} got={:?}",
            reference.as_ref().map(|r| r.0.len()),
            got.as_ref().map(|r| r.0.len())
        ),
    }
}

// ---- tests -------------------------------------------------------------------------------

#[test]
fn parallel_matches_sequential_on_generated_corpus() {
    let chars = Backend::from_name("chars").unwrap();
    let words_b = Backend::from_name("words").unwrap();
    for seed in 1..=5u64 {
        let doc = gen_doc(seed, 15_000 + (seed as usize) * 6_000);
        for (ci, c) in configs().iter().enumerate() {
            for (bn, backend) in [("chars", &chars), ("words", &words_b)] {
                let reference = run(&doc, c, backend, &ParallelOpts::never());
                for nseg in [2usize, 3, 8] {
                    let got = run(&doc, c, backend, &ParallelOpts::forced(nseg));
                    assert_same(
                        &reference,
                        &got,
                        &format!("seed={seed} cfg#{ci}={c:?} backend={bn} nseg={nseg}"),
                    );
                }
            }
        }
    }
}

#[test]
fn parallel_matches_sequential_with_real_tokenizer() {
    let backend = Backend::from_name("cl100k_base").unwrap();
    let doc = gen_doc(42, 40_000);
    let cases = [
        cfg(64, true, true, true, Normalization::Nfc),
        cfg(512, true, true, true, Normalization::None),
        cfg(128, true, false, true, Normalization::None),
        cfg(128, false, true, true, Normalization::Nfc),
    ];
    for (ci, c) in cases.iter().enumerate() {
        let reference = run(&doc, c, &backend, &ParallelOpts::never());
        for nseg in [2usize, 5] {
            let got = run(&doc, c, &backend, &ParallelOpts::forced(nseg));
            assert_same(
                &reference,
                &got,
                &format!("cl100k cfg#{ci}={c:?} nseg={nseg}"),
            );
        }
    }
}

#[test]
fn realistic_documents_resynchronize_at_seams() {
    let backend = Backend::from_name("words").unwrap();
    let c = cfg(128, true, true, true, Normalization::None);
    let (mut seams, mut resynced, mut restitched) = (0usize, 0usize, 0usize);
    for seed in 100..110u64 {
        let doc = gen_doc(seed, 200_000);
        let (chunks, stats) = chunk_context_parallel(&doc, &c, &backend, 6).unwrap();
        assert!(!chunks.is_empty());
        seams += stats.segments.saturating_sub(1);
        resynced += stats.resynced;
        restitched += stats.restitched_blocks;
    }
    eprintln!("parallel resync: seams={seams} resynced={resynced} restitched_blocks={restitched}");
    assert!(seams > 0);
    // Correctness never depends on this; it only guards against the speed-up silently vanishing.
    assert!(resynced >= 1, "no seam ever resynchronized");
}

#[test]
fn result_is_independent_of_thread_count() {
    let backend = Backend::from_name("words").unwrap();
    let doc = gen_doc(7, 120_000);
    let c = cfg(96, true, true, true, Normalization::Nfc);
    let reference = run(&doc, &c, &backend, &ParallelOpts::never());
    for threads in [1usize, 2, 5] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let got = pool.install(|| run(&doc, &c, &backend, &ParallelOpts::forced(6)));
        assert_same(&reference, &got, &format!("threads={threads}"));
    }
}

#[test]
fn planning_rules() {
    assert_eq!(plan(100, &ParallelOpts::default()), None);
    assert_eq!(plan(10, &ParallelOpts::never()), None);
    assert_eq!(plan(10, &ParallelOpts::forced(4)), Some(4));
    assert_eq!(plan(1, &ParallelOpts::forced(4)), None); // one byte cannot form two segments
}

#[test]
// Diagnostic (cargo feature `diagnostics`): benchmark diagnostic; run with --release --features diagnostics -- --nocapture
#[cfg(feature = "diagnostics")]
fn parallel_speedup_report() {
    use std::time::Instant;
    let backend = Backend::from_name("cl100k_base").unwrap();
    let c = cfg(512, true, true, true, Normalization::Nfc);
    let mut doc = String::new();
    let mut seed = 1000u64;
    while doc.len() < 32 * 1024 * 1024 {
        doc.push_str(&gen_doc(seed, 1024 * 1024));
        seed += 1;
    }
    let auto = ParallelOpts {
        min_bytes: 0,
        ..ParallelOpts::default()
    };
    let t0 = Instant::now();
    let reference = run(&doc, &c, &backend, &ParallelOpts::never());
    let seq = t0.elapsed();
    let t1 = Instant::now();
    let got = run(&doc, &c, &backend, &auto);
    let par = t1.elapsed();
    assert_same(&reference, &got, "speedup report");
    let mib = doc.len() as f64 / (1024.0 * 1024.0);
    println!(
        "PAR_SPEEDUP bytes_mib={mib:.1} threads={} seq_ms={} par_ms={} speedup={:.2}x",
        rayon::current_num_threads(),
        seq.as_millis(),
        par.as_millis(),
        seq.as_secs_f64() / par.as_secs_f64().max(1e-9)
    );
}

#[test]
fn par_map_ordered_returns_first_error_in_input_order() {
    use crate::{errors::ChipError, python::par_map_ordered};
    use std::{thread::sleep, time::Duration};

    let items: Vec<usize> = (0..64).collect();
    for _ in 0..20 {
        let r = par_map_ordered(&items, |&i| {
            // Later failing items finish first; the earliest failing item is the slowest.
            match i {
                3 => {
                    sleep(Duration::from_millis(30));
                    Err(ChipError::ChunkTooLarge {
                        required_tokens: 2,
                        max_tokens: 1,
                        start_byte: 3,
                    })
                }
                10 => {
                    sleep(Duration::from_millis(10));
                    Err(ChipError::ChunkTooLarge {
                        required_tokens: 2,
                        max_tokens: 1,
                        start_byte: 10,
                    })
                }
                50 => Err(ChipError::ChunkTooLarge {
                    required_tokens: 2,
                    max_tokens: 1,
                    start_byte: 50,
                }),
                _ => Ok(i * 2),
            }
        });
        match r {
            Err(ChipError::ChunkTooLarge { start_byte, .. }) => assert_eq!(start_byte, 3),
            other => panic!("unexpected result: {:?}", other.map(|v| v.len())),
        }
    }
    // Success path keeps input order.
    let ok = par_map_ordered(&items, |&i| Ok(i + 1)).unwrap();
    assert_eq!(ok, (1..=64).collect::<Vec<usize>>());
}

#[test]
fn dense_tiny_paragraphs_use_streaming_preflight_before_block_materialization() {
    let backend = Backend::from_name("chars").unwrap();
    let c = cfg(512, true, true, true, Normalization::Nfc);
    // Deliberately cross the automatic parallel threshold. The old post-parse density
    // fallback still allocated a Vec<Block> for every tiny paragraph before falling back,
    // which made the 2 MiB release-suite case dramatically slower than the 512 KiB case.
    let doc = "p q r.\n\n".repeat(262_145);
    assert!(doc.len() > DEFAULT_MIN_BYTES);
    assert!(has_dense_tiny_paragraph_prefix(&doc));

    let reference = run(&doc, &c, &backend, &ParallelOpts::never());
    let got = run(&doc, &c, &backend, &ParallelOpts::default());
    assert_same(
        &reference,
        &got,
        "dense tiny paragraphs above the parallel byte threshold",
    );
}
