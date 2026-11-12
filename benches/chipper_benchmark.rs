//! Criterion benchmark for `text_chipper`, run with `cargo bench`.
//!
//! Uses only the public API. For cross-library comparisons see `benchmarks/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use text_chipper::{smart_chunk, Backend, ChunkConfig};

/// Deterministic Markdown with headings, prose, a list and a code fence (~1 KiB per repeat).
fn sample_markdown(repeats: usize) -> String {
    let block = "# Section heading\n\n\
        The quick brown fox jumps over the lazy dog, and then it keeps going so that every \
        paragraph is long enough to span a few sentences. Retrieval pipelines split documents \
        like this one into token-bounded chunks.\n\n\
        - first list item with some text\n\
        - second list item with some more text\n\n\
        ```python\ndef hello():\n    print(\"world\")\n```\n\n";
    block.repeat(repeats)
}

fn bench_chunking(c: &mut Criterion) {
    let backend = Backend::from_name("cl100k_base").expect("tokenizer should load");
    let text = sample_markdown(2_000);

    let mut group = c.benchmark_group("smart_chunk");
    group.throughput(Throughput::Bytes(text.len() as u64));
    for &max_tokens in &[128usize, 512, 1024] {
        let cfg = ChunkConfig {
            max_tokens,
            overlap_tokens: 0,
            ..ChunkConfig::default()
        };
        group.bench_with_input(BenchmarkId::new("max_tokens", max_tokens), &cfg, |b, cfg| {
            b.iter(|| smart_chunk(black_box(&text), cfg, &backend).expect("chunking failed"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chunking);
criterion_main!(benches);
