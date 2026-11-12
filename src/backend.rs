use std::collections::HashMap;
#[cfg(feature = "stats")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "stats")]
use std::time::Instant;

use tiktoken::{get_encoding, CoreBpe};
use unicode_segmentation::UnicodeSegmentation;

use super::ChipError;

// Tokenizer backends (tiktoken BPE tables, chars, words and, optionally, Hugging Face).

/// Call-site tags used by the optional tokenizer profiler.
/// Keep these split by operation: broad "finalize" / "block_probe" buckets would hide
/// which callers pay for the tokenizer.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum CountSite {
    Other = 0,
    WindowProbe = 1,
    Atomize = 2,
    Settle = 3,
    Candidate = 4,
    Overlap = 5,
    FinalizeInput = 6,
    FinalizeRetry = 7,
    FinalizeHeaderCheck = 8,
    FinalizeSnap = 9,
    MergeCandidate = 10,
    ListItemCheck = 11,
    PendingListSplit = 12,
    BlockWhole = 13,
    BlockPrefix = 14,
    MergeSeparator = 15,
    MergePrefix = 16,
    Context = 17,
    Public = 18,
    // Control-flow events (not tokenizer calls), emitted only by the chunker. The retrospective
    // estimate/margin diagnostics (28..=38) are emitted only in `stats` builds.
    MergeFastDirect = 19,
    MergeFastPrefix = 20,
    MergeExactFallback = 21,
    MergeFastDisabled = 22,
    MergeEstimateRejected = 23,
    MergePrefixCandidate = 24,
    MergeCandidateTooLarge = 25,
    MergeExactAccept = 26,
    MergeExactReject = 27,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateUnder2Plus = 28,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateUnder1 = 29,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateEqual = 30,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateOver1 = 31,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateOver2 = 32,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateOver3 = 33,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeEstimateOver4Plus = 34,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeMargin1WouldAccept = 35,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeMargin1Unsafe = 36,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeMargin0WouldAccept = 37,
    #[cfg_attr(not(feature = "stats"), allow(dead_code))]
    MergeMargin0Unsafe = 38,
    MergeEarlyReject = 39,
}

impl CountSite {
    #[cfg(feature = "stats")]
    pub(crate) const ALL: [(&'static str, CountSite); 40] = [
        ("other", CountSite::Other),
        ("window_probe", CountSite::WindowProbe),
        ("atomize", CountSite::Atomize),
        ("settle", CountSite::Settle),
        ("candidate", CountSite::Candidate),
        ("overlap", CountSite::Overlap),
        ("finalize_input", CountSite::FinalizeInput),
        ("finalize_retry", CountSite::FinalizeRetry),
        ("finalize_header_check", CountSite::FinalizeHeaderCheck),
        ("finalize_snap", CountSite::FinalizeSnap),
        ("merge_candidate", CountSite::MergeCandidate),
        ("list_item_check", CountSite::ListItemCheck),
        ("pending_list_split", CountSite::PendingListSplit),
        ("block_whole", CountSite::BlockWhole),
        ("block_prefix", CountSite::BlockPrefix),
        ("merge_separator", CountSite::MergeSeparator),
        ("merge_prefix", CountSite::MergePrefix),
        ("context", CountSite::Context),
        ("public", CountSite::Public),
        ("merge_fast_direct", CountSite::MergeFastDirect),
        ("merge_fast_prefix", CountSite::MergeFastPrefix),
        ("merge_exact_fallback", CountSite::MergeExactFallback),
        ("merge_fast_disabled", CountSite::MergeFastDisabled),
        ("merge_estimate_rejected", CountSite::MergeEstimateRejected),
        ("merge_prefix_candidate", CountSite::MergePrefixCandidate),
        ("merge_candidate_too_large", CountSite::MergeCandidateTooLarge),
        ("merge_exact_accept", CountSite::MergeExactAccept),
        ("merge_exact_reject", CountSite::MergeExactReject),
        ("merge_estimate_under_2plus", CountSite::MergeEstimateUnder2Plus),
        ("merge_estimate_under_1", CountSite::MergeEstimateUnder1),
        ("merge_estimate_equal", CountSite::MergeEstimateEqual),
        ("merge_estimate_over_1", CountSite::MergeEstimateOver1),
        ("merge_estimate_over_2", CountSite::MergeEstimateOver2),
        ("merge_estimate_over_3", CountSite::MergeEstimateOver3),
        ("merge_estimate_over_4plus", CountSite::MergeEstimateOver4Plus),
        ("merge_margin1_would_accept", CountSite::MergeMargin1WouldAccept),
        ("merge_margin1_unsafe", CountSite::MergeMargin1Unsafe),
        ("merge_margin0_would_accept", CountSite::MergeMargin0WouldAccept),
        ("merge_margin0_unsafe", CountSite::MergeMargin0Unsafe),
        ("merge_early_reject", CountSite::MergeEarlyReject),
    ];
}

#[cfg(feature = "stats")]
struct CountStat {
    calls: AtomicU64,
    bytes: AtomicU64,
    nanos: AtomicU64,
}

#[cfg(feature = "stats")]
impl CountStat {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }
}

#[cfg(feature = "stats")]
static COUNT_STATS: [CountStat; 40] = [
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
    CountStat::new(),
];

/// Record a chunker control-flow event in the same report as tokenizer call sites.
/// These entries have zero elapsed time because they do not invoke the tokenizer.
#[inline]
pub(crate) fn record_event(site: CountSite, bytes: usize) {
    #[cfg(feature = "stats")]
    {
        let stat = &COUNT_STATS[site as usize];
        stat.calls.fetch_add(1, Ordering::Relaxed);
        stat.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    #[cfg(not(feature = "stats"))]
    {
        let _ = (site, bytes);
    }
}

#[cfg(feature = "stats")]
pub(crate) fn reset_count_stats() {
    for stat in &COUNT_STATS {
        stat.calls.store(0, Ordering::Relaxed);
        stat.bytes.store(0, Ordering::Relaxed);
        stat.nanos.store(0, Ordering::Relaxed);
    }
}

/// Returns `(site, calls, input_bytes, elapsed_nanoseconds)` for each call site.
#[cfg(feature = "stats")]
pub(crate) fn count_stats() -> Vec<(&'static str, u64, u64, u64)> {
    CountSite::ALL
        .iter()
        .map(|(name, site)| {
            let stat = &COUNT_STATS[*site as usize];
            (
                *name,
                stat.calls.load(Ordering::Relaxed),
                stat.bytes.load(Ordering::Relaxed),
                stat.nanos.load(Ordering::Relaxed),
            )
        })
        .collect()
}

/// A user-supplied token counter. Implement this to plug any tokenizer (a model-specific
/// SentencePiece/BPE, a remote counting service, a heuristic) into the chunker.
///
/// # Contract
///
/// The chunker's window search and strict-mode re-check assume a well-behaved counter:
///
/// * **Deterministic**: the same text always yields the same count.
/// * **Monotone on prefixes**: `count(prefix) <= count(prefix + more)`. Counters that violate
///   this can make window search settle on a suboptimal boundary; strict mode will still
///   re-check each delivered chunk, but cannot repair a non-monotone counter.
/// * **Thread-safe and cheap**: it is called many times per chunk, possibly from several
///   threads (`Send + Sync`). Do not do blocking I/O per call.
/// * `count_tokens("")` should return 0 (the chunker short-circuits empty input anyway).
pub trait Tokenizer: Send + Sync {
    /// Number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> usize;

    /// Return `true` only if `count(A + sep + B) <= count(A) + count(sep) + count(B)` holds
    /// for separators ending in a newline. When unsure, keep the default (`false`): the
    /// chunker then keeps a per-join safety slack when merging blocks.
    fn joins_are_additive(&self) -> bool {
        false
    }

    /// Rough bytes-per-token guess, used only to seed the window search. It affects speed,
    /// never correctness.
    fn initial_ratio(&self) -> f64 {
        4.0
    }
}

pub enum Backend {
    Tiktoken {
        bpe: &'static CoreBpe,
    },
    Chars,
    Words,
    #[cfg(feature = "hf")]
    Hf(Box<tokenizers::Tokenizer>),
    /// A caller-provided tokenizer (see [`Tokenizer`]).
    Custom(Arc<dyn Tokenizer>),
}

fn backend_err<E: std::fmt::Display>(e: E) -> ChipError {
    ChipError::Backend(e.to_string())
}

fn cache() -> &'static Mutex<HashMap<String, Arc<Backend>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Backend>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl Backend {
    pub fn from_name(name: &str) -> Result<Arc<Backend>, ChipError> {
        let key = name.to_ascii_lowercase();
        // Short critical section: never build the (slow) BPE while holding the lock.
        {
            let map = cache().lock().unwrap_or_else(|p| p.into_inner());
            if let Some(b) = map.get(&key) {
                return Ok(Arc::clone(b));
            }
        }
        let backend = match key.as_str() {
            "cl100k_base" | "o200k_base" | "p50k_base" | "r50k_base" => Backend::Tiktoken {
                bpe: get_encoding(key.as_str())
                    .ok_or_else(|| backend_err(format!("Unknown encoding: {name}")))?,
            },
            "chars" => Backend::Chars,
            "words" => Backend::Words,
            _ => return Err(ChipError::UnknownEncoding(name.to_string())),
        };
        let arc = Arc::new(backend);
        let mut map = cache().lock().unwrap_or_else(|p| p.into_inner());
        let entry = map.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Wrap a user-supplied [`Tokenizer`] as a backend.
    pub fn from_tokenizer(tokenizer: Arc<dyn Tokenizer>) -> Arc<Backend> {
        Arc::new(Backend::Custom(tokenizer))
    }

    /// Convenience: build a backend from a plain closure (default `Tokenizer` settings).
    pub fn from_fn<F>(f: F) -> Arc<Backend>
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        struct FnTokenizer<F>(F);
        impl<F: Fn(&str) -> usize + Send + Sync> Tokenizer for FnTokenizer<F> {
            fn count_tokens(&self, text: &str) -> usize {
                (self.0)(text)
            }
        }
        Arc::new(Backend::Custom(Arc::new(FnTokenizer(f))))
    }

    #[cfg(feature = "hf")]
    pub fn from_tokenizer_file(path: &str) -> Result<Arc<Backend>, ChipError> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ChipError::Backend(e.to_string()))?;
        Ok(Arc::new(Backend::Hf(Box::new(tok))))
    }

    #[cfg(not(feature = "hf"))]
    pub fn from_tokenizer_file(_path: &str) -> Result<Arc<Backend>, ChipError> {
        Err(ChipError::Backend(
            "this build has no HuggingFace tokenizer support; rebuild with `--features hf`"
                .to_string(),
        ))
    }

    pub fn count(&self, text: &str) -> usize {
        self.count_at(text, CountSite::Other)
    }

    pub(crate) fn count_at(&self, text: &str, site: CountSite) -> usize {
        if text.is_empty() {
            return 0;
        }
        #[cfg(feature = "stats")]
        let started = Instant::now();

        let tokens = match self {
            Backend::Tiktoken { bpe } => bpe.count(text),
            Backend::Chars => text.chars().count(),
            Backend::Words => text.unicode_words().count().max(1),
            #[cfg(feature = "hf")]
            Backend::Hf(tok) => match tok.encode(text, false) {
                Ok(enc) => enc.get_ids().len(),
                // Encoding a valid `&str` without special tokens is not expected to fail, and
                // `count()` has no error channel. If it ever does, fall back to the character
                // count instead of panicking. For most vocabularies that over-counts (safe for
                // the budget), but byte-level vocabularies can need more tokens than
                // characters, so treat the value as a best-effort estimate.
                Err(_) => text.chars().count(),
            },
            Backend::Custom(t) => t.count_tokens(text),
        };

        #[cfg(feature = "stats")]
        {
            let stat = &COUNT_STATS[site as usize];
            stat.calls.fetch_add(1, Ordering::Relaxed);
            stat.bytes.fetch_add(text.len() as u64, Ordering::Relaxed);
            stat.nanos.fetch_add(
                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        #[cfg(not(feature = "stats"))]
        let _ = site;

        tokens
    }

    /// True when `count(A + sep + B) <= count(A) + count(sep) + count(B)` holds for separators
    /// that end in a newline. Verified for tiktoken encodings by the additivity probe test;
    /// unknown for HuggingFace tokenizers, so they keep the per-join slack.
    pub(crate) fn joins_are_additive(&self) -> bool {
        match self {
            Backend::Tiktoken { .. } | Backend::Chars | Backend::Words => true,
            #[cfg(feature = "hf")]
            Backend::Hf(_) => false,
            Backend::Custom(t) => t.joins_are_additive(),
        }
    }

    /// Rough bytes-per-token guess used only to seed the window search.
    pub(super) fn initial_ratio(&self) -> f64 {
        match self {
            Backend::Tiktoken { .. } => 4.0,
            Backend::Chars => 1.2,
            Backend::Words => 6.0,
            #[cfg(feature = "hf")]
            Backend::Hf(_) => 4.0,
            Backend::Custom(t) => {
                let r = t.initial_ratio();
                if r.is_finite() && r > 0.0 {
                    r
                } else {
                    4.0
                }
            }
        }
    }
}
