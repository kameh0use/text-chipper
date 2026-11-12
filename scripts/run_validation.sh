#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="reports/validation_${STAMP}.log"
exec > >(tee "$LOG") 2>&1

echo "TEXT_CHIPPER_VALIDATION_START $(date -Is 2>/dev/null || date)"
echo "rustc: $(rustc --version)"
echo "cargo: $(cargo --version)"
echo "host: $(rustc -vV | sed -n 's/^host: //p')"

python3 scripts/fetch_commonmark.py

echo "=== A. Deterministic correctness suite ==="
cargo test

echo "=== B. Feature matrix ==="
cargo test --features hf,stats

echo "=== C. CommonMark corpus integration ==="
cargo test --features diagnostics commonmark_official_corpus_is_safe_to_scan -- --nocapture

echo "=== D. Differential report ==="
cargo test --features diagnostics commonmark_list_differential_report -- --nocapture

echo "=== E. Resource stress diagnostics ==="
cargo test --features diagnostics memory_stress_giant_single_line_and_many_short_lines -- --nocapture
cargo test --features diagnostics memory_stress_nested_depth_and_large_fence -- --nocapture

echo "=== F. Performance diagnostic ==="
cargo test --features diagnostics performance_regression_smoke_report -- --nocapture

echo "TEXT_CHIPPER_VALIDATION_END $(date -Is 2>/dev/null || date)"
echo "log=$LOG"
