#!/usr/bin/env bash
# Run every regular test plus the diagnostics (`--features diagnostics`) (including the release suite in
# src/tests/release_suite.rs) and write one combined log and one summary.
#
#   scripts/run_release_suite.sh            # full run from the crate root
#   RS_QUICK=1 scripts/run_release_suite.sh # faster smoke run
#
# Optional environment: RS_FUZZ_ITERS, RS_SEED, RS_ONLY=A|B|C|D|E, RS_PERF_SAVE=file,
# RS_PERF_BASELINE=file. The script never modifies the source tree.
set -uo pipefail

PROJECT=$(cd "$(dirname "$0")/.." && pwd)
TS=$(date +%Y%m%d_%H%M%S)
LOG="$PROJECT/release_report_$TS.log"
SUM="$PROJECT/release_summary_$TS.txt"

if [ ! -f "$PROJECT/Cargo.toml" ] || [ ! -f "$PROJECT/src/lib.rs" ]; then
  echo "error: $PROJECT is not the crate root (Cargo.toml / src/lib.rs missing)" >&2
  exit 2
fi

cd "$PROJECT"
{
  echo "=== environment ==="
  date; uname -a
  rustc --version 2>&1; cargo --version 2>&1
  echo "=== STEP 1: cargo test --release (all regular tests) ==="
} | tee "$LOG"

cargo test --release 2>&1 | tee -a "$LOG"
S1=${PIPESTATUS[0]}

echo "=== STEP 2: diagnostics + release suite (--features diagnostics) ===" | tee -a "$LOG"
cargo test --release --features diagnostics -- --nocapture --test-threads=1 2>&1 | tee -a "$LOG"
S2=${PIPESTATUS[0]}

{
  echo "step1_exit=$S1 step2_exit=$S2"
  grep -E '^(test result|RS_BEGIN|RS_SECTION|RS_TOTAL)' "$LOG"
  echo "--- failed tests ---"
  grep -E '^test .* FAILED|^failures:|panicked at' "$LOG" | head -40
  echo "--- RS rules with findings ---"
  grep -E '^RS_RULE' "$LOG" | grep -v 'fails=0$'
  echo "--- RS samples ---"
  grep -E '^RS_SAMPLE' "$LOG" | cut -c1-600
  echo "--- RS info (perf + counts) ---"
  grep -E '^RS_INFO' "$LOG" | grep -v 'SHRUNK' | cut -c1-300
  echo "--- minimal repro cases (shrunk) ---"
  grep -E '^RS_INFO.SHRUNK' "$LOG" | cut -c1-1500
} > "$SUM"

echo
echo "full log : $LOG"
echo "summary  : $SUM"
echo "exit codes: step1=$S1 step2=$S2"
if grep -q "^RS_TOTAL" "$LOG"; then
  grep "^RS_TOTAL" "$LOG"
else
  echo "RS_TOTAL not found: the suite did not finish (compile error or crash?)." >&2
  echo "If the error is a linker problem with Python symbols on macOS, add a build.rs that" >&2
  echo "passes -undefined dynamic_lookup to the test binaries." >&2
fi
[ "$S1" -eq 0 ] && [ "$S2" -eq 0 ]
