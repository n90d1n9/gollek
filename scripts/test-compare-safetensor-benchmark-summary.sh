#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-test-safetensor-benchmark-compare.XXXXXX")"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BASELINE_DIR="${TMP_DIR}/fused"
CURRENT_DIR="${TMP_DIR}/row-prefill"
COMPARE_DIR="${TMP_DIR}/compare"
mkdir -p "$BASELINE_DIR" "$CURRENT_DIR"

cat > "$BASELINE_DIR/summary.json" <<'JSON'
{
  "label": "fused",
  "runs": [
    {
      "case_id": "metal-deterministic",
      "status": "passed",
      "duration_s": 40,
      "speed_tps": 10,
      "chunks": 2,
      "profile_summary": "backend=metal ttft=100.00ms prefill=80.00ms decode=50.00ms",
      "profile_top_stage_ms": 100,
      "profile_prefill_ms": 80,
      "profile_decode_ms": 50,
      "profile_tpot_ms": 25,
      "profile_sampling_ms": 2,
      "profile_argmax_ms": 1,
      "profile_attention_ms": 20,
      "profile_ffn_ms": 60,
      "profile_logits_ms": 5,
      "profile_ffn_strategy": "fused_geglu_prefill_over_row_prefill",
      "profile_ffn_row_prefill_native_rows": null,
      "profile_ffn_row_prefill_variant": null,
      "log_file": "/tmp/fused.log"
    }
  ]
}
JSON

cat > "$CURRENT_DIR/summary.json" <<'JSON'
{
  "label": "row-prefill",
  "runs": [
    {
      "case_id": "metal-deterministic",
      "status": "passed",
      "duration_s": 35,
      "speed_tps": 12,
      "chunks": 2,
      "profile_summary": "backend=metal ttft=80.00ms prefill=50.00ms decode=52.00ms",
      "profile_top_stage_ms": 80,
      "profile_prefill_ms": 50,
      "profile_decode_ms": 52,
      "profile_tpot_ms": 26,
      "profile_sampling_ms": 2,
      "profile_argmax_ms": 1.2,
      "profile_attention_ms": 22,
      "profile_ffn_ms": 40,
      "profile_logits_ms": 5,
      "profile_ffn_strategy": "row_prefill_matvec_active",
      "profile_ffn_row_prefill_native_rows": 12,
      "profile_ffn_row_prefill_variant": "x4",
      "log_file": "/tmp/row-prefill.log"
    }
  ]
}
JSON

bash "$ROOT_DIR/scripts/compare-safetensor-benchmark-summary.sh" \
  --baseline "$BASELINE_DIR" \
  --current "$CURRENT_DIR" \
  --summary-dir "$COMPARE_DIR" \
  --case metal-deterministic > "$TMP_DIR/compare.out"

if ! grep -qx $'case\tmetric\tbaseline\tcurrent\tdelta\tdeltaPercent\tdirection\ttrend\tbaselineFfnStrategy\tcurrentFfnStrategy\tbaselineRowPrefill\tcurrentRowPrefill\tgateStatus\tgateReason' "$COMPARE_DIR/comparison.tsv" \
    || ! grep -qx $'metal-deterministic\tttftMs\t100.000\t80.000\t-20.000\t-20.000\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tnot-configured\t' "$COMPARE_DIR/comparison.tsv" \
    || ! grep -qx $'metal-deterministic\tffnMs\t60.000\t40.000\t-20.000\t-33.333\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tnot-configured\t' "$COMPARE_DIR/comparison.tsv" \
    || ! grep -qx $'metal-deterministic\tdecodeMs\t50.000\t52.000\t2.000\t4.000\tlower\tworse\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tnot-configured\t' "$COMPARE_DIR/comparison.tsv" \
    || ! grep -qx $'metal-deterministic\tspeedTps\t10.000\t12.000\t2.000\t20.000\thigher\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tnot-configured\t' "$COMPARE_DIR/comparison.tsv"; then
  echo "Expected safetensor benchmark comparison rows with FFN strategy context" >&2
  cat "$COMPARE_DIR/comparison.tsv" >&2
  exit 1
fi

if ! grep -qx $'status\tpass' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'recommendation\tpromote-current-with-watchlist' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'baselineLabel\tfused' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'currentLabel\trow-prefill' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'case\tmetal-deterministic' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'comparedCases\t1' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'gatedMetrics\t0' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'failedMetrics\t0' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestImprovementMetric\tprefillMs' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestImprovementDelta\t-30.000' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestImprovementPercent\t37.500' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestRegressionMetric\targmaxMs' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestRegressionDelta\t0.200' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'largestRegressionPercent\t20.000' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'baselineFfnStrategy\tfused_geglu_prefill_over_row_prefill' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'currentFfnStrategy\trow_prefill_matvec_active' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'baselineRowPrefill\tn/a/n/a' "$COMPARE_DIR/summary.tsv" \
    || ! grep -qx $'currentRowPrefill\t12/x4' "$COMPARE_DIR/summary.tsv" \
    || ! grep -Fqx "artifacts.comparison=$COMPARE_DIR/comparison.tsv" "$TMP_DIR/compare.out" \
    || ! grep -Fqx 'recommendation=promote-current-with-watchlist' "$TMP_DIR/compare.out" \
    || ! grep -Fqx 'largestImprovement=prefillMs:37.500%' "$TMP_DIR/compare.out" \
    || ! grep -Fqx 'largestRegression=argmaxMs:20.000%' "$TMP_DIR/compare.out" \
    || ! grep -Fqx 'ffnStrategyTransition=fused_geglu_prefill_over_row_prefill->row_prefill_matvec_active' "$TMP_DIR/compare.out" \
    || ! grep -Fqx 'rowPrefillTransition=n/a/n/a->12/x4' "$TMP_DIR/compare.out"; then
  echo "Expected safetensor benchmark comparison summary and report" >&2
  cat "$COMPARE_DIR/summary.tsv" >&2
  cat "$TMP_DIR/compare.out" >&2
  exit 1
fi

PASS_GATE_DIR="${TMP_DIR}/pass-gate"
bash "$ROOT_DIR/scripts/compare-safetensor-benchmark-summary.sh" \
  --baseline "$BASELINE_DIR" \
  --current "$CURRENT_DIR" \
  --summary-dir "$PASS_GATE_DIR" \
  --case metal-deterministic \
  --gate-metrics ttftMs,ffnMs \
  --max-regression-percent 10 \
  --max-regression-ms 5 > "$TMP_DIR/pass-gate.out"

if ! grep -qx $'status\tpass' "$PASS_GATE_DIR/summary.tsv" \
    || ! grep -qx $'recommendation\tpromote-current-with-watchlist' "$PASS_GATE_DIR/summary.tsv" \
    || ! grep -qx $'gatedMetrics\t2' "$PASS_GATE_DIR/summary.tsv" \
    || ! grep -qx $'failedMetrics\t0' "$PASS_GATE_DIR/summary.tsv" \
    || ! grep -qx $'metal-deterministic\tttftMs\t100.000\t80.000\t-20.000\t-20.000\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tpass\t' "$PASS_GATE_DIR/comparison.tsv" \
    || ! grep -qx $'metal-deterministic\tffnMs\t60.000\t40.000\t-20.000\t-33.333\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tpass\t' "$PASS_GATE_DIR/comparison.tsv" \
    || ! grep -Fqx 'status=pass' "$TMP_DIR/pass-gate.out" \
    || ! grep -Fqx 'recommendation=promote-current-with-watchlist' "$TMP_DIR/pass-gate.out" \
    || ! grep -Fqx 'gatedMetrics=2' "$TMP_DIR/pass-gate.out"; then
  echo "Expected safetensor benchmark gate to pass improved TTFT and FFN metrics" >&2
  cat "$PASS_GATE_DIR/summary.tsv" >&2
  cat "$PASS_GATE_DIR/comparison.tsv" >&2
  cat "$TMP_DIR/pass-gate.out" >&2
  exit 1
fi

FAIL_GATE_DIR="${TMP_DIR}/fail-gate"
if bash "$ROOT_DIR/scripts/compare-safetensor-benchmark-summary.sh" \
    --baseline "$BASELINE_DIR" \
    --current "$CURRENT_DIR" \
    --summary-dir "$FAIL_GATE_DIR" \
    --case metal-deterministic \
    --gate-metrics decodeMs \
    --max-regression-ms 1 > "$TMP_DIR/fail-gate.out" 2> "$TMP_DIR/fail-gate.err"; then
  echo "Expected safetensor benchmark gate failure for decode regression" >&2
  cat "$TMP_DIR/fail-gate.out" >&2
  cat "$TMP_DIR/fail-gate.err" >&2
  exit 1
fi

if ! grep -qx $'status\tfail' "$FAIL_GATE_DIR/summary.tsv" \
    || ! grep -qx $'reason\tmetric-regression' "$FAIL_GATE_DIR/summary.tsv" \
    || ! grep -qx $'recommendation\treject-current' "$FAIL_GATE_DIR/summary.tsv" \
    || ! grep -qx $'gatedMetrics\t1' "$FAIL_GATE_DIR/summary.tsv" \
    || ! grep -qx $'failedMetrics\t1' "$FAIL_GATE_DIR/summary.tsv" \
    || ! grep -qx $'metal-deterministic\tdecodeMs\t50.000\t52.000\t2.000\t4.000\tlower\tworse\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tfail\tdeltaMs=2.000 exceeded 1' "$FAIL_GATE_DIR/comparison.tsv" \
    || ! grep -Fqx 'status=fail' "$TMP_DIR/fail-gate.out" \
    || ! grep -Fqx 'recommendation=reject-current' "$TMP_DIR/fail-gate.out" \
    || ! grep -Fqx 'failedMetrics=1' "$TMP_DIR/fail-gate.out"; then
  echo "Expected safetensor benchmark gate to expose decode regression details" >&2
  cat "$FAIL_GATE_DIR/summary.tsv" >&2
  cat "$FAIL_GATE_DIR/comparison.tsv" >&2
  cat "$TMP_DIR/fail-gate.out" >&2
  cat "$TMP_DIR/fail-gate.err" >&2
  exit 1
fi

MISSING_DIR="${TMP_DIR}/missing"
if bash "$ROOT_DIR/scripts/compare-safetensor-benchmark-summary.sh" \
    --baseline "$BASELINE_DIR" \
    --current "$CURRENT_DIR" \
    --summary-dir "$MISSING_DIR" \
    --case missing > "$TMP_DIR/missing.out" 2> "$TMP_DIR/missing.err"; then
  echo "Expected missing benchmark case comparison failure" >&2
  cat "$TMP_DIR/missing.out" >&2
  cat "$TMP_DIR/missing.err" >&2
  exit 1
fi
if ! grep -qx 'Requested case not found in both summaries: missing' "$TMP_DIR/missing.err"; then
  echo "Expected missing benchmark case diagnostic" >&2
  cat "$TMP_DIR/missing.err" >&2
  exit 1
fi

printf 'safetensor benchmark summary comparison test passed\n'
