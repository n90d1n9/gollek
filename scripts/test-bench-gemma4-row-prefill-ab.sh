#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-test-gemma4-row-prefill-ab.XXXXXX")"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

GOLLEK_BIN="$TMP_DIR/gollek"
cat > "$GOLLEK_BIN" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

printf 'Platform: Metal\n'
printf '✓ GPU acceleration enabled (Apple M4)\n'
printf 'Model: fake-gemma4-12b\n'
printf 'Provider: safetensor\n'
printf 'GOLLEK_JAVA_OPTS=%s\n' "${GOLLEK_JAVA_OPTS:-}"
printf '%s\n' '--------------------------------------------------'
printf 'Jakarta is in Indonesia.\n'

if [[ "${GOLLEK_JAVA_OPTS:-}" == *"-Dgollek.safetensor.prefer_metal_matvec_ffn_prefill_rows=true"* ]]; then
  printf '\n[PROFILE] backend=metal mode=direct load=1.00ms tokenize=2.00ms session=3.00ms ttft=80.00ms engine_ttft=76.00ms prefill=50.00ms decode=52.00ms tpot=26.00ms sampling=2.00ms argmax=1.20ms attention=22.00ms ffn=40.00ms logits=5.00ms steps=2 linear_paths={attn_q_proj:metal_matmul_f16=2, ffn_down:metal_matmul_f16=2, logits:metal_matmul_f16=2} logits_paths={metal_matmul_f16=2} ffn_paths={matvec-gated-ffn-prefill-rows:accept:geglu:native_bf16=true:native_rows=12:variant=x4=2} attention_paths={paged_metal=2} argmax_paths={native_argmax_f32=2} ffn_strategy=row_prefill_matvec_active path_status={core=metal, linear=metal, logits=metal, ffn=metal, attention=metal, argmax=metal} path_coverage={core=12/12, linear=6/6, logits=2/2, ffn=2/2, attention=2/2, argmax=2/2}\n'
  printf '[Stream updates: 2, Duration: 35,00s, Speed: 12,00 t/s]\n'
else
  printf '\n[PROFILE] backend=metal mode=direct load=1.00ms tokenize=2.00ms session=3.00ms ttft=100.00ms engine_ttft=96.00ms prefill=80.00ms decode=50.00ms tpot=25.00ms sampling=2.00ms argmax=1.00ms attention=20.00ms ffn=60.00ms logits=5.00ms steps=2 linear_paths={attn_q_proj:metal_matmul_f16=2, ffn_down:metal_matmul_f16=2, logits:metal_matmul_f16=2} logits_paths={metal_matmul_f16=2} ffn_paths={matvec-gated-ffn-prefill-rows:skip:strategy_prefers_fused_geglu_prefill=2, fused-gated-ffn:accept:geglu:native_bf16=true=2} attention_paths={paged_metal=2} argmax_paths={native_argmax_f32=2} ffn_strategy=fused_geglu_prefill_over_row_prefill path_status={core=metal, linear=metal, logits=metal, ffn=metal, attention=metal, argmax=metal} path_coverage={core=12/12, linear=6/6, logits=2/2, ffn=2/2, attention=2/2, argmax=2/2}\n'
  printf '[Stream updates: 2, Duration: 40,00s, Speed: 10,00 t/s]\n'
fi
SH
chmod +x "$GOLLEK_BIN"

OUT_DIR="$TMP_DIR/ab"
bash "$ROOT_DIR/scripts/bench-gemma4-row-prefill-ab.sh" \
  --model fake-model \
  --gollek-bin "$GOLLEK_BIN" \
  --out-dir "$OUT_DIR" \
  --max-tokens 2 \
  --gate-metrics ttftMs,ffnMs \
  --max-regression-percent 10 \
  --max-regression-ms 5 > "$TMP_DIR/ab.out"

BASELINE_JSON="$OUT_DIR/runs/gemma4-12b-fused/summary.json"
CURRENT_JSON="$OUT_DIR/runs/gemma4-12b-row-prefill/summary.json"
COMPARE_SUMMARY="$OUT_DIR/compare/summary.tsv"
COMPARE_TSV="$OUT_DIR/compare/comparison.tsv"
REPORT="$OUT_DIR/report.txt"

if [[ ! -f "$BASELINE_JSON" || ! -f "$CURRENT_JSON" || ! -f "$COMPARE_SUMMARY" || ! -f "$COMPARE_TSV" || ! -f "$REPORT" ]]; then
  echo "Expected Gemma4 row-prefill A/B artifacts" >&2
  find "$OUT_DIR" -maxdepth 3 -type f -print >&2 || true
  exit 1
fi

if ! jq -e '
  .runs[0].status == "passed"
  and .runs[0].profile_ffn_strategy == "fused_geglu_prefill_over_row_prefill"
  and .runs[0].speed_tps == 10
  and .runs[0].profile_ffn_ms == 60
' "$BASELINE_JSON" >/dev/null \
    || ! jq -e '
  .runs[0].status == "passed"
  and .runs[0].profile_ffn_strategy == "row_prefill_matvec_active"
  and .runs[0].profile_ffn_row_prefill_native_rows == 12
  and .runs[0].profile_ffn_row_prefill_variant == "x4"
  and .runs[0].speed_tps == 12
  and .runs[0].profile_ffn_ms == 40
' "$CURRENT_JSON" >/dev/null; then
  echo "Expected A/B runner to preserve fused and row-prefill benchmark summaries" >&2
  cat "$BASELINE_JSON" >&2
  cat "$CURRENT_JSON" >&2
  cat "$TMP_DIR/ab.out" >&2
  exit 1
fi

if ! grep -qx $'status\tpass' "$COMPARE_SUMMARY" \
    || ! grep -qx $'recommendation\tpromote-current-with-watchlist' "$COMPARE_SUMMARY" \
    || ! grep -qx $'gatedMetrics\t2' "$COMPARE_SUMMARY" \
    || ! grep -qx $'failedMetrics\t0' "$COMPARE_SUMMARY" \
    || ! grep -qx $'largestImprovementMetric\tprefillMs' "$COMPARE_SUMMARY" \
    || ! grep -qx $'largestImprovementPercent\t37.500' "$COMPARE_SUMMARY" \
    || ! grep -qx $'largestRegressionMetric\targmaxMs' "$COMPARE_SUMMARY" \
    || ! grep -qx $'largestRegressionPercent\t20.000' "$COMPARE_SUMMARY" \
    || ! grep -qx $'baselineFfnStrategy\tfused_geglu_prefill_over_row_prefill' "$COMPARE_SUMMARY" \
    || ! grep -qx $'currentFfnStrategy\trow_prefill_matvec_active' "$COMPARE_SUMMARY" \
    || ! grep -qx $'currentRowPrefill\t12/x4' "$COMPARE_SUMMARY" \
    || ! grep -qx $'metal-deterministic\tttftMs\t100.000\t80.000\t-20.000\t-20.000\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tpass\t' "$COMPARE_TSV" \
    || ! grep -qx $'metal-deterministic\tffnMs\t60.000\t40.000\t-20.000\t-33.333\tlower\tbetter\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tpass\t' "$COMPARE_TSV"; then
  echo "Expected A/B comparison to pass improved TTFT and FFN gates" >&2
  cat "$COMPARE_SUMMARY" >&2
  cat "$COMPARE_TSV" >&2
  exit 1
fi

if ! grep -Fqx "artifacts.baselineSummary=$BASELINE_JSON" "$TMP_DIR/ab.out" \
    || ! grep -Fqx "artifacts.currentSummary=$CURRENT_JSON" "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'status=pass' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'recommendation=promote-current-with-watchlist' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'gatedMetrics=2' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'failedMetrics=0' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'largestImprovement=prefillMs:37.500%' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'largestRegression=argmaxMs:20.000%' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'ffnStrategyTransition=fused_geglu_prefill_over_row_prefill->row_prefill_matvec_active' "$TMP_DIR/ab.out" \
    || ! grep -Fqx 'rowPrefillTransition=n/a/n/a->12/x4' "$TMP_DIR/ab.out"; then
  echo "Expected A/B runner stdout to summarize artifacts and strategy transition" >&2
  cat "$TMP_DIR/ab.out" >&2
  exit 1
fi

FAIL_OUT_DIR="$TMP_DIR/ab-fail"
if bash "$ROOT_DIR/scripts/bench-gemma4-row-prefill-ab.sh" \
    --model fake-model \
    --gollek-bin "$GOLLEK_BIN" \
    --out-dir "$FAIL_OUT_DIR" \
    --max-tokens 2 \
    --gate-metrics decodeMs \
    --max-regression-ms 1 > "$TMP_DIR/ab-fail.out" 2> "$TMP_DIR/ab-fail.err"; then
  echo "Expected Gemma4 row-prefill A/B gate failure for decode regression" >&2
  cat "$TMP_DIR/ab-fail.out" >&2
  cat "$TMP_DIR/ab-fail.err" >&2
  exit 1
fi

if ! grep -qx $'status\tfail' "$FAIL_OUT_DIR/compare/summary.tsv" \
    || ! grep -qx $'recommendation\treject-current' "$FAIL_OUT_DIR/compare/summary.tsv" \
    || ! grep -qx $'failedMetrics\t1' "$FAIL_OUT_DIR/compare/summary.tsv" \
    || ! grep -qx $'metal-deterministic\tdecodeMs\t50.000\t52.000\t2.000\t4.000\tlower\tworse\tfused_geglu_prefill_over_row_prefill\trow_prefill_matvec_active\tn/a/n/a\t12/x4\tfail\tdeltaMs=2.000 exceeded 1' "$FAIL_OUT_DIR/compare/comparison.tsv" \
    || ! grep -Fqx 'status=fail' "$TMP_DIR/ab-fail.out" \
    || ! grep -Fqx 'recommendation=reject-current' "$TMP_DIR/ab-fail.out" \
    || ! grep -Fqx 'failedMetrics=1' "$TMP_DIR/ab-fail.out"; then
  echo "Expected A/B runner to surface decode regression gate failure" >&2
  cat "$FAIL_OUT_DIR/compare/summary.tsv" >&2
  cat "$FAIL_OUT_DIR/compare/comparison.tsv" >&2
  cat "$TMP_DIR/ab-fail.out" >&2
  cat "$TMP_DIR/ab-fail.err" >&2
  exit 1
fi

printf 'Gemma4 row-prefill A/B benchmark test passed\n'
