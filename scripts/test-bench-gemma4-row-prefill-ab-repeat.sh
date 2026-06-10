#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-test-gemma4-row-prefill-ab-repeat.XXXXXX")"

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

OUT_DIR="$TMP_DIR/repeat"
bash "$ROOT_DIR/scripts/bench-gemma4-row-prefill-ab-repeat.sh" \
  --model fake-model \
  --gollek-bin "$GOLLEK_BIN" \
  --out-dir "$OUT_DIR" \
  --repeat 2 \
  -- --max-tokens 2 --gate-metrics ttftMs,ffnMs --max-regression-percent 10 --max-regression-ms 5 > "$TMP_DIR/repeat.out"

if [[ ! -f "$OUT_DIR/summary.tsv" || ! -f "$OUT_DIR/samples.tsv" || ! -f "$OUT_DIR/metrics.tsv" || ! -f "$OUT_DIR/metric-rows.tsv" || ! -f "$OUT_DIR/report.txt" ]]; then
  echo "Expected Gemma4 row-prefill repeat A/B aggregate artifacts" >&2
  find "$OUT_DIR" -maxdepth 3 -type f -print >&2 || true
  exit 1
fi

if ! grep -qx $'status\tpass' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'recommendation\tpromote-current-with-watchlist' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'requestedSamples\t2' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'samples\t2' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'passedSamples\t2' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'failedSamples\t0' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'errorSamples\t0' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'failedMetrics\t0' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'unstableMetrics\t0' "$OUT_DIR/summary.tsv" \
    || ! grep -qx $'watchlistSamples\t2' "$OUT_DIR/summary.tsv"; then
  echo "Expected repeat A/B summary to aggregate passing watchlist samples" >&2
  cat "$OUT_DIR/summary.tsv" >&2
  cat "$TMP_DIR/repeat.out" >&2
  exit 1
fi

if ! grep -qx $'sample\texitCode\tstatus\trecommendation\tgatedMetrics\tfailedMetrics\tlargestImprovement\tlargestRegression\tffnStrategyTransition\trowPrefillTransition\tsampleDir\treport' "$OUT_DIR/samples.tsv" \
    || ! grep -q $'^sample-01	0	pass	promote-current-with-watchlist	2	0	prefillMs:37.500%	argmaxMs:20.000%	fused_geglu_prefill_over_row_prefill->row_prefill_matvec_active	n/a/n/a->12/x4	' "$OUT_DIR/samples.tsv" \
    || ! grep -q $'^sample-02	0	pass	promote-current-with-watchlist	2	0	prefillMs:37.500%	argmaxMs:20.000%	fused_geglu_prefill_over_row_prefill->row_prefill_matvec_active	n/a/n/a->12/x4	' "$OUT_DIR/samples.tsv"; then
  echo "Expected repeat A/B sample rows to preserve per-sample decisions" >&2
  cat "$OUT_DIR/samples.tsv" >&2
  exit 1
fi

if ! grep -qx $'metric\tsamples\tbetter\tworse\tsame\tavgDelta\tmedianDelta\tp95Delta\tstddevDelta\tavgDeltaPercent\tmedianDeltaPercent\tp95RegressionPercent\tstddevDeltaPercent\tmaxImprovementPercent\tmaxRegressionPercent\tfailedGateCount\tunstable' "$OUT_DIR/metrics.tsv" \
    || ! grep -qx $'ttftMs\t2\t2\t0\t0\t-20.000\t-20.000\t-20.000\t0.000\t-20.000\t-20.000\t0.000\t0.000\t20.000\t0.000\t0\tfalse' "$OUT_DIR/metrics.tsv" \
    || ! grep -qx $'prefillMs\t2\t2\t0\t0\t-30.000\t-30.000\t-30.000\t0.000\t-37.500\t-37.500\t0.000\t0.000\t37.500\t0.000\t0\tfalse' "$OUT_DIR/metrics.tsv" \
    || ! grep -qx $'decodeMs\t2\t0\t2\t0\t2.000\t2.000\t2.000\t0.000\t4.000\t4.000\t4.000\t0.000\t0.000\t4.000\t0\tfalse' "$OUT_DIR/metrics.tsv" \
    || ! grep -qx $'ffnMs\t2\t2\t0\t0\t-20.000\t-20.000\t-20.000\t0.000\t-33.333\t-33.333\t0.000\t0.000\t33.333\t0.000\t0\tfalse' "$OUT_DIR/metrics.tsv"; then
  echo "Expected repeat A/B metrics to aggregate deltas across samples" >&2
  cat "$OUT_DIR/metrics.tsv" >&2
  exit 1
fi

if ! grep -Fqx "artifacts.samples=$OUT_DIR/samples.tsv" "$TMP_DIR/repeat.out" \
    || ! grep -Fqx "artifacts.metrics=$OUT_DIR/metrics.tsv" "$TMP_DIR/repeat.out" \
    || ! grep -Fqx 'status=pass' "$TMP_DIR/repeat.out" \
    || ! grep -Fqx 'recommendation=promote-current-with-watchlist' "$TMP_DIR/repeat.out" \
    || ! grep -Fqx 'samples=2' "$TMP_DIR/repeat.out" \
    || ! grep -Fqx 'passedSamples=2' "$TMP_DIR/repeat.out" \
    || ! grep -Fqx 'unstableMetrics=0' "$TMP_DIR/repeat.out"; then
  echo "Expected repeat A/B stdout to summarize aggregate artifacts" >&2
  cat "$TMP_DIR/repeat.out" >&2
  exit 1
fi

FAIL_OUT_DIR="$TMP_DIR/repeat-fail"
if bash "$ROOT_DIR/scripts/bench-gemma4-row-prefill-ab-repeat.sh" \
    --model fake-model \
    --gollek-bin "$GOLLEK_BIN" \
    --out-dir "$FAIL_OUT_DIR" \
    --repeat 2 \
    -- --max-tokens 2 --gate-metrics decodeMs --max-regression-ms 1 > "$TMP_DIR/repeat-fail.out" 2> "$TMP_DIR/repeat-fail.err"; then
  echo "Expected repeat A/B aggregate failure for repeated decode regression" >&2
  cat "$TMP_DIR/repeat-fail.out" >&2
  cat "$TMP_DIR/repeat-fail.err" >&2
  exit 1
fi

if ! grep -qx $'status\tfail' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'recommendation\treject-current' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'samples\t2' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'failedSamples\t2' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'failedMetrics\t2' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'unstableMetrics\t0' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'rejectSamples\t2' "$FAIL_OUT_DIR/summary.tsv" \
    || ! grep -qx $'decodeMs\t2\t0\t2\t0\t2.000\t2.000\t2.000\t0.000\t4.000\t4.000\t4.000\t0.000\t0.000\t4.000\t2\tfalse' "$FAIL_OUT_DIR/metrics.tsv" \
    || ! grep -Fqx 'status=fail' "$TMP_DIR/repeat-fail.out" \
    || ! grep -Fqx 'recommendation=reject-current' "$TMP_DIR/repeat-fail.out" \
    || ! grep -Fqx 'failedSamples=2' "$TMP_DIR/repeat-fail.out" \
    || ! grep -Fqx 'unstableMetrics=0' "$TMP_DIR/repeat-fail.out"; then
  echo "Expected repeat A/B aggregate to reject repeated decode regression" >&2
  cat "$FAIL_OUT_DIR/summary.tsv" >&2
  cat "$FAIL_OUT_DIR/metrics.tsv" >&2
  cat "$TMP_DIR/repeat-fail.out" >&2
  cat "$TMP_DIR/repeat-fail.err" >&2
  exit 1
fi

printf 'Gemma4 row-prefill repeat A/B benchmark test passed\n'
