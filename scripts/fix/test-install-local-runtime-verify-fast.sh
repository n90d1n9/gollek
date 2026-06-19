#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-install-verify-fast.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

export HOME="$TMP_DIR/home"
export GOLLEK_BIN_DIR="$TMP_DIR/bin"
export VERIFY_FAST_LOG="$TMP_DIR/verify-fast.log"
mkdir -p "$HOME/.gollek/models" "$GOLLEK_BIN_DIR"

cat > "$GOLLEK_BIN_DIR/gollek" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$GOLLEK_BIN_DIR/gollek"

cat > "$HOME/.gollek/models/index.json" <<JSON
[ {
  "id" : "litert/model",
  "shortId" : "7c51c9",
  "name" : "litert-model",
  "format" : "litert",
  "runnable" : true,
  "path" : "$HOME/.gollek/models/blobs/7c51c9",
  "source" : "local"
}, {
  "id" : "gguf/model",
  "shortId" : "b71c9d",
  "name" : "gguf-model",
  "format" : "gguf",
  "runnable" : true,
  "path" : "$HOME/.gollek/models/blobs/b71c9d/model.gguf",
  "source" : "local"
}, {
  "id" : "safetensor/model",
  "shortId" : "6f469a",
  "name" : "safetensor-model",
  "format" : "safetensor",
  "runnable" : true,
  "path" : "$HOME/.gollek/models/blobs/6f469a",
  "source" : "local"
} ]
JSON

cat > "$TMP_DIR/fake-litert-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'litert %s\n' "$*" >> "$VERIFY_FAST_LOG"
exit 0
SH
cat > "$TMP_DIR/fake-gguf-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'gguf %s\n' "$*" >> "$VERIFY_FAST_LOG"
exit 0
SH
cat > "$TMP_DIR/fake-gguf-compare-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'gguf-compare %s\n' "$*" >> "$VERIFY_FAST_LOG"
exit 0
SH
cat > "$TMP_DIR/fake-safetensor-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'safetensor %s\n' "$*" >> "$VERIFY_FAST_LOG"
printf 'safetensor-env presetScript=%s\n' "${GOLLEK_BENCH_SAFETENSOR_PRESET_SCRIPT:-}" >> "$VERIFY_FAST_LOG"
exit 0
SH
cat > "$TMP_DIR/fake-gemma4-litert-smoke.sh" <<'SH'
#!/usr/bin/env bash
printf 'gemma4-litert %s\n' "$*" >> "$VERIFY_FAST_LOG"
exit 0
SH
cat > "$TMP_DIR/fake-aggregate-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'aggregate %s\n' "$*" >> "$VERIFY_FAST_LOG"
printf 'aggregate-env litertBench=%s ggufBench=%s compareBench=%s safetensorBench=%s prompt=%s litertExpected=%s ggufExpected=%s litertWarmOnly=%s ggufWarmOnly=%s ggufPromptCache=%s compareJavaRefusal=%s compareProbeRegex=%s safetensorMaxTokens=%s safetensorPreset=%s safetensorPresetScript=%s safetensorRequireProfile=%s safetensorRequireMetalPaths=%s safetensorRejectFallbackPaths=%s safetensorTopStage=%s safetensorPrefill=%s safetensorDecode=%s safetensorTpot=%s safetensorAttention=%s safetensorFfn=%s safetensorLogits=%s safetensorBaselineStages=%s safetensorBaselineRoot=%s safetensorBaselineName=%s safetensorMaxRegressionPercent=%s safetensorMaxRegressionMs=%s safetensorMinBaselineMs=%s safetensorFailPrimaryShift=%s continueOnFailure=%s\n' \
    "${GOLLEK_VERIFY_FAST_LITERT_BENCH:-}" \
    "${GOLLEK_VERIFY_FAST_GGUF_BENCH:-}" \
    "${GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH:-}" \
    "${GOLLEK_VERIFY_FAST_SAFETENSOR_BENCH:-}" \
    "${GOLLEK_VERIFY_PROMPT:-}" \
    "${GOLLEK_VERIFY_LITERT_EXPECTED:-}" \
    "${GOLLEK_VERIFY_GGUF_EXPECTED:-}" \
    "${GOLLEK_VERIFY_LITERT_WARM_ONLY:-}" \
    "${GOLLEK_VERIFY_GGUF_WARM_ONLY:-}" \
    "${GOLLEK_VERIFY_GGUF_PROMPT_CACHE:-}" \
    "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-}" \
    "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_MAX_TOKENS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_PRESET:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_PRESET_SCRIPT:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_REQUIRE_PROFILE:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_BASELINE_STAGES:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_BASELINE_ROOT:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_BASELINE_NAME:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_MAX_REGRESSION_PERCENT:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_MAX_REGRESSION_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_MIN_BASELINE_MS:-}" \
    "${GOLLEK_VERIFY_SAFETENSOR_FAIL_PRIMARY_SHIFT:-}" \
    "${GOLLEK_VERIFY_CONTINUE_ON_FAILURE:-}" >> "$VERIFY_FAST_LOG"
exit 0
SH
chmod +x "$TMP_DIR/fake-litert-bench.sh" "$TMP_DIR/fake-gguf-bench.sh" "$TMP_DIR/fake-gguf-compare-bench.sh" "$TMP_DIR/fake-safetensor-bench.sh" "$TMP_DIR/fake-gemma4-litert-smoke.sh" "$TMP_DIR/fake-aggregate-bench.sh"

HELP_OUTPUT="$TMP_DIR/help.txt"
bash "$ROOT_DIR/scripts/install-local-runtime.sh" --help >"$HELP_OUTPUT"
if ! grep -Fq -- '--verify-fast[=auto|all|litert|gguf|gguf-compare|safetensor|gemma4-litert]' "$HELP_OUTPUT" \
        || ! grep -Fq -- '--verify-fast-only[=auto|all|litert|gguf|gguf-compare|safetensor|gemma4-litert]' "$HELP_OUTPUT"; then
    echo "Expected install-local-runtime help to advertise Gemma4 LiteRT verification target" >&2
    cat "$HELP_OUTPUT" >&2
    exit 1
fi

SAFETENSOR_METAL_PATHS_DEFAULT=false
if [[ "$(uname -s)" == "Darwin" ]]; then
    SAFETENSOR_METAL_PATHS_DEFAULT=true
fi

GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_THRESHOLD_MS=1234 \
GOLLEK_INSTALL_VERIFY_LITERT_COLD_THRESHOLD_MS=5678 \
GOLLEK_INSTALL_VERIFY_LITERT_MAX_TOKENS=4 \
GOLLEK_INSTALL_VERIFY_LITERT_WARMUP_RUNS=2 \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS=44 \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS=55 \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS=66 \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_THRESHOLD_MS=2345 \
GOLLEK_INSTALL_VERIFY_GGUF_COLD_THRESHOLD_MS=6789 \
GOLLEK_INSTALL_VERIFY_GGUF_MAX_TOKENS=5 \
GOLLEK_INSTALL_VERIFY_GGUF_WARMUP_RUNS=3 \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS=11 \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS=222 \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS=333 \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_THRESHOLD_MS=4567 \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX=row-dot-primitives-ready \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX='type=gemma4, layers=35' \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS=12 \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS=77 \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS=88 \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_SUMMARY_FILE="$TMP_DIR/gguf-compare-summary.tsv" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS=6 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL=true \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS=true \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS=99 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS=98 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS=100 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS=101 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS=102 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS=103 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS=104 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_SUMMARY_FILE="$TMP_DIR/safetensor-summary.tsv" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_OUT_DIR="$TMP_DIR/safetensor-out" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_LABEL=install-smoke \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=all >/dev/null

if ! grep -q '^litert .*--gollek-bin .*/bin/gollek .*--model 7c51c9 .*--max-tokens 4 .*--warm-threshold-ms 1234 .*--cold-threshold-ms 5678 .*--warmup-runs 2 .*--warm-engine-init-threshold-ms 44 .*--warm-first-chunk-threshold-ms 55 .*--warm-profile-total-threshold-ms 66 .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected LiteRT verify benchmark invocation with configured args" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf .*--gollek-bin .*/bin/gollek .*--model b71c9d .*--max-tokens 5 .*--warm-threshold-ms 2345 .*--cold-threshold-ms 6789 .*--warmup-runs 3 .*--require-prompt-cache .*--warm-tokenize-threshold-ms 11 .*--warm-prefill-threshold-ms 222 .*--warm-decode-threshold-ms 333 .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF verify benchmark invocation with configured args" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf-compare .*--gollek-bin .*/bin/gollek .*--model b71c9d .*--threshold-ms 4567 .*--java-ready-regex row-dot-primitives-ready ' "$VERIFY_FAST_LOG" \
        || ! grep -Fq -- "--java-config-regex type=gemma4, layers=35 --java-matvec-threshold-ms 12 --fallback-prefill-threshold-ms 77 --fallback-decode-threshold-ms 88 --summary-file $TMP_DIR/gguf-compare-summary.tsv --verify-java-refusal" "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF compare benchmark invocation with configured args" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^safetensor .*--gollek-bin .*/bin/gollek .*--model 6f469a .*--out-dir $TMP_DIR/safetensor-out .*--label install-smoke .*--det-prompt where is jakarta .*--max-tokens 6 .*--quick .*--profile .*--summary-file $TMP_DIR/safetensor-summary.tsv .*--require-profile .*--require-metal .*--require-metal-paths .*--max-top-stage-ms 99 .*--max-prefill-ms 98 .*--max-decode-ms 100 .*--max-tpot-ms 101 .*--max-attention-ms 102 .*--max-ffn-ms 103 .*--max-logits-ms 104$" "$VERIFY_FAST_LOG"; then
    echo "Expected safetensor verify benchmark invocation with profile, Metal, and threshold args" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^litert .*--keep-daemon$' "$VERIFY_FAST_LOG" \
        || ! grep -q '^gguf .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast to keep daemons warm by default" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_SMOKE="$TMP_DIR/fake-gemma4-litert-smoke.sh" \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_MODEL=gemma4-local \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_PROMPT='what is jakarta' \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_EXPECTED='Jakarta|Indonesia' \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_MAX_TOKENS=31 \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_THRESHOLD_MS=901 \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS=251 \
GOLLEK_INSTALL_VERIFY_GEMMA4_LITERT_WARM_TOTAL_THRESHOLD_MS=1001 \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gemma4-litert >/dev/null

if ! grep -q '^gemma4-litert .*--gollek-bin .*/bin/gollek .*--model gemma4-local .*--prompt what is jakarta .*--expected Jakarta|Indonesia .*--max-tokens 31 .*--warm-threshold-ms 901 .*--warm-first-chunk-threshold-ms 251 .*--warm-profile-total-threshold-ms 1001 .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected Gemma4 LiteRT install verification to invoke the warm-daemon smoke with strict configured args" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET=m4-smoke \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS=true \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=safetensor >/dev/null

if ! grep -q '^safetensor .*--max-tokens 2 --preset m4-smoke --quick --profile .*--require-profile .*--reject-fallback-paths' "$VERIFY_FAST_LOG" \
        || ! grep -qx "safetensor-env presetScript=$ROOT_DIR/scripts/safetensor-performance-presets.sh" "$VERIFY_FAST_LOG"; then
    echo "Expected direct install safetensor verification to forward the m4-smoke preset, preset registry, and fallback rejection" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=safetensor >/dev/null

if [[ "$SAFETENSOR_METAL_PATHS_DEFAULT" == "true" ]]; then
    if ! grep -q '^safetensor .*--require-metal-paths' "$VERIFY_FAST_LOG"; then
        echo "Expected safetensor verify-fast to require Metal path proof by default on macOS" >&2
        cat "$VERIFY_FAST_LOG" >&2
        exit 1
    fi
else
    if grep -q -- '--require-metal-paths' "$VERIFY_FAST_LOG"; then
        echo "Expected safetensor verify-fast Metal path proof to default off away from macOS" >&2
        cat "$VERIFY_FAST_LOG" >&2
        exit 1
    fi
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_METAL_PATHS=false \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=safetensor >/dev/null

if grep -q -- '--require-metal-paths' "$VERIFY_FAST_LOG"; then
    echo "Expected safetensor Metal path proof to be explicitly disabled for diagnostics" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
SAFETENSOR_BASELINE_ROOT="$TMP_DIR/safetensor-baselines"
SAFETENSOR_BASELINE_NAME="install-safetensor"
SAFETENSOR_BASELINE_STAGES="$SAFETENSOR_BASELINE_ROOT/$SAFETENSOR_BASELINE_NAME/latest-stages.tsv"
mkdir -p "$SAFETENSOR_BASELINE_ROOT/$SAFETENSOR_BASELINE_NAME"
printf 'stage\tmetric\tvalueMs\tshareOfDurationPercent\tpriority\tpathEvidence\trecommendation\n' > "$SAFETENSOR_BASELINE_STAGES"
printf 'decode\tdecodeMs\t18.000\t12.000\tprimary\tmetal\tFocus decode.\n' >> "$SAFETENSOR_BASELINE_STAGES"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gguf >/dev/null

if grep -q -- '--keep-daemon' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast to allow explicit daemon cleanup" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf .*--max-tokens 1 .*--require-prompt-cache$' "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF verify-fast to use a first-token prompt-cache check by default" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE=false \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gguf >/dev/null

if grep -q -- '--require-prompt-cache' "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF verify-fast prompt-cache guard to be opt-out for diagnostics" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf .*--max-tokens 1 ' "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF verify-fast diagnostic opt-out to keep the first-token default" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=both >/dev/null

if ! grep -q '^litert .*--max-tokens 10 .*--warm-only$' "$VERIFY_FAST_LOG"; then
    echo "Expected LiteRT verify-fast warm-only mode to skip cold-load verification" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf .*--max-tokens 1 .*--require-prompt-cache .*--warm-only$' "$VERIFY_FAST_LOG"; then
    echo "Expected GGUF verify-fast warm-only mode to skip cold-load verification" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=true \
    bash "$ROOT_DIR/scripts/install-local-macos.sh" --verify-fast-only=gguf >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper verify-fast-only=gguf to skip LiteRT benchmark" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if grep -q '^safetensor ' "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper verify-fast-only=gguf to skip safetensor benchmark" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if grep -q '^gguf-compare ' "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper verify-fast-only=gguf to skip GGUF compare benchmark" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf .*--model b71c9d .*--max-tokens 1 .*--require-prompt-cache .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper verify-fast-only=gguf to honor explicit keep-daemon" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gguf-compare >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG" || grep -q '^gguf ' "$VERIFY_FAST_LOG" || grep -q '^safetensor ' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast-only=gguf-compare to skip regular LiteRT/GGUF/safetensor benchmarks" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf-compare .*--model b71c9d .*--threshold-ms 10000 .*--java-ready-regex row-dot-primitives-ready ' "$VERIFY_FAST_LOG" \
        || ! grep -Fq -- '--java-config-regex type=[^,]+, layers=[1-9][0-9]*, hidden=[1-9][0-9]*, heads=[1-9][0-9]*/[1-9][0-9]*, headDim=[1-9][0-9]*, context=[1-9][0-9]*, vocab=[1-9][0-9]* --java-matvec-threshold-ms 50 --verify-java-refusal' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast-only=gguf-compare to invoke only the comparison benchmark" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_REFUSAL=false \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gguf-compare >/dev/null

if ! grep -q '^gguf-compare .*--no-verify-java-refusal$' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast-only=gguf-compare to allow disabling Java fallback refusal check" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE_BENCH="$TMP_DIR/fake-aggregate-bench.sh" \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-m4-smoke-only >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG" || grep -q '^gguf ' "$VERIFY_FAST_LOG" || grep -q '^gguf-compare ' "$VERIFY_FAST_LOG" || grep -q '^safetensor ' "$VERIFY_FAST_LOG"; then
    echo "Expected m4-smoke install shortcut to use the aggregate verifier instead of direct benchmark scripts" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^aggregate --only safetensor --gollek-bin .*/bin/gollek --litert-model 7c51c9 --gguf-model b71c9d --safetensor-model 6f469a --require-metal --keep-daemon$" "$VERIFY_FAST_LOG" \
        || ! grep -q "^aggregate-env .*safetensorMaxTokens=3 .*safetensorPreset=m4-smoke .*safetensorRequireProfile=true .*safetensorRequireMetalPaths=true .*safetensorRejectFallbackPaths=true " "$VERIFY_FAST_LOG"; then
    echo "Expected m4-smoke install shortcut to enforce aggregate safetensor Metal/no-fallback verification" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE_BENCH="$TMP_DIR/fake-aggregate-bench.sh" \
    bash "$ROOT_DIR/scripts/install-local-macos.sh" --verify-fast-m4-smoke-only >/dev/null

if ! grep -q "^aggregate --only safetensor --gollek-bin .*/bin/gollek --litert-model 7c51c9 --gguf-model b71c9d --safetensor-model 6f469a --require-metal --keep-daemon$" "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper to forward m4-smoke verify-only shortcut without running the full install wrapper" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BENCH="$TMP_DIR/fake-safetensor-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE=true \
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE_BENCH="$TMP_DIR/fake-aggregate-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_SUMMARY_DIR="$TMP_DIR/aggregate-summaries" \
GOLLEK_INSTALL_VERIFY_FAST_SLOWEST_LIMIT=2 \
GOLLEK_INSTALL_VERIFY_FAST_REQUIRE_METAL=false \
GOLLEK_INSTALL_VERIFY_PROMPT='where is jakarta?' \
GOLLEK_INSTALL_VERIFY_LITERT_EXPECTED='litert-ok|Jakarta' \
GOLLEK_INSTALL_VERIFY_GGUF_EXPECTED='gguf-ok|Indonesia' \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE=false \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_REFUSAL=false \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX='preparedMatVecReady=true' \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MODEL=safetensor-local \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_TOKENS=7 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_PRESET=m4-smoke \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REQUIRE_PROFILE=false \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_REJECT_FALLBACK_PATHS=true \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_TOP_STAGE_THRESHOLD_MS=123 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_PREFILL_THRESHOLD_MS=234 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_DECODE_THRESHOLD_MS=456 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_TPOT_THRESHOLD_MS=567 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_ATTENTION_THRESHOLD_MS=678 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_FFN_THRESHOLD_MS=789 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_LOGITS_THRESHOLD_MS=890 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_STAGES=latest \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_ROOT="$SAFETENSOR_BASELINE_ROOT" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_BASELINE_NAME="$SAFETENSOR_BASELINE_NAME" \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_REGRESSION_PERCENT=12 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MAX_REGRESSION_MS=34 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_MIN_BASELINE_MS=5 \
GOLLEK_INSTALL_VERIFY_SAFETENSOR_FAIL_PRIMARY_SHIFT=true \
GOLLEK_INSTALL_VERIFY_FAST_CONTINUE_ON_FAILURE=true \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=all >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG" || grep -q '^gguf ' "$VERIFY_FAST_LOG" || grep -q '^gguf-compare ' "$VERIFY_FAST_LOG" || grep -q '^safetensor ' "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast mode to skip direct benchmark scripts" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^aggregate --only all --gollek-bin .*/bin/gollek --litert-model 7c51c9 --gguf-model b71c9d --safetensor-model safetensor-local --no-require-metal --summary-dir $TMP_DIR/aggregate-summaries --slowest-limit 2 --keep-daemon$" "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast invocation with summary directory and keep-daemon" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^aggregate-env litertBench=$TMP_DIR/fake-litert-bench.sh ggufBench=$TMP_DIR/fake-gguf-bench.sh compareBench=$TMP_DIR/fake-gguf-compare-bench.sh safetensorBench=$TMP_DIR/fake-safetensor-bench.sh prompt=where is jakarta? litertExpected=litert-ok|Jakarta ggufExpected=gguf-ok|Indonesia litertWarmOnly=true ggufWarmOnly=true ggufPromptCache=false compareJavaRefusal=false compareProbeRegex=preparedMatVecReady=true safetensorMaxTokens=7 safetensorPreset=m4-smoke safetensorPresetScript=$ROOT_DIR/scripts/safetensor-performance-presets.sh safetensorRequireProfile=false safetensorRequireMetalPaths=$SAFETENSOR_METAL_PATHS_DEFAULT safetensorRejectFallbackPaths=true safetensorTopStage=123 safetensorPrefill=234 safetensorDecode=456 safetensorTpot=567 safetensorAttention=678 safetensorFfn=789 safetensorLogits=890 safetensorBaselineStages=latest safetensorBaselineRoot=$SAFETENSOR_BASELINE_ROOT safetensorBaselineName=$SAFETENSOR_BASELINE_NAME safetensorMaxRegressionPercent=12 safetensorMaxRegressionMs=34 safetensorMinBaselineMs=5 safetensorFailPrimaryShift=true continueOnFailure=true$" "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast environment mapping" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

printf 'install-local-runtime verify-fast test passed\n'
