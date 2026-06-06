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
cat > "$TMP_DIR/fake-aggregate-bench.sh" <<'SH'
#!/usr/bin/env bash
printf 'aggregate %s\n' "$*" >> "$VERIFY_FAST_LOG"
printf 'aggregate-env litertBench=%s ggufBench=%s compareBench=%s prompt=%s litertExpected=%s ggufExpected=%s litertWarmOnly=%s ggufWarmOnly=%s ggufPromptCache=%s compareJavaRefusal=%s compareProbeRegex=%s continueOnFailure=%s\n' \
    "${GOLLEK_VERIFY_FAST_LITERT_BENCH:-}" \
    "${GOLLEK_VERIFY_FAST_GGUF_BENCH:-}" \
    "${GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH:-}" \
    "${GOLLEK_VERIFY_PROMPT:-}" \
    "${GOLLEK_VERIFY_LITERT_EXPECTED:-}" \
    "${GOLLEK_VERIFY_GGUF_EXPECTED:-}" \
    "${GOLLEK_VERIFY_LITERT_WARM_ONLY:-}" \
    "${GOLLEK_VERIFY_GGUF_WARM_ONLY:-}" \
    "${GOLLEK_VERIFY_GGUF_PROMPT_CACHE:-}" \
    "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-}" \
    "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX:-}" \
    "${GOLLEK_VERIFY_CONTINUE_ON_FAILURE:-}" >> "$VERIFY_FAST_LOG"
exit 0
SH
chmod +x "$TMP_DIR/fake-litert-bench.sh" "$TMP_DIR/fake-gguf-bench.sh" "$TMP_DIR/fake-gguf-compare-bench.sh" "$TMP_DIR/fake-aggregate-bench.sh"

GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
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
if ! grep -q '^litert .*--keep-daemon$' "$VERIFY_FAST_LOG" \
        || ! grep -q '^gguf .*--keep-daemon$' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast to keep daemons warm by default" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_LOG"
GOLLEK_INSTALL_VERIFY_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
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
GOLLEK_INSTALL_VERIFY_FAST_KEEP_DAEMON=true \
    bash "$ROOT_DIR/scripts/run-install-local-macos.sh" --verify-fast-only=gguf >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG"; then
    echo "Expected macOS wrapper verify-fast-only=gguf to skip LiteRT benchmark" >&2
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
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=gguf-compare >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG" || grep -q '^gguf ' "$VERIFY_FAST_LOG"; then
    echo "Expected verify-fast-only=gguf-compare to skip regular LiteRT/GGUF benchmarks" >&2
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
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE=true \
GOLLEK_INSTALL_VERIFY_FAST_AGGREGATE_BENCH="$TMP_DIR/fake-aggregate-bench.sh" \
GOLLEK_INSTALL_VERIFY_FAST_SUMMARY_DIR="$TMP_DIR/aggregate-summaries" \
GOLLEK_INSTALL_VERIFY_FAST_SLOWEST_LIMIT=2 \
GOLLEK_INSTALL_VERIFY_PROMPT='where is jakarta?' \
GOLLEK_INSTALL_VERIFY_LITERT_EXPECTED='litert-ok|Jakarta' \
GOLLEK_INSTALL_VERIFY_GGUF_EXPECTED='gguf-ok|Indonesia' \
GOLLEK_INSTALL_VERIFY_LITERT_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_GGUF_WARM_ONLY=true \
GOLLEK_INSTALL_VERIFY_GGUF_PROMPT_CACHE=false \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_REFUSAL=false \
GOLLEK_INSTALL_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX='preparedMatVecReady=true' \
GOLLEK_INSTALL_VERIFY_FAST_CONTINUE_ON_FAILURE=true \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" --verify-fast-only=all >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_LOG" || grep -q '^gguf ' "$VERIFY_FAST_LOG" || grep -q '^gguf-compare ' "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast mode to skip direct benchmark scripts" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^aggregate --only all --gollek-bin .*/bin/gollek --litert-model 7c51c9 --gguf-model b71c9d --summary-dir $TMP_DIR/aggregate-summaries --slowest-limit 2 --keep-daemon$" "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast invocation with summary directory and keep-daemon" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi
if ! grep -q "^aggregate-env litertBench=$TMP_DIR/fake-litert-bench.sh ggufBench=$TMP_DIR/fake-gguf-bench.sh compareBench=$TMP_DIR/fake-gguf-compare-bench.sh prompt=where is jakarta? litertExpected=litert-ok|Jakarta ggufExpected=gguf-ok|Indonesia litertWarmOnly=true ggufWarmOnly=true ggufPromptCache=false compareJavaRefusal=false compareProbeRegex=preparedMatVecReady=true continueOnFailure=true$" "$VERIFY_FAST_LOG"; then
    echo "Expected aggregate verify-fast environment mapping" >&2
    cat "$VERIFY_FAST_LOG" >&2
    exit 1
fi

printf 'install-local-runtime verify-fast test passed\n'
