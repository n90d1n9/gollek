#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-verify-fast-speed-gates.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

export VERIFY_FAST_GATES_LOG="$TMP_DIR/verify-fast-gates.log"
GOLLEK_BIN="$TMP_DIR/gollek"
cat > "$GOLLEK_BIN" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$GOLLEK_BIN"

make_fake_bench() {
    local path="$1"
    local name="$2"
    cat > "$path" <<SH
#!/usr/bin/env bash
set -euo pipefail
summary=""
args=("\$@")
for ((i = 0; i < \${#args[@]}; i++)); do
    if [[ "\${args[\$i]}" == "--summary-file" ]]; then
        summary="\${args[\$((i + 1))]}"
    fi
done
printf '$name %s\n' "\$*" >> "\$VERIFY_FAST_GATES_LOG"
if [[ -n "\$summary" ]]; then
    case "$name" in
        litert)
            printf 'case\tdurationMs\tbackend\twarmEngine\tengineInitMs\tfirstChunkMs\ttotalMs\tlog\n' > "\$summary"
            printf 'warm\t100\tMTL0 (Apple M4)\ttrue\t1\t2\t3\t/tmp/fake-litert.log\n' >> "\$summary"
            ;;
        gguf)
            printf 'case\tdurationMs\tbackend\twarmSession\ttokenizeCache\tpromptCache\tpromptCacheEagerShort\toutputBytes\tjavaRetries\tmodelLoadMs\ttokenizeMs\tprefillMs\tdecodeMs\tlog\n' > "\$summary"
            printf 'warm\t120\tMTL0 (Apple M4)\ttrue\thit\tprefix-hit\tfalse\t64\t0\t0\t4\t5\t6\t/tmp/fake-gguf.log\n' >> "\$summary"
            ;;
        *)
            printf 'case\tdurationMs\tbackend\twarmSession\tpromptCache\tmodelLoadMs\ttokenizeMs\tprefillMs\tdecodeMs\tjavaStatus\tloaderReady\tdecoderTensorsReady\trowDotReady\tgenerationReady\tbenchmarkParallelMatVecMs\tjavaOnlyParallelMatVecMs\tbestParallelMatVecMs\tjavaMatVecThresholdMs\tjavaFallbackRefusal\tlog\n' > "\$summary"
            printf 'compare\t130\tMTL0 (Apple M4)\ttrue\thit\t0\t4\t5\t6\trow-dot-primitives-ready\ttrue\ttrue\ttrue\tfalse\t7\t8\t7\t9\tchecked\t/tmp/fake-compare.log\n' >> "\$summary"
            ;;
    esac
fi
if [[ "\${GOLLEK_FAKE_FAIL_GATE:-}" == "$name" ]]; then
    exit 42
fi
SH
    chmod +x "$path"
}

make_fake_bench "$TMP_DIR/fake-litert-bench.sh" litert
make_fake_bench "$TMP_DIR/fake-gguf-bench.sh" gguf
make_fake_bench "$TMP_DIR/fake-gguf-compare-bench.sh" gguf-compare

SUMMARY_DIR="$TMP_DIR/summaries"
ALL_OUT="$TMP_DIR/all.out"
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
GOLLEK_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS=3 \
GOLLEK_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS=4 \
GOLLEK_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS=5 \
GOLLEK_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS=6 \
GOLLEK_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS=7 \
GOLLEK_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS=8 \
GOLLEK_VERIFY_GGUF_COMPARE_MAX_TOKENS=12 \
GOLLEK_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS=9 \
GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS=10 \
GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS=11 \
GOLLEK_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX=row-dot-primitives-ready \
GOLLEK_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX='type=gemma4, layers=35' \
GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX='preparedMatVecReady=true' \
GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL=false \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --summary-dir "$SUMMARY_DIR" \
        --gollek-bin "$GOLLEK_BIN" \
        --litert-model litert-local \
        --gguf-model gguf-local \
        --prompt "where is jakarta?" \
        --litert-expected 'litert-ok|Jakarta' \
        --gguf-expected 'gguf-ok|Jakarta' \
        --require-metal \
        --warm-only \
        --keep-daemon >"$ALL_OUT"

if [[ ! -f "$SUMMARY_DIR/litert-fast.tsv" \
        || ! -f "$SUMMARY_DIR/gguf-fast.tsv" \
        || ! -f "$SUMMARY_DIR/gguf-compare.tsv" \
        || ! -f "$SUMMARY_DIR/config.tsv" \
        || ! -f "$SUMMARY_DIR/manifest.tsv" \
        || ! -f "$SUMMARY_DIR/rollup.tsv" \
        || ! -f "$SUMMARY_DIR/backend.tsv" \
        || ! -f "$SUMMARY_DIR/contracts.tsv" \
        || ! -f "$SUMMARY_DIR/slowest.tsv" \
        || ! -f "$SUMMARY_DIR/results.tsv" ]]; then
    echo "Expected aggregate speed gate summaries, config, manifest, rollup, backend, contracts, slowest, and results" >&2
    find "$SUMMARY_DIR" -maxdepth 1 -type f -print >&2 || true
    exit 1
fi

if ! grep -qx $'key\tvalue' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'only\tall' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx "summaryDir	$SUMMARY_DIR" "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'litert.model\tlitert-local' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'gguf.model\tgguf-local' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'prompt\twhere is jakarta?' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'requireMetal\ttrue' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'warmOnly\ttrue' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'keepDaemon\ttrue' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'slowestLimit\t5' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'gguf.promptCache\ttrue' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'litert.warmEngineInitThresholdMs\t3' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'gguf.warmPrefillThresholdMs\t7' "$SUMMARY_DIR/config.tsv" \
        || ! grep -qx $'ggufCompare.javaConfigRegex\ttype=gemma4, layers=35' "$SUMMARY_DIR/config.tsv"; then
    echo "Expected config artifact to capture resolved aggregate gate settings" >&2
    cat "$SUMMARY_DIR/config.tsv" >&2
    exit 1
fi
if ! grep -qx $'gate\tcase\tdurationMs\tbackend\tmetrics\tlog' "$SUMMARY_DIR/rollup.tsv" \
        || ! grep -qx "litert	warm	100	MTL0 (Apple M4)	warmEngine=true,engineInitMs=1,firstChunkMs=2,totalMs=3	/tmp/fake-litert.log" "$SUMMARY_DIR/rollup.tsv" \
        || ! grep -qx "gguf	warm	120	MTL0 (Apple M4)	warmSession=true,tokenizeCache=hit,promptCache=prefix-hit,promptCacheEagerShort=false,outputBytes=64,javaRetries=0,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6	/tmp/fake-gguf.log" "$SUMMARY_DIR/rollup.tsv" \
        || ! grep -qx "gguf-compare	compare	130	MTL0 (Apple M4)	warmSession=true,promptCache=hit,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6,javaStatus=row-dot-primitives-ready,loaderReady=true,decoderTensorsReady=true,rowDotReady=true,generationReady=false,benchmarkParallelMatVecMs=7,javaOnlyParallelMatVecMs=8,bestParallelMatVecMs=7,javaMatVecThresholdMs=9,javaFallbackRefusal=checked	/tmp/fake-compare.log" "$SUMMARY_DIR/rollup.tsv"; then
    echo "Expected normalized rollup rows for all speed gates" >&2
    cat "$SUMMARY_DIR/rollup.tsv" >&2
    exit 1
fi
if ! grep -qx $'gate\tcase\tbackend\tmetal\twarmReuse\tpromptCache\tjavaFallbackRefusal\tlog' "$SUMMARY_DIR/backend.tsv" \
        || ! grep -qx "litert	warm	MTL0 (Apple M4)	true	true	n/a	n/a	/tmp/fake-litert.log" "$SUMMARY_DIR/backend.tsv" \
        || ! grep -qx "gguf	warm	MTL0 (Apple M4)	true	true	prefix-hit	n/a	/tmp/fake-gguf.log" "$SUMMARY_DIR/backend.tsv" \
        || ! grep -qx "gguf-compare	compare	MTL0 (Apple M4)	true	true	hit	checked	/tmp/fake-compare.log" "$SUMMARY_DIR/backend.tsv"; then
    echo "Expected backend proof rows for Metal, warm reuse, prompt cache, and fallback refusal" >&2
    cat "$SUMMARY_DIR/backend.tsv" >&2
    exit 1
fi
if ! grep -qx $'gate\tcase\tmetalRequired\tmetal\tmetalStatus\twarmRequired\twarmReuse\twarmStatus\tpromptCacheRequired\tpromptCache\tpromptCacheStatus\tjavaFallbackRefusalRequired\tjavaFallbackRefusal\tjavaFallbackRefusalStatus\tlog' "$SUMMARY_DIR/contracts.tsv" \
        || ! grep -qx "litert	warm	true	true	pass	true	true	pass	false	n/a	n/a	false	n/a	n/a	/tmp/fake-litert.log" "$SUMMARY_DIR/contracts.tsv" \
        || ! grep -qx "gguf	warm	true	true	pass	true	true	pass	true	prefix-hit	pass	false	n/a	n/a	/tmp/fake-gguf.log" "$SUMMARY_DIR/contracts.tsv" \
        || ! grep -qx "gguf-compare	compare	true	true	pass	false	true	n/a	false	hit	n/a	false	checked	n/a	/tmp/fake-compare.log" "$SUMMARY_DIR/contracts.tsv"; then
    echo "Expected backend contract rows to mark required checks as pass or n/a" >&2
    cat "$SUMMARY_DIR/contracts.tsv" >&2
    exit 1
fi
if ! grep -qx $'rank\tgate\tcase\tdurationMs\tbackend\tmetrics\tlog' "$SUMMARY_DIR/slowest.tsv" \
        || ! grep -qx "1	gguf-compare	compare	130	MTL0 (Apple M4)	warmSession=true,promptCache=hit,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6,javaStatus=row-dot-primitives-ready,loaderReady=true,decoderTensorsReady=true,rowDotReady=true,generationReady=false,benchmarkParallelMatVecMs=7,javaOnlyParallelMatVecMs=8,bestParallelMatVecMs=7,javaMatVecThresholdMs=9,javaFallbackRefusal=checked	/tmp/fake-compare.log" "$SUMMARY_DIR/slowest.tsv" \
        || ! grep -qx "2	gguf	warm	120	MTL0 (Apple M4)	warmSession=true,tokenizeCache=hit,promptCache=prefix-hit,promptCacheEagerShort=false,outputBytes=64,javaRetries=0,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6	/tmp/fake-gguf.log" "$SUMMARY_DIR/slowest.tsv" \
        || ! grep -qx "3	litert	warm	100	MTL0 (Apple M4)	warmEngine=true,engineInitMs=1,firstChunkMs=2,totalMs=3	/tmp/fake-litert.log" "$SUMMARY_DIR/slowest.tsv"; then
    echo "Expected slowest report to rank normalized rollup rows by duration" >&2
    cat "$SUMMARY_DIR/slowest.tsv" >&2
    exit 1
fi
if ! grep -qx 'PASS: fast speed gates passed' "$ALL_OUT" \
        || ! grep -qx "summaryDir=$SUMMARY_DIR" "$ALL_OUT" \
        || ! grep -qx "artifacts.config=$SUMMARY_DIR/config.tsv" "$ALL_OUT" \
        || ! grep -qx "artifacts.results=$SUMMARY_DIR/results.tsv" "$ALL_OUT" \
        || ! grep -qx "artifacts.rollup=$SUMMARY_DIR/rollup.tsv" "$ALL_OUT" \
        || ! grep -qx "artifacts.backend=$SUMMARY_DIR/backend.tsv" "$ALL_OUT" \
        || ! grep -qx "artifacts.contracts=$SUMMARY_DIR/contracts.tsv" "$ALL_OUT" \
        || ! grep -qx "artifacts.slowest=$SUMMARY_DIR/slowest.tsv" "$ALL_OUT" \
        || ! grep -qx 'backend:' "$ALL_OUT" \
        || ! grep -qx '  gguf/warm metal=true warmReuse=true promptCache=prefix-hit javaFallbackRefusal=n/a backend=MTL0 (Apple M4) log=/tmp/fake-gguf.log' "$ALL_OUT" \
        || ! grep -qx 'contracts:' "$ALL_OUT" \
        || ! grep -qx '  gguf/warm metal=pass warm=pass promptCache=pass javaRefusal=n/a log=/tmp/fake-gguf.log' "$ALL_OUT" \
        || ! grep -qx 'slowest:' "$ALL_OUT" \
        || ! grep -qx '  #1 gguf-compare/compare duration=130ms backend=MTL0 (Apple M4) log=/tmp/fake-compare.log' "$ALL_OUT"; then
    echo "Expected aggregate output to print artifact paths and slowest rows" >&2
    cat "$ALL_OUT" >&2
    exit 1
fi

if ! grep -qx $'gate\tsummary' "$SUMMARY_DIR/manifest.tsv" \
        || ! grep -qx "litert	$SUMMARY_DIR/litert-fast.tsv" "$SUMMARY_DIR/manifest.tsv" \
        || ! grep -qx "gguf	$SUMMARY_DIR/gguf-fast.tsv" "$SUMMARY_DIR/manifest.tsv" \
        || ! grep -qx "gguf-compare	$SUMMARY_DIR/gguf-compare.tsv" "$SUMMARY_DIR/manifest.tsv"; then
    echo "Expected manifest rows for all speed gates" >&2
    cat "$SUMMARY_DIR/manifest.tsv" >&2
    exit 1
fi
if ! grep -qx $'gate\tstatus\texitCode\telapsedMs\tsummary\tstdout\tstderr\targv\treason' "$SUMMARY_DIR/results.tsv" \
        || ! grep -q "^litert	pass	0	[0-9][0-9]*	$SUMMARY_DIR/litert-fast.tsv	$SUMMARY_DIR/litert.out	$SUMMARY_DIR/litert.err	$SUMMARY_DIR/litert.argv.tsv	$" "$SUMMARY_DIR/results.tsv" \
        || ! grep -q "^gguf	pass	0	[0-9][0-9]*	$SUMMARY_DIR/gguf-fast.tsv	$SUMMARY_DIR/gguf.out	$SUMMARY_DIR/gguf.err	$SUMMARY_DIR/gguf.argv.tsv	$" "$SUMMARY_DIR/results.tsv" \
        || ! grep -q "^gguf-compare	pass	0	[0-9][0-9]*	$SUMMARY_DIR/gguf-compare.tsv	$SUMMARY_DIR/gguf-compare.out	$SUMMARY_DIR/gguf-compare.err	$SUMMARY_DIR/gguf-compare.argv.tsv	$" "$SUMMARY_DIR/results.tsv"; then
    echo "Expected pass rows for all speed gates" >&2
    cat "$SUMMARY_DIR/results.tsv" >&2
    exit 1
fi
if [[ ! -f "$SUMMARY_DIR/litert.out" || ! -f "$SUMMARY_DIR/litert.err" \
        || ! -f "$SUMMARY_DIR/litert.argv.tsv" \
        || ! -f "$SUMMARY_DIR/gguf.out" || ! -f "$SUMMARY_DIR/gguf.err" \
        || ! -f "$SUMMARY_DIR/gguf.argv.tsv" \
        || ! -f "$SUMMARY_DIR/gguf-compare.out" || ! -f "$SUMMARY_DIR/gguf-compare.err" \
        || ! -f "$SUMMARY_DIR/gguf-compare.argv.tsv" ]]; then
    echo "Expected stdout/stderr/argv logs for all speed gates" >&2
    find "$SUMMARY_DIR" -maxdepth 1 -type f -print >&2
    exit 1
fi
if ! grep -qx $'0\tbash' "$SUMMARY_DIR/litert.argv.tsv" \
        || ! grep -qx $'4\t--model' "$SUMMARY_DIR/litert.argv.tsv" \
        || ! grep -qx $'5\tlitert-local' "$SUMMARY_DIR/litert.argv.tsv" \
        || ! grep -qx $'7\twhere is jakarta?' "$SUMMARY_DIR/litert.argv.tsv"; then
    echo "Expected LiteRT argv artifact to preserve command arguments" >&2
    cat "$SUMMARY_DIR/litert.argv.tsv" >&2
    exit 1
fi

if ! grep -q "^litert .*--gollek-bin $GOLLEK_BIN .*--model litert-local .*--prompt where is jakarta? .*--expected litert-ok|Jakarta .*--require-metal .*--warm-engine-init-threshold-ms 3 .*--warm-first-chunk-threshold-ms 4 .*--warm-profile-total-threshold-ms 5 .*--warm-only .*--summary-file $SUMMARY_DIR/litert-fast.tsv .*--keep-daemon$" "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected LiteRT aggregate gate invocation with thresholds and summary" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi

if ! grep -q "^gguf .*--gollek-bin $GOLLEK_BIN .*--model gguf-local .*--prompt where is jakarta? .*--expected gguf-ok|Jakarta .*--require-metal .*--require-prompt-cache .*--warm-tokenize-threshold-ms 6 .*--warm-prefill-threshold-ms 7 .*--warm-decode-threshold-ms 8 .*--warm-only .*--summary-file $SUMMARY_DIR/gguf-fast.tsv .*--keep-daemon$" "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected GGUF aggregate gate invocation with cache guard, thresholds, and summary" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi

if ! grep -q "^gguf-compare .*--gollek-bin $GOLLEK_BIN .*--model gguf-local .*--prompt where is jakarta? .*--expected gguf-ok|Jakarta .*--max-tokens 12 .*--java-matvec-threshold-ms 9 .*--require-metal .*--fallback-prefill-threshold-ms 10 .*--fallback-decode-threshold-ms 11 .*--java-ready-regex row-dot-primitives-ready .*--java-config-regex type=gemma4, layers=35 .*--java-probe-regex preparedMatVecReady=true .*--no-verify-java-refusal .*--summary-file $SUMMARY_DIR/gguf-compare.tsv$" "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected GGUF compare aggregate gate invocation with thresholds and summary" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_GATES_LOG"
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --only gguf-compare \
        --summary-dir "$TMP_DIR/compare-only" \
        --gollek-bin "$GOLLEK_BIN" \
        --no-require-metal >/dev/null

if grep -q '^litert ' "$VERIFY_FAST_GATES_LOG" || grep -q '^gguf ' "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected --only gguf-compare to skip LiteRT and GGUF fast gates" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi
if ! grep -q '^gguf-compare .*--no-require-metal .*--summary-file ' "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected --only gguf-compare to run compare gate with requested Metal policy" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_GATES_LOG"
LIMIT_DIR="$TMP_DIR/slowest-limit"
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --summary-dir "$LIMIT_DIR" \
        --gollek-bin "$GOLLEK_BIN" \
        --slowest-limit 2 \
        --no-require-metal >"$TMP_DIR/slowest-limit.out"

if ! grep -qx $'slowestLimit\t2' "$LIMIT_DIR/config.tsv" \
        || ! grep -qx "1	gguf-compare	compare	130	MTL0 (Apple M4)	warmSession=true,promptCache=hit,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6,javaStatus=row-dot-primitives-ready,loaderReady=true,decoderTensorsReady=true,rowDotReady=true,generationReady=false,benchmarkParallelMatVecMs=7,javaOnlyParallelMatVecMs=8,bestParallelMatVecMs=7,javaMatVecThresholdMs=9,javaFallbackRefusal=checked	/tmp/fake-compare.log" "$LIMIT_DIR/slowest.tsv" \
        || ! grep -qx "2	gguf	warm	120	MTL0 (Apple M4)	warmSession=true,tokenizeCache=hit,promptCache=prefix-hit,promptCacheEagerShort=false,outputBytes=64,javaRetries=0,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6	/tmp/fake-gguf.log" "$LIMIT_DIR/slowest.tsv" \
        || grep -q '^3	' "$LIMIT_DIR/slowest.tsv" \
        || grep -q '^  #3 ' "$TMP_DIR/slowest-limit.out"; then
    echo "Expected --slowest-limit to trim slowest report and terminal footer" >&2
    cat "$LIMIT_DIR/config.tsv" >&2
    cat "$LIMIT_DIR/slowest.tsv" >&2
    cat "$TMP_DIR/slowest-limit.out" >&2
    exit 1
fi

>"$VERIFY_FAST_GATES_LOG"
cat > "$TMP_DIR/auto-index.json" <<JSON
[ {
  "id" : "litert/local",
  "shortId" : "litert-local",
  "name" : "litert-local",
  "format" : "litert",
  "runnable" : true,
  "path" : "$TMP_DIR/litert-local"
} ]
JSON
GOLLEK_VERIFY_MODEL_INDEX="$TMP_DIR/auto-index.json" \
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --only auto \
        --summary-dir "$TMP_DIR/auto-only" \
        --gollek-bin "$GOLLEK_BIN" \
        --litert-model litert-local \
        --gguf-model gguf-local \
        --no-require-metal >/dev/null

if ! grep -q '^litert ' "$VERIFY_FAST_GATES_LOG" \
        || grep -q '^gguf ' "$VERIFY_FAST_GATES_LOG" \
        || grep -q '^gguf-compare ' "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected --only auto to run only locally indexed LiteRT gate and skip compare" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi
if ! grep -q "^litert	pass	0	[0-9][0-9]*	$TMP_DIR/auto-only/litert-fast.tsv	$TMP_DIR/auto-only/litert.out	$TMP_DIR/auto-only/litert.err	$TMP_DIR/auto-only/litert.argv.tsv	$" "$TMP_DIR/auto-only/results.tsv" \
        || ! grep -q '^gguf	skip	0	0					model-not-found$' "$TMP_DIR/auto-only/results.tsv" \
        || ! grep -q '^gguf-compare	skip	0	0					auto-skips-compare$' "$TMP_DIR/auto-only/results.tsv"; then
    echo "Expected auto results to include LiteRT pass and explicit skip rows" >&2
    cat "$TMP_DIR/auto-only/results.tsv" >&2
    exit 1
fi

>"$VERIFY_FAST_GATES_LOG"
FAIL_DIR="$TMP_DIR/failure"
set +e
GOLLEK_FAKE_FAIL_GATE=gguf \
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --summary-dir "$FAIL_DIR" \
        --gollek-bin "$GOLLEK_BIN" \
        --no-require-metal >"$TMP_DIR/failure.out" 2>"$TMP_DIR/failure.err"
failure_exit=$?
set -e
if [[ "$failure_exit" -ne 42 ]]; then
    echo "Expected failing aggregate gate to preserve benchmark exit code" >&2
    cat "$TMP_DIR/failure.out" >&2
    cat "$TMP_DIR/failure.err" >&2
    exit 1
fi
if ! grep -q "^gguf	fail	42	[0-9][0-9]*	$FAIL_DIR/gguf-fast.tsv	$FAIL_DIR/gguf.out	$FAIL_DIR/gguf.err	$FAIL_DIR/gguf.argv.tsv	$" "$FAIL_DIR/results.tsv"; then
    echo "Expected failed GGUF gate row in aggregate results" >&2
    cat "$FAIL_DIR/results.tsv" >&2
    exit 1
fi
if ! grep -qx $'rank\tgate\tcase\tdurationMs\tbackend\tmetrics\tlog' "$FAIL_DIR/slowest.tsv" \
        || ! grep -qx "1	gguf	warm	120	MTL0 (Apple M4)	warmSession=true,tokenizeCache=hit,promptCache=prefix-hit,promptCacheEagerShort=false,outputBytes=64,javaRetries=0,modelLoadMs=0,tokenizeMs=4,prefillMs=5,decodeMs=6	/tmp/fake-gguf.log" "$FAIL_DIR/slowest.tsv" \
        || ! grep -qx "2	litert	warm	100	MTL0 (Apple M4)	warmEngine=true,engineInitMs=1,firstChunkMs=2,totalMs=3	/tmp/fake-litert.log" "$FAIL_DIR/slowest.tsv"; then
    echo "Expected early failure to leave a populated slowest report" >&2
    cat "$FAIL_DIR/slowest.tsv" >&2
    exit 1
fi
if ! grep -qx "summaryDir=$FAIL_DIR" "$TMP_DIR/failure.out" \
        || ! grep -qx "artifacts.backend=$FAIL_DIR/backend.tsv" "$TMP_DIR/failure.out" \
        || ! grep -qx "artifacts.contracts=$FAIL_DIR/contracts.tsv" "$TMP_DIR/failure.out" \
        || ! grep -qx "artifacts.slowest=$FAIL_DIR/slowest.tsv" "$TMP_DIR/failure.out" \
        || ! grep -qx '  gguf/warm metal=true warmReuse=true promptCache=prefix-hit javaFallbackRefusal=n/a backend=MTL0 (Apple M4) log=/tmp/fake-gguf.log' "$TMP_DIR/failure.out" \
        || ! grep -qx '  gguf/warm metal=n/a warm=pass promptCache=pass javaRefusal=n/a log=/tmp/fake-gguf.log' "$TMP_DIR/failure.out" \
        || ! grep -qx '  #1 gguf/warm duration=120ms backend=MTL0 (Apple M4) log=/tmp/fake-gguf.log' "$TMP_DIR/failure.out"; then
    echo "Expected early failure output to print artifact paths and slowest rows" >&2
    cat "$TMP_DIR/failure.out" >&2
    cat "$TMP_DIR/failure.err" >&2
    exit 1
fi
if grep -q '^gguf-compare ' "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected aggregate gate to stop after GGUF failure" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi

>"$VERIFY_FAST_GATES_LOG"
CONTINUE_DIR="$TMP_DIR/continue-failure"
set +e
GOLLEK_FAKE_FAIL_GATE=gguf \
GOLLEK_VERIFY_FAST_LITERT_BENCH="$TMP_DIR/fake-litert-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_BENCH="$TMP_DIR/fake-gguf-bench.sh" \
GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH="$TMP_DIR/fake-gguf-compare-bench.sh" \
    bash "$ROOT_DIR/scripts/verify-fast-speed-gates.sh" \
        --summary-dir "$CONTINUE_DIR" \
        --gollek-bin "$GOLLEK_BIN" \
        --no-require-metal \
        --continue-on-failure >"$TMP_DIR/continue.out" 2>"$TMP_DIR/continue.err"
continue_exit=$?
set -e
if [[ "$continue_exit" -ne 42 ]]; then
    echo "Expected continue-on-failure aggregate gate to preserve first failure exit code" >&2
    cat "$TMP_DIR/continue.out" >&2
    cat "$TMP_DIR/continue.err" >&2
    exit 1
fi
if ! grep -q '^litert ' "$VERIFY_FAST_GATES_LOG" \
        || ! grep -q '^gguf ' "$VERIFY_FAST_GATES_LOG" \
        || ! grep -q '^gguf-compare ' "$VERIFY_FAST_GATES_LOG"; then
    echo "Expected continue-on-failure to run all selected gates" >&2
    cat "$VERIFY_FAST_GATES_LOG" >&2
    exit 1
fi
if ! grep -q "^litert	pass	0	[0-9][0-9]*	$CONTINUE_DIR/litert-fast.tsv	$CONTINUE_DIR/litert.out	$CONTINUE_DIR/litert.err	$CONTINUE_DIR/litert.argv.tsv	$" "$CONTINUE_DIR/results.tsv" \
        || ! grep -q "^gguf	fail	42	[0-9][0-9]*	$CONTINUE_DIR/gguf-fast.tsv	$CONTINUE_DIR/gguf.out	$CONTINUE_DIR/gguf.err	$CONTINUE_DIR/gguf.argv.tsv	$" "$CONTINUE_DIR/results.tsv" \
        || ! grep -q "^gguf-compare	pass	0	[0-9][0-9]*	$CONTINUE_DIR/gguf-compare.tsv	$CONTINUE_DIR/gguf-compare.out	$CONTINUE_DIR/gguf-compare.err	$CONTINUE_DIR/gguf-compare.argv.tsv	$" "$CONTINUE_DIR/results.tsv"; then
    echo "Expected continue-on-failure results to include pass, fail, pass rows" >&2
    cat "$CONTINUE_DIR/results.tsv" >&2
    exit 1
fi

printf 'verify-fast-speed-gates test passed\n'
