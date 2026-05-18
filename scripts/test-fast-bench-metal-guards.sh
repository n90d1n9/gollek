#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-fast-bench-metal-guards.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

FAKE_GOLLEK="$TMP_DIR/gollek"
cat > "$FAKE_GOLLEK" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

state_dir="${GOLLEK_FAKE_STATE_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "$state_dir"
litert_state="$state_dir/litert-daemon-state"
gguf_state="$state_dir/gguf-daemon-state"

if [[ "${1:-}" == "__daemon-stop" ]]; then
    rm -f "$litert_state"
    exit 0
fi
if [[ "${1:-}" == "__gguf-daemon-stop" ]]; then
    rm -f "$gguf_state"
    exit 0
fi

engine="auto"
provider=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) engine="$2"; shift 2 ;;
        --provider) provider="$2"; shift 2 ;;
        *) shift ;;
    esac
done

backend="${GOLLEK_FAKE_BACKEND:-CPU}"
probe="${GOLLEK_FAKE_PROBE:-tensor=blk.0.attn_q.weight, type=Q4_K, rows=2304, cols=1536, sampledRows=16, dot=1.000ms, dotChecksum=1.0, matVecRows=4096, cache=1.000ms, preparedMatVecReady=true, parallelMatVec=5.0ms, matVecChecksum=1.0, cachedGenericMatVec=6.0ms, cachedChecksum=1.0}"
java_probe="${GOLLEK_FAKE_JAVA_PROBE:-$probe}"
case "$engine" in
    benchmark)
        cat <<EOF
GGUF engine benchmark: Java-native loader/probe vs llama.cpp generation fallback
Java-native GGUF loader: status=loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled.
Java-native GGUF loader readiness: loaderReady=true, decoderTensorsReady=true, rowDotReady=true, generationReady=false.
Java-native GGUF loader config: ${GOLLEK_FAKE_CONFIG:-type=gemma4, layers=35, hidden=1536, heads=8/1, headDim=256, context=131072, vocab=262144}.
Java-native GGUF loader tensor probe: ${probe}
Using llama.cpp GGUF
Jakarta is in Indonesia.
nativeTiming(modelLoad=1.0ms, backend=${backend}, warmSession=true, promptCache=hit, prefill=1.0ms, decode=1.0ms)
[Stream updates: 1, Duration: 1.00s, Speed: 10.00 t/s]
EOF
        ;;
    java)
        cat <<EOF
Java-native GGUF loader: status=loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled.
Java-native GGUF loader readiness: loaderReady=true, decoderTensorsReady=true, rowDotReady=true, generationReady=false.
Java-native GGUF loader config: ${GOLLEK_FAKE_JAVA_CONFIG:-type=gemma4, layers=35, hidden=1536, heads=8/1, headDim=256, context=131072, vocab=262144}.
Java-native GGUF loader tensor probe: ${java_probe}
Java-native GGUF generation is not enabled yet; refusing to silently use llama.cpp.
EOF
        exit 1
        ;;
    *)
        if [[ "$provider" == "gguf" ]]; then
            warm_session=false
            prompt_cache=stored
            if [[ -f "$gguf_state" ]]; then
                warm_session=true
                prompt_cache=hit
            fi
            printf 'warm\n' > "$gguf_state"
            cat <<EOF
Jakarta is in Indonesia.
nativeTiming(modelLoad=1.0ms, backend=${backend}, warmSession=${warm_session}, promptCache=${prompt_cache}, prefill=1.0ms, decode=1.0ms)
[Stream updates: 1, Duration: 1.00s, Speed: 10.00 t/s]
EOF
            exit 0
        fi
        warm_engine=false
        if [[ -f "$litert_state" ]]; then
            warm_engine=true
        fi
        printf 'warm\n' > "$litert_state"
        cat <<EOF
Jakarta is in Indonesia.
profile(engineInit=0.001s, firstChunk=0.001s, total=0.001s, backend=${backend}, warmEngine=${warm_engine})
[Stream updates: 1, Duration: 1.00s, Speed: 10.00 t/s]
EOF
        ;;
esac
SH
chmod +x "$FAKE_GOLLEK"

expect_failure() {
    local name="$1"
    shift
    expect_failure_message "$name" 'did not use Metal backend' "$@"
}

expect_failure_message() {
    local name="$1"
    local message="$2"
    shift 2
    if "$@" >"$TMP_DIR/${name}.out" 2>"$TMP_DIR/${name}.err"; then
        echo "Expected failure: $name" >&2
        cat "$TMP_DIR/${name}.out" >&2
        cat "$TMP_DIR/${name}.err" >&2
        exit 1
    fi
    if ! grep -q "$message" "$TMP_DIR/${name}.err"; then
        echo "Expected failure message '$message': $name" >&2
        cat "$TMP_DIR/${name}.out" >&2
        cat "$TMP_DIR/${name}.err" >&2
        exit 1
    fi
}

expect_success() {
    local name="$1"
    shift
    "$@" >"$TMP_DIR/${name}.out" 2>"$TMP_DIR/${name}.err"
}

COMMON_ARGS=(--gollek-bin "$FAKE_GOLLEK" --warm-threshold-ms 2000 --cold-threshold-ms 2000 --warmup-runs 0 --require-metal)
export GOLLEK_FAKE_STATE_DIR="$TMP_DIR"

GOLLEK_FAKE_BACKEND=GPU expect_failure litert_generic_gpu \
    bash "$ROOT_DIR/scripts/bench-litert-fast-run.sh" "${COMMON_ARGS[@]}"
GOLLEK_FAKE_BACKEND=CPU expect_failure gguf_cpu \
    bash "$ROOT_DIR/scripts/bench-gguf-fast-run.sh" "${COMMON_ARGS[@]}"
GOLLEK_FAKE_BACKEND=CPU expect_failure gguf_compare_cpu \
    bash "$ROOT_DIR/scripts/bench-gguf-engine-compare.sh" --gollek-bin "$FAKE_GOLLEK" --threshold-ms 2000 --require-metal
GOLLEK_FAKE_BACKEND='MTL0 (Apple M4)' GOLLEK_FAKE_CONFIG='type=unknown, layers=0, hidden=0, heads=0/0, headDim=0, context=0, vocab=0' \
    expect_failure_message gguf_compare_bad_config 'mapped config did not match required regex' \
    bash "$ROOT_DIR/scripts/bench-gguf-engine-compare.sh" --gollek-bin "$FAKE_GOLLEK" --threshold-ms 2000 --require-metal
GOLLEK_FAKE_BACKEND='MTL0 (Apple M4)' GOLLEK_FAKE_JAVA_CONFIG='type=gemma4, layers=34, hidden=1536, heads=8/1, headDim=256, context=131072, vocab=262144' \
    expect_failure_message gguf_compare_java_config_drift 'mapped config differed from benchmark mode' \
    bash "$ROOT_DIR/scripts/bench-gguf-engine-compare.sh" --gollek-bin "$FAKE_GOLLEK" --threshold-ms 2000 --require-metal
GOLLEK_FAKE_BACKEND='MTL0 (Apple M4)' GOLLEK_FAKE_PROBE='parallelMatVec=5.0ms' \
    expect_failure_message gguf_compare_shallow_probe 'tensor probe did not match required regex' \
    bash "$ROOT_DIR/scripts/bench-gguf-engine-compare.sh" --gollek-bin "$FAKE_GOLLEK" --threshold-ms 2000 --require-metal

GOLLEK_FAKE_BACKEND='GPU/Metal' expect_success litert_metal \
    bash "$ROOT_DIR/scripts/bench-litert-fast-run.sh" "${COMMON_ARGS[@]}"
GOLLEK_FAKE_BACKEND='MTL0 (Apple M4)' expect_success gguf_metal \
    bash "$ROOT_DIR/scripts/bench-gguf-fast-run.sh" "${COMMON_ARGS[@]}"
GOLLEK_FAKE_BACKEND='MTL0 (Apple M4)' expect_success gguf_compare_metal \
    bash "$ROOT_DIR/scripts/bench-gguf-engine-compare.sh" --gollek-bin "$FAKE_GOLLEK" --threshold-ms 2000 --require-metal

printf 'fast benchmark Metal guard test passed\n'
