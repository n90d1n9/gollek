#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-gguf-fast-run.sh [options]

Run a focused smoke/benchmark for the installed GGUF fast path. The check
starts from a cold daemon, verifies the backend, then verifies warm latency.

Options:
  --model ID              Model id or local alias (default: b71c9d)
  --prompt TEXT           Prompt to run (default: "where is jakarta")
  --expected REGEX        Expected answer regex (default: Indonesia|Jakarta)
  --max-tokens N          Max generated tokens (default: 10)
  --gollek-bin PATH       Gollek executable (default: ~/.local/bin/gollek)
  --warm-threshold-ms N   Fail if warm duration exceeds this (default: 1500)
  --cold-threshold-ms N   Fail if cold duration exceeds this (default: 60000)
  --warmup-runs N         Stabilization runs before measured warm check (default: 1)
  --require-metal         Require Metal backend
  --no-require-metal      Do not require Metal backend
  --keep-daemon           Leave daemon running after benchmark
  --help                  Show this help
USAGE
}

MODEL_ID="b71c9d"
PROMPT="where is jakarta"
EXPECTED_REGEX="Indonesia|Jakarta"
MAX_TOKENS=10
GOLLEK_BIN="${HOME}/.local/bin/gollek"
WARM_THRESHOLD_MS=1500
COLD_THRESHOLD_MS=60000
WARMUP_RUNS=1
KEEP_DAEMON=0
case "$(uname -s)" in
  Darwin) REQUIRE_METAL=1 ;;
  *) REQUIRE_METAL=0 ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --expected) EXPECTED_REGEX="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --gollek-bin) GOLLEK_BIN="$2"; shift 2 ;;
    --warm-threshold-ms) WARM_THRESHOLD_MS="$2"; shift 2 ;;
    --cold-threshold-ms) COLD_THRESHOLD_MS="$2"; shift 2 ;;
    --warmup-runs) WARMUP_RUNS="$2"; shift 2 ;;
    --require-metal) REQUIRE_METAL=1; shift ;;
    --no-require-metal) REQUIRE_METAL=0; shift ;;
    --keep-daemon) KEEP_DAEMON=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

for cmd in awk grep mktemp sed tail; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 127
  fi
done

if [[ ! -x "$GOLLEK_BIN" ]]; then
  echo "gollek executable not found or not executable: $GOLLEK_BIN" >&2
  exit 2
fi

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-gguf-fast-bench.XXXXXX")"
cleanup() {
  if [[ "$KEEP_DAEMON" -ne 1 ]]; then
    "$GOLLEK_BIN" __gguf-daemon-stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

strip_ansi() {
  sed -E $'s/\x1B\\[[0-9;?]*[ -/]*[@-~]//g'
}

extract_duration_ms() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "$file" \
    | { grep -oE 'Duration: [0-9]+([,.][0-9]+)?s' || true; } \
    | tail -n 1 \
    | sed -E 's/^Duration: //; s/s$//; s/,/./')"
  if [[ -z "$raw" ]]; then
    echo ""
    return 0
  fi
  awk -v seconds="$raw" 'BEGIN { printf "%.0f", seconds * 1000 }'
}

extract_native_metric_ms() {
  local file="$1"
  local key="$2"
  strip_ansi < "$file" \
    | { grep -oE "${key}=[0-9]+([.][0-9]+)?ms" || true; } \
    | tail -n 1 \
    | sed -E "s/^${key}=//; s/ms$//"
}

extract_native_metric_value() {
  local file="$1"
  local key="$2"
  strip_ansi < "$file" \
    | { grep -oE "${key}=[^,)} ]+" || true; } \
    | tail -n 1 \
    | sed -E "s/^${key}=//"
}

extract_backend() {
  local file="$1"
  strip_ansi < "$file" \
    | { grep -oE 'backend=[^,)]+(\([^)]*\))?' || true; } \
    | head -n 1 \
    | sed 's/^backend=//'
}

is_metal_backend() {
  local backend="$1"
  [[ "$backend" == *Metal* || "$backend" == *MTL* ]]
}

run_case() {
  local name="$1"
  local output="$RUN_DIR/${name}.log"
  GOLLEK_GGUF_FAST_RUN_TIMING=true "$GOLLEK_BIN" run \
    --model "$MODEL_ID" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0 \
    --top-k 1 \
    --top-p 1 \
    --provider gguf \
    --engine auto >"$output" 2>&1
  RUN_CASE_OUTPUT="$output"
}

assert_case() {
  local name="$1"
  local output="$2"
  local threshold_ms="$3"
  local require_warm_state="${4:-0}"
  local duration_ms backend model_load_ms prefill_ms decode_ms warm_session prompt_cache

  if grep -E 'GGML_ASSERT|FATAL:|Exception|ERROR:' "$output" >/dev/null; then
    echo "FAIL: $name produced runtime errors" >&2
    tail -n 80 "$output" >&2
    exit 1
  fi
  if ! strip_ansi < "$output" | grep -Eiq "$EXPECTED_REGEX"; then
    echo "FAIL: $name answer did not match expected regex: $EXPECTED_REGEX" >&2
    tail -n 80 "$output" >&2
    exit 1
  fi

  backend="$(extract_backend "$output")"
  if [[ "$REQUIRE_METAL" -eq 1 ]] && ! is_metal_backend "$backend"; then
    echo "FAIL: $name did not use Metal backend (backend=${backend:-missing})" >&2
    tail -n 80 "$output" >&2
    exit 1
  fi

  duration_ms="$(extract_duration_ms "$output")"
  if [[ -z "$duration_ms" ]]; then
    echo "FAIL: $name did not report duration" >&2
    tail -n 80 "$output" >&2
    exit 1
  fi
  if (( duration_ms > threshold_ms )); then
    echo "FAIL: $name took ${duration_ms}ms, threshold is ${threshold_ms}ms" >&2
    tail -n 80 "$output" >&2
    exit 1
  fi

  if [[ "$require_warm_state" -eq 1 ]]; then
    warm_session="$(extract_native_metric_value "$output" warmSession || true)"
    prompt_cache="$(extract_native_metric_value "$output" promptCache || true)"
    if [[ "$warm_session" != "true" ]]; then
      echo "FAIL: $name did not reuse the GGUF daemon session (warmSession=$warm_session)" >&2
      tail -n 80 "$output" >&2
      exit 1
    fi
    if [[ "$prompt_cache" != "hit" ]]; then
      echo "FAIL: $name did not hit the GGUF prompt cache (promptCache=$prompt_cache)" >&2
      tail -n 80 "$output" >&2
      exit 1
    fi
  fi

  model_load_ms="$(extract_native_metric_ms "$output" modelLoad || true)"
  prefill_ms="$(extract_native_metric_ms "$output" prefill || true)"
  decode_ms="$(extract_native_metric_ms "$output" decode || true)"
  warm_session="$(extract_native_metric_value "$output" warmSession || true)"
  prompt_cache="$(extract_native_metric_value "$output" promptCache || true)"
  printf '%-8s duration=%sms backend=%s warmSession=%s promptCache=%s modelLoad=%sms prefill=%sms decode=%sms log=%s\n' \
    "$name" "$duration_ms" "${backend:-unknown}" "${warm_session:-n/a}" "${prompt_cache:-n/a}" "${model_load_ms:-n/a}" "${prefill_ms:-n/a}" "${decode_ms:-n/a}" "$output"
}

"$GOLLEK_BIN" __gguf-daemon-stop >/dev/null 2>&1 || true

cold_log=""
RUN_CASE_OUTPUT=""
run_case cold
cold_log="$RUN_CASE_OUTPUT"
assert_case cold "$cold_log" "$COLD_THRESHOLD_MS"

for ((i = 1; i <= WARMUP_RUNS; i++)); do
  RUN_CASE_OUTPUT=""
  run_case "warmup-${i}"
  warmup_log="$RUN_CASE_OUTPUT"
  assert_case "warmup${i}" "$warmup_log" "$COLD_THRESHOLD_MS"
done

RUN_CASE_OUTPUT=""
run_case warm
warm_log="$RUN_CASE_OUTPUT"
assert_case warm "$warm_log" "$WARM_THRESHOLD_MS" 1

echo "PASS: GGUF fast path benchmark passed"
