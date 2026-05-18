#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-litert-fast-run.sh [options]

Run a focused smoke/benchmark for the installed LiteRT-LM fast path. The check
starts from a cold daemon, verifies the backend, then verifies warm latency.

Options:
  --model ID              Model id or local alias (default: 7c51c9)
  --prompt TEXT           Prompt to run (default: "where is jakarta")
  --expected REGEX        Expected answer regex (default: Jakarta|Indonesia)
  --max-tokens N          Max generated tokens (default: 10)
  --gollek-bin PATH       Gollek executable (default: ~/.local/bin/gollek)
  --warm-threshold-ms N   Fail if warm duration exceeds this (default: 1500)
  --cold-threshold-ms N   Fail if cold duration exceeds this (default: 60000)
  --warmup-runs N         Stabilization runs before measured warm check (default: 1)
  --require-metal         Require Metal backend
  --no-require-metal      Do not require Metal backend
  --require-warm-engine   Require daemon engine reuse on warm runs (default)
  --no-require-warm-engine
  --keep-daemon           Leave daemon running after benchmark
  --help                  Show this help
USAGE
}

MODEL_ID="7c51c9"
PROMPT="where is jakarta"
EXPECTED_REGEX="Jakarta|Indonesia"
MAX_TOKENS=10
GOLLEK_BIN="${HOME}/.local/bin/gollek"
WARM_THRESHOLD_MS=1500
COLD_THRESHOLD_MS=60000
WARMUP_RUNS=1
KEEP_DAEMON=0
REQUIRE_WARM_ENGINE=1
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
    --require-warm-engine) REQUIRE_WARM_ENGINE=1; shift ;;
    --no-require-warm-engine) REQUIRE_WARM_ENGINE=0; shift ;;
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

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-litert-fast-bench.XXXXXX")"
cleanup() {
  if [[ "$KEEP_DAEMON" -ne 1 ]]; then
    "$GOLLEK_BIN" __daemon-stop >/dev/null 2>&1 || true
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

extract_profile_metric_ms() {
  local file="$1"
  local key="$2"
  strip_ansi < "$file" \
    | { grep -oE "${key}=[0-9]+([.][0-9]+)?s" || true; } \
    | tail -n 1 \
    | sed -E "s/^${key}=//; s/s$//" \
    | awk 'NF { printf "%.0f", $1 * 1000 }'
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

extract_warm_engine() {
  local file="$1"
  strip_ansi < "$file" \
    | { grep -oE 'warmEngine=(true|false)' || true; } \
    | tail -n 1 \
    | sed 's/^warmEngine=//'
}

run_case() {
  local name="$1"
  local output="$RUN_DIR/${name}.log"
  GOLLEK_JAVA_OPTS="${GOLLEK_JAVA_OPTS:-} -Dgollek.litert.fast_run.profile=true" \
    "$GOLLEK_BIN" run \
      --model "$MODEL_ID" \
      --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" \
      --temperature 0 \
      --top-k 1 \
      --top-p 1 >"$output" 2>&1
  RUN_CASE_OUTPUT="$output"
}

assert_case() {
  local name="$1"
  local output="$2"
  local threshold_ms="$3"
  local require_warm_engine="${4:-0}"
  local duration_ms backend engine_ms first_chunk_ms total_ms warm_engine

  if grep -E 'DISPATCH_OP|GGML_ASSERT|FATAL:|Exception|ERROR:' "$output" >/dev/null; then
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

  warm_engine="$(extract_warm_engine "$output")"
  if [[ "$require_warm_engine" -eq 1 && "$REQUIRE_WARM_ENGINE" -eq 1 && "$warm_engine" != "true" ]]; then
    echo "FAIL: $name did not reuse the LiteRT daemon engine (warmEngine=${warm_engine:-missing})" >&2
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

  engine_ms="$(extract_profile_metric_ms "$output" engineInit || true)"
  first_chunk_ms="$(extract_profile_metric_ms "$output" firstChunk || true)"
  total_ms="$(extract_profile_metric_ms "$output" total || true)"
  printf '%-7s duration=%sms backend=%s warmEngine=%s engineInit=%sms firstChunk=%sms total=%sms log=%s\n' \
    "$name" "$duration_ms" "${backend:-unknown}" "${warm_engine:-n/a}" "${engine_ms:-n/a}" "${first_chunk_ms:-n/a}" "${total_ms:-n/a}" "$output"
}

"$GOLLEK_BIN" __daemon-stop >/dev/null 2>&1 || true

cold_log=""
RUN_CASE_OUTPUT=""
run_case cold
cold_log="$RUN_CASE_OUTPUT"
assert_case cold "$cold_log" "$COLD_THRESHOLD_MS" 0

for ((i = 1; i <= WARMUP_RUNS; i++)); do
  RUN_CASE_OUTPUT=""
  run_case "warmup-${i}"
  warmup_log="$RUN_CASE_OUTPUT"
  # Warmup runs are stabilizers: after install or daemon replacement they may
  # be the request that repopulates the engine. The measured warm run below is
  # the hard steady-state daemon-reuse contract.
  assert_case "warmup${i}" "$warmup_log" "$COLD_THRESHOLD_MS" 0
done

RUN_CASE_OUTPUT=""
run_case warm
warm_log="$RUN_CASE_OUTPUT"
assert_case warm "$warm_log" "$WARM_THRESHOLD_MS" 1

echo "PASS: LiteRT-LM fast path benchmark passed"
