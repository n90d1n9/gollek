#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bench-safetensor-inference.sh --model MODEL [options]

Run a repeatable local benchmark for safetensor text inference using the
installed `gollek` CLI and emit raw logs plus a compact JSON/CSV summary.

Options:
  --model ID              Model id or local id alias (required)
  --gollek-bin PATH       Gollek executable (default: ~/.local/bin/gollek)
  --out-dir PATH          Output root (default: gollek/ops/benchmarks/safetensor)
  --label NAME            Optional run label
  --det-prompt TEXT       Deterministic prompt
                          (default: "where is jakarta")
  --normal-prompt TEXT    Natural prompt
                          (default: "Tell me briefly where Jakarta is located.")
  --max-tokens N          Max tokens per run (default: 3)
  --cases LIST            Comma-separated case ids to run
                          (example: metal-deterministic,cpu-deterministic)
  --quick                 Deterministic-only benchmark plan
                          (metal first, plus CPU/turbo variants if requested)
  --profile               Enable -Dgollek.profile=true (default: off)
  --with-cpu              Include CPU reference run
  --with-quantize-turbo   Include Metal and CPU turbo-quantized runs
  --help                  Show this help

Examples:
  ./gollek/scripts/bench-safetensor-inference.sh --model 6f469a --with-cpu
  ./gollek/scripts/bench-safetensor-inference.sh --model 1a008d --with-cpu --with-quantize-turbo
USAGE
}

MODEL_ID=""
GOLLEK_BIN="${HOME}/.local/bin/gollek"
OUT_DIR="gollek/ops/benchmarks/safetensor"
LABEL=""
DET_PROMPT="where is jakarta"
NORMAL_PROMPT="Tell me briefly where Jakarta is located."
MAX_TOKENS=3
PROFILE_ENABLED=0
WITH_CPU=0
WITH_QUANT_TURBO=0
HEARTBEAT_SECONDS=10
CASES_FILTER=""
QUICK_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --gollek-bin) GOLLEK_BIN="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --det-prompt) DET_PROMPT="$2"; shift 2 ;;
    --normal-prompt) NORMAL_PROMPT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --cases) CASES_FILTER="$2"; shift 2 ;;
    --quick) QUICK_MODE=1; shift ;;
    --profile) PROFILE_ENABLED=1; shift ;;
    --with-cpu) WITH_CPU=1; shift ;;
    --with-quantize-turbo) WITH_QUANT_TURBO=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MODEL_ID}" ]]; then
  echo "--model is required" >&2
  exit 2
fi

for cmd in date mkdir mktemp awk sed grep jq; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 127
  fi
done

if [[ ! -x "${GOLLEK_BIN}" ]]; then
  echo "gollek executable not found or not executable: ${GOLLEK_BIN}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${LABEL}" ]]; then
  LABEL="safetensor-${MODEL_ID//\//_}-${timestamp}"
fi
RUN_DIR="${OUT_DIR%/}/${LABEL}"
mkdir -p "${RUN_DIR}/logs"

jsonl_file="$(mktemp)"
touch "${jsonl_file}"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
REPORT_TXT="${RUN_DIR}/report.txt"

jsonl_objects_only() {
  local file="$1"
  grep -E '^[[:space:]]*\{' "${file}" || true
}

generate_outputs() {
  local objects_file
  objects_file="$(mktemp)"
  jsonl_objects_only "${jsonl_file}" > "${objects_file}"

  jq -s \
    --arg generated_at "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --arg model "${MODEL_ID}" \
    --arg label "${LABEL}" \
    '{
      generated_at_utc: $generated_at,
      label: $label,
      model: $model,
      runs: .
    }' "${objects_file}" > "${SUMMARY_JSON}"

  {
    echo "case_id,mode,prompt_kind,quantize,status,exit_code,duration_s,speed_tps,chunks,answer,profile_summary,fatal_line,log_file"
    jq -r '.[] | [
      .case_id,
      .mode,
      .prompt_kind,
      .quantize,
      .status,
      .exit_code,
      (.duration_s // ""),
      (.speed_tps // ""),
      (.chunks // ""),
      (.answer // "" | gsub("\""; "\"\"")),
      (.profile_summary // "" | gsub("\""; "\"\"")),
      (.fatal_line // "" | gsub("\""; "\"\"")),
      .log_file
    ] | @csv' "${objects_file}"
  } > "${SUMMARY_CSV}"

  {
    echo "label=${LABEL}"
    echo "generated_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "model=${MODEL_ID}"
    echo
    printf "%-26s %-8s %-14s %-10s %-8s %-12s %-10s %-8s %s\n" \
      "case_id" "mode" "prompt_kind" "quantize" "status" "duration_s" "speed_tps" "chunks" "answer"
    printf "%-26s %-8s %-14s %-10s %-8s %-12s %-10s %-8s %s\n" \
      "--------------------------" "--------" "--------------" "----------" "--------" "------------" "----------" "--------" "------"
    jq -r '.[] | [
      .case_id,
      .mode,
      .prompt_kind,
      .quantize,
      .status,
      (.duration_s // "n/a"),
      (.speed_tps // "n/a"),
      (.chunks // "n/a"),
      (.answer // "")
    ] | @tsv' "${objects_file}" | while IFS=$'\t' read -r case_id mode prompt_kind quantize status duration speed chunks answer; do
      printf "%-26s %-8s %-14s %-10s %-8s %-12s %-10s %-8s %s\n" \
        "${case_id}" "${mode}" "${prompt_kind}" "${quantize}" "${status}" "${duration}" "${speed}" "${chunks}" "${answer}"
    done
  } > "${REPORT_TXT}"

  rm -f "${objects_file}"
}

cleanup() {
  generate_outputs
  rm -f "${jsonl_file}"
}

on_interrupt() {
  echo
  echo "[bench] interrupted; partial results saved:"
  echo "[bench]   - ${REPORT_TXT}"
  echo "[bench]   - ${SUMMARY_JSON}"
  echo "[bench]   - ${SUMMARY_CSV}"
  exit 130
}

trap on_interrupt INT TERM
trap cleanup EXIT

extract_duration_seconds() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "${file}" | grep -oE 'Duration: [0-9]+,[0-9]+s|Duration: [0-9]+\.[0-9]+s|Duration: [0-9]+s' || true)"
  raw="$(printf '%s\n' "${raw}" | tail -n 1 | sed -E 's/^Duration: //; s/s$//')"
  raw="${raw/,/.}"
  if [[ -n "${raw}" ]]; then
    printf '%s' "${raw}"
  fi
}

extract_speed() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "${file}" | grep -oE 'Speed: [0-9]+,[0-9]+ t/s|Speed: [0-9]+\.[0-9]+ t/s|Speed: [0-9]+ t/s' || true)"
  raw="$(printf '%s\n' "${raw}" | tail -n 1 | sed -E 's/^Speed: //; s/ t\/s$//')"
  raw="${raw/,/.}"
  if [[ -n "${raw}" ]]; then
    printf '%s' "${raw}"
  fi
}

extract_chunks() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "${file}" | grep -oE 'Chunks: [0-9]+' || true)"
  printf '%s\n' "${raw}" | tail -n 1 | awk '{print $2}'
}

extract_profile_summary() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "${file}" | grep '^\[PROFILE\]' || true)"
  printf '%s\n' "${raw}" | tail -n 1 | sed 's/^\[PROFILE\] //'
}

extract_fatal_line() {
  local file="$1"
  local raw
  raw="$(strip_ansi < "${file}" | grep '^\[FATAL\]' || true)"
  printf '%s\n' "${raw}" | tail -n 1
}

extract_answer() {
  local file="$1"
  strip_ansi < "${file}" | awk '
    /--------------------------------------------------/ {capture=1; next}
    capture && /^\[PROFILE\]/ {capture=0}
    capture && /^\[Chunks:/ {capture=0}
    capture && /^Platform: / {next}
    capture && /^✓ GPU acceleration enabled/ {next}
    capture && /^⚡ / {next}
    capture && /^   / {next}
    capture && /^Add --quantize turbo/ {next}
    capture && /^Example: / {next}
    capture && /^Using local model: / {next}
    capture && /^Model: / {next}
    capture && /^Provider: / {next}
    capture && NF {print}
  ' | sed '/^[[:space:]]*$/d' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
}

strip_ansi() {
  sed -E $'s/\x1B\\[[0-9;]*[A-Za-z]//g'
}

run_case() {
  local case_id="$1"
  local mode="$2"
  local prompt_kind="$3"
  local prompt_text="$4"
  local quant_strategy="${5:-}"
  local log_file="${RUN_DIR}/logs/${case_id}.log"
  local status="passed"
  local exit_code=0
  local heartbeat_pid=""

  local -a cmd=( "${GOLLEK_BIN}" )
  local java_opts=()
  if (( PROFILE_ENABLED == 1 )); then
    java_opts+=( "-Dgollek.profile=true" )
  fi

  if [[ "${mode}" == "cpu" ]]; then
    cmd+=( "--use-cpu" )
  fi

  cmd+=( run --model "${MODEL_ID}" --prompt "${prompt_text}" --max-tokens "${MAX_TOKENS}" --temperature 0 --top-k 1 --top-p 1 --direct )

  if [[ -n "${quant_strategy}" ]]; then
    cmd+=( --quantize "${quant_strategy}" )
  fi

  echo "[bench] running ${case_id} ..."
  echo "[bench] log: ${log_file}"
  set +e
  (
    local elapsed=0
    while true; do
      sleep "${HEARTBEAT_SECONDS}" || exit 0
      elapsed=$((elapsed + HEARTBEAT_SECONDS))
      echo "[bench] ${case_id} still running (${elapsed}s elapsed)"
    done
  ) &
  heartbeat_pid=$!
  {
    echo "case_id=${case_id}"
    echo "mode=${mode}"
    echo "prompt_kind=${prompt_kind}"
    echo "quantize=${quant_strategy:-none}"
    echo "command=${cmd[*]}"
    echo
    if (( ${#java_opts[@]} > 0 )); then
      GOLLEK_JAVA_OPTS="${java_opts[*]}" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
  } > "${log_file}" 2>&1
  exit_code=$?
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" >/dev/null 2>&1 || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi

  local duration speed chunks answer profile fatal_line
  duration="$(extract_duration_seconds "${log_file}")"
  speed="$(extract_speed "${log_file}")"
  chunks="$(extract_chunks "${log_file}")"
  answer="$(extract_answer "${log_file}")"
  profile="$(extract_profile_summary "${log_file}")"
  fatal_line="$(extract_fatal_line "${log_file}")"

  if [[ "${status}" == "failed" ]]; then
    echo "[bench] ${case_id} failed (exit=${exit_code})"
    if [[ -n "${fatal_line}" ]]; then
      echo "[bench] ${fatal_line}"
    fi
    echo "[bench] log: ${log_file}"
  else
    echo "[bench] ${case_id} done: duration=${duration:-n/a}s speed=${speed:-n/a} t/s"
  fi

  jq -nc \
    --arg case_id "${case_id}" \
    --arg model "${MODEL_ID}" \
    --arg mode "${mode}" \
    --arg prompt_kind "${prompt_kind}" \
    --arg prompt "${prompt_text}" \
    --arg quantize "${quant_strategy:-none}" \
    --arg duration_s "${duration:-}" \
    --arg speed_tps "${speed:-}" \
    --arg chunks "${chunks:-}" \
    --arg answer "${answer:-}" \
    --arg profile "${profile:-}" \
    --arg status "${status}" \
    --arg fatal_line "${fatal_line:-}" \
    --argjson exit_code "${exit_code}" \
    --arg log_file "${log_file}" \
    '{
      case_id: $case_id,
      model: $model,
      mode: $mode,
      prompt_kind: $prompt_kind,
      prompt: $prompt,
      quantize: $quantize,
      status: $status,
      exit_code: $exit_code,
      duration_s: (if $duration_s == "" then null else ($duration_s | tonumber) end),
      speed_tps: (if $speed_tps == "" then null else ($speed_tps | tonumber) end),
      chunks: (if $chunks == "" then null else ($chunks | tonumber) end),
      answer: $answer,
      profile_summary: $profile,
      fatal_line: $fatal_line,
      log_file: $log_file
    }' >> "${jsonl_file}"

  generate_outputs
}

declare -a CASE_PLAN=()

if (( QUICK_MODE == 1 )); then
  CASE_PLAN+=(
    "metal-deterministic|metal|deterministic|${DET_PROMPT}|"
  )
  if (( WITH_CPU == 1 )); then
    CASE_PLAN+=(
      "cpu-deterministic|cpu|deterministic|${DET_PROMPT}|"
    )
  fi
  if (( WITH_QUANT_TURBO == 1 )); then
    CASE_PLAN+=(
      "metal-deterministic-turbo|metal|deterministic|${DET_PROMPT}|turbo"
    )
    if (( WITH_CPU == 1 )); then
      CASE_PLAN+=(
        "cpu-deterministic-turbo|cpu|deterministic|${DET_PROMPT}|turbo"
      )
    fi
  fi
else
  CASE_PLAN+=(
    "metal-deterministic|metal|deterministic|${DET_PROMPT}|"
    "metal-normal|metal|normal|${NORMAL_PROMPT}|"
  )

  if (( WITH_CPU == 1 )); then
    CASE_PLAN+=(
      "cpu-deterministic|cpu|deterministic|${DET_PROMPT}|"
      "cpu-normal|cpu|normal|${NORMAL_PROMPT}|"
    )
  fi

  if (( WITH_QUANT_TURBO == 1 )); then
    CASE_PLAN+=(
      "metal-deterministic-turbo|metal|deterministic|${DET_PROMPT}|turbo"
      "metal-normal-turbo|metal|normal|${NORMAL_PROMPT}|turbo"
    )
    if (( WITH_CPU == 1 )); then
      CASE_PLAN+=(
        "cpu-deterministic-turbo|cpu|deterministic|${DET_PROMPT}|turbo"
        "cpu-normal-turbo|cpu|normal|${NORMAL_PROMPT}|turbo"
      )
    fi
  fi
fi

echo "[bench] label: ${LABEL}"
echo "[bench] output: ${RUN_DIR}"
echo "[bench] cases: ${#CASE_PLAN[@]}"
echo "[bench] quick_mode: $([[ ${QUICK_MODE} -eq 1 ]] && echo on || echo off)"
echo "[bench] note: large Gemma runs can take several minutes per case"
for case_spec in "${CASE_PLAN[@]}"; do
  IFS='|' read -r case_id mode prompt_kind _prompt_text quant_strategy <<< "${case_spec}"
  echo "[bench] plan: ${case_id} mode=${mode} prompt=${prompt_kind} quantize=${quant_strategy:-none}"
done

if [[ -n "${CASES_FILTER}" ]]; then
  IFS=',' read -r -a allowed_cases <<< "${CASES_FILTER}"
  declare -a FILTERED_CASE_PLAN=()
  for case_spec in "${CASE_PLAN[@]}"; do
    IFS='|' read -r case_id _mode _prompt_kind _prompt_text _quant_strategy <<< "${case_spec}"
    for allowed_case in "${allowed_cases[@]}"; do
      if [[ "${case_id}" == "${allowed_case}" ]]; then
        FILTERED_CASE_PLAN+=( "${case_spec}" )
        break
      fi
    done
  done
  CASE_PLAN=( "${FILTERED_CASE_PLAN[@]}" )
fi

if (( ${#CASE_PLAN[@]} == 0 )); then
  echo "No cases selected to run" >&2
  exit 2
fi

echo "[bench] selected cases: ${#CASE_PLAN[@]}"
for case_spec in "${CASE_PLAN[@]}"; do
  IFS='|' read -r case_id mode prompt_kind _prompt_text quant_strategy <<< "${case_spec}"
  echo "[bench] selected: ${case_id} mode=${mode} prompt=${prompt_kind} quantize=${quant_strategy:-none}"
done

for case_spec in "${CASE_PLAN[@]}"; do
  IFS='|' read -r case_id mode prompt_kind prompt_text quant_strategy <<< "${case_spec}"
  run_case "${case_id}" "${mode}" "${prompt_kind}" "${prompt_text}" "${quant_strategy}"
done

echo "Safetensor benchmark generated:"
echo "  - ${REPORT_TXT}"
echo "  - ${SUMMARY_JSON}"
echo "  - ${SUMMARY_CSV}"
