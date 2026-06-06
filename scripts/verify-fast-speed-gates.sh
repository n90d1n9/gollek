#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: verify-fast-speed-gates.sh [options]

Run the installed fast inference speed gates and collect TSV summaries, a
normalized rollup, backend proof and contract reports, ranked slowest-case
report, configuration, command argv, stdout, and stderr in one directory. This
wraps the LiteRT fast gate, GGUF fast gate, and GGUF Java-native-vs-llama.cpp
comparison gate.

Options:
  --only MODE        auto|all|both|litert|gguf|gguf-compare (default: all)
  --summary-dir DIR  Directory for TSV summary artifacts (default: temp dir)
  --gollek-bin PATH  Gollek executable (default: ~/.local/bin/gollek)
  --litert-model ID  LiteRT model id or alias (default: 7c51c9)
  --gguf-model ID    GGUF model id or alias (default: b71c9d)
  --prompt TEXT      Prompt shared by all gates (default: "where is jakarta")
  --litert-expected REGEX
                     Expected LiteRT answer regex (default: Jakarta|Indonesia)
  --gguf-expected REGEX
                     Expected GGUF answer regex (default: Indonesia|Jakarta)
  --require-metal    Require Metal backend (default on macOS)
  --no-require-metal Do not require Metal backend
  --warm-only        Require already-warm LiteRT/GGUF daemons and skip cold runs
  --keep-daemon      Leave LiteRT/GGUF daemons running after checks
  --slowest-limit N  Number of slowest rows to print and write (default: 5)
  --continue-on-failure
                     Run remaining selected gates after a failure, then exit non-zero
  --help             Show this help

Environment overrides:
  GOLLEK_VERIFY_FAST_{LITERT,GGUF,GGUF_COMPARE}_BENCH
  GOLLEK_VERIFY_LITERT_{MAX_TOKENS,WARM_THRESHOLD_MS,COLD_THRESHOLD_MS,WARMUP_RUNS}
  GOLLEK_VERIFY_LITERT_WARM_ONLY=true
  GOLLEK_VERIFY_LITERT_WARM_{ENGINE_INIT,FIRST_CHUNK,TOTAL}_THRESHOLD_MS
  GOLLEK_VERIFY_GGUF_{MAX_TOKENS,WARM_THRESHOLD_MS,COLD_THRESHOLD_MS,WARMUP_RUNS}
  GOLLEK_VERIFY_GGUF_WARM_ONLY=true
  GOLLEK_VERIFY_GGUF_WARM_{TOKENIZE,PREFILL,DECODE}_THRESHOLD_MS
  GOLLEK_VERIFY_GGUF_COMPARE_{MAX_TOKENS,THRESHOLD_MS,JAVA_MATVEC_THRESHOLD_MS}
  GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_{PREFILL,DECODE}_THRESHOLD_MS
  GOLLEK_VERIFY_GGUF_COMPARE_JAVA_{READY,CONFIG,PROBE}_REGEX
  GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL=false
  GOLLEK_VERIFY_SLOWEST_LIMIT=5
  GOLLEK_VERIFY_CONTINUE_ON_FAILURE=true
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONLY="all"
SUMMARY_DIR=""
GOLLEK_BIN="${HOME}/.local/bin/gollek"
LITERT_MODEL="7c51c9"
GGUF_MODEL="b71c9d"
PROMPT="${GOLLEK_VERIFY_PROMPT:-where is jakarta}"
LITERT_EXPECTED="${GOLLEK_VERIFY_LITERT_EXPECTED:-Jakarta|Indonesia}"
GGUF_EXPECTED="${GOLLEK_VERIFY_GGUF_EXPECTED:-Indonesia|Jakarta}"
KEEP_DAEMON=0
WARM_ONLY=0
SLOWEST_LIMIT="${GOLLEK_VERIFY_SLOWEST_LIMIT:-5}"
CONTINUE_ON_FAILURE="${GOLLEK_VERIFY_CONTINUE_ON_FAILURE:-false}"
case "$(uname -s)" in
  Darwin) REQUIRE_METAL=1 ;;
  *) REQUIRE_METAL=0 ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --only=*) ONLY="${1#*=}"; shift ;;
    --summary-dir) SUMMARY_DIR="$2"; shift 2 ;;
    --summary-dir=*) SUMMARY_DIR="${1#*=}"; shift ;;
    --gollek-bin) GOLLEK_BIN="$2"; shift 2 ;;
    --gollek-bin=*) GOLLEK_BIN="${1#*=}"; shift ;;
    --litert-model) LITERT_MODEL="$2"; shift 2 ;;
    --litert-model=*) LITERT_MODEL="${1#*=}"; shift ;;
    --gguf-model) GGUF_MODEL="$2"; shift 2 ;;
    --gguf-model=*) GGUF_MODEL="${1#*=}"; shift ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --prompt=*) PROMPT="${1#*=}"; shift ;;
    --litert-expected) LITERT_EXPECTED="$2"; shift 2 ;;
    --litert-expected=*) LITERT_EXPECTED="${1#*=}"; shift ;;
    --gguf-expected) GGUF_EXPECTED="$2"; shift 2 ;;
    --gguf-expected=*) GGUF_EXPECTED="${1#*=}"; shift ;;
    --require-metal) REQUIRE_METAL=1; shift ;;
    --no-require-metal) REQUIRE_METAL=0; shift ;;
    --warm-only) WARM_ONLY=1; shift ;;
    --keep-daemon) KEEP_DAEMON=1; shift ;;
    --slowest-limit) SLOWEST_LIMIT="$2"; shift 2 ;;
    --slowest-limit=*) SLOWEST_LIMIT="${1#*=}"; shift ;;
    --continue-on-failure) CONTINUE_ON_FAILURE=true; shift ;;
    --stop-on-failure) CONTINUE_ON_FAILURE=false; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

case "$ONLY" in
  auto|all|both|litert|gguf|gguf-compare) ;;
  *) echo "Unknown --only mode: $ONLY" >&2; usage; exit 2 ;;
esac
if [[ ! "$SLOWEST_LIMIT" =~ ^[0-9]+$ ]] || (( SLOWEST_LIMIT < 1 )); then
  echo "Invalid --slowest-limit value: $SLOWEST_LIMIT" >&2
  usage
  exit 2
fi

for cmd in awk mkdir mktemp tr; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 127
  fi
done

if [[ ! -x "$GOLLEK_BIN" ]]; then
  echo "gollek executable not found or not executable: $GOLLEK_BIN" >&2
  exit 2
fi

if [[ -z "$SUMMARY_DIR" ]]; then
  SUMMARY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-fast-speed-gates.XXXXXX")"
else
  mkdir -p "$SUMMARY_DIR"
fi

LITERT_BENCH="${GOLLEK_VERIFY_FAST_LITERT_BENCH:-${ROOT_DIR}/scripts/bench-litert-fast-run.sh}"
GGUF_BENCH="${GOLLEK_VERIFY_FAST_GGUF_BENCH:-${ROOT_DIR}/scripts/bench-gguf-fast-run.sh}"
GGUF_COMPARE_BENCH="${GOLLEK_VERIFY_FAST_GGUF_COMPARE_BENCH:-${ROOT_DIR}/scripts/bench-gguf-engine-compare.sh}"
MANIFEST="${SUMMARY_DIR}/manifest.tsv"
RESULTS="${SUMMARY_DIR}/results.tsv"
CONFIG="${SUMMARY_DIR}/config.tsv"
ROLLUP="${SUMMARY_DIR}/rollup.tsv"
BACKEND="${SUMMARY_DIR}/backend.tsv"
CONTRACTS="${SUMMARY_DIR}/contracts.tsv"
SLOWEST="${SUMMARY_DIR}/slowest.tsv"
RAN_ANY=0
FINAL_EXIT_CODE=0
printf 'gate\tsummary\n' > "$MANIFEST"
printf 'gate\tstatus\texitCode\telapsedMs\tsummary\tstdout\tstderr\targv\treason\n' > "$RESULTS"
printf 'gate\tcase\tdurationMs\tbackend\tmetrics\tlog\n' > "$ROLLUP"
printf 'gate\tcase\tbackend\tmetal\twarmReuse\tpromptCache\tjavaFallbackRefusal\tlog\n' > "$BACKEND"
printf 'gate\tcase\tmetalRequired\tmetal\tmetalStatus\twarmRequired\twarmReuse\twarmStatus\tpromptCacheRequired\tpromptCache\tpromptCacheStatus\tjavaFallbackRefusalRequired\tjavaFallbackRefusal\tjavaFallbackRefusalStatus\tlog\n' > "$CONTRACTS"
printf 'rank\tgate\tcase\tdurationMs\tbackend\tmetrics\tlog\n' > "$SLOWEST"

should_run() {
  local gate="$1"
  case "$ONLY" in
    auto)
      case "$gate" in
        litert) model_present "$LITERT_MODEL" ;;
        gguf) model_present "$GGUF_MODEL" ;;
        *) return 1 ;;
      esac
      ;;
    all) return 0 ;;
    both) [[ "$gate" == "litert" || "$gate" == "gguf" ]] ;;
    *) [[ "$gate" == "$ONLY" ]] ;;
  esac
}

skip_reason() {
  local gate="$1"
  case "$ONLY" in
    auto)
      case "$gate" in
        litert)
          model_present "$LITERT_MODEL" && printf 'not-selected' || printf 'model-not-found'
          ;;
        gguf)
          model_present "$GGUF_MODEL" && printf 'not-selected' || printf 'model-not-found'
          ;;
        *) printf 'auto-skips-compare' ;;
      esac
      ;;
    both)
      case "$gate" in
        litert|gguf) printf 'not-selected' ;;
        *) printf 'mode-both-skips-compare' ;;
      esac
      ;;
    *) printf 'not-selected' ;;
  esac
}

model_present() {
  local ref="$1"
  local index_path="${GOLLEK_VERIFY_MODEL_INDEX:-${HOME}/.gollek/models/index.json}"
  [[ -f "$index_path" ]] || return 1
  awk -v ref="$ref" '
    function json_value(line, value) {
      value = line
      sub(/^[^:]*:[[:space:]]*"/, "", value)
      sub(/".*$/, "", value)
      return value
    }
    {
      lower = tolower($0)
      if (lower ~ /"(id|shortid|name|path)"[[:space:]]*:/) {
        value = json_value($0)
        if (value == ref || tolower(value) == tolower(ref)) {
          found = 1
        }
      }
    }
    END { exit found ? 0 : 1 }
  ' "$index_path"
}

truthy_env() {
  case "$(printf '%s' "${1:-false}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

record_manifest() {
  local gate="$1"
  local summary="$2"
  RAN_ANY=1
  printf '%s\t%s\n' "$gate" "$summary" >> "$MANIFEST"
}

record_skip() {
  local gate="$1"
  local reason="$2"
  printf '%s\tskip\t0\t0\t\t\t\t\t%s\n' "$gate" "$reason" >> "$RESULTS"
}

run_gate_command() {
  local gate="$1"
  local summary="$2"
  shift 2
  local stdout_log="${SUMMARY_DIR}/${gate}.out"
  local stderr_log="${SUMMARY_DIR}/${gate}.err"
  local argv_log="${SUMMARY_DIR}/${gate}.argv.tsv"
  local start_seconds="$SECONDS"
  local exit_code elapsed_ms status
  write_argv_log "$argv_log" "$@"
  set +e
  "$@" >"$stdout_log" 2>"$stderr_log"
  exit_code=$?
  set -e
  elapsed_ms=$(( (SECONDS - start_seconds) * 1000 ))
  if [[ "$exit_code" -eq 0 ]]; then
    status="pass"
  else
    status="fail"
  fi
  append_rollup "$gate" "$summary"
  refresh_backend_report
  refresh_contract_report
  refresh_slowest_report
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' "$gate" "$status" "$exit_code" "$elapsed_ms" "$summary" "$stdout_log" "$stderr_log" "$argv_log" >> "$RESULTS"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "FAIL: ${gate} speed gate failed; artifacts are in ${SUMMARY_DIR}; stderr=${stderr_log}" >&2
    if truthy_env "$CONTINUE_ON_FAILURE"; then
      if [[ "$FINAL_EXIT_CODE" -eq 0 ]]; then
        FINAL_EXIT_CODE="$exit_code"
      fi
      return 0
    fi
    print_artifact_footer
    exit "$exit_code"
  fi
}

safe_tsv_field() {
  local value="${1:-}"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  printf '%s' "$value"
}

write_argv_log() {
  local path="$1"
  shift
  local index=0
  printf 'index\targ\n' > "$path"
  while [[ $# -gt 0 ]]; do
    printf '%s\t%s\n' "$index" "$(safe_tsv_field "$1")" >> "$path"
    index=$((index + 1))
    shift
  done
}

append_rollup() {
  local gate="$1"
  local summary="$2"
  [[ -f "$summary" && -s "$summary" ]] || return 0
  awk -v gate="$gate" '
    BEGIN { FS = OFS = "\t" }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        header[i] = $i
      }
      next
    }
    NF > 0 {
      metrics = ""
      for (i = 4; i < NF; i++) {
        if (header[i] == "" || header[i] == "log") {
          continue
        }
        if (metrics != "") {
          metrics = metrics ","
        }
        metrics = metrics header[i] "=" $i
      }
      print gate, $1, $2, $3, metrics, $NF
    }
  ' "$summary" >> "$ROLLUP"
}

refresh_slowest_report() {
  local limit="$SLOWEST_LIMIT"
  awk -v limit="$limit" '
    BEGIN {
      FS = OFS = "\t"
      if (limit !~ /^[0-9]+$/ || limit < 1) {
        limit = 5
      }
      print "rank", "gate", "case", "durationMs", "backend", "metrics", "log"
    }
    NR == 1 {
      next
    }
    NF >= 6 && $3 ~ /^[0-9]+([.][0-9]+)?$/ {
      row_count++
      duration[row_count] = $3 + 0
      row[row_count] = $0
    }
    END {
      max_rank = (row_count < limit) ? row_count : limit
      for (rank = 1; rank <= max_rank; rank++) {
        best = 0
        for (i = 1; i <= row_count; i++) {
          if (used[i]) {
            continue
          }
          if (best == 0 || duration[i] > duration[best]) {
            best = i
          }
        }
        if (best == 0) {
          break
        }
        used[best] = 1
        print rank, row[best]
      }
    }
  ' "$ROLLUP" > "$SLOWEST"
}

refresh_backend_report() {
  awk '
    BEGIN {
      FS = OFS = "\t"
      print "gate", "case", "backend", "metal", "warmReuse", "promptCache", "javaFallbackRefusal", "log"
    }
    function metric_value(metrics, key, parts, count, i, prefix) {
      count = split(metrics, parts, ",")
      prefix = key "="
      for (i = 1; i <= count; i++) {
        if (index(parts[i], prefix) == 1) {
          return substr(parts[i], length(prefix) + 1)
        }
      }
      return "n/a"
    }
    NR == 1 {
      next
    }
    NF >= 6 {
      metal = ($4 ~ /(Metal|MTL)/) ? "true" : "false"
      warm = metric_value($5, "warmEngine")
      if (warm == "n/a") {
        warm = metric_value($5, "warmSession")
      }
      prompt_cache = metric_value($5, "promptCache")
      java_refusal = metric_value($5, "javaFallbackRefusal")
      print $1, $2, $4, metal, warm, prompt_cache, java_refusal, $6
    }
  ' "$ROLLUP" > "$BACKEND"
}

refresh_contract_report() {
  local require_metal="false"
  local litert_warm_only="false"
  local gguf_warm_only="false"
  local gguf_prompt_cache="true"
  local compare_java_refusal="true"
  if [[ "$REQUIRE_METAL" -eq 1 ]]; then
    require_metal="true"
  fi
  if [[ "$WARM_ONLY" -eq 1 ]] || truthy_env "${GOLLEK_VERIFY_LITERT_WARM_ONLY:-false}"; then
    litert_warm_only="true"
  fi
  if [[ "$WARM_ONLY" -eq 1 ]] || truthy_env "${GOLLEK_VERIFY_GGUF_WARM_ONLY:-false}"; then
    gguf_warm_only="true"
  fi
  if ! truthy_env "${GOLLEK_VERIFY_GGUF_PROMPT_CACHE:-true}"; then
    gguf_prompt_cache="false"
  fi
  if ! truthy_env "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-true}"; then
    compare_java_refusal="false"
  fi

  awk \
    -v requireMetal="$require_metal" \
    -v litertWarmOnly="$litert_warm_only" \
    -v ggufWarmOnly="$gguf_warm_only" \
    -v ggufPromptCache="$gguf_prompt_cache" \
    -v compareJavaRefusal="$compare_java_refusal" '
    BEGIN {
      FS = OFS = "\t"
      print "gate", "case", "metalRequired", "metal", "metalStatus", "warmRequired", "warmReuse", "warmStatus", "promptCacheRequired", "promptCache", "promptCacheStatus", "javaFallbackRefusalRequired", "javaFallbackRefusal", "javaFallbackRefusalStatus", "log"
    }
    function bool_status(required, actual) {
      if (required != "true") {
        return "n/a"
      }
      return (actual == "true") ? "pass" : "fail"
    }
    function prompt_cache_status(required, actual) {
      if (required != "true") {
        return "n/a"
      }
      return (actual == "hit" || actual == "prefix-hit") ? "pass" : "fail"
    }
    function java_refusal_status(required, actual) {
      if (required != "true") {
        return "n/a"
      }
      return (actual == "checked") ? "pass" : "fail"
    }
    NR == 1 {
      next
    }
    NF >= 8 {
      gate = $1
      case_name = $2
      metal = $4
      warm = $5
      prompt_cache = $6
      java_refusal = $7
      log_path = $8

      warm_required = "false"
      if (gate == "litert" && (case_name == "warm" || (litertWarmOnly == "true" && case_name ~ /^warmup/))) {
        warm_required = "true"
      }
      if (gate == "gguf" && (case_name == "warm" || (ggufWarmOnly == "true" && case_name ~ /^warmup/))) {
        warm_required = "true"
      }

      prompt_required = "false"
      if (gate == "gguf" && ggufPromptCache == "true" && (case_name == "warm" || case_name == "warmup1" || (ggufWarmOnly == "true" && case_name ~ /^warmup/))) {
        prompt_required = "true"
      }

      java_required = "false"
      if (gate == "gguf-compare" && compareJavaRefusal == "true") {
        java_required = "true"
      }

      print gate, case_name, requireMetal, metal, bool_status(requireMetal, metal), warm_required, warm, bool_status(warm_required, warm), prompt_required, prompt_cache, prompt_cache_status(prompt_required, prompt_cache), java_required, java_refusal, java_refusal_status(java_required, java_refusal), log_path
    }
  ' "$BACKEND" > "$CONTRACTS"
}

print_artifact_footer() {
  echo "summaryDir=$SUMMARY_DIR"
  echo "artifacts.config=$CONFIG"
  echo "artifacts.results=$RESULTS"
  echo "artifacts.rollup=$ROLLUP"
  echo "artifacts.backend=$BACKEND"
  echo "artifacts.contracts=$CONTRACTS"
  echo "artifacts.slowest=$SLOWEST"
  if awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }' "$BACKEND"; then
    echo "backend:"
    awk '
      BEGIN { FS = "\t" }
      NR == 1 { next }
      {
        printf "  %s/%s metal=%s warmReuse=%s promptCache=%s javaFallbackRefusal=%s backend=%s log=%s\n", $1, $2, $4, $5, $6, $7, $3, $8
      }
    ' "$BACKEND"
  fi
  if awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }' "$CONTRACTS"; then
    echo "contracts:"
    awk '
      BEGIN { FS = "\t" }
      NR == 1 { next }
      {
        printf "  %s/%s metal=%s warm=%s promptCache=%s javaRefusal=%s log=%s\n", $1, $2, $5, $8, $11, $14, $15
      }
    ' "$CONTRACTS"
  fi
  if awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }' "$SLOWEST"; then
    echo "slowest:"
    awk '
      BEGIN { FS = "\t" }
      NR == 1 { next }
      {
        printf "  #%s %s/%s duration=%sms backend=%s log=%s\n", $1, $2, $3, $4, $5, $7
      }
    ' "$SLOWEST"
  fi
}

config_row() {
  printf '%s\t%s\n' "$(safe_tsv_field "$1")" "$(safe_tsv_field "${2:-}")" >> "$CONFIG"
}

write_config() {
  printf 'key\tvalue\n' > "$CONFIG"
  config_row rootDir "$ROOT_DIR"
  config_row platform "$(uname -s)"
  config_row only "$ONLY"
  config_row summaryDir "$SUMMARY_DIR"
  config_row gollekBin "$GOLLEK_BIN"
  config_row litert.model "$LITERT_MODEL"
  config_row gguf.model "$GGUF_MODEL"
  config_row prompt "$PROMPT"
  config_row litert.expected "$LITERT_EXPECTED"
  config_row gguf.expected "$GGUF_EXPECTED"
  config_row requireMetal "$([[ "$REQUIRE_METAL" -eq 1 ]] && printf true || printf false)"
  config_row warmOnly "$([[ "$WARM_ONLY" -eq 1 ]] && printf true || printf false)"
  config_row keepDaemon "$([[ "$KEEP_DAEMON" -eq 1 ]] && printf true || printf false)"
  config_row continueOnFailure "$CONTINUE_ON_FAILURE"
  config_row slowestLimit "$SLOWEST_LIMIT"
  config_row modelIndex "${GOLLEK_VERIFY_MODEL_INDEX:-${HOME}/.gollek/models/index.json}"
  config_row litert.bench "$LITERT_BENCH"
  config_row gguf.bench "$GGUF_BENCH"
  config_row ggufCompare.bench "$GGUF_COMPARE_BENCH"
  config_row litert.maxTokens "${GOLLEK_VERIFY_LITERT_MAX_TOKENS:-10}"
  config_row litert.warmThresholdMs "${GOLLEK_VERIFY_LITERT_WARM_THRESHOLD_MS:-1500}"
  config_row litert.coldThresholdMs "${GOLLEK_VERIFY_LITERT_COLD_THRESHOLD_MS:-60000}"
  config_row litert.warmupRuns "${GOLLEK_VERIFY_LITERT_WARMUP_RUNS:-1}"
  config_row litert.warmOnly "${GOLLEK_VERIFY_LITERT_WARM_ONLY:-false}"
  config_row litert.warmEngineInitThresholdMs "${GOLLEK_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS:-}"
  config_row litert.warmFirstChunkThresholdMs "${GOLLEK_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS:-}"
  config_row litert.warmProfileTotalThresholdMs "${GOLLEK_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS:-}"
  config_row gguf.maxTokens "${GOLLEK_VERIFY_GGUF_MAX_TOKENS:-1}"
  config_row gguf.warmThresholdMs "${GOLLEK_VERIFY_GGUF_WARM_THRESHOLD_MS:-1500}"
  config_row gguf.coldThresholdMs "${GOLLEK_VERIFY_GGUF_COLD_THRESHOLD_MS:-60000}"
  config_row gguf.warmupRuns "${GOLLEK_VERIFY_GGUF_WARMUP_RUNS:-1}"
  config_row gguf.warmOnly "${GOLLEK_VERIFY_GGUF_WARM_ONLY:-false}"
  config_row gguf.promptCache "${GOLLEK_VERIFY_GGUF_PROMPT_CACHE:-true}"
  config_row gguf.warmTokenizeThresholdMs "${GOLLEK_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS:-}"
  config_row gguf.warmPrefillThresholdMs "${GOLLEK_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS:-}"
  config_row gguf.warmDecodeThresholdMs "${GOLLEK_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS:-}"
  config_row ggufCompare.maxTokens "${GOLLEK_VERIFY_GGUF_COMPARE_MAX_TOKENS:-10}"
  config_row ggufCompare.thresholdMs "${GOLLEK_VERIFY_GGUF_COMPARE_THRESHOLD_MS:-10000}"
  config_row ggufCompare.javaMatvecThresholdMs "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS:-50}"
  config_row ggufCompare.fallbackPrefillThresholdMs "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS:-}"
  config_row ggufCompare.fallbackDecodeThresholdMs "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS:-}"
  config_row ggufCompare.javaReadyRegex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX:-}"
  config_row ggufCompare.javaConfigRegex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX:-}"
  config_row ggufCompare.javaProbeRegex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX:-}"
  config_row ggufCompare.javaRefusal "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-true}"
}

run_litert_gate() {
  local summary="${SUMMARY_DIR}/litert-fast.tsv"
  local cmd=(
    bash "$LITERT_BENCH"
    --gollek-bin "$GOLLEK_BIN"
    --model "$LITERT_MODEL"
    --prompt "$PROMPT"
    --expected "$LITERT_EXPECTED"
    --max-tokens "${GOLLEK_VERIFY_LITERT_MAX_TOKENS:-10}"
    --warm-threshold-ms "${GOLLEK_VERIFY_LITERT_WARM_THRESHOLD_MS:-1500}"
    --cold-threshold-ms "${GOLLEK_VERIFY_LITERT_COLD_THRESHOLD_MS:-60000}"
    --warmup-runs "${GOLLEK_VERIFY_LITERT_WARMUP_RUNS:-1}"
  )
  if [[ "$REQUIRE_METAL" -eq 1 ]]; then
    cmd+=(--require-metal)
  else
    cmd+=(--no-require-metal)
  fi
  if [[ -n "${GOLLEK_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-engine-init-threshold-ms "${GOLLEK_VERIFY_LITERT_WARM_ENGINE_INIT_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-first-chunk-threshold-ms "${GOLLEK_VERIFY_LITERT_WARM_FIRST_CHUNK_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-profile-total-threshold-ms "${GOLLEK_VERIFY_LITERT_WARM_TOTAL_THRESHOLD_MS}")
  fi
  if [[ "$WARM_ONLY" -eq 1 ]] || truthy_env "${GOLLEK_VERIFY_LITERT_WARM_ONLY:-false}"; then
    cmd+=(--warm-only)
  fi
  cmd+=(--summary-file "$summary")
  if [[ "$KEEP_DAEMON" -eq 1 ]]; then
    cmd+=(--keep-daemon)
  fi
  echo ">>> LiteRT fast speed gate"
  record_manifest litert "$summary"
  run_gate_command litert "$summary" "${cmd[@]}"
}

run_gguf_gate() {
  local summary="${SUMMARY_DIR}/gguf-fast.tsv"
  local cmd=(
    bash "$GGUF_BENCH"
    --gollek-bin "$GOLLEK_BIN"
    --model "$GGUF_MODEL"
    --prompt "$PROMPT"
    --expected "$GGUF_EXPECTED"
    --max-tokens "${GOLLEK_VERIFY_GGUF_MAX_TOKENS:-1}"
    --warm-threshold-ms "${GOLLEK_VERIFY_GGUF_WARM_THRESHOLD_MS:-1500}"
    --cold-threshold-ms "${GOLLEK_VERIFY_GGUF_COLD_THRESHOLD_MS:-60000}"
    --warmup-runs "${GOLLEK_VERIFY_GGUF_WARMUP_RUNS:-1}"
  )
  if [[ "$REQUIRE_METAL" -eq 1 ]]; then
    cmd+=(--require-metal)
  else
    cmd+=(--no-require-metal)
  fi
  case "$(printf '%s' "${GOLLEK_VERIFY_GGUF_PROMPT_CACHE:-true}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) ;;
    *) cmd+=(--require-prompt-cache) ;;
  esac
  if [[ -n "${GOLLEK_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-tokenize-threshold-ms "${GOLLEK_VERIFY_GGUF_WARM_TOKENIZE_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-prefill-threshold-ms "${GOLLEK_VERIFY_GGUF_WARM_PREFILL_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS:-}" ]]; then
    cmd+=(--warm-decode-threshold-ms "${GOLLEK_VERIFY_GGUF_WARM_DECODE_THRESHOLD_MS}")
  fi
  if [[ "$WARM_ONLY" -eq 1 ]] || truthy_env "${GOLLEK_VERIFY_GGUF_WARM_ONLY:-false}"; then
    cmd+=(--warm-only)
  fi
  cmd+=(--summary-file "$summary")
  if [[ "$KEEP_DAEMON" -eq 1 ]]; then
    cmd+=(--keep-daemon)
  fi
  echo ">>> GGUF fast speed gate"
  record_manifest gguf "$summary"
  run_gate_command gguf "$summary" "${cmd[@]}"
}

run_gguf_compare_gate() {
  local summary="${SUMMARY_DIR}/gguf-compare.tsv"
  local cmd=(
    bash "$GGUF_COMPARE_BENCH"
    --gollek-bin "$GOLLEK_BIN"
    --model "$GGUF_MODEL"
    --prompt "$PROMPT"
    --expected "$GGUF_EXPECTED"
    --max-tokens "${GOLLEK_VERIFY_GGUF_COMPARE_MAX_TOKENS:-10}"
    --threshold-ms "${GOLLEK_VERIFY_GGUF_COMPARE_THRESHOLD_MS:-10000}"
    --java-matvec-threshold-ms "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_MATVEC_THRESHOLD_MS:-50}"
  )
  if [[ "$REQUIRE_METAL" -eq 1 ]]; then
    cmd+=(--require-metal)
  else
    cmd+=(--no-require-metal)
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS:-}" ]]; then
    cmd+=(--fallback-prefill-threshold-ms "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_PREFILL_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS:-}" ]]; then
    cmd+=(--fallback-decode-threshold-ms "${GOLLEK_VERIFY_GGUF_COMPARE_FALLBACK_DECODE_THRESHOLD_MS}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX:-}" ]]; then
    cmd+=(--java-ready-regex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_READY_REGEX}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX:-}" ]]; then
    cmd+=(--java-config-regex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_CONFIG_REGEX}")
  fi
  if [[ -n "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX:-}" ]]; then
    cmd+=(--java-probe-regex "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_PROBE_REGEX}")
  fi
  case "$(printf '%s' "${GOLLEK_VERIFY_GGUF_COMPARE_JAVA_REFUSAL:-true}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off) cmd+=(--no-verify-java-refusal) ;;
    *) cmd+=(--verify-java-refusal) ;;
  esac
  cmd+=(--summary-file "$summary")
  echo ">>> GGUF Java-native vs llama.cpp speed gate"
  record_manifest gguf-compare "$summary"
  run_gate_command gguf-compare "$summary" "${cmd[@]}"
}

write_config

if should_run litert; then
  run_litert_gate
else
  record_skip litert "$(skip_reason litert)"
fi
if should_run gguf; then
  run_gguf_gate
else
  record_skip gguf "$(skip_reason gguf)"
fi
if should_run gguf-compare; then
  run_gguf_compare_gate
else
  record_skip gguf-compare "$(skip_reason gguf-compare)"
fi

if [[ "$RAN_ANY" -ne 1 ]]; then
  echo "WARN: no fast speed gates matched --only=${ONLY}; summaryDir=${SUMMARY_DIR}" >&2
fi
if [[ "$FINAL_EXIT_CODE" -ne 0 ]]; then
  echo "FAIL: one or more fast speed gates failed; artifacts are in ${SUMMARY_DIR}" >&2
  print_artifact_footer
  exit "$FINAL_EXIT_CODE"
fi
echo "PASS: fast speed gates passed"
print_artifact_footer
