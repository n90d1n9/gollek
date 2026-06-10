#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: bench-gemma4-row-prefill-ab-repeat.sh --model MODEL [options] [-- RUNNER_ARGS...]

Run the Gemma4 row-prefill A/B benchmark multiple times, keep each sample's
artifacts, and aggregate the comparison decisions into sample and metric TSVs
with average, median, p95, standard-deviation, and unstable-metric signals.

Options:
  --model ID             Model id or local id alias (required)
  --gollek-bin PATH      Gollek executable (default: ~/.local/bin/gollek)
  --out-dir PATH         Output directory for repeat artifacts
                         (default: gollek/ops/benchmarks/safetensor/gemma4-row-prefill-ab-repeat)
  --repeat N             Number of A/B samples to run (default: 3)
  --runner-script PATH   Single-sample A/B runner path
  --runner-arg ARG       Extra argument passed to each A/B runner; repeatable
  --max-p95-regression-percent N
                         Fail when any metric p95 regression exceeds N percent
  --fail-unstable-metrics
                         Fail when any metric has mixed better/worse samples or high variance
  --help                 Show this help

Any arguments after `--` are forwarded to every single-sample A/B run.

Examples:
  ./scripts/bench-gemma4-row-prefill-ab-repeat.sh --model 53f473 --repeat 3 -- \
    --max-regression-ms 250 --gate-metrics ttftMs,ffnMs
USAGE
}

MODEL_ID=""
GOLLEK_BIN="${HOME}/.local/bin/gollek"
OUT_DIR="gollek/ops/benchmarks/safetensor/gemma4-row-prefill-ab-repeat"
REPEAT_COUNT=3
RUNNER_SCRIPT="${GOLLEK_GEMMA4_ROW_PREFILL_AB_RUNNER_SCRIPT:-${SCRIPT_DIR}/bench-gemma4-row-prefill-ab.sh}"
MAX_P95_REGRESSION_PERCENT=""
FAIL_UNSTABLE_METRICS=0
RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_ID="$2"; shift 2 ;;
    --model=*) MODEL_ID="${1#*=}"; shift ;;
    --gollek-bin) GOLLEK_BIN="$2"; shift 2 ;;
    --gollek-bin=*) GOLLEK_BIN="${1#*=}"; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --out-dir=*) OUT_DIR="${1#*=}"; shift ;;
    --repeat) REPEAT_COUNT="$2"; shift 2 ;;
    --repeat=*) REPEAT_COUNT="${1#*=}"; shift ;;
    --runner-script) RUNNER_SCRIPT="$2"; shift 2 ;;
    --runner-script=*) RUNNER_SCRIPT="${1#*=}"; shift ;;
    --runner-arg) RUNNER_ARGS+=("$2"); shift 2 ;;
    --runner-arg=*) RUNNER_ARGS+=("${1#*=}"); shift ;;
    --max-p95-regression-percent) MAX_P95_REGRESSION_PERCENT="$2"; shift 2 ;;
    --max-p95-regression-percent=*) MAX_P95_REGRESSION_PERCENT="${1#*=}"; shift ;;
    --fail-unstable-metrics) FAIL_UNSTABLE_METRICS=1; shift ;;
    --) shift; RUNNER_ARGS+=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  echo "--model is required" >&2
  usage
  exit 2
fi
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "A/B runner script not found: $RUNNER_SCRIPT" >&2
  exit 2
fi
if [[ ! "$REPEAT_COUNT" =~ ^[0-9]+$ || "$REPEAT_COUNT" -lt 1 ]]; then
  echo "Invalid repeat count: $REPEAT_COUNT" >&2
  exit 2
fi
if [[ -n "$MAX_P95_REGRESSION_PERCENT" && ! "$MAX_P95_REGRESSION_PERCENT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Invalid max p95 regression percent: $MAX_P95_REGRESSION_PERCENT" >&2
  exit 2
fi
for cmd in awk date mkdir tee; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 127
  fi
done

SAMPLES_DIR="${OUT_DIR%/}/samples"
SAMPLES_TSV="${OUT_DIR%/}/samples.tsv"
METRIC_ROWS_TSV="${OUT_DIR%/}/metric-rows.tsv"
METRICS_TSV="${OUT_DIR%/}/metrics.tsv"
SUMMARY_TSV="${OUT_DIR%/}/summary.tsv"
CONFIG_TSV="${OUT_DIR%/}/config.tsv"
REPORT="${OUT_DIR%/}/report.txt"

mkdir -p "$OUT_DIR" "$SAMPLES_DIR"

safe_tsv_field() {
  local value="${1:-}"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  printf '%s' "$value"
}

config_row() {
  printf '%s\t%s\n' "$(safe_tsv_field "$1")" "$(safe_tsv_field "${2:-}")" >> "$CONFIG_TSV"
}

{
  printf 'key\tvalue\n'
} > "$CONFIG_TSV"
config_row generatedAt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
config_row model "$MODEL_ID"
config_row gollekBin "$GOLLEK_BIN"
config_row outDir "$OUT_DIR"
config_row repeat "$REPEAT_COUNT"
config_row runnerScript "$RUNNER_SCRIPT"
config_row runnerArgs "${RUNNER_ARGS[*]:-}"
config_row maxP95RegressionPercent "$MAX_P95_REGRESSION_PERCENT"
config_row failUnstableMetrics "$FAIL_UNSTABLE_METRICS"

report_value() {
  local key="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "$file"
  fi
}

append_metric_rows() {
  local sample="$1"
  local comparison="$2"
  if [[ ! -f "$comparison" ]]; then
    return
  fi
  if [[ ! -f "$METRIC_ROWS_TSV" ]]; then
    awk 'BEGIN { FS = OFS = "\t" } NR == 1 { print "sample", $0; exit }' "$comparison" > "$METRIC_ROWS_TSV"
  fi
  awk -v sample="$sample" 'BEGIN { FS = OFS = "\t" } NR > 1 { print sample, $0 }' "$comparison" >> "$METRIC_ROWS_TSV"
}

{
  printf 'sample\texitCode\tstatus\trecommendation\tgatedMetrics\tfailedMetrics\tlargestImprovement\tlargestRegression\tffnStrategyTransition\trowPrefillTransition\tsampleDir\treport\n'
} > "$SAMPLES_TSV"
rm -f "$METRIC_ROWS_TSV"

for ((sample_index = 1; sample_index <= REPEAT_COUNT; sample_index++)); do
  sample="$(printf 'sample-%02d' "$sample_index")"
  sample_dir="${SAMPLES_DIR}/${sample}"
  stdout_file="${sample_dir}/stdout.log"
  stderr_file="${sample_dir}/stderr.log"
  report_file="${sample_dir}/report.txt"
  comparison_file="${sample_dir}/compare/comparison.tsv"
  mkdir -p "$sample_dir"

  echo "[repeat] running ${sample}/${REPEAT_COUNT}"
  cmd=(
    bash "$RUNNER_SCRIPT"
    --model "$MODEL_ID"
    --gollek-bin "$GOLLEK_BIN"
    --out-dir "$sample_dir"
  )
  if (( ${#RUNNER_ARGS[@]} > 0 )); then
    cmd+=("${RUNNER_ARGS[@]}")
  fi

  set +e
  "${cmd[@]}" > "$stdout_file" 2> "$stderr_file"
  exit_code=$?
  set -e

  status="$(report_value status "$report_file")"
  recommendation="$(report_value recommendation "$report_file")"
  gated_metrics="$(report_value gatedMetrics "$report_file")"
  failed_metrics="$(report_value failedMetrics "$report_file")"
  largest_improvement="$(report_value largestImprovement "$report_file")"
  largest_regression="$(report_value largestRegression "$report_file")"
  ffn_transition="$(report_value ffnStrategyTransition "$report_file")"
  row_transition="$(report_value rowPrefillTransition "$report_file")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(safe_tsv_field "$sample")" \
    "$(safe_tsv_field "$exit_code")" \
    "$(safe_tsv_field "${status:-error}")" \
    "$(safe_tsv_field "${recommendation:-unknown}")" \
    "$(safe_tsv_field "${gated_metrics:-0}")" \
    "$(safe_tsv_field "${failed_metrics:-0}")" \
    "$(safe_tsv_field "$largest_improvement")" \
    "$(safe_tsv_field "$largest_regression")" \
    "$(safe_tsv_field "$ffn_transition")" \
    "$(safe_tsv_field "$row_transition")" \
    "$(safe_tsv_field "$sample_dir")" \
    "$(safe_tsv_field "$report_file")" >> "$SAMPLES_TSV"

  append_metric_rows "$sample" "$comparison_file"
done

if [[ -f "$METRIC_ROWS_TSV" ]]; then
  awk '
    BEGIN { FS = OFS = "\t" }
    function numeric(value) { return value ~ /^-?[0-9]+([.][0-9]+)?$/ }
    function abs(value) { return value < 0 ? -value : value }
    function clear_sorted(key) {
      for (key in sorted) {
        delete sorted[key]
      }
    }
    function bubble_sort(n, i, j, tmp) {
      for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
          if (sorted[j] < sorted[i]) {
            tmp = sorted[i]
            sorted[i] = sorted[j]
            sorted[j] = tmp
          }
        }
      }
    }
    function percentile_rank(n, percentile, raw) {
      raw = n * percentile
      return raw == int(raw) ? int(raw) : int(raw) + 1
    }
    function sort_delta(metric, n, i) {
      clear_sorted()
      for (i = 1; i <= n; i++) {
        sorted[i] = deltaValue[metric, i] + 0
      }
      bubble_sort(n)
    }
    function sort_percent(metric, n, i) {
      clear_sorted()
      for (i = 1; i <= n; i++) {
        sorted[i] = percentValue[metric, i] + 0
      }
      bubble_sort(n)
    }
    function sort_regression(metric, n, i) {
      clear_sorted()
      for (i = 1; i <= n; i++) {
        sorted[i] = regressionValue[metric, i] + 0
      }
      bubble_sort(n)
    }
    function median_delta(metric, n, mid) {
      n = samples[metric] + 0
      if (n == 0) return ""
      sort_delta(metric, n)
      mid = int((n + 1) / 2)
      return n % 2 ? sorted[mid] : (sorted[mid] + sorted[mid + 1]) / 2
    }
    function median_percent(metric, n, mid) {
      n = samples[metric] + 0
      if (n == 0) return ""
      sort_percent(metric, n)
      mid = int((n + 1) / 2)
      return n % 2 ? sorted[mid] : (sorted[mid] + sorted[mid + 1]) / 2
    }
    function p95_delta(metric, n, rank) {
      n = samples[metric] + 0
      if (n == 0) return ""
      sort_delta(metric, n)
      rank = percentile_rank(n, 0.95)
      if (rank < 1) rank = 1
      if (rank > n) rank = n
      return sorted[rank]
    }
    function p95_regression(metric, n, rank) {
      n = samples[metric] + 0
      if (n == 0) return ""
      sort_regression(metric, n)
      rank = percentile_rank(n, 0.95)
      if (rank < 1) rank = 1
      if (rank > n) rank = n
      return sorted[rank]
    }
    function stddev_delta(metric, n, mean, i, sumsq, diff) {
      n = samples[metric] + 0
      if (n <= 1) return 0
      mean = sumDelta[metric] / n
      for (i = 1; i <= n; i++) {
        diff = deltaValue[metric, i] - mean
        sumsq += diff * diff
      }
      return sqrt(sumsq / n)
    }
    function stddev_percent(metric, n, mean, i, sumsq, diff) {
      n = samples[metric] + 0
      if (n <= 1) return 0
      mean = sumPercent[metric] / n
      for (i = 1; i <= n; i++) {
        diff = percentValue[metric, i] - mean
        sumsq += diff * diff
      }
      return sqrt(sumsq / n)
    }
    function unstable_metric(metric, percentStddev) {
      if (samples[metric] + 0 <= 1) {
        return "false"
      }
      if (better[metric] + 0 > 0 && worse[metric] + 0 > 0) {
        return "true"
      }
      if (percentStddev + 0 > 10) {
        return "true"
      }
      return "false"
    }
    function remember(metric) {
      if (metric == "" || seen[metric]) {
        return
      }
      seen[metric] = 1
      order[++orderCount] = metric
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        col[$i] = i
      }
      print "metric", "samples", "better", "worse", "same", "avgDelta", "medianDelta", "p95Delta", "stddevDelta", "avgDeltaPercent", "medianDeltaPercent", "p95RegressionPercent", "stddevDeltaPercent", "maxImprovementPercent", "maxRegressionPercent", "failedGateCount", "unstable"
      next
    }
    {
      metric = $(col["metric"])
      direction = $(col["direction"])
      trend = $(col["trend"])
      gate = $(col["gateStatus"])
      delta = $(col["delta"])
      percent = $(col["deltaPercent"])
      remember(metric)
      if (numeric(delta) && numeric(percent)) {
        samples[metric]++
        deltaValue[metric, samples[metric]] = delta + 0
        percentValue[metric, samples[metric]] = percent + 0
        sumDelta[metric] += delta
        sumPercent[metric] += percent
        if (trend == "better") {
          better[metric]++
        } else if (trend == "worse") {
          worse[metric]++
        } else if (trend == "same") {
          same[metric]++
        }
        improvement = 0
        regression = 0
        if (direction == "higher") {
          if (percent + 0 > 0) improvement = percent + 0
          if (percent + 0 < 0) regression = abs(percent + 0)
        } else {
          if (percent + 0 < 0) improvement = abs(percent + 0)
          if (percent + 0 > 0) regression = percent + 0
        }
        if (improvement > maxImprovement[metric]) {
          maxImprovement[metric] = improvement
        }
        if (regression > maxRegression[metric]) {
          maxRegression[metric] = regression
        }
        regressionValue[metric, samples[metric]] = regression
      }
      if (gate == "fail") {
        failedGate[metric]++
      }
    }
    END {
      for (i = 1; i <= orderCount; i++) {
        metric = order[i]
        if (samples[metric] + 0 == 0) {
          continue
        }
        deltaStddev = stddev_delta(metric)
        percentStddev = stddev_percent(metric)
        print metric, samples[metric] + 0, better[metric] + 0, worse[metric] + 0, same[metric] + 0, sprintf("%.3f", sumDelta[metric] / samples[metric]), sprintf("%.3f", median_delta(metric)), sprintf("%.3f", p95_delta(metric)), sprintf("%.3f", deltaStddev), sprintf("%.3f", sumPercent[metric] / samples[metric]), sprintf("%.3f", median_percent(metric)), sprintf("%.3f", p95_regression(metric)), sprintf("%.3f", percentStddev), sprintf("%.3f", maxImprovement[metric] + 0), sprintf("%.3f", maxRegression[metric] + 0), failedGate[metric] + 0, unstable_metric(metric, percentStddev)
      }
    }
  ' "$METRIC_ROWS_TSV" > "$METRICS_TSV"
else
  printf 'metric\tsamples\tbetter\tworse\tsame\tavgDelta\tmedianDelta\tp95Delta\tstddevDelta\tavgDeltaPercent\tmedianDeltaPercent\tp95RegressionPercent\tstddevDeltaPercent\tmaxImprovementPercent\tmaxRegressionPercent\tfailedGateCount\tunstable\n' > "$METRICS_TSV"
fi

UNSTABLE_METRICS="$(awk 'BEGIN { FS = "\t" } NR > 1 && $NF == "true" { count++ } END { print count + 0 }' "$METRICS_TSV")"
P95_REGRESSION_GATE_FAILURES=0
if [[ -n "$MAX_P95_REGRESSION_PERCENT" ]]; then
  P95_REGRESSION_GATE_FAILURES="$(
    awk -v max="$MAX_P95_REGRESSION_PERCENT" '
      BEGIN { FS = "\t" }
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          col[$i] = i
        }
        next
      }
      $(col["p95RegressionPercent"]) + 0 > max + 0 { count++ }
      END { print count + 0 }
    ' "$METRICS_TSV"
  )"
fi
UNSTABLE_GATE_FAILURES=0
if (( FAIL_UNSTABLE_METRICS == 1 )); then
  UNSTABLE_GATE_FAILURES="$UNSTABLE_METRICS"
fi
AGGREGATE_GATE_FAILURES=$((P95_REGRESSION_GATE_FAILURES + UNSTABLE_GATE_FAILURES))
AGGREGATE_GATE_REASON=""
if (( P95_REGRESSION_GATE_FAILURES > 0 )); then
  AGGREGATE_GATE_REASON="p95-regression"
fi
if (( UNSTABLE_GATE_FAILURES > 0 )); then
  if [[ -n "$AGGREGATE_GATE_REASON" ]]; then
    AGGREGATE_GATE_REASON="${AGGREGATE_GATE_REASON}; unstable-metrics"
  else
    AGGREGATE_GATE_REASON="unstable-metrics"
  fi
fi

awk \
  -v repeat="$REPEAT_COUNT" \
  -v unstableMetrics="$UNSTABLE_METRICS" \
  -v aggregateGateFailures="$AGGREGATE_GATE_FAILURES" \
  -v aggregateGateReason="$AGGREGATE_GATE_REASON" \
  -v p95RegressionGateFailures="$P95_REGRESSION_GATE_FAILURES" \
  -v unstableGateFailures="$UNSTABLE_GATE_FAILURES" \
  -v summaryOut="$SUMMARY_TSV" '
  BEGIN { FS = OFS = "\t" }
  function write_summary(key, value) { print key, value >> summaryOut }
  function append_reason(existing, nextReason) { return existing == "" ? nextReason : existing "; " nextReason }
  NR == 1 { next }
  {
    samples++
    exitCode = $2 + 0
    status = $3
    recommendation = $4
    failedMetrics += $6 + 0
    if (exitCode == 0 && status == "pass") {
      passedSamples++
    } else if (status == "error" || recommendation == "unknown") {
      errorSamples++
    } else {
      failedSamples++
    }
    if (recommendation == "promote-current") {
      promoteSamples++
    } else if (recommendation == "promote-current-with-watchlist") {
      watchlistSamples++
    } else if (recommendation == "hold-current") {
      holdSamples++
    } else if (recommendation == "reject-current") {
      rejectSamples++
    } else {
      collectSamples++
    }
  }
  END {
    print "key", "value" > summaryOut
    reason = ""
    if (failedSamples + errorSamples + rejectSamples > 0) {
      reason = append_reason(reason, "sample-regression")
    }
    if (aggregateGateReason != "") {
      reason = append_reason(reason, aggregateGateReason)
    }
    status = (failedSamples + errorSamples + rejectSamples + aggregateGateFailures == 0) ? "pass" : "fail"
    if (status == "fail") {
      recommendation = "reject-current"
    } else if (promoteSamples == samples) {
      recommendation = "promote-current"
    } else if (promoteSamples + watchlistSamples == samples) {
      recommendation = "promote-current-with-watchlist"
    } else if (promoteSamples + watchlistSamples > holdSamples + collectSamples) {
      recommendation = "promote-current-with-watchlist"
    } else if (holdSamples > 0) {
      recommendation = "hold-current"
    } else {
      recommendation = "collect-more-samples"
    }
    write_summary("status", status)
    write_summary("reason", reason)
    write_summary("recommendation", recommendation)
    write_summary("requestedSamples", repeat)
    write_summary("samples", samples + 0)
    write_summary("passedSamples", passedSamples + 0)
    write_summary("failedSamples", failedSamples + 0)
    write_summary("errorSamples", errorSamples + 0)
    write_summary("failedMetrics", failedMetrics + 0)
    write_summary("unstableMetrics", unstableMetrics + 0)
    write_summary("aggregateGateFailures", aggregateGateFailures + 0)
    write_summary("p95RegressionGateFailures", p95RegressionGateFailures + 0)
    write_summary("unstableGateFailures", unstableGateFailures + 0)
    write_summary("promoteCurrentSamples", promoteSamples + 0)
    write_summary("watchlistSamples", watchlistSamples + 0)
    write_summary("holdSamples", holdSamples + 0)
    write_summary("rejectSamples", rejectSamples + 0)
    write_summary("collectMoreSamples", collectSamples + 0)
  }
' "$SAMPLES_TSV"

STATUS="$(awk 'BEGIN { FS = "\t" } $1 == "status" { print $2; exit }' "$SUMMARY_TSV")"
REASON="$(awk 'BEGIN { FS = "\t" } $1 == "reason" { print $2; exit }' "$SUMMARY_TSV")"
RECOMMENDATION="$(awk 'BEGIN { FS = "\t" } $1 == "recommendation" { print $2; exit }' "$SUMMARY_TSV")"
SAMPLES="$(awk 'BEGIN { FS = "\t" } $1 == "samples" { print $2; exit }' "$SUMMARY_TSV")"
PASSED_SAMPLES="$(awk 'BEGIN { FS = "\t" } $1 == "passedSamples" { print $2; exit }' "$SUMMARY_TSV")"
FAILED_SAMPLES="$(awk 'BEGIN { FS = "\t" } $1 == "failedSamples" { print $2; exit }' "$SUMMARY_TSV")"
ERROR_SAMPLES="$(awk 'BEGIN { FS = "\t" } $1 == "errorSamples" { print $2; exit }' "$SUMMARY_TSV")"
FAILED_METRICS="$(awk 'BEGIN { FS = "\t" } $1 == "failedMetrics" { print $2; exit }' "$SUMMARY_TSV")"
UNSTABLE_METRICS="$(awk 'BEGIN { FS = "\t" } $1 == "unstableMetrics" { print $2; exit }' "$SUMMARY_TSV")"
AGGREGATE_GATE_FAILURES="$(awk 'BEGIN { FS = "\t" } $1 == "aggregateGateFailures" { print $2; exit }' "$SUMMARY_TSV")"
P95_REGRESSION_GATE_FAILURES="$(awk 'BEGIN { FS = "\t" } $1 == "p95RegressionGateFailures" { print $2; exit }' "$SUMMARY_TSV")"
UNSTABLE_GATE_FAILURES="$(awk 'BEGIN { FS = "\t" } $1 == "unstableGateFailures" { print $2; exit }' "$SUMMARY_TSV")"

{
  echo "Gollek Gemma4 row-prefill repeat A/B"
  echo "outDir=$OUT_DIR"
  echo "samplesDir=$SAMPLES_DIR"
  echo "artifacts.config=$CONFIG_TSV"
  echo "artifacts.samples=$SAMPLES_TSV"
  echo "artifacts.metricRows=$METRIC_ROWS_TSV"
  echo "artifacts.metrics=$METRICS_TSV"
  echo "artifacts.summary=$SUMMARY_TSV"
  echo "status=$STATUS"
  echo "reason=$REASON"
  echo "recommendation=$RECOMMENDATION"
  echo "samples=$SAMPLES"
  echo "passedSamples=$PASSED_SAMPLES"
  echo "failedSamples=$FAILED_SAMPLES"
  echo "errorSamples=$ERROR_SAMPLES"
  echo "failedMetrics=$FAILED_METRICS"
  echo "unstableMetrics=$UNSTABLE_METRICS"
  echo "aggregateGateFailures=$AGGREGATE_GATE_FAILURES"
  echo "p95RegressionGateFailures=$P95_REGRESSION_GATE_FAILURES"
  echo "unstableGateFailures=$UNSTABLE_GATE_FAILURES"
} | tee "$REPORT"

if [[ "$STATUS" == "fail" ]]; then
  exit 1
fi
