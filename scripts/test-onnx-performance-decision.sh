#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-test-onnx-performance-decision.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

PASS_DIR="${TMP_DIR}/pass"
mkdir -p "$PASS_DIR/compare"
{
  printf 'key\tvalue\n'
  printf 'bundle\t%s\n' "$PASS_DIR/bundle.tsv"
  printf 'currentNoise\t%s\n' "$PASS_DIR/current-noise.tsv"
} > "$PASS_DIR/config.tsv"
{
  printf 'stage\tstatus\texitCode\tartifact\tstdout\tstderr\treason\n'
  printf 'environment\tpass\t0\t%s/environment.tsv\t\t\t\n' "$PASS_DIR"
  printf 'compare\tpass\t0\t%s/compare/comparison.tsv\t%s/compare.stdout.log\t%s/compare.stderr.log\t\n' "$PASS_DIR" "$PASS_DIR" "$PASS_DIR"
} > "$PASS_DIR/results.tsv"
{
  printf 'case\tmetric\tbaseline\tcurrent\tdirection\tregressionPercent\tmaxRegressionPercent\tstatus\treason\n'
  printf 'measured-mean\tdurationMs\t1000\t1015\tlower-is-better\t1.500\t10\tpass\t\n'
  printf 'measured-mean\tgenerationTps\t10\t10.2\thigher-is-better\t-2.000\t10\tpass\t\n'
} > "$PASS_DIR/compare/comparison.tsv"
{
  printf 'metric\tcases\tcompared\tpasses\tfailures\tskips\tworstCase\tworstRegressionPercent\tmaxRegressionPercent\tstatus\n'
  printf 'durationMs\t1\t1\t1\t0\t0\tmeasured-mean\t1.500\t10\tpass\n'
} > "$PASS_DIR/compare/metric-summary.tsv"
{
  printf 'metric\tmean\tbest\tworst\tnoisePercent\tmaxNoisePercent\tstatus\treason\n'
  printf 'durationMs\t1015\t1000\t1030\t2.956\t10\tpass\t\n'
} > "$PASS_DIR/current-noise.tsv"
{
  printf 'kind\tpath\trequired\tstatus\tdescription\n'
  printf 'config\t%s/config.tsv\trequired\tpresent\tConfig artifact\n' "$PASS_DIR"
  printf 'comparison\t%s/compare/comparison.tsv\trequired\tpresent\tComparison artifact\n' "$PASS_DIR"
  printf 'optionalMissing\t%s/optional-missing.tsv\toptional\tmissing\tOptional missing artifact\n' "$PASS_DIR"
} > "$PASS_DIR/bundle.tsv"

bash "$ROOT_DIR/scripts/onnx-performance-decision.sh" \
  --config "$PASS_DIR/config.tsv" \
  --results "$PASS_DIR/results.tsv" \
  --out "$PASS_DIR/decision.tsv" \
  --comparison "$PASS_DIR/compare/comparison.tsv" \
  --metric-summary "$PASS_DIR/compare/metric-summary.tsv" \
  --noise "$PASS_DIR/current-noise.tsv" \
  --bundle "$PASS_DIR/bundle.tsv"

if ! grep -qx $'key\tvalue' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'status\tpass' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'failedStages\t0' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'lastStage\tcompare' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'regressionFailures\t0' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'regressionCompared\t2' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'worstRegressionMetric\tdurationMs' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'worstRegressionPercent\t1.500' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'noiseFailures\t0' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'worstNoiseMetric\tdurationMs' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'bundleArtifacts\t3' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'bundlePresent\t2' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'bundleMissing\t1' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'bundleRequiredMissing\t0' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'bundle\t'"$PASS_DIR"'/bundle.tsv' "$PASS_DIR/decision.tsv" \
    || ! grep -qx $'json\t'"$PASS_DIR"'/decision.json' "$PASS_DIR/decision.tsv" \
    || ! grep -q '"status": "pass"' "$PASS_DIR/decision.json" \
    || ! grep -q '"worstRegressionMetric": "durationMs"' "$PASS_DIR/decision.json"; then
  echo "Expected passing decision summary" >&2
  cat "$PASS_DIR/decision.tsv" >&2
  [[ -f "$PASS_DIR/decision.json" ]] && cat "$PASS_DIR/decision.json" >&2
  exit 1
fi

NOISY_DIR="${TMP_DIR}/noisy"
mkdir -p "$NOISY_DIR"
{
  printf 'key\tvalue\n'
  printf 'bundle\t%s\n' "$NOISY_DIR/bundle.tsv"
  printf 'currentNoise\t%s\n' "$NOISY_DIR/current-noise.tsv"
} > "$NOISY_DIR/config.tsv"
{
  printf 'stage\tstatus\texitCode\tartifact\tstdout\tstderr\treason\n'
  printf 'environment\tpass\t0\t%s/environment.tsv\t\t\t\n' "$NOISY_DIR"
  printf 'current-noise\tfail\t47\t%s/current-noise.tsv\t%s/noise.stdout.log\t%s/noise.stderr.log\tnoise-failed\n' "$NOISY_DIR" "$NOISY_DIR" "$NOISY_DIR"
} > "$NOISY_DIR/results.tsv"
{
  printf 'metric\tmean\tbest\tworst\tnoisePercent\tmaxNoisePercent\tstatus\treason\n'
  printf 'durationMs\t1120\t1080\t1160\t7.143\t5\tfail\t\n'
} > "$NOISY_DIR/current-noise.tsv"

bash "$ROOT_DIR/scripts/onnx-performance-decision.sh" \
  --config "$NOISY_DIR/config.tsv" \
  --results "$NOISY_DIR/results.tsv" \
  --out "$NOISY_DIR/decision.tsv" \
  --json-out "$NOISY_DIR/decision-output.json"

if ! grep -qx $'status\tfail' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'failedStages\t1' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'failureStage\tcurrent-noise' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'failureReason\tnoise-failed' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'regressionCompared\t0' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'noiseFailures\t1' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'worstNoisePercent\t7.143' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'bundleRequiredMissing\t1' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'firstBundleMissingKind\tbundle' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'firstBundleMissingPath\t'"$NOISY_DIR"'/bundle.tsv' "$NOISY_DIR/decision.tsv" \
    || ! grep -qx $'json\t'"$NOISY_DIR"'/decision-output.json' "$NOISY_DIR/decision.tsv" \
    || ! grep -q '"status": "fail"' "$NOISY_DIR/decision-output.json" \
    || ! grep -q '"failureStage": "current-noise"' "$NOISY_DIR/decision-output.json"; then
  echo "Expected noisy failure decision summary" >&2
  cat "$NOISY_DIR/decision.tsv" >&2
  [[ -f "$NOISY_DIR/decision-output.json" ]] && cat "$NOISY_DIR/decision-output.json" >&2
  exit 1
fi

GATE_DIR="${TMP_DIR}/gate"
mkdir -p "$GATE_DIR"
{
  printf 'key\tvalue\n'
  printf 'contracts\t%s\n' "$GATE_DIR/contracts.tsv"
} > "$GATE_DIR/config.tsv"
{
  printf 'stage\tstatus\texitCode\tartifact\tstdout\tstderr\treason\n'
  printf 'benchmark\tpass\t0\t%s/bench/profile/summary.tsv\t%s/bench.stdout.log\t%s/bench.stderr.log\t\n' "$GATE_DIR" "$GATE_DIR" "$GATE_DIR"
  printf 'contracts\tfail\t42\t%s/contracts.tsv\t%s/bench.stdout.log\t%s/bench.stderr.log\tcontract-failures=2\n' "$GATE_DIR" "$GATE_DIR" "$GATE_DIR"
} > "$GATE_DIR/results.tsv"
{
  printf 'case\tmetric\tactual\toperator\tthreshold\tstatus\treason\n'
  printf 'run-1\tdurationMs\t1250\t<=\t1000\tfail\t\n'
  printf 'run-1\tgenerationTps\t6\t>=\t10\tfail\t\n'
  printf 'run-1\tonnxDecodeRunMs\t20\t<=\t25\tpass\t\n'
} > "$GATE_DIR/contracts.tsv"

bash "$ROOT_DIR/scripts/onnx-performance-decision.sh" \
  --config "$GATE_DIR/config.tsv" \
  --results "$GATE_DIR/results.tsv" \
  --out "$GATE_DIR/decision.tsv" \
  --contracts "$GATE_DIR/contracts.tsv"

if ! grep -qx $'status\tfail' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'failureStage\tcontracts' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'failureReason\tcontract-failures=2' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'contractFailures\t2' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'contractPasses\t1' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'contractChecks\t3' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'firstContractFailureMetric\tdurationMs' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'worstContractMetric\tgenerationTps' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'worstContractActual\t6' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'worstContractOperator\t>=' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'worstContractThreshold\t10' "$GATE_DIR/decision.tsv" \
    || ! grep -qx $'worstContractPercent\t40.000' "$GATE_DIR/decision.tsv"; then
  echo "Expected gate contract decision summary" >&2
  cat "$GATE_DIR/decision.tsv" >&2
  exit 1
fi

printf 'ONNX performance decision test passed\n'
