#!/bin/bash
# Unified runner for Gollek recursive-reasoning demo, comparison, and bundle flow.
# Usage: ./run_recursive_reasoning_workflow.sh [full] [output-root]

set -euo pipefail

BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
OUTPUT_ROOT="${2:-$SCRIPT_DIR/../trainer_checkpoints/recursive_reasoning_workflow}"

BASELINE_DIR="${OUTPUT_ROOT}-baseline"
CURRENT_DIR="${OUTPUT_ROOT}-current"
RESULT_FILE="${OUTPUT_ROOT}-compare-result.json"
BUNDLE_DIR="${OUTPUT_ROOT}-compare-bundle"

print_usage() {
    echo "Usage: ./run_recursive_reasoning_workflow.sh [full] [output-root]"
    echo
    echo "Examples:"
    echo "  ./run_recursive_reasoning_workflow.sh"
    echo "  ./run_recursive_reasoning_workflow.sh full"
    echo "  ./run_recursive_reasoning_workflow.sh full /tmp/recursive-reasoning"
}

require_mode() {
    if [[ "$MODE" != "full" ]]; then
        echo "Only 'full' mode is currently supported."
        print_usage
        exit 1
    fi
}

ensure_parent() {
    local path="$1"
    mkdir -p "$(dirname "$path")"
}

run_demo() {
    local task_id="$1"
    local output_dir="$2"
    echo -e "${BLUE}Running recursive reasoning demo:${NC} taskId=$task_id outputDir=$output_dir"
    jbang "$SCRIPT_DIR/trainer_recursive_reasoning_demo.java" "$task_id" "$output_dir"
}

materialize_current_variant() {
    local baseline_report="$1"
    local current_report="$2"
    echo -e "${BLUE}Synthesizing current report variant:${NC} $current_report"
    jq '.taskId="sudoku-demo-b"
        | .summary.selectedStateId="state-s1-e1-t2"
        | .selectedTrajectoryIndex=1
        | .selectedCumulativeLogProbability=-2.5
        | .selectedRewardScore=0.75' \
        "$baseline_report" > "$current_report"
}

export_compare_result() {
    echo -e "${BLUE}Exporting recursive reasoning compare result:${NC} $RESULT_FILE"
    jbang "$SCRIPT_DIR/trainer_recursive_reasoning_compare_inspector.java" \
        "$BASELINE_DIR" \
        "$CURRENT_DIR" \
        result=standard \
        json \
        "$RESULT_FILE"
}

export_compare_bundle() {
    echo -e "${BLUE}Exporting recursive reasoning compare bundle:${NC} $BUNDLE_DIR"
    jbang "$SCRIPT_DIR/trainer_recursive_reasoning_compare_inspector.java" \
        "$BASELINE_DIR" \
        "$CURRENT_DIR" \
        bundle=standard \
        json \
        "$BUNDLE_DIR"
}

print_summary() {
    echo "===================================================="
    echo " Recursive Reasoning Workflow Summary"
    echo "===================================================="
    echo "baselineDir=$BASELINE_DIR"
    echo "currentDir=$CURRENT_DIR"
    echo "resultFile=$RESULT_FILE"
    echo "bundleDir=$BUNDLE_DIR"
    echo "inspectBaseline=jbang trainer/trainer_recursive_reasoning_inspector.java \"$BASELINE_DIR\" overview"
    echo "inspectCurrent=jbang trainer/trainer_recursive_reasoning_inspector.java \"$CURRENT_DIR\" overview"
    echo "inspectCompare=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$BASELINE_DIR\" \"$CURRENT_DIR\" overview"
    echo "inspectCompareCi=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$BASELINE_DIR\" \"$CURRENT_DIR\" comparison:ci"
    echo "inspectResult=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$RESULT_FILE\" resultSummary"
    echo "inspectResultHealth=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$RESULT_FILE\" resultHealth:ci"
    echo "inspectBundleSummary=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$BUNDLE_DIR\" bundleSummary"
    echo "inspectBundleHealth=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$BUNDLE_DIR\" bundleHealth:ci"
    echo "inspectBundleComparison=jbang trainer/trainer_recursive_reasoning_compare_inspector.java \"$BUNDLE_DIR\" loadfile:comparison json"
}

main() {
    require_mode
    ensure_parent "$OUTPUT_ROOT"
    run_demo "sudoku-demo" "$BASELINE_DIR"
    mkdir -p "$CURRENT_DIR"
    materialize_current_variant \
        "$BASELINE_DIR/recursive-reasoning-report.json" \
        "$CURRENT_DIR/recursive-reasoning-report.json"
    export_compare_result
    export_compare_bundle
    echo -e "${YELLOW}Workflow complete.${NC}"
    print_summary
}

main "$@"
