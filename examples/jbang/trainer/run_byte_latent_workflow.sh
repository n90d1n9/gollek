#!/bin/bash
# Unified runner for Gollek byte-latent fresh, resume, and full smoke flows.
# Usage: ./run_byte_latent_workflow.sh [fresh|resume|full] [checkpoint-root]

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-full}"
EXPLICIT_ROOT="${2:-}"
declare -a COMPLETED_MODES=()

resolve_manifest_file() {
    if [[ -n "$EXPLICIT_ROOT" ]]; then
        echo "${EXPLICIT_ROOT}-workflow-manifest.json"
        return
    fi
    echo "$SCRIPT_DIR/../trainer_checkpoints/byte_latent_workflow_manifest.json"
}

resolve_bundle_dir() {
    local manifest_file
    manifest_file="$(resolve_manifest_file)"
    echo "$(dirname "$manifest_file")/byte-latent-workflow-bundle"
}

json_escape() {
    local value="${1:-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "$value"
}

print_usage() {
    echo "Usage: ./run_byte_latent_workflow.sh [fresh|resume|full] [checkpoint-root]"
    echo
    echo "Examples:"
    echo "  ./run_byte_latent_workflow.sh"
    echo "  ./run_byte_latent_workflow.sh fresh"
    echo "  ./run_byte_latent_workflow.sh resume /tmp/gollek-byte-latent"
    echo "  ./run_byte_latent_workflow.sh full /tmp/gollek-byte-latent"
}

resolve_checkpoint_dir() {
    local kind="$1"
    if [[ -n "$EXPLICIT_ROOT" ]]; then
        echo "${EXPLICIT_ROOT}-${kind}"
        return
    fi
    if [[ "$kind" == "fresh" ]]; then
        echo "$SCRIPT_DIR/../trainer_checkpoints/byte_latent_smoke"
        return
    fi
    echo "$SCRIPT_DIR/../trainer_checkpoints/byte_latent_resume_smoke"
}

summary_value() {
    local file="$1"
    local key="$2"
    if [[ ! -f "$file" ]]; then
        echo ""
        return
    fi
    sed -n "s/^${key}=//p" "$file" | head -n 1
}

history_row_count() {
    local history_file="$1"
    if [[ ! -f "$history_file" ]]; then
        echo "0"
        return
    fi
    local line_count
    line_count=$(wc -l < "$history_file")
    if [[ "$line_count" -le 1 ]]; then
        echo "0"
        return
    fi
    echo $((line_count - 1))
}

emit_run_manifest_json() {
    local kind="$1"
    local checkpoint_dir
    local summary_file
    local history_file
    local report_file
    local metadata_file
    local history_rows
    local epoch_count
    local latest_train_loss
    local best_validation_loss
    checkpoint_dir="$(resolve_checkpoint_dir "$kind")"
    summary_file="$checkpoint_dir/byte-latent-summary.json"
    history_file="$checkpoint_dir/byte-latent-history.csv"
    report_file="$checkpoint_dir/byte-latent-report.json"
    metadata_file="$checkpoint_dir/byte-latent-checkpoint.metadata"
    history_rows="$(history_row_count "$history_file")"
    epoch_count="$(summary_value "$summary_file" "epochCount")"
    latest_train_loss="$(summary_value "$summary_file" "latestTrainLoss")"
    best_validation_loss="$(summary_value "$summary_file" "bestValidationLoss")"
    cat <<EOF
    {
      "mode": "$(json_escape "$kind")",
      "checkpointDir": "$(json_escape "$checkpoint_dir")",
      "summaryFile": "$(json_escape "$summary_file")",
      "historyFile": "$(json_escape "$history_file")",
      "reportFile": "$(json_escape "$report_file")",
      "metadataFile": "$(json_escape "$metadata_file")",
      "historyRows": $history_rows,
      "epochCount": "$(json_escape "${epoch_count:-unknown}")",
      "latestTrainLoss": "$(json_escape "${latest_train_loss:-unknown}")",
      "bestValidationLoss": "$(json_escape "${best_validation_loss:-unknown}")",
      "inspectRunAll": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$(resolve_manifest_file)")\" runcommands:all",
      "inspectRunFields": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$(resolve_manifest_file)")\" inspectfields:ci",
      "inspectRunCommands": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$(resolve_manifest_file)")\" runcommands:$(json_escape "$kind")",
      "inspectWorkflowCommands": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$(resolve_manifest_file)")\" commands:ci:mode=$(json_escape "$kind")",
      "inspectStatus": "jbang trainer/trainer_byte_latent_history_inspector.java \"$(json_escape "$checkpoint_dir")\" status",
      "inspectHealth": "jbang trainer/trainer_byte_latent_history_inspector.java \"$(json_escape "$checkpoint_dir")\" health",
      "inspectCi": "jbang trainer/trainer_byte_latent_history_inspector.java \"$(json_escape "$checkpoint_dir")\" ci",
      "inspectOverview": "jbang trainer/trainer_byte_latent_history_inspector.java \"$(json_escape "$checkpoint_dir")\" overview json",
      "inspectHistory": "jbang trainer/trainer_byte_latent_history_inspector.java \"$(json_escape "$checkpoint_dir")\" history:summary"
    }
EOF
}

write_workflow_manifest() {
    local manifest_file
    local bundle_dir
    local bundle_status
    local created_at
    local completed_json=""
    local runs_json=""
    local kind
    manifest_file="$(resolve_manifest_file)"
    bundle_dir="$(resolve_bundle_dir)"
    if [[ "$MODE" == "full" ]]; then
        bundle_status="materialized"
    else
        bundle_status="on-demand"
    fi
    created_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    for kind in "${COMPLETED_MODES[@]}"; do
        if [[ -n "$completed_json" ]]; then
            completed_json+=", "
            runs_json+=",\n"
        fi
        completed_json+="\"$(json_escape "$kind")\""
        runs_json+="$(emit_run_manifest_json "$kind")"
    done
    cat > "$manifest_file" <<EOF
{
  "workflow": "byte-latent",
  "mode": "$(json_escape "$MODE")",
  "checkpointRoot": "$(json_escape "${EXPLICIT_ROOT:-}")",
  "workflowBundleDir": "$(json_escape "$bundle_dir")",
  "workflowBundleStatus": "$(json_escape "$bundle_status")",
  "inspectWorkflowStatus": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$manifest_file")\" status",
  "inspectWorkflowHealth": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$manifest_file")\" health",
  "inspectBundleStatus": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$bundle_dir")\" status",
  "inspectBundleHealth": "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$(json_escape "$bundle_dir")\" health",
  "createdAt": "$(json_escape "$created_at")",
  "completedModes": [$completed_json],
  "runs": [
$runs_json
  ]
}
EOF
    echo "$manifest_file"
}

materialize_workflow_bundle_if_needed() {
    local manifest_file="$1"
    local bundle_dir
    if [[ "$MODE" != "full" ]]; then
        return
    fi
    bundle_dir="$(resolve_bundle_dir)"
    echo -e "${BLUE}Materializing workflow bundle:${NC} $bundle_dir"
    jbang "$SCRIPT_DIR/trainer_byte_latent_workflow_inspector.java" \
        "$manifest_file" \
        bundle=standard \
        json \
        "$bundle_dir" >/dev/null
}

print_run_summary() {
    local kind="$1"
    local checkpoint_dir
    local summary_file
    local history_file
    local report_file
    local history_rows
    local epoch_count
    local latest_train_loss
    local best_validation_loss
    local manifest_file
    checkpoint_dir="$(resolve_checkpoint_dir "$kind")"
    manifest_file="$(resolve_manifest_file)"
    summary_file="$checkpoint_dir/byte-latent-summary.json"
    history_file="$checkpoint_dir/byte-latent-history.csv"
    report_file="$checkpoint_dir/byte-latent-report.json"
    history_rows="$(history_row_count "$history_file")"
    epoch_count="$(summary_value "$summary_file" "epochCount")"
    latest_train_loss="$(summary_value "$summary_file" "latestTrainLoss")"
    best_validation_loss="$(summary_value "$summary_file" "bestValidationLoss")"
    echo "----------------------------------------------------"
    echo -e "${YELLOW}Run summary:${NC} mode=$kind"
    echo "checkpointDir=$checkpoint_dir"
    echo "summaryFile=$summary_file"
    echo "historyFile=$history_file"
    echo "reportFile=$report_file"
    echo "historyRows=$history_rows"
    echo "epochCount=${epoch_count:-unknown}"
    echo "latestTrainLoss=${latest_train_loss:-unknown}"
    echo "bestValidationLoss=${best_validation_loss:-unknown}"
    echo "inspectRunDiscovery=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:ci"
    echo "inspectRunAll=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:all"
    echo "inspectRunFields=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runfields:ci"
    echo "inspectRunCommands=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:$kind"
    echo "inspectWorkflowCommands=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" commands:ci:mode=$kind"
    echo "inspectStatus=jbang trainer/trainer_byte_latent_history_inspector.java \"$checkpoint_dir\" status"
    echo "inspectHealth=jbang trainer/trainer_byte_latent_history_inspector.java \"$checkpoint_dir\" health"
    echo "inspectCi=jbang trainer/trainer_byte_latent_history_inspector.java \"$checkpoint_dir\" ci"
    echo "inspectOverview=jbang trainer/trainer_byte_latent_history_inspector.java \"$checkpoint_dir\" overview json"
    echo "inspectHistory=jbang trainer/trainer_byte_latent_history_inspector.java \"$checkpoint_dir\" history:summary"
}

print_workflow_summary() {
    local kind
    local manifest_file
    local bundle_dir
    manifest_file="$(resolve_manifest_file)"
    bundle_dir="$(resolve_bundle_dir)"
    echo "===================================================="
    echo " Byte-Latent Workflow Summary"
    echo "===================================================="
    echo "manifestFile=$manifest_file"
    if [[ "$MODE" == "full" ]]; then
        echo "bundleDir=$bundle_dir"
        echo "bundleStatus=materialized"
    else
        echo "bundleDir=$bundle_dir"
        echo "bundleStatus=on-demand"
    fi
    echo "inspectWorkflowOverview=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" overview"
    echo "inspectWorkflowRuns=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runs json"
    echo "inspectWorkflowStatus=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" status"
    echo "inspectWorkflowHealth=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" health"
    echo "inspectWorkflowDelta=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" delta:historyRows"
    echo "inspectBundleStatus=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" status"
    echo "inspectBundleHealth=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" health"
    echo "exportWorkflowBundle=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" bundle=standard json \"$bundle_dir\""
    echo "loadWorkflowBundle=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" loadfile:section=runs"
    echo "loadWorkflowBundleIndexed=jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" loadfile:section=runs:index=0 json"
    for kind in "${COMPLETED_MODES[@]}"; do
        print_run_summary "$kind"
    done
}

run_mode() {
    local kind="$1"
    local script_name
    local checkpoint_dir
    if [[ "$kind" == "fresh" ]]; then
        script_name="run_byte_latent_smoke.sh"
    else
        script_name="run_byte_latent_resume_smoke.sh"
    fi
    checkpoint_dir="$(resolve_checkpoint_dir "$kind")"
    echo -e "${BLUE}Running mode:${NC} $kind"
    echo "checkpointDir=$checkpoint_dir"
    "$SCRIPT_DIR/$script_name" "$checkpoint_dir"
    COMPLETED_MODES+=("$kind")
}

print_header() {
    echo "===================================================="
    echo " Gollek Byte-Latent Workflow Runner"
    echo "===================================================="
    echo "mode=$MODE"
    if [[ -n "$EXPLICIT_ROOT" ]]; then
        echo "checkpointRoot=$EXPLICIT_ROOT"
    fi
}

print_next_steps() {
    local manifest_file
    local bundle_dir
    manifest_file="$(resolve_manifest_file)"
    bundle_dir="$(resolve_bundle_dir)"
    echo "----------------------------------------------------"
    echo -e "${YELLOW}Workflow modes:${NC}"
    echo "./trainer/run_byte_latent_workflow.sh fresh"
    echo "./trainer/run_byte_latent_workflow.sh resume"
    echo "./trainer/run_byte_latent_workflow.sh full"
    echo -e "${YELLOW}Checkpoint status:${NC}"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir fresh)\" status"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir fresh)\" health"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir fresh)\" ci"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir resume)\" status"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir resume)\" health"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$(resolve_checkpoint_dir resume)\" ci"
    echo -e "${YELLOW}Workflow manifest:${NC}"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" inspectfields"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" inspectfields:ci"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:ci"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:all"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:fresh"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runcommands:resume"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runfields:ci"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runfields"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" commands:summary"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" overview"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" runs json"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" status"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" health"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" delta:historyRows"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$manifest_file\" bundle=standard json \"$bundle_dir\""
    echo -e "${YELLOW}Workflow bundle:${NC}"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" status"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" health"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" loadfile:section=runs"
    echo "jbang trainer/trainer_byte_latent_workflow_inspector.java \"$bundle_dir\" loadfile:section=runs:index=0 json"
}

case "$MODE" in
    fresh|resume|full)
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Unsupported mode:${NC} $MODE"
        print_usage
        exit 1
        ;;
esac

print_header

if [[ "$MODE" == "fresh" ]]; then
    run_mode "fresh"
elif [[ "$MODE" == "resume" ]]; then
    run_mode "resume"
else
    run_mode "fresh"
    run_mode "resume"
fi

WORKFLOW_MANIFEST_FILE="$(write_workflow_manifest)"
materialize_workflow_bundle_if_needed "$WORKFLOW_MANIFEST_FILE"
print_workflow_summary
print_next_steps
