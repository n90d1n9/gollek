#!/bin/bash
# Smoke runner for the Gollek byte-latent resume demo + inspector flow.
# Usage: ./run_byte_latent_resume_smoke.sh [checkpoint-dir]

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="${1:-$EXAMPLES_DIR/trainer_checkpoints/byte_latent_resume_smoke}"

print_header() {
    echo "===================================================="
    echo " Gollek Byte-Latent Resume Smoke Runner"
    echo "===================================================="
    echo "examplesDir=$EXAMPLES_DIR"
    echo "checkpointDir=$CHECKPOINT_DIR"
}

check_jbang() {
    if ! command -v jbang >/dev/null 2>&1; then
        echo -e "${RED}JBang is required but was not found on PATH.${NC}"
        echo "Install it with: brew install jbang/tap/jbang"
        exit 1
    fi
}

run_resume_demo() {
    echo -e "${BLUE}Running byte-latent resume demo...${NC}"
    (
        cd "$EXAMPLES_DIR"
        jbang --fresh trainer/trainer_byte_latent_resume_demo.java "$CHECKPOINT_DIR"
    )
}

assert_artifacts() {
    local files=(
        "$CHECKPOINT_DIR/byte-latent-summary.json"
        "$CHECKPOINT_DIR/byte-latent-history.csv"
        "$CHECKPOINT_DIR/byte-latent-report.json"
        "$CHECKPOINT_DIR/byte-latent-checkpoint.metadata"
    )
    echo -e "${BLUE}Checking resumed checkpoint artifacts...${NC}"
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}Missing required artifact:${NC} $file"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} $file"
    done
}

assert_history_growth() {
    local history_file="$CHECKPOINT_DIR/byte-latent-history.csv"
    local line_count
    line_count=$(wc -l < "$history_file")
    echo -e "${BLUE}Checking resumed history growth...${NC}"
    if [[ "$line_count" -lt 4 ]]; then
        echo -e "${RED}Expected resumed history to contain at least 4 lines:${NC} $history_file"
        echo "actualLineCount=$line_count"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} history rows persisted (lineCount=$line_count)"
}

run_inspector() {
    local section="$1"
    local format="${2:-text}"
    echo -e "${BLUE}Inspecting:${NC} section=$section format=$format"
    (
        cd "$EXAMPLES_DIR"
        jbang --fresh trainer/trainer_byte_latent_history_inspector.java "$CHECKPOINT_DIR" "$section" "$format"
    )
}

print_next_steps() {
    echo "----------------------------------------------------"
    echo -e "${YELLOW}Manual follow-ups:${NC}"
    echo "cd $EXAMPLES_DIR"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" status"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" health"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" ci"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" summary:metadata:globalStep"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" history:sort=-trainLoss:top=3 json"
}

print_header
check_jbang
run_resume_demo
assert_artifacts
assert_history_growth
run_inspector "status" "text"
run_inspector "health" "text"
run_inspector "ci" "text"
run_inspector "summary:metadata:resumeLoaded" "text"
run_inspector "history:sort=-trainLoss:top=3" "json"
print_next_steps
