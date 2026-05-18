#!/bin/bash
# Smoke runner for the Gollek byte-latent JBang demo + inspector flow.
# Usage: ./run_byte_latent_smoke.sh [checkpoint-dir]

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="${1:-$EXAMPLES_DIR/trainer_checkpoints/byte_latent_smoke}"

print_header() {
    echo "===================================================="
    echo " Gollek Byte-Latent Smoke Runner"
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

run_demo() {
    echo -e "${BLUE}Running byte-latent demo...${NC}"
    (
        cd "$EXAMPLES_DIR"
        jbang --fresh trainer/trainer_byte_latent_demo.java "$CHECKPOINT_DIR"
    )
}

assert_artifacts() {
    local files=(
        "$CHECKPOINT_DIR/byte-latent-summary.json"
        "$CHECKPOINT_DIR/byte-latent-history.csv"
        "$CHECKPOINT_DIR/byte-latent-report.json"
        "$CHECKPOINT_DIR/byte-latent-checkpoint.metadata"
    )
    echo -e "${BLUE}Checking checkpoint artifacts...${NC}"
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo -e "${RED}Missing required artifact:${NC} $file"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} $file"
    done
}

run_inspector() {
    local section="$1"
    local format="$2"
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
    echo "jbang trainer/trainer_byte_latent_resume_demo.java \"$CHECKPOINT_DIR\""
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" status"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" health"
    echo "jbang trainer/trainer_byte_latent_history_inspector.java \"$CHECKPOINT_DIR\" ci"
}

print_header
check_jbang
run_demo
assert_artifacts
run_inspector "status" "text"
run_inspector "health" "text"
run_inspector "ci" "text"
run_inspector "history:sort=-trainLoss:top=3" "json"
print_next_steps
