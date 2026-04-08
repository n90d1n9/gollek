#!/bin/bash
# Run Gollek Quantizer JBang Examples
# Usage: ./run.sh [example-name] [options]

set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example definitions
declare -A EXAMPLES
EXAMPLES=(
    ["gptq"]="gollek-quantizer-gptq.java|GPTQ 4-bit quantization (Hessian-based)"
    ["awq"]="gollek-quantizer-awq.java|AWQ quantization (Activation-Aware)"
    ["autoround"]="gollek-quantizer-autoround.java|AutoRound quantization (SignSGD)"
    ["turboquant"]="gollek-quantizer-turboquant.java|TurboQuant (Edge-optimized)"
    ["comparison"]="gollek-quantizer-comparison.java|Quantizer comparison & benchmark"
)

# Functions
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}       Gollek Quantizer JBang Examples Runner             ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo -e "${BOLD}Usage:${NC} $0 <example-name> [options]"
    echo ""
    echo -e "${BOLD}Available Examples:${NC}"
    for key in "${!EXAMPLES[@]}"; do
        IFS='|' read -r file desc <<< "${EXAMPLES[$key]}"
        printf "  ${GREEN}%-15s${NC} %s\n" "$key" "$desc"
    done
    echo ""
    echo -e "${BOLD}Common Options:${NC}"
    echo "  --demo              Run in demo mode (no model required)"
    echo "  --model <path>      Path to model file"
    echo "  --output <path>     Output path for quantized model"
    echo "  --bits <n>          Quantization bits (2, 3, 4, 8)"
    echo "  --group-size <n>    Group size (default: 128)"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 gptq --demo"
    echo "  $0 awq --demo"
    echo "  $0 autoround --demo"
    echo "  $0 turboquant --detect /path/to/model.safetensors"
    echo "  $0 comparison --demo"
    echo "  $0 comparison --benchmark --dimension 128 --iterations 1000"
    echo "  $0 comparison --recommend \"LLM serving\""
    echo ""
}

check_jbang_installed() {
    if ! command -v jbang &> /dev/null; then
        echo -e "${RED}❌ JBang is not installed${NC}"
        echo ""
        echo -e "${BLUE}Install JBang:${NC}"
        echo "  curl -Ls https://sh.jbang.dev | bash -s -"
        echo ""
        exit 1
    fi
}

check_java_version() {
    local java_version
    java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d. -f1)
    if [ "$java_version" -lt 25 ]; then
        echo -e "${RED}❌ Java 25 or higher is required (found Java $java_version)${NC}"
        echo ""
        echo -e "${BLUE}Current Java version:${NC} $(java -version 2>&1)"
        exit 1
    fi
}

run_example() {
    local example_name="$1"
    shift
    local options="$@"

    # Get example file
    local example_info="${EXAMPLES[$example_name]}"
    if [ -z "$example_info" ]; then
        echo -e "${RED}❌ Unknown example: '$example_name'${NC}"
        echo ""
        print_usage
        exit 1
    fi

    IFS='|' read -r example_file description <<< "$example_info"
    local example_path="$SCRIPT_DIR/$example_file"

    if [ ! -f "$example_path" ]; then
        echo -e "${RED}❌ Example file not found: $example_path${NC}"
        exit 1
    fi

    echo -e "${BLUE}Running example:${NC} $example_name"
    echo -e "${BLUE}Description:${NC} $description"
    echo -e "${BLUE}File:${NC} $example_file"
    echo ""

    # Run with JBang
    cd "$SCRIPT_DIR"
    if [ -n "$options" ]; then
        echo -e "${BLUE}Command:${NC} jbang --fresh $example_file $options"
        echo ""
        jbang --fresh "$example_file" $options
    else
        echo -e "${BLUE}Command:${NC} jbang --fresh $example_file"
        echo ""
        jbang --fresh "$example_file"
    fi
}

# Main execution
print_header

# Check prerequisites
check_java_version
check_jbang_installed

# Parse arguments
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

EXAMPLE_NAME="$1"
shift

# Handle help flag
if [ "$EXAMPLE_NAME" = "help" ] || [ "$EXAMPLE_NAME" = "--help" ] || [ "$EXAMPLE_NAME" = "-h" ]; then
    print_usage
    exit 0
fi

# Run example
run_example "$EXAMPLE_NAME" "$@"
