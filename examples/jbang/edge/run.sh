#!/bin/bash
# Enhanced Gollek LiteRT Edge Examples Runner
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
GOLLEK_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Example definitions
declare -A EXAMPLES
EXAMPLES=(
    ["edge"]="LiteRTEdgeExample.java|Basic LiteRT edge inference"
    ["image"]="ImageClassificationExample.java|Image classification with MobileNet"
    ["object"]="ObjectDetectionExample.java|Object detection with SSD MobileNet"
    ["text"]="TextGenerationExample.java|Text generation with language models"
    ["benchmark"]="BenchmarkExample.java|Performance benchmarking"
)

# Functions
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}       Gollek LiteRT Edge Examples Runner               ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo -e "${BOLD}Usage:${NC} $0 <example-name> [options]"
    echo ""
    echo -e "${BOLD}Available Examples:${NC}"
    for key in "${!EXAMPLES[@]}"; do
        IFS='|' read -r file desc <<< "${EXAMPLES[$key]}"
        printf "  ${GREEN}%-12s${NC} %s\n" "$key" "$desc"
    done
    echo ""
    echo -e "${BOLD}Common Options:${NC}"
    echo "  --model <path>          Path to model file (.litertlm or .litert)"
    echo "  --delegate <type>       Hardware delegate (NONE, CPU, GPU, NNAPI, COREML, AUTO)"
    echo "  --threads <count>       Number of CPU threads (default: 4)"
    echo "  --xnnpack <true|false>  Enable XNNPACK optimization (default: true)"
    echo ""
    echo -e "${BOLD}Example-specific Options:${NC}"
    echo "  Image Classification:"
    echo "    --image <path>          Input image path (required)"
    echo "    --topk <count>          Number of top predictions (default: 5)"
    echo ""
    echo "  Object Detection:"
    echo "    --image <path>          Input image path (required)"
    echo "    --threshold <value>     Confidence threshold (default: 0.5)"
    echo "    --output <path>         Output image path"
    echo ""
    echo "  Text Generation:"
    echo "    --prompt <text>         Input prompt"
    echo "    --maxtokens <count>     Max tokens to generate (default: 50)"
    echo "    --temperature <value>   Sampling temperature (default: 0.7)"
    echo "    --topk <count>          Top-K sampling (default: 40)"
    echo "    --interactive           Enable interactive chat mode"
    echo ""
    echo "  Benchmark:"
    echo "    --iterations <count>    Number of iterations (default: 50)"
    echo "    --batch <size>          Batch size (default: 1)"
    echo "    --warmup <count>        Warmup runs (default: 10)"
    echo "    --output <file>         CSV output file"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 edge"
    echo "  $0 image --model mobilenet.litert --image cat.jpg"
    echo "  $0 object --model ssd_mobilenet.litert --image street.jpg --threshold 0.6"
    echo "  $0 text --model gemma.litertlm --prompt \"Hello world\" --interactive"
    echo "  $0 benchmark --model model.litert --iterations 100 --batch 8"
    echo ""
}

check_sdk_built() {
    if [ ! -d "$GOLLEK_DIR/sdk/lib/gollek-sdk-litert/target" ]; then
        echo -e "${YELLOW}⚠️  SDK module not built yet.${NC}"
        echo ""
        echo -e "${BLUE}Building LiteRT SDK module...${NC}"
        echo ""
        cd "$GOLLEK_DIR"
        if mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests -q; then
            echo ""
            echo -e "${GREEN}✓ SDK module built successfully${NC}"
            echo ""
        else
            echo -e "${RED}❌ SDK build failed. Please build manually:${NC}"
            echo "  cd $GOLLEK_DIR && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests"
            exit 1
        fi
    fi
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
    if [ "$java_version" -lt 21 ]; then
        echo -e "${RED}❌ Java 21 or higher is required (found Java $java_version)${NC}"
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

# Check SDK
check_sdk_built

# Run example
run_example "$EXAMPLE_NAME" "$@"
