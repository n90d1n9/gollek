#!/bin/bash
# 🎨 Stable Diffusion - Quick Run Script
# Usage: ./sd_generate.sh "a cat" 10 cat.png

PROMPT="${1:-a cat playing ball}"
STEPS="${2:-10}"
OUTPUT="${3:-output.png}"
SEED="${4:-42}"
CFG="${5:-7.5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER_JAR="$PROJECT_ROOT/gollek/ui/gollek-cli/target/gollek-runner.jar"

if [ ! -f "$RUNNER_JAR" ]; then
    echo "❌ Runner JAR not found. Build first:"
    echo "   mvn clean package -pl gollek/ui/gollek-cli -am -DskipTests -Dmaven.javadoc.skip=true"
    exit 1
fi

echo "🎨 Gollek Stable Diffusion"
echo "Prompt: $PROMPT"
echo "Steps: $STEPS | Seed: $SEED | CFG: $CFG"
echo "Output: $OUTPUT"
echo

java -jar "$RUNNER_JAR" run \
    --model CompVis/stable-diffusion-v1-4 \
    --branch onnx \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --guidance-scale "$CFG" \
    --output "$OUTPUT"
