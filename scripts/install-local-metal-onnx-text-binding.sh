#!/bin/bash
set -euo pipefail

RESET="\033[0m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
BOLD="\033[1m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo -e "${BOLD}${CYAN}:) Resolved backend targets:${RESET} metal"
echo -e "${BOLD}${CYAN}:) Resolved format targets:${RESET} onnx"
echo -e "${BOLD}${CYAN}:) Resolved LLM targets:${RESET} text"
echo -e "${BOLD}${MAGENTA}:) Architecture:${RESET} binding"
echo -e "${BOLD}${MAGENTA}:) Profile:${RESET} snapshot"
echo -e "${BOLD}${GREEN}:) Selection manifest:${RESET} /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/scripts/module-selection-metal-onnx-text-binding.json"
echo ":) Module manifest resolved cleanly."

./gradlew :backend:metal:gollek-backend-metal:publishToMavenLocal :backend:metal:gollek-mlx-binding:publishToMavenLocal :core:gollek-core:publishToMavenLocal :core:gollek-error-code:publishToMavenLocal :core:gollek-model-repository:publishToMavenLocal :core:gollek-provider-core:publishToMavenLocal :core:gollek-tokenizer-core:publishToMavenLocal :core:plugin:gollek-plugin-kernel-core:publishToMavenLocal :models:gollek-model-gemma:publishToMavenLocal :models:gollek-model-llama:publishToMavenLocal :models:gollek-model-mistral:publishToMavenLocal :models:gollek-model-phi:publishToMavenLocal :models:gollek-model-qwen:publishToMavenLocal :models:gollek-model-repo-hf:publishToMavenLocal :models:gollek-model-repo-local:publishToMavenLocal :models:gollek-model-runner:publishToMavenLocal :runner:litert:gollek-litert-core:publishToMavenLocal :runner:onnx:gollek-ml-export-onnx:publishToMavenLocal :runner:onnx:gollek-ml-onnx:publishToMavenLocal :runner:onnx:gollek-plugin-runner-onnx:publishToMavenLocal :runner:onnx:gollek-runner-onnx:publishToMavenLocal :sdk:gollek-sdk:publishToMavenLocal :sdk:gollek-sdk-api:publishToMavenLocal :sdk:gollek-sdk-core:publishToMavenLocal :sdk:gollek-sdk-local:publishToMavenLocal :sdk:gollek-sdk-session:publishToMavenLocal :spi:gollek-spi:publishToMavenLocal :spi:gollek-spi-inference:publishToMavenLocal :spi:gollek-spi-model:publishToMavenLocal :spi:gollek-spi-multimodal:publishToMavenLocal :spi:gollek-spi-plugin:publishToMavenLocal :spi:gollek-spi-provider:publishToMavenLocal :spi:gollek-spi-runtime:publishToMavenLocal :ui:gollek-api:publishToMavenLocal :ui:gollek-cli:publishToMavenLocal \
  -Pgollek.backend=cpu,metal \
  -Pgollek.profile=snapshot \
  -Pgollek.model.formats=onnx \
  -Pgollek.llm.types=text \
  -Pgollek.architecture=binding
