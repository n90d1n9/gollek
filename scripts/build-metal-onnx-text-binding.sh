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

./gradlew clean :backend:metal:gollek-backend-metal:build :backend:metal:gollek-mlx-binding:build :core:gollek-core:build :core:gollek-error-code:build :core:gollek-model-repository:build :core:gollek-provider-core:build :core:gollek-tokenizer-core:build :core:plugin:gollek-plugin-kernel-core:build :models:gollek-model-gemma:build :models:gollek-model-llama:build :models:gollek-model-mistral:build :models:gollek-model-phi:build :models:gollek-model-qwen:build :models:gollek-model-repo-hf:build :models:gollek-model-repo-local:build :models:gollek-model-runner:build :runner:litert:gollek-litert-core:build :runner:onnx:gollek-ml-export-onnx:build :runner:onnx:gollek-ml-onnx:build :runner:onnx:gollek-plugin-runner-onnx:build :runner:onnx:gollek-runner-onnx:build :sdk:gollek-sdk:build :sdk:gollek-sdk-api:build :sdk:gollek-sdk-core:build :sdk:gollek-sdk-local:build :sdk:gollek-sdk-session:build :spi:gollek-spi:build :spi:gollek-spi-inference:build :spi:gollek-spi-model:build :spi:gollek-spi-multimodal:build :spi:gollek-spi-plugin:build :spi:gollek-spi-provider:build :spi:gollek-spi-runtime:build :ui:gollek-api:build :ui:gollek-cli:build \
  -Pgollek.backend=cpu,metal \
  -Pgollek.profile=snapshot \
  -Pgollek.model.formats=onnx \
  -Pgollek.llm.types=text \
  -Pgollek.architecture=binding
