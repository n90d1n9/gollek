#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Backend targets: metal"
echo "Resolved backend targets: metal"
echo "Resolved format targets: safetensor,gguf,litert,onnx"
echo "Resolved LLM targets: text,vision,audio,multimodal,stable-diffusion,whispering"
echo "Architecture: binding"
echo "Profile: snapshot"
echo "Selection manifest: /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/scripts/module-selection-metal-safetensor-gguf-litert-onnx-text-vision-audio-multimodal-stable-diffusion-whispering-binding.json"


./gradlew clean :backend:metal:gollek-backend-metal:build :backend:metal:gollek-mlx-binding:build :core:gollek-core:build :core:gollek-error-code:build :core:gollek-model-repository:build :core:gollek-provider-core:build :core:gollek-tokenizer-core:build :core:plugin:gollek-plugin-kernel-core:build :core:plugin:gollek-plugin-runner-gguf:build :models:gollek-model-gemma:build :models:gollek-model-llama:build :models:gollek-model-mistral:build :models:gollek-model-phi:build :models:gollek-model-qwen:build :models:gollek-model-repo-hf:build :models:gollek-model-repo-local:build :models:gollek-model-runner:build :runner:diffuser:gollek-diffuser:build :runner:gguf:gollek-gguf-converter:build :runner:gguf:gollek-gguf-converter-java:build :runner:gguf:gollek-gguf-core:build :runner:litert:gollek-litert-core:build :runner:litert:gollek-plugin-runner-litert:build :runner:litert:gollek-runner-litert:build :runner:onnx:gollek-ml-export-onnx:build :runner:onnx:gollek-ml-onnx:build :runner:onnx:gollek-plugin-runner-onnx:build :runner:onnx:gollek-runner-onnx:build :runner:safetensor:gollek-runner-safetensor:build :runner:safetensor:gollek-runner-stable-diffusion:build :runner:safetensor:gollek-safetensor-api:build :runner:safetensor:gollek-safetensor-core:build :runner:safetensor:gollek-safetensor-engine:build :runner:safetensor:gollek-safetensor-loader:build :runner:safetensor:gollek-safetensor-quantization:build :runner:safetensor:gollek-safetensor-spi:build :sdk:gollek-sdk:build :sdk:gollek-sdk-api:build :sdk:gollek-sdk-core:build :sdk:gollek-sdk-local:build :sdk:gollek-sdk-session:build :spi:gollek-spi:build :spi:gollek-spi-inference:build :spi:gollek-spi-model:build :spi:gollek-spi-multimodal:build :spi:gollek-spi-plugin:build :spi:gollek-spi-provider:build :spi:gollek-spi-runtime:build :ml:gollek-ml-nlp:build :ml:gollek-ml-audio:build :ml:gollek-ml-cnn:build :ml:gollek-ml-multimodal:build :ml:gollek-ml-vision:build :ui:gollek-api:build :ui:gollek-cli:build \
  -Pgollek.backend=cpu,metal \
  -Pgollek.profile=snapshot \
  -Pgollek.model.formats=all \
  -Pgollek.llm.types=all \
  -Pgollek.architecture=binding
