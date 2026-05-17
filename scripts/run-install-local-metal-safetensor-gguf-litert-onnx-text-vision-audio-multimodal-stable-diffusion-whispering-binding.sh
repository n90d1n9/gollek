#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Resolved backend targets: metal"
echo "Resolved format targets: safetensor,gguf,litert,onnx"
echo "Resolved LLM targets: text,vision,audio,multimodal,stable-diffusion,whispering"
echo "Architecture: binding"
echo "Profile: snapshot"
echo "Selection manifest: /Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/scripts/module-selection-metal-safetensor-gguf-litert-onnx-text-vision-audio-multimodal-stable-diffusion-whispering-binding.json"


./gradlew :backend:metal:gollek-backend-metal:publishToMavenLocal :backend:metal:gollek-mlx-binding:publishToMavenLocal :core:gollek-core:publishToMavenLocal :core:gollek-error-code:publishToMavenLocal :core:gollek-model-repository:publishToMavenLocal :core:gollek-provider-core:publishToMavenLocal :core:gollek-tokenizer-core:publishToMavenLocal :core:plugin:gollek-plugin-kernel-core:publishToMavenLocal :core:plugin:gollek-plugin-runner-gguf:publishToMavenLocal :models:gollek-model-gemma:publishToMavenLocal :models:gollek-model-llama:publishToMavenLocal :models:gollek-model-mistral:publishToMavenLocal :models:gollek-model-phi:publishToMavenLocal :models:gollek-model-qwen:publishToMavenLocal :models:gollek-model-repo-hf:publishToMavenLocal :models:gollek-model-repo-local:publishToMavenLocal :models:gollek-model-runner:publishToMavenLocal :runner:diffuser:gollek-diffuser:publishToMavenLocal :runner:gguf:gollek-gguf-converter:publishToMavenLocal :runner:gguf:gollek-gguf-converter-java:publishToMavenLocal :runner:gguf:gollek-gguf-core:publishToMavenLocal :runner:litert:gollek-litert-core:publishToMavenLocal :runner:litert:gollek-plugin-runner-litert:publishToMavenLocal :runner:litert:gollek-runner-litert:publishToMavenLocal :runner:onnx:gollek-ml-export-onnx:publishToMavenLocal :runner:onnx:gollek-ml-onnx:publishToMavenLocal :runner:onnx:gollek-plugin-runner-onnx:publishToMavenLocal :runner:onnx:gollek-runner-onnx:publishToMavenLocal :runner:safetensor:gollek-runner-safetensor:publishToMavenLocal :runner:safetensor:gollek-runner-stable-diffusion:publishToMavenLocal :runner:safetensor:gollek-safetensor-api:publishToMavenLocal :runner:safetensor:gollek-safetensor-core:publishToMavenLocal :runner:safetensor:gollek-safetensor-engine:publishToMavenLocal :runner:safetensor:gollek-safetensor-loader:publishToMavenLocal :runner:safetensor:gollek-safetensor-quantization:publishToMavenLocal :runner:safetensor:gollek-safetensor-spi:publishToMavenLocal :sdk:gollek-sdk:publishToMavenLocal :sdk:gollek-sdk-api:publishToMavenLocal :sdk:gollek-sdk-core:publishToMavenLocal :sdk:gollek-sdk-local:publishToMavenLocal :sdk:gollek-sdk-session:publishToMavenLocal :spi:gollek-spi:publishToMavenLocal :spi:gollek-spi-inference:publishToMavenLocal :spi:gollek-spi-model:publishToMavenLocal :spi:gollek-spi-multimodal:publishToMavenLocal :spi:gollek-spi-plugin:publishToMavenLocal :spi:gollek-spi-provider:publishToMavenLocal :spi:gollek-spi-runtime:publishToMavenLocal :ml:gollek-ml-nlp:publishToMavenLocal :ml:gollek-ml-audio:publishToMavenLocal :ml:gollek-ml-cnn:publishToMavenLocal :ml:gollek-ml-multimodal:publishToMavenLocal :ml:gollek-ml-vision:publishToMavenLocal :ui:gollek-api:publishToMavenLocal :ui:gollek-cli:publishToMavenLocal \
  -Pgollek.backend=cpu,metal \
  -Pgollek.profile=snapshot \
  -Pgollek.model.formats=all \
  -Pgollek.llm.types=all \
  -Pgollek.architecture=binding
