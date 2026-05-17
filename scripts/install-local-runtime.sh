#!/bin/bash
# Gollek - Local Development Installer
# Builds the CLI from source and installs a local shim for testing.
# Usage: ./scripts/install-local-runtime.sh [--native]

set -e

# --- Default Options ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRADLEW="${PROJECT_ROOT}/gradlew"
CLI_MODULE="ui/gollek-cli"
SDK_MODULE="sdk/gollek-sdk-local"
SAFETENSOR_ENGINE_MODULE="runner/safetensor/gollek-safetensor-engine"
METAL_MODULE="backend/metal/gollek-backend-metal"
BIN_DIR="${GOLLEK_BIN_DIR:-${HOME}/.local/bin}"
LIB_DIR="${GOLLEK_LIB_DIR:-${HOME}/.gollek/libs}"
RUNTIME_DIR="${GOLLEK_RUNTIME_DIR:-${HOME}/.gollek/runtime}"
GOLLEK_RUNTIME_JAR="${RUNTIME_DIR}/gollek.jar"
GOLLEK_CLI_BIN="${BIN_DIR}/gollek"
GGUF_FAST_BRIDGE_SRC="${PROJECT_ROOT}/runner/gguf/gguf-fast-bridge/gollek_gguf_fast.cpp"
NATIVE_MODE=false
BUILD_ARGS="-c"
SKIP_LEGACY_SOURCE_HOTPATCHES="${GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES:-true}"
SKIP_BUILD="${GOLLEK_SKIP_BUILD:-false}"

if command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JAVA_25_HOME="$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
    if [ -n "$JAVA_25_HOME" ]; then
        export JAVA_HOME="$JAVA_25_HOME"
        export PATH="${JAVA_HOME}/bin:${PATH}"
    fi
fi

append_java_tool_option() {
    local option="$1"
    case " ${JAVA_TOOL_OPTIONS:-} " in
        *" ${option} "*) ;;
        *) JAVA_TOOL_OPTIONS="${option}${JAVA_TOOL_OPTIONS:+ ${JAVA_TOOL_OPTIONS}}" ;;
    esac
}

append_java_tool_option "--enable-native-access=ALL-UNNAMED"
append_java_tool_option "--add-modules=jdk.incubator.vector"
export JAVA_TOOL_OPTIONS

# --- Visuals ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--native) NATIVE_MODE=true; BUILD_ARGS="$BUILD_ARGS -n" ;;
        -t|--tests) BUILD_ARGS="$BUILD_ARGS -t" ;;
        --skip-tests) BUILD_ARGS="$BUILD_ARGS" ;; # build.sh skips by default
        -h|--help) 
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -n, --native     Build and install as native executable (requires GraalVM)"
            echo "  -t, --tests      Run tests during build"
            echo "  --skip-tests     Skip tests during build (default)"
            echo "  -h, --help       Show this help message"
            exit 0 
            ;;
    esac
    shift
done

echo -e "${BLUE}--------------------------------------------------${NC}"
echo -e "${GREEN} Gollek Local Installer (Dev Mode) ${NC}"
if [ "$NATIVE_MODE" = true ]; then
    echo -e "${YELLOW} Mode: Native (GraalVM) ${NC}"
else
    echo -e "${YELLOW} Mode: JVM (Standard JAR) ${NC}"
fi
echo -e "${YELLOW} Build: Gradle-only ${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"

# Tokenization is pure Java on the Gradle install path. Historical native
# SentencePiece install/copy logic is intentionally disabled so local installs
# do not imply an external tokenizer runtime dependency.
install_tokenizer_native_runtime() {
    mkdir -p "${HOME}/.gollek/native" "${LIB_DIR}"
    echo -e "${GREEN}✓ Tokenizer runtime: pure Java (no SentencePiece native dependency)${NC}"
}

stop_daemon_via_cli() {
    local command="$1"
    if [ ! -x "$GOLLEK_CLI_BIN" ]; then
        return 0
    fi
    GOLLEK_GGUF_FAST_RUN=true \
        GOLLEK_GGUF_FAST_DAEMON=true \
        GOLLEK_LITERT_FAST_RUN=true \
        GOLLEK_LITERT_FAST_DAEMON=true \
        "$GOLLEK_CLI_BIN" "$command" >/dev/null 2>&1 || true
}

stop_daemon_from_port_file() {
    local port_file="$1"
    local class_name="$2"
    local label="$3"
    local pid command attempt

    if [ ! -f "$port_file" ]; then
        return 0
    fi
    pid="$(sed -n '2p' "$port_file" 2>/dev/null | tr -d '[:space:]' || true)"
    case "$pid" in
        ''|*[!0-9]*) return 0 ;;
    esac
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
    fi
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    case "$command" in
        *"$class_name"*"__daemon"*) ;;
        *) return 0 ;;
    esac

    echo -e "${YELLOW}⚠ ${label} fast daemon did not stop via protocol; terminating pid ${pid}.${NC}"
    kill "$pid" >/dev/null 2>&1 || true
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    kill -9 "$pid" >/dev/null 2>&1 || true
}

stop_existing_fast_daemons() {
    mkdir -p "${HOME}/.gollek/run" "${HOME}/.gollek/logs"
    echo -e "${BLUE}>>> Stopping existing fast inference daemons...${NC}"
    stop_daemon_via_cli "__gguf-daemon-stop"
    stop_daemon_via_cli "__daemon-stop"
    stop_daemon_from_port_file \
        "${HOME}/.gollek/run/gguf-fast-daemon.port" \
        "tech.kayys.gollek.cli.commands.GgufFastRun" \
        "GGUF"
    stop_daemon_from_port_file \
        "${HOME}/.gollek/run/litert-fast-daemon.port" \
        "tech.kayys.gollek.cli.commands.LiteRtLmFastRun" \
        "LiteRT"
    if [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1; then
        launchctl remove tech.kayys.gollek.gguf-fast-daemon >/dev/null 2>&1 || true
        launchctl remove tech.kayys.gollek.litert-fast-daemon >/dev/null 2>&1 || true
    fi
    rm -f "${HOME}/.gollek/run/gguf-fast-daemon.port" \
          "${HOME}/.gollek/run/litert-fast-daemon.port"
}

build_gguf_fast_bridge() {
    if [ "$(uname -s)" != "Darwin" ]; then
        return 0
    fi
    if [ ! -f "$GGUF_FAST_BRIDGE_SRC" ]; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (source file not found).${NC}"
        return 0
    fi
    if ! command -v c++ >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (c++ compiler not found).${NC}"
        return 0
    fi

    local gemma4_llama_src="${HOME}/.gollek/source/vendor/llama.cpp-gemma4/llama.cpp"
    local gemma4_llama_lib_dir="${gemma4_llama_src}/build-gollek/bin"
    local default_llama_src="${HOME}/.gollek/source/vendor/llama.cpp/llama.cpp"
    local llama_src="${GOLLEK_LLAMA_SOURCE_DIR:-}"
    local llama_lib_dir="${GOLLEK_LLAMA_LIB_DIR:-}"
    if [ -z "$llama_src" ] && [ -f "${gemma4_llama_src}/include/llama.h" ]; then
        llama_src="$gemma4_llama_src"
    fi
    if [ -z "$llama_src" ]; then
        llama_src="$default_llama_src"
    fi
    if [ -z "$llama_lib_dir" ] && [ -f "${gemma4_llama_lib_dir}/libllama.dylib" ]; then
        llama_lib_dir="$gemma4_llama_lib_dir"
    fi
    if [ -z "$llama_lib_dir" ]; then
        llama_lib_dir="${LIB_DIR}/llama"
    fi
    local llama_header="${llama_src}/include/llama.h"
    local ggml_header="${llama_src}/ggml/include/ggml-backend.h"
    if [ ! -f "$llama_header" ] || [ ! -f "$ggml_header" ] || [ ! -f "${llama_lib_dir}/libllama.dylib" ]; then
        echo -e "${YELLOW}⚠ Skipping GGUF fast bridge build (llama.cpp headers/libs not found).${NC}"
        echo -e "${YELLOW}  Expected headers under ${llama_src} and libs under ${llama_lib_dir}.${NC}"
        return 0
    fi

    local runtime_llama_lib_dir="${LIB_DIR}/llama"
    if [ "$llama_lib_dir" != "$runtime_llama_lib_dir" ]; then
        echo -e "${BLUE}>>> Installing llama.cpp runtime libraries for GGUF fast path...${NC}"
        mkdir -p "$runtime_llama_lib_dir"
        local copied_any=false
        for lib in "$llama_lib_dir"/libllama*.dylib "$llama_lib_dir"/libggml*.dylib "$llama_lib_dir"/libmtmd*.dylib; do
            if [ -e "$lib" ]; then
                cp -PR "$lib" "$runtime_llama_lib_dir/"
                copied_any=true
            fi
        done
        if [ "$copied_any" = true ]; then
            echo -e "${GREEN}✓ Installed llama.cpp runtime libraries to ${runtime_llama_lib_dir}${NC}"
            llama_lib_dir="$runtime_llama_lib_dir"
        else
            echo -e "${YELLOW}⚠ No llama.cpp runtime libraries were copied from ${llama_lib_dir}.${NC}"
        fi
    fi

    echo -e "${BLUE}>>> Building and installing GGUF llama.cpp fast bridge...${NC}"
    mkdir -p "$LIB_DIR"
    if c++ -std=c++17 -O3 -DNDEBUG -fPIC -dynamiclib \
        -I"${llama_src}/include" \
        -I"${llama_src}/ggml/include" \
        "$GGUF_FAST_BRIDGE_SRC" \
        -L"${llama_lib_dir}" \
        -lllama -lggml -lggml-cpu -lggml-blas -lggml-metal -lggml-base \
        -Wl,-rpath,"${llama_lib_dir}" \
        -o "${LIB_DIR}/libgollek_gguf_fast.dylib"; then
        echo -e "${GREEN}✓ Installed libgollek_gguf_fast.dylib to ${LIB_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠ GGUF fast bridge build failed; full CLI fallback remains available.${NC}"
    fi
}

# 1. Build the project
stop_existing_fast_daemons
echo -e "${BLUE}>>> Building Gollek...${NC}"

METAL_MAKE_DIR="${PROJECT_ROOT}/${METAL_MODULE}/src/main/cpp/metal"
CLI_GRADLE_JAR="${PROJECT_ROOT}/${CLI_MODULE}/build/gollek.jar"
SAFETENSOR_ENGINE_CLASSES_GRADLE="${PROJECT_ROOT}/${SAFETENSOR_ENGINE_MODULE}/build/classes/java/main"
SAFETENSOR_ENGINE_CLASSES="$SAFETENSOR_ENGINE_CLASSES_GRADLE"
METAL_CLASSES_GRADLE="${PROJECT_ROOT}/${METAL_MODULE}/build/classes/java/main"

if [ ! -x "$GRADLEW" ]; then
    echo -e "${RED}❌ Gradle wrapper not found at ${GRADLEW}${NC}"
    exit 1
fi

GRADLE_ARGS=("--no-daemon" "--no-build-cache")
if [ -n "${JAVA_HOME:-}" ]; then
    GRADLE_ARGS=("-Dorg.gradle.java.home=${JAVA_HOME}" "${GRADLE_ARGS[@]}")
fi
if [[ "$BUILD_ARGS" != *"-t"* ]]; then
    GRADLE_ARGS+=("-x" "test")
fi

GRADLE_TASKS=(":ui:gollek-cli:quarkusBuild")
if [[ "$BUILD_ARGS" == *"-c"* ]]; then
    GRADLE_TASKS=("clean" "${GRADLE_TASKS[@]}")
fi
if [ "$NATIVE_MODE" = true ]; then
    echo -e "${YELLOW}ℹ Building native image (this may take a while)...${NC}"
    GRADLE_ARGS+=("-Dquarkus.native.enabled=true" "-Dquarkus.native.container-build=false" "-Dgraalvm.metadataRepository.enabled=false")
fi

if [ "$SKIP_BUILD" = "true" ]; then
    echo -e "${YELLOW}⚠ Skipping Gradle build because GOLLEK_SKIP_BUILD=true${NC}"
else
    echo -e "${BLUE}>>> Running: ${GRADLEW} ${GRADLE_ARGS[*]} ${GRADLE_TASKS[*]}${NC}"
    if "$GRADLEW" "${GRADLE_ARGS[@]}" "${GRADLE_TASKS[@]}"; then
        echo -e "${GREEN}✓ Built Gollek CLI via Gradle${NC}"
    else
        echo -e "${RED}❌ Gradle build failed. Maven fallback has been removed from this installer.${NC}"
        exit 1
    fi
fi

if [ "$NATIVE_MODE" = false ] && [ "$(uname -s)" = "Darwin" ] && [ -f "${METAL_MAKE_DIR}/Makefile" ]; then
    echo -e "${BLUE}>>> Building and installing Metal native bridge...${NC}"
    if make -C "${METAL_MAKE_DIR}" install INSTALL_DIR="${LIB_DIR}"; then
        echo -e "${GREEN}✓ Installed libgollek_metal.dylib to ${LIB_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠ Metal native bridge build/install failed; Java fallback path will still be available.${NC}"
    fi
fi

if [ "$NATIVE_MODE" = false ]; then
    build_gguf_fast_bridge
fi

# 2. Locate Artifacts
echo -e "${BLUE}>>> Locating artifacts...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # 2a. Locate Native Executable
    NATIVE_PATH="${PROJECT_ROOT}/${CLI_MODULE}/build/gollek"
    if [ ! -f "$NATIVE_PATH" ]; then
        NATIVE_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/build" -type f -perm -111 -name "*-runner" -maxdepth 3 | head -n 1)
    fi
    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        # Compatibility fallback for legacy Maven-native layouts.
        NATIVE_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek"
    fi
    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        NATIVE_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -type f -perm -111 -not -name "*.jar" -not -name "*.sh" -maxdepth 2 | head -n 1)
    fi

    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        echo -e "${RED}❌ Could not find native executable in ${CLI_MODULE}/build or ${CLI_MODULE}/target${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found Native Binary: $(basename "$NATIVE_PATH")${NC}"

    # 2b. Locate SDK Shared Library
    echo -e "${BLUE}>>> Locating SDK Shared Library...${NC}"
    LIB_NAME="libgollek_sdk_local.dylib"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then LIB_NAME="libgollek_sdk_local.so"; fi
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then LIB_NAME="gollek_sdk_local.dll"; fi
    
    SDK_LIB_PATH="${PROJECT_ROOT}/${SDK_MODULE}/target/${LIB_NAME}"
    if [ ! -f "$SDK_LIB_PATH" ]; then
        # Fallback search
        SDK_LIB_PATH=$(find "${PROJECT_ROOT}/${SDK_MODULE}/target" -name "*gollek_sdk_local*" -not -name "*.jar" -maxdepth 1 | head -n 1)
    fi
    
    if [ -n "$SDK_LIB_PATH" ] && [ -f "$SDK_LIB_PATH" ]; then
        echo -e "${GREEN}✓ Found SDK Library: $(basename "$SDK_LIB_PATH")${NC}"
        mkdir -p "$LIB_DIR"
        cp "$SDK_LIB_PATH" "$LIB_DIR/"
        echo -e "${GREEN}✓ Installed SDK Library to ${LIB_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠ Could not find SDK shared library. Native SDK features might be unavailable.${NC}"
    fi
else
    is_runnable_cli_jar() {
        local candidate="$1"
        [ -n "$candidate" ] && [ -f "$candidate" ] || return 1
        jar tf "$candidate" >/dev/null 2>&1 || return 1
        jar tf "$candidate" | grep -q "tech/kayys/gollek/cli/commands/LiteRtLmFastRun.class" || return 1
        jar tf "$candidate" | grep -q "com/google/ai/edge/litertlm/Engine.class" || return 1
        jar tf "$candidate" | grep -q "com/google/ai/edge/litertlm/Backend.class" || return 1
    }

    # Locate JAR
    JAR_PATH=""
    if is_runnable_cli_jar "${CLI_GRADLE_JAR}"; then
        JAR_PATH="${CLI_GRADLE_JAR}"
    fi
    if [ -z "$JAR_PATH" ]; then
        while IFS= read -r candidate; do
            if is_runnable_cli_jar "$candidate"; then
                JAR_PATH="$candidate"
                break
            fi
        done < <(find "${PROJECT_ROOT}/${CLI_MODULE}/build" -maxdepth 2 -name "gollek*.jar" -not -path "*/build/libs/*" 2>/dev/null | sort)
    fi
    if [ -z "$JAR_PATH" ]; then
        # Compatibility fallback for legacy Maven packagers.
        if is_runnable_cli_jar "${PROJECT_ROOT}/${CLI_MODULE}/target/gollek.jar"; then
            JAR_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek.jar"
        fi
    fi
    if [ -z "$JAR_PATH" ]; then
        while IFS= read -r candidate; do
            if is_runnable_cli_jar "$candidate"; then
                JAR_PATH="$candidate"
                break
            fi
        done < <(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -maxdepth 2 -name "gollek*.jar" 2>/dev/null | sort)
    fi
    if is_runnable_cli_jar "${CLI_GRADLE_JAR}"; then
        JAR_PATH="${CLI_GRADLE_JAR}"
    fi

    if [ -z "$JAR_PATH" ] || [ ! -f "$JAR_PATH" ]; then
        echo -e "${RED}❌ Could not find a runnable Gollek CLI fat JAR in ${CLI_MODULE}/build or ${CLI_MODULE}/target${NC}"
        echo -e "${YELLOW}  The slim Gradle jar under build/libs is intentionally rejected because it lacks LiteRT-LM runtime classes.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found JAR: $(basename "$JAR_PATH")${NC}"

    if [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
        echo -e "${YELLOW}⚠ Skipping legacy post-build jar hot-patches on Gradle install path.${NC}"
        echo -e "${YELLOW}  Gradle/Quarkus output is treated as the source of truth. Set GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES=false to re-enable the old patch path.${NC}"
    else
    # Patch GGUFTokenizer directly into the packaged jar so local runs use the
    # updated runtime-facing tokenizer even when the standalone GGUF reactor
    # is not buildable in this checkout.
    TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/runner/gguf/gollek-gguf-core/src/main/java/tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.java"
    TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-tokenizer"
    BPE_TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/core/gollek-tokenizer-core/src/main/java/tech/kayys/gollek/tokenizer/impl/BpeTokenizer.java"
    BPE_TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-bpe-tokenizer"
    LITERT_PATCH_OUT="${PROJECT_ROOT}/tools/.build-litert-runner"
    LITERT_TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTTokenizer.java"
    LITERT_BINDINGS_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTNativeBindings.java"
    LITERT_INFERENCE_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTInferenceRunner.java"
    LITERT_NATIVE_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTGemmaNativeRunner.java"
    LITERT_GEMMA_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTGemmaMetalRunner.java"
    LITERT_CPU_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTCpuRunner.java"
    LITERT_PROVIDER_PATCH_SRC="${PROJECT_ROOT}/runner/litert/gollek-runner-litert/src/main/java/tech/kayys/gollek/provider/litert/LiteRTProvider.java"
    CHAT_TEMPLATE_PATCH_SRC="${PROJECT_ROOT}/core/gollek-tokenizer-core/src/main/java/tech/kayys/gollek/tokenizer/template/ChatTemplateFormatter.java"
    CHAT_TEMPLATE_PATCH_OUT="${PROJECT_ROOT}/tools/.build-chat-template"
    GEMMA_FAMILY_PATCH_SRC="${PROJECT_ROOT}/models/gollek-model-gemma/src/main/java/tech/kayys/gollek/models/GemmaFamily.java"
    GEMMA_FAMILY_PATCH_OUT="${PROJECT_ROOT}/tools/.build-gemma-family"
    MODEL_CONFIG_PATCH_SRC="${PROJECT_ROOT}/spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/ModelConfig.java"
    MODEL_CONFIG_PATCH_OUT="${PROJECT_ROOT}/tools/.build-model-config"
    ACCEL_TENSOR_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-core/src/main/java/tech/kayys/gollek/safetensor/core/tensor/AccelTensor.java"
    ACCEL_TENSOR_PATCH_OUT="${PROJECT_ROOT}/tools/.build-accel-tensor"
    ACCEL_OPS_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-core/src/main/java/tech/kayys/gollek/safetensor/core/tensor/AccelOps.java"
    ACCEL_OPS_PATCH_OUT="${PROJECT_ROOT}/tools/.build-accel-ops"
    DIRECT_FORWARD_PASS_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass.java"
    DIRECT_INFERENCE_ENGINE_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine.java"
    FLASH_ATTENTION_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel.java"
    ROPE_FREQUENCY_CACHE_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache.java"
    KV_CACHE_MANAGER_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager.java"
    SAFETENSOR_ENGINE_PATCH_OUT="${PROJECT_ROOT}/tools/.build-safetensor-engine"
    METAL_FA4_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/binding/MetalFlashAttentionBinding.java"
    METAL_FA4_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-fa4"
    METAL_RUNNER_MODE_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/config/MetalRunnerMode.java"
    METAL_CAPABILITIES_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/detection/MetalCapabilities.java"
    METAL_APPLE_DETECTOR_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/detection/AppleSiliconDetector.java"
    METAL_RUNNER_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalRunner.java"
    METAL_RUNNER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-runner"
    METAL_OFFLOAD_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalWeightOffloadingRunner.java"
    METAL_OFFLOAD_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-offload"
    RUNNER_METADATA_PATCH_SRC="${PROJECT_ROOT}/spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/RunnerMetadata.java"
    METAL_FA4_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class"
    METAL_RUNNER_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/runner/MetalRunner.class"
    METAL_OFFLOAD_PREBUILT_CLASS="${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class"
    if [ -f "$TOKENIZER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling GGUF tokenizer patch...${NC}"
        mkdir -p "$TOKENIZER_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$TOKENIZER_PATCH_OUT" "$TOKENIZER_PATCH_SRC"
        (
            cd "$TOKENIZER_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.class
        )
        echo -e "${GREEN}✓ Patched GGUFTokenizer.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping GGUF tokenizer patch (source file not found).${NC}"
    fi

    if [ -f "$BPE_TOKENIZER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling BPE tokenizer patch...${NC}"
        mkdir -p "$BPE_TOKENIZER_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$BPE_TOKENIZER_PATCH_OUT" "$BPE_TOKENIZER_PATCH_SRC"
        (
            cd "$BPE_TOKENIZER_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/tokenizer/impl/BpeTokenizer.class
        )
        echo -e "${GREEN}✓ Patched BpeTokenizer.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping BPE tokenizer patch (source file not found).${NC}"
    fi

    if [ -f "$LITERT_TOKENIZER_PATCH_SRC" ] && [ -f "$LITERT_INFERENCE_PATCH_SRC" ] \
        && [ -f "$LITERT_GEMMA_PATCH_SRC" ] && [ -f "$LITERT_CPU_PATCH_SRC" ] \
        && [ -f "$LITERT_PROVIDER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling LiteRT runtime patches...${NC}"
        mkdir -p "$LITERT_PATCH_OUT"
        javac --release 25 -cp "$JAR_PATH" -d "$LITERT_PATCH_OUT" \
            "$LITERT_TOKENIZER_PATCH_SRC" \
            "$LITERT_BINDINGS_PATCH_SRC" \
            "$LITERT_INFERENCE_PATCH_SRC" \
            "$LITERT_NATIVE_PATCH_SRC" \
            "$LITERT_GEMMA_PATCH_SRC" \
            "$LITERT_CPU_PATCH_SRC" \
            "$LITERT_PROVIDER_PATCH_SRC"
        (
            cd "$LITERT_PATCH_OUT"
            zip -q "$JAR_PATH" \
                tech/kayys/gollek/provider/litert/LiteRTTokenizer*.class \
                tech/kayys/gollek/provider/litert/LiteRTNativeBindings*.class \
                tech/kayys/gollek/provider/litert/LiteRTInferenceRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTGemmaNativeRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTGemmaMetalRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTCpuRunner*.class \
                tech/kayys/gollek/provider/litert/LiteRTProvider*.class
        )
        echo -e "${GREEN}✓ Patched LiteRT runtime classes into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping LiteRT runtime patches (source file not found).${NC}"
    fi

    if [ -f "$CHAT_TEMPLATE_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling chat template patch...${NC}"
        mkdir -p "$CHAT_TEMPLATE_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$CHAT_TEMPLATE_PATCH_OUT" "$CHAT_TEMPLATE_PATCH_SRC"
        (
            cd "$CHAT_TEMPLATE_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/models/core/ChatTemplateFormatter.class
        )
        echo -e "${GREEN}✓ Patched ChatTemplateFormatter.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping chat template patch (source file not found).${NC}"
    fi

    if [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
        echo -e "${YELLOW}⚠ Skipping legacy Gemma family jar patch on Gradle install path (keeps CDI bean discovery owned by model modules).${NC}"
    elif [ -f "$GEMMA_FAMILY_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling Gemma family patch...${NC}"
        mkdir -p "$GEMMA_FAMILY_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$GEMMA_FAMILY_PATCH_OUT" "$GEMMA_FAMILY_PATCH_SRC"
        (
            cd "$GEMMA_FAMILY_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/models/GemmaFamily.class
        )
        echo -e "${GREEN}✓ Patched GemmaFamily.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping Gemma family patch (source file not found).${NC}"
    fi

    if [ -f "$MODEL_CONFIG_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling ModelConfig patch...${NC}"
        mkdir -p "$MODEL_CONFIG_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$MODEL_CONFIG_PATCH_OUT" "$MODEL_CONFIG_PATCH_SRC"
        (
            cd "$MODEL_CONFIG_PATCH_OUT"
            zip -q "$JAR_PATH" \
                tech/kayys/gollek/spi/model/ModelConfig.class \
                tech/kayys/gollek/spi/model/ModelConfig\$RopeScaling.class
        )
        echo -e "${GREEN}✓ Patched ModelConfig.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping ModelConfig patch (source file not found).${NC}"
    fi

    if [ -f "$ACCEL_TENSOR_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling AccelTensor patch...${NC}"
        mkdir -p "$ACCEL_TENSOR_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$ACCEL_TENSOR_PATCH_OUT" "$ACCEL_TENSOR_PATCH_SRC"
        (
            cd "$ACCEL_TENSOR_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/core/tensor/AccelTensor*.class
        )
        echo -e "${GREEN}✓ Patched AccelTensor*.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping AccelTensor patch (source file not found).${NC}"
    fi

    if [ -f "$ACCEL_OPS_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling AccelOps patch...${NC}"
        mkdir -p "$ACCEL_OPS_PATCH_OUT"
        javac --release 25 --enable-preview --add-modules jdk.incubator.vector \
            -cp "$JAR_PATH" -d "$ACCEL_OPS_PATCH_OUT" "$ACCEL_OPS_PATCH_SRC"
        (
            cd "$ACCEL_OPS_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/core/tensor/AccelOps*.class
        )
        echo -e "${GREEN}✓ Patched AccelOps*.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping AccelOps patch (source file not found).${NC}"
    fi

    if [ -d "$SAFETENSOR_ENGINE_CLASSES" ]; then
            echo -e "${BLUE}>>> Patching safetensor engine class set from Gradle build output...${NC}"
            (
                cd "$SAFETENSOR_ENGINE_CLASSES"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                    tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                    tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
            )
            echo -e "${GREEN}✓ Patched safetensor engine class set into jar via Gradle module classes${NC}"
    else
        echo -e "${BLUE}>>> Compiling safetensor engine patch set directly against the packaged jar...${NC}"
        mkdir -p "$SAFETENSOR_ENGINE_PATCH_OUT"
        if javac --release 25 --enable-preview --add-modules jdk.incubator.vector \
            -cp "$JAR_PATH" -d "$SAFETENSOR_ENGINE_PATCH_OUT" \
            "$DIRECT_FORWARD_PASS_PATCH_SRC" \
            "$DIRECT_INFERENCE_ENGINE_PATCH_SRC" \
            "$FLASH_ATTENTION_PATCH_SRC" \
            "$ROPE_FREQUENCY_CACHE_PATCH_SRC" \
            "$KV_CACHE_MANAGER_PATCH_SRC"; then
            (
                cd "$SAFETENSOR_ENGINE_PATCH_OUT"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                    tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                    tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                    tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
            )
            echo -e "${GREEN}✓ Patched safetensor engine class set into jar via direct source compile${NC}"
        else
            echo -e "${YELLOW}⚠ Direct safetensor engine source patch compile failed; trying prebuilt module classes fallback.${NC}"
            if [ -d "$SAFETENSOR_ENGINE_CLASSES" ]; then
                (
                    cd "$SAFETENSOR_ENGINE_CLASSES"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass*.class \
                        tech/kayys/gollek/safetensor/engine/generation/DirectInferenceEngine*.class \
                        tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel*.class \
                        tech/kayys/gollek/safetensor/engine/generation/attention/RopeFrequencyCache*.class \
                        tech/kayys/gollek/safetensor/engine/generation/kv/KVCacheManager*.class
                )
                echo -e "${GREEN}✓ Patched safetensor engine class set into jar via prebuilt module classes${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping safetensor engine patch set (direct source compile and prebuilt class fallback both failed).${NC}"
            fi
        fi
    fi

    if [ -f "$METAL_FA4_PATCH_SRC" ]; then
        if [ -f "$METAL_FA4_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal FA4 binding from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class
            )
            echo -e "${GREEN}✓ Patched MetalFlashAttentionBinding.class into jar via Gradle module classes${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal FA4 patch...${NC}"
            mkdir -p "$METAL_FA4_PATCH_OUT"
            javac -cp "$JAR_PATH" -d "$METAL_FA4_PATCH_OUT" "$METAL_FA4_PATCH_SRC"
            (
                cd "$METAL_FA4_PATCH_OUT"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class
            )
            echo -e "${GREEN}✓ Patched MetalFlashAttentionBinding.class into jar${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal FA4 patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_RUNNER_PATCH_SRC" ]; then
        if [ -f "$METAL_RUNNER_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal runner from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/metal/runner/MetalRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalRunner.class into jar via Gradle module classes${NC}"
        elif [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
            echo -e "${YELLOW}⚠ Skipping legacy Metal runner hot-patch on Gradle install path (runner slice is not part of the maintained Gradle build output).${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal runner patch...${NC}"
            mkdir -p "$METAL_RUNNER_PATCH_OUT"
            if javac -cp "$JAR_PATH" -d "$METAL_RUNNER_PATCH_OUT" \
                "$RUNNER_METADATA_PATCH_SRC" \
                "$METAL_RUNNER_MODE_SRC" \
                "$METAL_CAPABILITIES_SRC" \
                "$METAL_APPLE_DETECTOR_SRC" \
                "$METAL_RUNNER_PATCH_SRC"; then
                (
                    cd "$METAL_RUNNER_PATCH_OUT"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/spi/model/RunnerMetadata.class \
                        tech/kayys/gollek/metal/config/MetalRunnerMode.class \
                        tech/kayys/gollek/metal/detection/MetalCapabilities.class \
                        tech/kayys/gollek/metal/detection/AppleSiliconDetector.class \
                        tech/kayys/gollek/metal/runner/MetalRunner.class
                )
                echo -e "${GREEN}✓ Patched MetalRunner.class into jar${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping Metal runner patch (compile failed in current packaged classpath).${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal runner patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_OFFLOAD_PATCH_SRC" ]; then
        if [ -f "$METAL_OFFLOAD_PREBUILT_CLASS" ]; then
            echo -e "${BLUE}>>> Patching Metal offload runner from Gradle build output...${NC}"
            (
                cd "$METAL_CLASSES_GRADLE"
                zip -q "$JAR_PATH" \
                    tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalWeightOffloadingRunner.class into jar via Gradle module classes${NC}"
        elif [ "$SKIP_LEGACY_SOURCE_HOTPATCHES" = "true" ]; then
            echo -e "${YELLOW}⚠ Skipping legacy Metal offload runner hot-patch on Gradle install path (runner slice is not part of the maintained Gradle build output).${NC}"
        else
            echo -e "${BLUE}>>> Compiling Metal offload runner patch...${NC}"
            mkdir -p "$METAL_OFFLOAD_PATCH_OUT"
            if javac -cp "$JAR_PATH" -d "$METAL_OFFLOAD_PATCH_OUT" \
                "$RUNNER_METADATA_PATCH_SRC" \
                "$METAL_RUNNER_MODE_SRC" \
                "$METAL_CAPABILITIES_SRC" \
                "$METAL_APPLE_DETECTOR_SRC" \
                "$METAL_OFFLOAD_PATCH_SRC"; then
                (
                    cd "$METAL_OFFLOAD_PATCH_OUT"
                    zip -q "$JAR_PATH" \
                        tech/kayys/gollek/spi/model/RunnerMetadata.class \
                        tech/kayys/gollek/metal/config/MetalRunnerMode.class \
                        tech/kayys/gollek/metal/detection/MetalCapabilities.class \
                        tech/kayys/gollek/metal/detection/AppleSiliconDetector.class \
                        tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class
                )
                echo -e "${GREEN}✓ Patched MetalWeightOffloadingRunner.class into jar${NC}"
            else
                echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (compile failed in current packaged classpath).${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (source file not found).${NC}"
    fi

    # Ensure packaged jar includes Metal backend classes whenever safetensor
    # engine classes that reference tech.kayys.gollek.metal.* are patched in.
    if [ -d "${METAL_CLASSES_GRADLE}/tech/kayys/gollek/metal" ]; then
        echo -e "${BLUE}>>> Injecting full Metal backend class tree into packaged jar...${NC}"
        (
            cd "$METAL_CLASSES_GRADLE"
            zip -q -r "$JAR_PATH" tech/kayys/gollek/metal
        )
        echo -e "${GREEN}✓ Injected tech/kayys/gollek/metal/** into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping full Metal class injection (Gradle metal classes not found).${NC}"
    fi

    # Patch bundled classes for known dev-runtime issues in the current uber-jar.
    ASM_JAR="${HOME}/.m2/repository/org/ow2/asm/asm/9.9.1/asm-9.9.1.jar"
    ASM_TREE_JAR="${HOME}/.m2/repository/org/ow2/asm/asm-tree/9.9.1/asm-tree-9.9.1.jar"
    PATCH_TOOL_SRC="${PROJECT_ROOT}/tools/DisableDirectForwardMetal.java"
    PATCH_TOOL_OUT="${PROJECT_ROOT}/tools/.build"
    if [ -f "$ASM_JAR" ] && [ -f "$ASM_TREE_JAR" ] && [ -f "$PATCH_TOOL_SRC" ]; then
        echo -e "${BLUE}>>> Applying dev jar patches...${NC}"
        mkdir -p "$PATCH_TOOL_OUT"
        javac -cp "${ASM_JAR}:${ASM_TREE_JAR}" -d "$PATCH_TOOL_OUT" "$PATCH_TOOL_SRC"
        if java -cp "${PATCH_TOOL_OUT}:${ASM_JAR}:${ASM_TREE_JAR}" DisableDirectForwardMetal "$JAR_PATH"; then
            echo -e "${GREEN}✓ Applied jar patches${NC}"
        else
            echo -e "${YELLOW}⚠ Dev jar patch tool failed on the current packaged layout; continuing with the built jar.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping jar patches (ASM tooling not available).${NC}"
    fi

    echo -e "${BLUE}>>> Verifying patched runtime classes...${NC}"
    if ! javap -p -classpath "$JAR_PATH" tech.kayys.gollek.safetensor.core.tensor.AccelTensor \
        | grep -q "dequantizeCachedUpTo(long)"; then
        echo -e "${RED}❌ Patched jar verification failed: AccelTensor.dequantizeCachedUpTo(long) missing${NC}"
        exit 1
    fi
    if ! javap -p -classpath "$JAR_PATH" tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass \
        | grep -q "tryCachedFfnDownHalfLinear"; then
        echo -e "${RED}❌ Patched jar verification failed: DirectForwardPass.tryCachedFfnDownHalfLinear missing${NC}"
        exit 1
    fi
    if ! javap -classpath "$JAR_PATH" tech.kayys.gollek.models.core.ChatTemplateFormatter \
        | grep -q "format"; then
        echo -e "${RED}❌ Patched jar verification failed: ChatTemplateFormatter not loadable${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Verified patched runtime symbols in jar${NC}"
    fi
fi

# Install native tokenizer dependencies for local runtime usage.
install_tokenizer_native_runtime

# 3. Create Local Bin Directory
mkdir -p "$BIN_DIR"

if [ "$NATIVE_MODE" = false ]; then
    mkdir -p "$RUNTIME_DIR"
    cp "$JAR_PATH" "${GOLLEK_RUNTIME_JAR}.tmp"
    mv "${GOLLEK_RUNTIME_JAR}.tmp" "$GOLLEK_RUNTIME_JAR"
    JAR_PATH="$GOLLEK_RUNTIME_JAR"
    echo -e "${GREEN}✓ Installed CLI runtime jar to ${GOLLEK_RUNTIME_JAR}${NC}"
fi

# 4. Create the Shim/Binary link
echo -e "${BLUE}>>> Creating executable at ${GOLLEK_CLI_BIN}...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # For native, we create a shim that sets library paths but calls the binary directly
    cat <<EOF > "$GOLLEK_CLI_BIN"
#!/bin/bash
# Local Gollek CLI Native Shim (Generated by run-install-local-macos.sh)
# Points to: $NATIVE_PATH

# Ensure log directory exists
mkdir -p "\$HOME/.gollek/logs"

# Launch native binary
exec "$NATIVE_PATH" "\$@"
EOF
else
    # Standard JAR Shim - Optimized for Java 25 with FFM/Vector support
    cat <<EOF > "$GOLLEK_CLI_BIN"
#!/bin/bash
# Local Gollek CLI Shim (Generated by run-install-local-macos.sh)
# Points to: $JAR_PATH

# Ensure Java 25 is used if possible without paying SDKMAN startup cost on every run.
if [ -n "\${GOLLEK_JAVA_HOME:-}" ]; then
    export JAVA_HOME="\$GOLLEK_JAVA_HOME"
    export PATH="\$JAVA_HOME/bin:\$PATH"
elif command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JAVA_25_HOME="\$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
    if [ -n "\$JAVA_25_HOME" ]; then
        export JAVA_HOME="\$JAVA_25_HOME"
        export PATH="\$JAVA_HOME/bin:\$PATH"
    fi
elif [ -s "\$HOME/.sdkman/bin/sdkman-init.sh" ]; then
    source "\$HOME/.sdkman/bin/sdkman-init.sh"
    sdk use java 25-open >/dev/null 2>&1 || true
fi

# Ensure log directory exists
mkdir -p "\$HOME/.gollek/logs"

# Give native GGUF runs enough direct-memory headroom by default.
: "\${GOLLEK_MAX_DIRECT_MEMORY:=24g}"
# Gemma LiteRT-LM CPU fallback moves large KV-cache tensors through Java.
# A 4 GiB default heap can trigger GC pressure on Apple Silicon.
: "\${GOLLEK_JAVA_HEAP:=8g}"
# Keep the official LiteRT-LM engine warm between repeated local run calls.
# Set GOLLEK_LITERT_FAST_DAEMON=false to force one-shot process behavior.
: "\${GOLLEK_LITERT_FAST_DAEMON:=true}"
# Enable the native llama.cpp GGUF shortcut for simple local run calls.
: "\${GOLLEK_GGUF_FAST_RUN:=true}"
# Keep the llama.cpp GGUF fast path warm between repeated local run calls.
# Set GOLLEK_GGUF_FAST_DAEMON=false to force one-shot process behavior.
: "\${GOLLEK_GGUF_FAST_DAEMON:=true}"

# LiteRT accelerator plugins are loaded by native code via dlopen, so they need
# the macOS dynamic loader path in addition to Java's library path.
export DYLD_LIBRARY_PATH="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export DYLD_FALLBACK_LIBRARY_PATH="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama\${DYLD_FALLBACK_LIBRARY_PATH:+:\$DYLD_FALLBACK_LIBRARY_PATH}"

# Simple local text runs avoid the full Quarkus CLI startup by default.
# Fast runners use status 42 as an explicit "not mine; continue to the next
# CLI path" sentinel. Other non-zero statuses are real command failures and
# must not be hidden by silently launching another engine.
if [ "\${GOLLEK_GGUF_FAST_RUN:-true}" != "false" ] && { [ "\${1:-}" = "run" ] || [ "\${1:-}" = "__gguf-daemon-stop" ]; }; then
    java \\
        -Xmx"\${GOLLEK_JAVA_HEAP}" \\
        -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
        --enable-preview \\
        --add-modules jdk.incubator.vector \\
        --enable-native-access=ALL-UNNAMED \\
        \${GOLLEK_JAVA_OPTS:-} \\
        -Dgollek.gguf.fast_run.daemon="\${GOLLEK_GGUF_FAST_DAEMON}" \\
        -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
        -cp "$JAR_PATH" tech.kayys.gollek.cli.commands.GgufFastRun "\$@" 2> >(grep -v "WARNING: Using incubator modules" >&2)
    FAST_STATUS=\$?
    if [ "\$FAST_STATUS" -eq 0 ]; then
        exit 0
    fi
    if [ "\$FAST_STATUS" -ne 42 ]; then
        exit "\$FAST_STATUS"
    fi
    if [ "\${GOLLEK_GGUF_FAST_RUN_DEBUG:-false}" = "true" ]; then
        echo "GGUF fast path returned \$FAST_STATUS; falling back to remaining CLI paths." >&2
    fi
fi

if [ "\${GOLLEK_LITERT_FAST_RUN:-true}" != "false" ] && { [ "\${1:-}" = "run" ] || [ "\${1:-}" = "__daemon-stop" ]; }; then
    java \\
        -Xmx"\${GOLLEK_JAVA_HEAP}" \\
        -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
        --enable-preview \\
        --add-modules jdk.incubator.vector \\
        --enable-native-access=ALL-UNNAMED \\
        \${GOLLEK_JAVA_OPTS:-} \\
        -Dgollek.litert.fast_run.daemon="\${GOLLEK_LITERT_FAST_DAEMON}" \\
        -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
        -cp "$JAR_PATH" tech.kayys.gollek.cli.commands.LiteRtLmFastRun "\$@" 2> >(grep -v "WARNING: Using incubator modules" >&2)
    FAST_STATUS=\$?
    if [ "\$FAST_STATUS" -eq 0 ]; then
        exit 0
    fi
    if [ "\$FAST_STATUS" -ne 42 ]; then
        exit "\$FAST_STATUS"
    fi
    if [ "\${GOLLEK_LITERT_FAST_RUN_DEBUG:-false}" = "true" ]; then
        echo "LiteRT fast path returned \$FAST_STATUS; falling back to full CLI." >&2
    fi
fi

# Launch — enabling FFM and Vector API for pure Java converter
exec java \\
    -Xmx"\${GOLLEK_JAVA_HEAP}" \\
    -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
    --enable-preview \\
    --add-modules jdk.incubator.vector \\
    --enable-native-access=ALL-UNNAMED \\
    \${GOLLEK_JAVA_OPTS:-} \\
    -Dgollek.litert.fast_run.daemon="\${GOLLEK_LITERT_FAST_DAEMON}" \\
    -Djava.library.path="\$HOME/.gollek/libs:\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
    -jar "$JAR_PATH" "\$@" 2> >(grep -v "WARNING: Using incubator modules" >&2)
EOF
fi

chmod +x "$GOLLEK_CLI_BIN"

# 5. Final Instructions
echo ""
echo -e "${GREEN}✅ Gollek CLI installed locally to ${GOLLEK_CLI_BIN}${NC}"
echo ""

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo -e "${YELLOW}⚠ ${BIN_DIR} is not in your PATH.${NC}"
    echo -e "Add this to your .zshrc or .bashrc:"
    echo -e "  ${BLUE}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo ""
fi

echo -e "Try running: ${YELLOW}gollek --version${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"
