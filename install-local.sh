#!/bin/bash
# Gollek - Local Development Installer
# Builds the CLI from source and installs a local shim for testing.
# Usage: ./install-local.sh [--native]

set -e

# --- Default Options ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_MODULE="ui/gollek-cli"
SDK_MODULE="sdk/gollek-sdk-local"
BIN_DIR="${HOME}/.local/bin"
LIB_DIR="${HOME}/.gollek/libs"
GOLLEK_CLI_BIN="${BIN_DIR}/gollek"
NATIVE_MODE=false
BUILD_ARGS="-c"

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
echo -e "${BLUE}--------------------------------------------------${NC}"

# 1. Build the project
echo -e "${BLUE}>>> Building Gollek...${NC}"

SDK_POM="${PROJECT_ROOT}/sdk/pom.xml"
CLI_POM="${PROJECT_ROOT}/${CLI_MODULE}/pom.xml"

SDK_ARGS="-f ${SDK_POM} -pl gollek-sdk-local -am"
CLI_ARGS="-f ${CLI_POM}"

if [[ "$BUILD_ARGS" == *"-c"* ]]; then
    SDK_ARGS="$SDK_ARGS clean"
    CLI_ARGS="$CLI_ARGS clean"
fi

SDK_ARGS="$SDK_ARGS install"
CLI_ARGS="$CLI_ARGS package"

if [[ "$BUILD_ARGS" != *"-t"* ]]; then
    SDK_ARGS="$SDK_ARGS -DskipTests"
    CLI_ARGS="$CLI_ARGS -DskipTests"
fi

if [ "$NATIVE_MODE" = true ]; then
    echo -e "${YELLOW}ℹ Building native image and shared library (this may take a while)...${NC}"
    CLI_ARGS="$CLI_ARGS -Pnative -Dquarkus.native.container-build=false -Dgraalvm.metadataRepository.enabled=false"
fi

echo -e "${BLUE}>>> Running: mvn $SDK_ARGS${NC}"
mvn $SDK_ARGS
echo -e "${BLUE}>>> Running: mvn $CLI_ARGS${NC}"
mvn $CLI_ARGS

# 2. Locate Artifacts
echo -e "${BLUE}>>> Locating artifacts...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # 2a. Locate Native Executable
    NATIVE_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek"
    if [ ! -f "$NATIVE_PATH" ]; then
        # Try alternate location or final name variation
        NATIVE_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -type f -perm +111 -not -name "*.jar" -not -name "*.sh" -maxdepth 1 | head -n 1)
    fi

    if [ -z "$NATIVE_PATH" ] || [ ! -f "$NATIVE_PATH" ]; then
        echo -e "${RED}❌ Could not find native executable in ${CLI_MODULE}/target${NC}"
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
    # Locate JAR
    JAR_PATH="${PROJECT_ROOT}/${CLI_MODULE}/target/gollek.jar"
    if [ ! -f "$JAR_PATH" ]; then
        JAR_PATH=$(find "${PROJECT_ROOT}/${CLI_MODULE}/target" -name "gollek*.jar" -maxdepth 1 | head -n 1)
    fi

    if [ -z "$JAR_PATH" ] || [ ! -f "$JAR_PATH" ]; then
        echo -e "${RED}❌ Could not find Gollek CLI JAR in ${CLI_MODULE}/target${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found JAR: $(basename "$JAR_PATH")${NC}"

    # Patch GGUFTokenizer directly into the packaged jar so local runs use the
    # updated runtime-facing tokenizer even when the standalone GGUF reactor
    # is not buildable in this checkout.
    TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/runner/gguf/gollek-gguf-core/src/main/java/tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.java"
    TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-tokenizer"
    BPE_TOKENIZER_PATCH_SRC="${PROJECT_ROOT}/core/gollek-tokenizer-core/src/main/java/tech/kayys/gollek/tokenizer/impl/BpeTokenizer.java"
    BPE_TOKENIZER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-bpe-tokenizer"
    GEMMA_FAMILY_PATCH_SRC="${PROJECT_ROOT}/models/gollek-model-gemma/src/main/java/tech/kayys/gollek/models/GemmaFamily.java"
    GEMMA_FAMILY_PATCH_OUT="${PROJECT_ROOT}/tools/.build-gemma-family"
    MODEL_CONFIG_PATCH_SRC="${PROJECT_ROOT}/spi/gollek-spi-model/src/main/java/tech/kayys/gollek/spi/model/ModelConfig.java"
    MODEL_CONFIG_PATCH_OUT="${PROJECT_ROOT}/tools/.build-model-config"
    ACCEL_TENSOR_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-core/src/main/java/tech/kayys/gollek/safetensor/core/tensor/AccelTensor.java"
    ACCEL_TENSOR_PATCH_OUT="${PROJECT_ROOT}/tools/.build-accel-tensor"
    DIRECT_FORWARD_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass.java"
    DIRECT_FORWARD_PATCH_OUT="${PROJECT_ROOT}/tools/.build-direct-forward"
    FLASH_ATTENTION_PATCH_SRC="${PROJECT_ROOT}/runner/safetensor/gollek-safetensor-engine/src/main/java/tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel.java"
    FLASH_ATTENTION_PATCH_OUT="${PROJECT_ROOT}/tools/.build-flash-attention"
    METAL_FA4_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/binding/MetalFlashAttentionBinding.java"
    METAL_FA4_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-fa4"
    METAL_RUNNER_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalRunner.java"
    METAL_RUNNER_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-runner"
    METAL_OFFLOAD_PATCH_SRC="${PROJECT_ROOT}/backend/metal/gollek-backend-metal/src/main/java/tech/kayys/gollek/backend/metal/runner/MetalWeightOffloadingRunner.java"
    METAL_OFFLOAD_PATCH_OUT="${PROJECT_ROOT}/tools/.build-metal-offload"
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

    if [ -f "$GEMMA_FAMILY_PATCH_SRC" ]; then
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
            zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/core/tensor/AccelTensor.class
        )
        echo -e "${GREEN}✓ Patched AccelTensor.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping AccelTensor patch (source file not found).${NC}"
    fi

    if [ -f "$DIRECT_FORWARD_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling DirectForwardPass patch...${NC}"
        mkdir -p "$DIRECT_FORWARD_PATCH_OUT"
        if javac --add-modules jdk.incubator.vector -cp "$JAR_PATH" -d "$DIRECT_FORWARD_PATCH_OUT" "$DIRECT_FORWARD_PATCH_SRC"; then
            (
                cd "$DIRECT_FORWARD_PATCH_OUT"
                zip -q "$JAR_PATH" tech/kayys/gollek/safetensor/engine/forward/DirectForwardPass.class
            )
            echo -e "${GREEN}✓ Patched DirectForwardPass.class into jar${NC}"
        else
            echo -e "${YELLOW}⚠ Skipping DirectForwardPass patch (compile failed in current packaged classpath).${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping DirectForwardPass patch (source file not found).${NC}"
    fi

    if [ -f "$FLASH_ATTENTION_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling FlashAttentionKernel patch...${NC}"
        mkdir -p "$FLASH_ATTENTION_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$FLASH_ATTENTION_PATCH_OUT" "$FLASH_ATTENTION_PATCH_SRC"
        (
            cd "$FLASH_ATTENTION_PATCH_OUT"
            zip -q "$JAR_PATH" \
                tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel.class \
                tech/kayys/gollek/safetensor/engine/generation/attention/FlashAttentionKernel\$AttentionInput.class
        )
        echo -e "${GREEN}✓ Patched FlashAttentionKernel.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping FlashAttentionKernel patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_FA4_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling Metal FA4 patch...${NC}"
        mkdir -p "$METAL_FA4_PATCH_OUT"
        javac -cp "$JAR_PATH" -d "$METAL_FA4_PATCH_OUT" "$METAL_FA4_PATCH_SRC"
        (
            cd "$METAL_FA4_PATCH_OUT"
            zip -q "$JAR_PATH" tech/kayys/gollek/metal/binding/MetalFlashAttentionBinding.class
        )
        echo -e "${GREEN}✓ Patched MetalFlashAttentionBinding.class into jar${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping Metal FA4 patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_RUNNER_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling Metal runner patch...${NC}"
        mkdir -p "$METAL_RUNNER_PATCH_OUT"
        if javac -cp "$JAR_PATH" -d "$METAL_RUNNER_PATCH_OUT" "$METAL_RUNNER_PATCH_SRC"; then
            (
                cd "$METAL_RUNNER_PATCH_OUT"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/runner/MetalRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalRunner.class into jar${NC}"
        else
            echo -e "${YELLOW}⚠ Skipping Metal runner patch (compile failed in current packaged classpath).${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal runner patch (source file not found).${NC}"
    fi

    if [ -f "$METAL_OFFLOAD_PATCH_SRC" ]; then
        echo -e "${BLUE}>>> Compiling Metal offload runner patch...${NC}"
        mkdir -p "$METAL_OFFLOAD_PATCH_OUT"
        if javac -cp "$JAR_PATH" -d "$METAL_OFFLOAD_PATCH_OUT" "$METAL_OFFLOAD_PATCH_SRC"; then
            (
                cd "$METAL_OFFLOAD_PATCH_OUT"
                zip -q "$JAR_PATH" tech/kayys/gollek/metal/runner/MetalWeightOffloadingRunner.class
            )
            echo -e "${GREEN}✓ Patched MetalWeightOffloadingRunner.class into jar${NC}"
        else
            echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (compile failed in current packaged classpath).${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Skipping Metal offload runner patch (source file not found).${NC}"
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
        java -cp "${PATCH_TOOL_OUT}:${ASM_JAR}:${ASM_TREE_JAR}" DisableDirectForwardMetal "$JAR_PATH"
        echo -e "${GREEN}✓ Applied jar patches${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping jar patches (ASM tooling not available).${NC}"
    fi
fi

# 3. Create Local Bin Directory
mkdir -p "$BIN_DIR"

# 4. Create the Shim/Binary link
echo -e "${BLUE}>>> Creating executable at ${GOLLEK_CLI_BIN}...${NC}"

if [ "$NATIVE_MODE" = true ]; then
    # For native, we create a shim that sets library paths but calls the binary directly
    cat <<EOF > "$GOLLEK_CLI_BIN"
#!/bin/bash
# Local Gollek CLI Native Shim (Generated by install-local.sh)
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
# Local Gollek CLI Shim (Generated by install-local.sh)
# Points to: $JAR_PATH

# Ensure Java 25 is used if possible
if [ -s "\$HOME/.sdkman/bin/sdkman-init.sh" ]; then
    source "\$HOME/.sdkman/bin/sdkman-init.sh"
    sdk use java 25-open >/dev/null 2>&1 || true
fi

# Ensure log directory exists
mkdir -p "\$HOME/.gollek/logs"

# Give native GGUF runs enough direct-memory headroom by default.
: "\${GOLLEK_MAX_DIRECT_MEMORY:=24g}"

# Launch — enabling FFM and Vector API for pure Java converter
exec java \\
    -XX:MaxDirectMemorySize="\${GOLLEK_MAX_DIRECT_MEMORY}" \\
    --enable-preview \\
    --add-modules jdk.incubator.vector \\
    --enable-native-access=ALL-UNNAMED \\
    \${GOLLEK_JAVA_OPTS:-} \\
    -Djava.library.path="\$HOME/.gollek/libs/llama:\$HOME/.gollek/libs/onnxruntime:\$HOME/.gollek/libs/libtorch" \\
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
