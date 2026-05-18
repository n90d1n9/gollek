# ==============================================================================
# Gollek Metal Bridge — Makefile
# ==============================================================================
#
# Compiles the Metal Objective-C sources into a shared dylib for Java FFM loading.
#
# Requirements:
#   - macOS 13 (Ventura) or later
#   - Apple Silicon (arm64) or Intel Mac with Metal GPU
#   - Xcode Command Line Tools
#
# Usage:
#   make                 — build libgollek_metal.dylib
#   make install         — copy to ~/.gollek/libs/ and resources/native/darwin-aarch64/
#   make clean           — remove build artifacts
#   make info            — print detected SDK and device info
#
# Output:
#   target/native/darwin-aarch64/libgollek_metal.dylib
# Install:
#   ~/.gollek/libs/libgollek_metal.dylib
# ==============================================================================

SDK        := $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null)
CC         := clang
TARGET_DIR := ../../../../target/native/darwin-aarch64
OBJ_DIR    := $(TARGET_DIR)/obj
OUTPUT     := libgollek_metal.dylib
SRC        := gollek_metal_support.m gollek_metal_buffers.m gollek_metal_scratch.m gollek_metal_cpu_fallback.m gollek_metal_elementwise.m gollek_metal_gated_ffn.m gollek_metal_attention.m gollek_metal_runtime.m gollek_metal_kernel_source.m gollek_metal_pipelines.m gollek_metal_mps_cache.m gollek_metal_mps_matmul.m gollek_metal_mps_validation.m gollek_metal_mps_matvec_policy.m gollek_metal_matvec_dispatch.m gollek_metal_matvec_tuning.m gollek_metal_matvec_api.m gollek_metal_bridge.m gollek_metal_fa4.m
OBJ        := $(addprefix $(OBJ_DIR)/,$(SRC:.m=.o))
INSTALL_DIR ?= $(HOME)/.gollek/libs

# Detect architecture: default arm64 (Apple Silicon), allow override for x86_64
ARCH ?= arm64

CFLAGS := -arch $(ARCH) \
           -fPIC -fobjc-arc \
           -O3 -isysroot $(SDK) \
           -mmacosx-version-min=15.0

LDFLAGS := -arch $(ARCH) \
           -shared -isysroot $(SDK) \
           -mmacosx-version-min=15.0

FRAMEWORKS := -framework Metal \
               -framework MetalPerformanceShaders \
               -framework MetalPerformanceShadersGraph \
               -framework Foundation \
               -framework Accelerate

.PHONY: all install clean info

all: $(TARGET_DIR)/$(OUTPUT)

$(TARGET_DIR)/$(OUTPUT): $(OBJ) | $(TARGET_DIR)
	$(CC) $(LDFLAGS) $(FRAMEWORKS) -o $@ $(OBJ)
	@echo "✓ Built $(OUTPUT) → $(TARGET_DIR)/"
	@codesign --sign - $@ 2>/dev/null || true  # ad-hoc sign for local use

$(TARGET_DIR) $(OBJ_DIR):
	@mkdir -p $@

$(OBJ_DIR)/%.o: %.m gollek_metal_support.h gollek_metal_buffers.h gollek_metal_scratch.h gollek_metal_cpu_fallback.h gollek_metal_elementwise.h gollek_metal_gated_ffn.h gollek_metal_attention.h gollek_metal_runtime.h gollek_metal_kernel_source.h gollek_metal_pipelines.h gollek_metal_mps_cache.h gollek_metal_mps_matmul.h gollek_metal_mps_validation.h gollek_metal_mps_matvec_policy.h gollek_metal_matvec_dispatch.h gollek_metal_matvec_tuning.h gollek_metal_matvec_api.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

install: all
	@mkdir -p $(INSTALL_DIR)
	cp $(TARGET_DIR)/$(OUTPUT) $(INSTALL_DIR)/
	@mkdir -p ../resources/native/darwin-aarch64
	cp $(TARGET_DIR)/$(OUTPUT) ../resources/native/darwin-aarch64/
	@echo "✓ Installed to $(INSTALL_DIR)/ and resources/native/darwin-aarch64/"

clean:
	rm -f $(TARGET_DIR)/$(OUTPUT) $(SRC:.m=.o)
	rm -rf $(OBJ_DIR)
	@echo "✓ Cleaned"

info:
	@echo "SDK:    $(SDK)"
	@echo "ARCH:   $(ARCH)"
	@echo "OUTPUT: $(TARGET_DIR)/$(OUTPUT)"
	@system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|VRAM" | head -4 || true
