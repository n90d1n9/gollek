#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTION_FILE="${GOLLEK_MOBILE_SELECTION_FILE:-$ROOT_DIR/scripts/module-selection-current.env}"

if [[ ! -f "$SELECTION_FILE" ]]; then
  echo ":| Mobile selection file not found: $SELECTION_FILE" >&2
  exit 1
fi

source "$SELECTION_FILE"

MOBILE_PLUGIN_DIR_VALUE="${MOBILE_PLUGIN_DIR:-$ROOT_DIR/../mobile/gollek_edge}"
NATIVE_DIR="$MOBILE_PLUGIN_DIR_VALUE/native"
GENERATED_DIR="$NATIVE_DIR/generated"
GENERATED_INCLUDE_DIR="$GENERATED_DIR/include"
PUBLIC_CONFIG_HEADER="$NATIVE_DIR/include/gollek_edge_runtime_config.h"
DIST_DIR="$NATIVE_DIR/dist"
BUILD_DIR="$NATIVE_DIR/build/host"
ANDROID_JNILIBS_DIR="$MOBILE_PLUGIN_DIR_VALUE/android/src/main/jniLibs"
PROFILE_JSON="${MOBILE_PROFILE_FILE:-$MOBILE_PLUGIN_DIR_VALUE/assets/gollek_edge_profile.json}"
BUILD_INFO_JSON="$MOBILE_PLUGIN_DIR_VALUE/assets/gollek_edge_build.json"
FORMAT_TARGETS_VALUE="${MOBILE_FORMAT_TARGETS:-${FORMAT_TARGETS:-litert}}"
MOBILE_FEATURES_VALUE="${MOBILE_FEATURES:-text,speech-to-text,vision}"
MODEL_SIDECAR_FILES_VALUE="${GOLLEK_EDGE_MODEL_SIDECAR_FILES:-${MODEL_SIDECAR_FILES:-tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json,config.json}}"
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
HOST_ARCH="$(uname -m)"
MACOS_DEPLOYMENT_TARGET_VALUE="${GOLLEK_EDGE_MACOS_DEPLOYMENT_TARGET:-14.0}"
REQUIRED_RUNTIME_SYMBOLS=(
  gollek_edge_runtime_version
  gollek_edge_runtime_profile_json
  gollek_edge_runtime_model_formats_csv
  gollek_edge_runtime_mobile_features_csv
  gollek_edge_runtime_build_profile
  gollek_edge_runtime_delegates_json
  gollek_edge_runtime_last_error
  gollek_edge_runtime_supports_format
  gollek_edge_runtime_supports_feature
  gollek_edge_runtime_open_model
  gollek_edge_runtime_close_model
  gollek_edge_runtime_warm_model
  gollek_edge_runtime_probe_model
  gollek_edge_runtime_adapter_diagnostics
  gollek_edge_runtime_infer
  gollek_edge_runtime_infer_with_options
  gollek_edge_runtime_infer_text
)
RUNTIME_EXPORTS_VERIFIED=false
RUNTIME_EXPORT_SYMBOL_COUNT=0

mkdir -p "$GENERATED_INCLUDE_DIR" "$DIST_DIR" "$(dirname "$PROFILE_JSON")" "$(dirname "$PUBLIC_CONFIG_HEADER")"

csv_has() {
  local csv=",$1,"
  local needle="$2"
  [[ "$csv" == *",$needle,"* ]]
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

native_shared_library() {
  local path="$1"
  local kind
  [[ -f "$path" ]] || return 1
  kind="$(native_library_kind "$path")"
  case "$HOST_OS" in
    darwin)
      [[ "$kind" == Mach-O* && "$kind" == *"dynamically linked shared library"* ]]
      ;;
    linux)
      [[ "$kind" == ELF* && "$kind" == *"shared object"* ]]
      ;;
    *)
      return 0
      ;;
  esac
}

native_library_kind() {
  local path="$1"
  file -Lb "$path" 2>/dev/null || printf 'unreadable'
}

darwin_dylib_install_name() {
  local path="$1"
  otool -D "$path" 2>/dev/null | tail -n 1 | xargs basename 2>/dev/null || true
}

runtime_symbol_table() {
  local path="$1"
  case "$HOST_OS" in
    darwin)
      nm -gU "$path" 2>/dev/null || true
      ;;
    linux)
      nm -D --defined-only "$path" 2>/dev/null || true
      ;;
    *)
      printf ''
      ;;
  esac
}

verify_runtime_exports() {
  local path="$1"
  local symbols
  local symbol
  local needle
  local -a missing=()

  if [[ ! -f "$path" ]]; then
    echo ":| Native runtime library was not produced: $path" >&2
    return 1
  fi
  if ! native_shared_library "$path"; then
    echo ":| Native runtime output is not a shared library: $path ($(native_library_kind "$path"))" >&2
    return 1
  fi
  if ! command -v nm >/dev/null 2>&1; then
    echo ":| Cannot verify native runtime exports because nm is not installed" >&2
    return 1
  fi

  symbols="$(runtime_symbol_table "$path")"
  if [[ -z "$symbols" ]]; then
    echo ":| Cannot read native runtime exports from: $path" >&2
    return 1
  fi

  for symbol in "${REQUIRED_RUNTIME_SYMBOLS[@]}"; do
    case "$HOST_OS" in
      darwin) needle="_$symbol" ;;
      *) needle="$symbol" ;;
    esac
    if [[ "$symbols" != *"$needle"* ]]; then
      missing+=("$symbol")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo ":| Native runtime is missing required exports:" >&2
    for symbol in "${missing[@]}"; do
      echo "   - $symbol" >&2
    done
    echo ":| Check mobile/gollek_edge/native/CMakeLists.txt is building src/gollek_edge_runtime.c" >&2
    return 1
  fi

  RUNTIME_EXPORTS_VERIFIED=true
  RUNTIME_EXPORT_SYMBOL_COUNT=${#REQUIRED_RUNTIME_SYMBOLS[@]}
  echo ":) Verified Gollek Edge runtime exports: $RUNTIME_EXPORT_SYMBOL_COUNT symbols"
}

json_array_from_csv() {
  local csv="$1"
  local out="["
  local item
  IFS=',' read -r -a parts <<< "$csv"
  for item in "${parts[@]}"; do
    [[ -z "$item" ]] && continue
    if [[ "$out" != "[" ]]; then
      out+=", "
    fi
    out+="\"${item}\""
  done
  out+="]"
  printf '%s' "$out"
}

json_string_literal() {
  local input="$1"
  printf '%s' "$input" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

json_native_libraries_array() {
  local out="["
  local entry
  local label
  local path
  local kind
  for entry in "$@"; do
    IFS='|' read -r label path kind <<< "$entry"
    if [[ "$out" != "[" ]]; then
      out+=", "
    fi
    out+="{\"label\":\"$(json_string_literal "$label")\",\"path\":\"$(json_string_literal "$path")\",\"kind\":\"$(json_string_literal "$kind")\"}"
  done
  out+="]"
  printf '%s' "$out"
}

c_string_literal() {
  local input="$1"
  printf '%s' "$input" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

record_invalid_native_library() {
  local label="$1"
  local path="$2"
  local optional="${3:-false}"
  local kind
  kind="$(native_library_kind "$path")"
  if [[ "$optional" == "true" ]]; then
    OPTIONAL_INVALID_NATIVE_LIBS+=("$label|$path|$kind")
    echo ":| Warning: ignoring optional non-native $label library: $path ($kind)" >&2
  else
    INVALID_NATIVE_LIBS+=("$label|$path|$kind")
    echo ":| Warning: skipping non-native $label library: $path ($kind)" >&2
  fi
}

PROFILE_JSON_TEXT="$(printf '{"runtimeTarget":"%s","pluginPath":"mobile/gollek_edge","modelFormats":%s,"mobileFeatures":%s,"modelSidecarFiles":%s}' \
  "${RUNTIME_TARGET:-mobile}" \
  "$(json_array_from_csv "$FORMAT_TARGETS_VALUE")" \
  "$(json_array_from_csv "$MOBILE_FEATURES_VALUE")" \
  "$(json_array_from_csv "$MODEL_SIDECAR_FILES_VALUE")")"

{
  printf '{\n'
  printf '  "runtimeTarget": "%s",\n' "${RUNTIME_TARGET:-mobile}"
  printf '  "pluginPath": "mobile/gollek_edge",\n'
  printf '  "modelFormats": %s,\n' "$(json_array_from_csv "$FORMAT_TARGETS_VALUE")"
  printf '  "mobileFeatures": %s,\n' "$(json_array_from_csv "$MOBILE_FEATURES_VALUE")"
  printf '  "modelSidecarFiles": %s\n' "$(json_array_from_csv "$MODEL_SIDECAR_FILES_VALUE")"
  printf '}\n'
} > "$PROFILE_JSON"

for config_header in "$GENERATED_INCLUDE_DIR/gollek_edge_runtime_config.h" "$PUBLIC_CONFIG_HEADER"; do
  {
    printf '#ifndef GOLLEK_EDGE_RUNTIME_CONFIG_H_\n'
    printf '#define GOLLEK_EDGE_RUNTIME_CONFIG_H_\n\n'
    printf '#define GOLLEK_EDGE_PROFILE_JSON "%s"\n' "$(c_string_literal "$PROFILE_JSON_TEXT")"
    printf '#define GOLLEK_EDGE_MODEL_FORMATS_CSV "%s"\n' "$FORMAT_TARGETS_VALUE"
    printf '#define GOLLEK_EDGE_MOBILE_FEATURES_CSV "%s"\n' "$MOBILE_FEATURES_VALUE"
    printf '#define GOLLEK_EDGE_BUILD_PROFILE "%s"\n' "${BUILD_PROFILE:-snapshot}"
    printf '\n#endif  // GOLLEK_EDGE_RUNTIME_CONFIG_H_\n'
  } > "$config_header"
done

LITERT_LIB_DIR="${GOLLEK_EDGE_LITERT_LIB_DIR:-$HOME/.gollek/libs}"
ONNX_LIB_DIR="${GOLLEK_EDGE_ONNX_LIB_DIR:-$HOME/.gollek/libs}"
ANDROID_LIB_DIR="${GOLLEK_EDGE_ANDROID_LIB_DIR:-$HOME/.gollek/libs/android}"
LITERT_RUNTIME_AVAILABLE=false
ONNX_RUNTIME_AVAILABLE=false
ANDROID_LITERT_RUNTIME_AVAILABLE=false
ANDROID_ONNX_RUNTIME_AVAILABLE=false
LITERT_TEXT_GENERATION_AVAILABLE=false
ONNX_TEXT_GENERATION_AVAILABLE=false
declare -a LITERT_PACKAGED_LIBS=()
declare -a ONNX_PACKAGED_LIBS=()
declare -a ANDROID_PACKAGED_LIBS=()
declare -a INVALID_NATIVE_LIBS=()
declare -a OPTIONAL_INVALID_NATIVE_LIBS=()
declare -a ONNX_SEARCH_DIRS=("$ONNX_LIB_DIR")
if [[ "$ONNX_LIB_DIR" != "$HOME/.gollek/libs/onnxruntime" ]]; then
  ONNX_SEARCH_DIRS+=("$HOME/.gollek/libs/onnxruntime")
fi

if csv_has "$FORMAT_TARGETS_VALUE" "litert"; then
  case "$HOST_OS" in
    darwin)
      if [[ -f "$LITERT_LIB_DIR/libLiteRt.dylib" ]]; then
        LITERT_RUNTIME_AVAILABLE=true
        for lib_name in libLiteRt.dylib libLiteRtMetalAccelerator.dylib libtensorflowlite_c.dylib; do
          lib_path="$LITERT_LIB_DIR/$lib_name"
          if [[ -f "$lib_path" ]]; then
            if native_shared_library "$lib_path"; then
              LITERT_PACKAGED_LIBS+=("$lib_path")
            else
              if [[ "$lib_name" == "libtensorflowlite_c.dylib" ]]; then
                record_invalid_native_library "LiteRT" "$lib_path" true
              else
                record_invalid_native_library "LiteRT" "$lib_path" false
              fi
            fi
          fi
        done
      fi
      ;;
    linux)
      if [[ -f "$LITERT_LIB_DIR/libLiteRt.so" ]]; then
        LITERT_RUNTIME_AVAILABLE=true
        for lib_name in libLiteRt.so libLiteRtMetalAccelerator.so libtensorflowlite_c.so; do
          lib_path="$LITERT_LIB_DIR/$lib_name"
          if [[ -f "$lib_path" ]]; then
            if native_shared_library "$lib_path"; then
              LITERT_PACKAGED_LIBS+=("$lib_path")
            else
              if [[ "$lib_name" == "libtensorflowlite_c.so" ]]; then
                record_invalid_native_library "LiteRT" "$lib_path" true
              else
                record_invalid_native_library "LiteRT" "$lib_path" false
              fi
            fi
          fi
        done
      fi
      ;;
  esac
fi

if csv_has "$FORMAT_TARGETS_VALUE" "onnx"; then
  case "$HOST_OS" in
    darwin)
      for lib_dir in "${ONNX_SEARCH_DIRS[@]}"; do
        if [[ "$ONNX_RUNTIME_AVAILABLE" == false && -f "$lib_dir/libonnxruntime.dylib" ]] && native_shared_library "$lib_dir/libonnxruntime.dylib"; then
          install_name="$(darwin_dylib_install_name "$lib_dir/libonnxruntime.dylib")"
          ONNX_RUNTIME_AVAILABLE=true
          ONNX_PACKAGED_LIBS+=("$lib_dir/libonnxruntime.dylib")
          if [[ -n "$install_name" && "$install_name" != "libonnxruntime.dylib" && -f "$lib_dir/$install_name" ]] && native_shared_library "$lib_dir/$install_name"; then
            ONNX_PACKAGED_LIBS+=("$lib_dir/$install_name")
          fi
        fi
      done
      ;;
    linux)
      for lib_dir in "${ONNX_SEARCH_DIRS[@]}"; do
        if [[ "$ONNX_RUNTIME_AVAILABLE" == false && -f "$lib_dir/libonnxruntime.so" ]] && native_shared_library "$lib_dir/libonnxruntime.so"; then
          ONNX_RUNTIME_AVAILABLE=true
          ONNX_PACKAGED_LIBS+=("$lib_dir/libonnxruntime.so")
        fi
      done
      ;;
  esac
fi

if [[ "$LITERT_RUNTIME_AVAILABLE" == true ]] && truthy "${GOLLEK_EDGE_ENABLE_LITERT_TEXT_GENERATION:-false}"; then
  LITERT_TEXT_GENERATION_AVAILABLE=true
fi
if [[ "$ONNX_RUNTIME_AVAILABLE" == true ]] && truthy "${GOLLEK_EDGE_ENABLE_ONNX_TEXT_GENERATION:-false}"; then
  ONNX_TEXT_GENERATION_AVAILABLE=true
fi

litert_runtime_cmake=OFF
litert_text_cmake=OFF
onnx_runtime_cmake=OFF
onnx_text_cmake=OFF
if [[ "$LITERT_RUNTIME_AVAILABLE" == true ]]; then
  litert_runtime_cmake=ON
fi
if [[ "$LITERT_TEXT_GENERATION_AVAILABLE" == true ]]; then
  litert_text_cmake=ON
fi
if [[ "$ONNX_RUNTIME_AVAILABLE" == true ]]; then
  onnx_runtime_cmake=ON
fi
if [[ "$ONNX_TEXT_GENERATION_AVAILABLE" == true ]]; then
  onnx_text_cmake=ON
fi

cmake_args=(
  -S "$NATIVE_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$DIST_DIR"
  -DGOLLEK_EDGE_GENERATED_INCLUDE_DIR="$GENERATED_INCLUDE_DIR"
  -DGOLLEK_EDGE_ENABLE_LITERT_RUNTIME="$litert_runtime_cmake"
  -DGOLLEK_EDGE_ENABLE_LITERT_TEXT_GENERATION="$litert_text_cmake"
  -DGOLLEK_EDGE_ENABLE_ONNX_RUNTIME="$onnx_runtime_cmake"
  -DGOLLEK_EDGE_ENABLE_ONNX_TEXT_GENERATION="$onnx_text_cmake"
)
if [[ "$HOST_OS" == "darwin" ]]; then
  cmake_args+=(
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOS_DEPLOYMENT_TARGET_VALUE}"
  )
fi
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config Release
ctest --test-dir "$BUILD_DIR" --output-on-failure
cmake --install "$BUILD_DIR" --config Release

case "$HOST_OS" in
  darwin) LIB_NAME="libgollek_edge_runtime.dylib" ;;
  linux) LIB_NAME="libgollek_edge_runtime.so" ;;
  *) LIB_NAME="gollek_edge_runtime.dll" ;;
esac

LIB_PATH="$DIST_DIR/lib/$LIB_NAME"
verify_runtime_exports "$LIB_PATH"

mkdir -p "$MOBILE_PLUGIN_DIR_VALUE/macos/Libraries"
for destination_lib in "$MOBILE_PLUGIN_DIR_VALUE/macos/Libraries"/*.dylib; do
  [[ -e "$destination_lib" ]] || continue
  if ! native_shared_library "$destination_lib"; then
    kind="$(native_library_kind "$destination_lib")"
    INVALID_NATIVE_LIBS+=("packaged|$destination_lib|$kind")
    echo ":| Warning: removing non-native packaged library: $destination_lib ($kind)" >&2
    rm -f "$destination_lib"
  fi
done
if [[ -f "$LIB_PATH" && "$HOST_OS" == "darwin" ]]; then
  cp "$LIB_PATH" "$MOBILE_PLUGIN_DIR_VALUE/macos/Libraries/$LIB_NAME"
  for packaged_lib in "${LITERT_PACKAGED_LIBS[@]}" "${ONNX_PACKAGED_LIBS[@]}"; do
    if [[ -f "$packaged_lib" ]]; then
      cp -fL "$packaged_lib" "$MOBILE_PLUGIN_DIR_VALUE/macos/Libraries/$(basename "$packaged_lib")"
    fi
  done
fi

if [[ -d "$ANDROID_LIB_DIR" ]]; then
  while IFS= read -r -d '' android_lib; do
    abi="$(basename "$(dirname "$android_lib")")"
    case "$abi" in
      arm64-v8a|armeabi-v7a|x86|x86_64) ;;
      *) continue ;;
    esac

    case "$(basename "$android_lib")" in
      libLiteRt.so|libtensorflowlite_c.so)
        ANDROID_LITERT_RUNTIME_AVAILABLE=true
        ;;
      libonnxruntime.so)
        ANDROID_ONNX_RUNTIME_AVAILABLE=true
        ;;
      *)
        continue
        ;;
    esac

    mkdir -p "$ANDROID_JNILIBS_DIR/$abi"
    cp -fL "$android_lib" "$ANDROID_JNILIBS_DIR/$abi/$(basename "$android_lib")"
    ANDROID_PACKAGED_LIBS+=("$ANDROID_JNILIBS_DIR/$abi/$(basename "$android_lib")")
  done < <(find "$ANDROID_LIB_DIR" -type f \( -name 'libLiteRt.so' -o -name 'libtensorflowlite_c.so' -o -name 'libonnxruntime.so' \) -print0)
fi

invalid_native_libraries_json="[]"
optional_invalid_native_libraries_json="[]"
if (( ${#INVALID_NATIVE_LIBS[@]} > 0 )); then
  invalid_native_libraries_json="$(json_native_libraries_array "${INVALID_NATIVE_LIBS[@]}")"
fi
if (( ${#OPTIONAL_INVALID_NATIVE_LIBS[@]} > 0 )); then
  optional_invalid_native_libraries_json="$(json_native_libraries_array "${OPTIONAL_INVALID_NATIVE_LIBS[@]}")"
fi

{
  printf '{\n'
  printf '  "runtime": "gollek_edge_native",\n'
  printf '  "version": "0.1.0",\n'
  printf '  "runtimeOrigin": "gollek-builder-native-c",\n'
  printf '  "runtimeIntegration": "flutter-method-channel-ffi-native-c",\n'
  printf '  "gollekRelationship": "generated-by-gollek-builder",\n'
  printf '  "usesGollekCli": false,\n'
  printf '  "usesGollekJvmRuntime": false,\n'
  printf '  "host": "%s-%s",\n' "$HOST_OS" "$HOST_ARCH"
  printf '  "buildProfile": "%s",\n' "${BUILD_PROFILE:-snapshot}"
  printf '  "macosDeploymentTarget": "%s",\n' "$MACOS_DEPLOYMENT_TARGET_VALUE"
  printf '  "libraryPath": "%s",\n' "$LIB_PATH"
  printf '  "headerPath": "%s",\n' "$DIST_DIR/include/gollek_edge_runtime.h"
  printf '  "compiled": true,\n'
  printf '  "runtimeExportsVerified": %s,\n' "$RUNTIME_EXPORTS_VERIFIED"
  printf '  "runtimeExportSymbolCount": %s,\n' "$RUNTIME_EXPORT_SYMBOL_COUNT"
  printf '  "litertRuntimeLinked": %s,\n' "$LITERT_RUNTIME_AVAILABLE"
  printf '  "litertTextGenerationRunnable": %s,\n' "$LITERT_TEXT_GENERATION_AVAILABLE"
  printf '  "onnxRuntimeLinked": %s,\n' "$ONNX_RUNTIME_AVAILABLE"
  printf '  "onnxTextGenerationRunnable": %s,\n' "$ONNX_TEXT_GENERATION_AVAILABLE"
  printf '  "androidLiteRtRuntimePackaged": %s,\n' "$ANDROID_LITERT_RUNTIME_AVAILABLE"
  printf '  "androidOnnxRuntimePackaged": %s,\n' "$ANDROID_ONNX_RUNTIME_AVAILABLE"
  printf '  "invalidNativeLibraries": %s,\n' "$invalid_native_libraries_json"
  printf '  "optionalInvalidNativeLibraries": %s,\n' "$optional_invalid_native_libraries_json"
  printf '  "modelFormats": %s,\n' "$(json_array_from_csv "$FORMAT_TARGETS_VALUE")"
  printf '  "mobileFeatures": %s,\n' "$(json_array_from_csv "$MOBILE_FEATURES_VALUE")"
  printf '  "modelSidecarFiles": %s\n' "$(json_array_from_csv "$MODEL_SIDECAR_FILES_VALUE")"
  printf '}\n'
} > "$BUILD_INFO_JSON"

echo ":) Built Gollek Edge native runtime: $LIB_PATH"
echo ":) LiteRT runtime: linked=$LITERT_RUNTIME_AVAILABLE runnable=$LITERT_TEXT_GENERATION_AVAILABLE libs=${#LITERT_PACKAGED_LIBS[@]}"
echo ":) ONNX runtime: linked=$ONNX_RUNTIME_AVAILABLE runnable=$ONNX_TEXT_GENERATION_AVAILABLE libs=${#ONNX_PACKAGED_LIBS[@]}"
echo ":) Android native libs: litert=$ANDROID_LITERT_RUNTIME_AVAILABLE onnx=$ANDROID_ONNX_RUNTIME_AVAILABLE libs=${#ANDROID_PACKAGED_LIBS[@]}"
echo ":) Wrote Flutter build metadata: $BUILD_INFO_JSON"
