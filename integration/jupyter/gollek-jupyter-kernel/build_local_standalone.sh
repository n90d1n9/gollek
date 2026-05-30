#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/local-standalone"
CLASSES_DIR="$BUILD_DIR/classes"
STAGE_DIR="$BUILD_DIR/stage"
LIB_DIR="$SCRIPT_DIR/build/libs"
OUTPUT_JAR="$LIB_DIR/gollek-kernel.jar"
M2_REPO="${GOLLEK_JUPYTER_MANUAL_M2_REPO:-$HOME/.m2/repository}"
JAVAC_BIN="${JAVAC_BIN:-$(command -v javac || true)}"
JAR_BIN="${JAR_BIN:-$(command -v jar || true)}"
UNZIP_BIN="${UNZIP_BIN:-$(command -v unzip || true)}"

usage() {
  cat <<'EOF'
Usage:
  ./build_local_standalone.sh

Environment:
  GOLLEK_JUPYTER_MANUAL_M2_REPO Override the Maven repository to read jars from.
  JAVAC_BIN                     Override the javac executable.
  JAR_BIN                       Override the jar executable.
  UNZIP_BIN                     Override the unzip executable.
EOF
}

require_bin() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "$name executable not found."
    exit 1
  fi
}

jar_path() {
  local rel="$1"
  printf '%s/%s\n' "$M2_REPO" "$rel"
}

require_jar() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required jar not found: $path"
    exit 1
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  require_bin "javac" "$JAVAC_BIN"
  require_bin "jar" "$JAR_BIN"
  require_bin "unzip" "$UNZIP_BIN"

  local jjava_jar tensor_jar autograd_jar nn_jar gson_jar slf4j_jar
  jjava_jar="$(jar_path "org/dflib/jjava/jjava/1.0-a4/jjava-1.0-a4.jar")"
  tensor_jar="$(jar_path "tech/kayys/gollek/gollek-tensor/0.1.0-SNAPSHOT/gollek-tensor-0.1.0-SNAPSHOT.jar")"
  autograd_jar="$(jar_path "tech/kayys/gollek/gollek-ml-autograd/0.1.0-SNAPSHOT/gollek-ml-autograd-0.1.0-SNAPSHOT.jar")"
  nn_jar="$(jar_path "tech/kayys/gollek/gollek-ml-nn/0.1.0-SNAPSHOT/gollek-ml-nn-0.1.0-SNAPSHOT.jar")"
  gson_jar="$(jar_path "com/google/code/gson/gson/2.11.0/gson-2.11.0.jar")"
  slf4j_jar="$(jar_path "org/slf4j/slf4j-api/2.0.13/slf4j-api-2.0.13.jar")"

  require_jar "$jjava_jar"
  require_jar "$tensor_jar"
  require_jar "$autograd_jar"
  require_jar "$nn_jar"
  require_jar "$gson_jar"
  require_jar "$slf4j_jar"

  rm -rf "$BUILD_DIR"
  mkdir -p "$CLASSES_DIR" "$STAGE_DIR" "$LIB_DIR"

  local compile_cp
  compile_cp="$jjava_jar"

  echo "Compiling standalone Gollek Jupyter kernel"
  echo "  m2 repo: $M2_REPO"
  "$JAVAC_BIN" \
    --enable-preview \
    --release 25 \
    -cp "$compile_cp" \
    -d "$CLASSES_DIR" \
    $(find "$SCRIPT_DIR/src/main/java" -name '*.java' | sort)

  cp -R "$CLASSES_DIR"/. "$STAGE_DIR"/
  cp "$SCRIPT_DIR/src/main/resources/kernel.json" "$STAGE_DIR"/

  local dep
  for dep in \
    "$jjava_jar" \
    "$tensor_jar" \
    "$autograd_jar" \
    "$nn_jar" \
    "$gson_jar" \
    "$slf4j_jar"
  do
    "$UNZIP_BIN" -oq "$dep" -d "$STAGE_DIR"
  done

  rm -f "$OUTPUT_JAR"
  "$JAR_BIN" --create \
    --file "$OUTPUT_JAR" \
    --main-class tech.kayys.gollek.jupyter.KernelLauncher \
    -C "$STAGE_DIR" .

  echo "✅ Standalone kernel jar built"
  echo "   $OUTPUT_JAR"
}

main "$@"
