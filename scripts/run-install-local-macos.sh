#!/bin/bash
set -euo pipefail

RESET="$(printf '\033[0m')"
BOLD="$(printf '\033[1m')"
GREEN="$(printf '\033[32m')"
CYAN="$(printf '\033[36m')"
YELLOW="$(printf '\033[33m')"
MAGENTA="$(printf '\033[35m')"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/module-selection-current.env"
ARCHITECTURE_VALUE="${ARCHITECTURE_TARGETS:-${ARCHITECTURE_TARGET:-native-java,binding}}"
PUBLISH_RUNTIME_LOCAL="${GOLLEK_PUBLISH_RUNTIME_LOCAL:-false}"
export GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES="${GOLLEK_SKIP_LEGACY_SOURCE_HOTPATCHES:-true}"

if command -v /usr/libexec/java_home >/dev/null 2>&1; then
  JAVA_25_HOME="$(/usr/libexec/java_home -v 25 2>/dev/null || true)"
  if [[ -n "${JAVA_25_HOME}" ]]; then
    export JAVA_HOME="${JAVA_25_HOME}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
  fi
fi

GRADLE_JAVA_HOME_ARG=()
if [[ -n "${JAVA_HOME:-}" ]]; then
  GRADLE_JAVA_HOME_ARG+=("-Dorg.gradle.java.home=${JAVA_HOME}")
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

is_legacy_gradle_module() {
  case "$1" in
    :ml:*|:trainer:*|:examples:gollek-ml-examples|:models:gollek-model-common|:runner:onnx:gollek-ml-export-onnx|:runner:onnx:gollek-ml-onnx|:runner:diffuser:gollek-diffuser) return 0 ;;
    *) return 1 ;;
  esac
}

read -r -a MODULE_ARRAY <<< "${MODULES:-}"
SUPPORTED_MODULES=()
SKIPPED_MODULES=()
for module in "${MODULE_ARRAY[@]}"; do
  if is_legacy_gradle_module "$module"; then
    SKIPPED_MODULES+=("$module")
  else
    SUPPORTED_MODULES+=("$module")
  fi
done

GRADLE_INSTALL_TASKS=()
for module in "${SUPPORTED_MODULES[@]}"; do
  GRADLE_INSTALL_TASKS+=("${module}:publishToMavenLocal")
done

echo "${BOLD}${CYAN}:) Resolved backend targets:${RESET} ${BACKEND_TARGETS}"
echo "${BOLD}${CYAN}:) Resolved format targets:${RESET} ${FORMAT_TARGETS}"
echo "${BOLD}${CYAN}:) Resolved LLM targets:${RESET} ${LLM_TARGETS}"
echo "${BOLD}${MAGENTA}:) Architecture:${RESET} ${ARCHITECTURE_VALUE}"
echo "${BOLD}${MAGENTA}:) Profile:${RESET} ${BUILD_PROFILE}"
echo "${BOLD}${MAGENTA}:) Java home:${RESET} ${JAVA_HOME:-$(command -v java)}"
echo "${BOLD}${MAGENTA}:) Build mode:${RESET} Gradle-only install"
echo "${BOLD}${GREEN}:) Selection manifest:${RESET} ${ROOT_DIR}/scripts/module-selection-current.env"
echo "$GREEN:) Module manifest resolved cleanly.$RESET"

if [[ ${#SKIPPED_MODULES[@]} -gt 0 ]]; then
  echo "${BOLD}${YELLOW}:| Warning:${RESET} skipping legacy Gradle modules in local install wrapper:"
  for module in "${SKIPPED_MODULES[@]}"; do
    echo "   - ${module}"
  done
fi

echo "${BOLD}${GREEN}:) Step 1/2:${RESET} packaging and installing gollek via the macOS local installer"
echo "${BOLD}${YELLOW}:| Install path:${RESET} ${GOLLEK_BIN_DIR:-$HOME/.local/bin}/gollek"
bash "$ROOT_DIR/scripts/install-local-runtime.sh"

echo "${BOLD}${GREEN}:) Step 2/2:${RESET} optional compatibility publish to local artifact cache"
if [[ "${PUBLISH_RUNTIME_LOCAL}" == "true" ]]; then
  if [[ ${#GRADLE_INSTALL_TASKS[@]} -gt 0 ]]; then
    if ./gradlew "${GRADLE_JAVA_HOME_ARG[@]}" "${GRADLE_INSTALL_TASKS[@]}" \
      -Pgollek.backend="${RESOLVED_BACKEND_PROPERTY}" \
      -Pgollek.profile="${BUILD_PROFILE}" \
      -Pgollek.model.formats="${FORMAT_TARGETS}" \
      -Pgollek.llm.types="${LLM_TARGETS}" \
      -Pgollek.architecture="${ARCHITECTURE_VALUE}"; then
      echo "${GREEN}✓ Published supported runtime artifacts to local artifact cache${RESET}"
    else
      echo "${YELLOW}⚠ Optional Gradle publish failed after local install; CLI install is still complete.${RESET}"
    fi
  else
    echo "${YELLOW}⚠ No supported Gradle runtime modules were selected for publish.${RESET}"
  fi
else
  echo "${YELLOW}⚠ Skipping local artifact publish by default. Set GOLLEK_PUBLISH_RUNTIME_LOCAL=true to enable it.${RESET}"
fi

echo "${BOLD}${GREEN}:D Done.${RESET} Gollek is installed locally, and compatibility publish is optional."
