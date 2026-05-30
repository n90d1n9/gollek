#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOCAL_CACHE_DIR="$MODULE_DIR/.cache"
LOCAL_M2="$LOCAL_CACHE_DIR/m2"
LOCAL_GRADLE_HOME="$LOCAL_CACHE_DIR/gradle"
DEFAULT_MVN_BIN="${MVN_BIN:-$(command -v mvn || true)}"
DEFAULT_GRADLE_BIN="${GRADLE_BIN:-$(command -v gradle || true)}"
KERNEL_JAR="$MODULE_DIR/target/gollek-kernel.jar"

usage() {
  cat <<'EOF'
Usage:
  ./rebuild_kernel.sh doctor
  ./rebuild_kernel.sh seed-home-m2
  ./rebuild_kernel.sh maven
  ./rebuild_kernel.sh maven-diagnose
  ./rebuild_kernel.sh gradle
  ./rebuild_kernel.sh gradle-fat
  ./rebuild_kernel.sh local-fat

Environment:
  MVN_BIN                  Override the Maven executable.
  GRADLE_BIN               Override the Gradle executable.
  GOLLEK_JUPYTER_SEED_M2   Seed the local Maven repo cache before Maven build. Default: false
  GOLLEK_JUPYTER_SEED_MODE focused|full. Default: focused
  GOLLEK_JUPYTER_LOCAL_M2  Override the local Maven cache path.
  GOLLEK_JUPYTER_GRADLE_HOME Override the local Gradle user home path.
  GOLLEK_JUPYTER_MAVEN_OFFLINE Run Maven with -o after seeding. Default: true
  GOLLEK_JUPYTER_MANUAL_M2_REPO Override the Maven repo for local-fat jar assembly.
EOF
}

local_m2() {
  printf '%s\n' "${GOLLEK_JUPYTER_LOCAL_M2:-$LOCAL_M2}"
}

local_gradle_home() {
  printf '%s\n' "${GOLLEK_JUPYTER_GRADLE_HOME:-$LOCAL_GRADLE_HOME}"
}

maven_log_path() {
  printf '%s\n' "$LOCAL_CACHE_DIR/maven-build.log"
}

latest_source() {
  find "$MODULE_DIR/src" "$MODULE_DIR/pom.xml" "$MODULE_DIR/build.gradle.kts" -type f 2>/dev/null | xargs stat -f '%m %N' | sort -nr | head -n1
}

jar_state() {
  if [[ -f "$KERNEL_JAR" ]]; then
    stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' "$KERNEL_JAR"
  else
    echo "missing $KERNEL_JAR"
  fi
}

seed_home_m2() {
  local src="$HOME/.m2/repository"
  local dst
  dst="$(local_m2)"
  local mode="${GOLLEK_JUPYTER_SEED_MODE:-focused}"
  if [[ "${GOLLEK_JUPYTER_SEED_M2:-false}" != "true" ]]; then
    return
  fi
  if [[ ! -d "$src" ]]; then
    echo "Home Maven repository not found: $src"
    return
  fi
  mkdir -p "$dst"
  echo "Seeding local Maven cache from $src"
  echo "  mode: $mode"
  if [[ "$mode" == "full" ]]; then
    rsync -a --ignore-existing "$src/" "$dst/"
    return
  fi

  local paths=(
    "tech/kayys/gollek"
    "org/dflib"
    "org/apache/maven"
    "org/apache/maven/plugins"
    "org/apache/apache"
    "org/apache/commons"
    "org/codehaus/plexus"
    "org/eclipse"
    "org/jboss"
    "org/junit"
    "org/reactivestreams"
    "org/jetbrains"
    "org/apache/ivy"
    "org/ow2"
    "org/slf4j"
    "org/zeromq"
    "io/quarkus"
    "io/grpc"
    "io/smallrye"
    "jakarta"
    "com/fasterxml"
    "com/google/code/gson"
    "software/amazon/awssdk"
    "eu/neilalexander"
    "commons-codec"
    "commons-io"
    "commons-cli"
  )
  local rel
  for rel in "${paths[@]}"; do
    if [[ -d "$src/$rel" ]]; then
      mkdir -p "$(dirname "$dst/$rel")"
      rsync -a --ignore-existing "$src/$rel/" "$dst/$rel/"
    fi
  done
}

seed_home_m2_command() {
  GOLLEK_JUPYTER_SEED_M2=true seed_home_m2
  echo "Seed complete:"
  echo "  $(local_m2)"
}

doctor() {
  echo "Gollek Jupyter kernel rebuild doctor"
  echo "  module:        $MODULE_DIR"
  echo "  repo root:     $REPO_ROOT"
  echo "  local m2:      $(local_m2)"
  echo "  local gradle:  $(local_gradle_home)"
  echo "  mvn:           ${DEFAULT_MVN_BIN:-missing}"
  echo "  gradle:        ${DEFAULT_GRADLE_BIN:-missing}"
  echo "  kernel jar:    $(jar_state)"
  echo "  latest source: $(latest_source)"
  echo "  seed home m2:  ${GOLLEK_JUPYTER_SEED_M2:-false}"
  echo "  seed mode:     ${GOLLEK_JUPYTER_SEED_MODE:-focused}"
  echo "  maven offline: ${GOLLEK_JUPYTER_MAVEN_OFFLINE:-true}"
  echo "  maven log:     $(maven_log_path)"
  echo
  echo "Recommended next steps:"
  echo "  1. ./rebuild_kernel.sh local-fat"
  echo "  2. ./rebuild_kernel.sh seed-home-m2"
  echo "  3. ./rebuild_kernel.sh maven"
  echo "  4. ./rebuild_kernel.sh maven-diagnose"
  echo "  5. python3 install.py --dry-run"
  echo "  6. python3 selftest.py"
}

maven_build_impl() {
  if [[ -z "$DEFAULT_MVN_BIN" ]]; then
    echo "Maven executable not found."
    exit 1
  fi
  mkdir -p "$(local_m2)"
  seed_home_m2
  echo "Building Gollek Jupyter kernel with Maven"
  echo "  maven repo local: $(local_m2)"
  echo "  offline: ${GOLLEK_JUPYTER_MAVEN_OFFLINE:-true}"
  mkdir -p "$LOCAL_CACHE_DIR"
  local log_file
  log_file="$(maven_log_path)"
  : >"$log_file"
  (
    cd "$MODULE_DIR"
    "$DEFAULT_MVN_BIN" \
      -Dmaven.repo.local="$(local_m2)" \
      $( [[ "${GOLLEK_JUPYTER_MAVEN_OFFLINE:-true}" == "true" ]] && printf '%s' "-o" ) \
      -DskipTests \
      package 2>&1
  ) | tee "$log_file"
}

maven_build() {
  maven_build_impl
}

artifact_path_from_coord() {
  local repo_root="$1"
  local coord="$2"
  IFS=':' read -r group artifact packaging version <<<"$coord"
  local group_path="${group//./\/}"
  printf '%s/%s/%s/%s/%s-%s.%s\n' "$repo_root" "$group_path" "$artifact" "$version" "$artifact" "$version" "$packaging"
}

diagnose_log() {
  local log_file
  log_file="$(maven_log_path)"
  if [[ ! -f "$log_file" ]]; then
    echo "No Maven log found at $log_file"
    echo "Run: ./rebuild_kernel.sh maven-diagnose"
    exit 1
  fi

  echo
  echo "Maven offline diagnostics"
  echo "  log: $log_file"

  local artifacts=()
  while IFS= read -r coord; do
    if [[ -n "$coord" ]]; then
      artifacts+=("$coord")
    fi
  done < <(python3 - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='replace').read()
coords = re.findall(r'([A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:(?:pom|jar|xml|module):[A-Za-z0-9_.-]+)', text)
seen = []
for c in coords:
    if c not in seen:
        seen.append(c)
for c in seen:
    print(c)
PY
)

  if (( ${#artifacts[@]} == 0 )); then
    echo "  No missing artifact coordinates were parsed from the log."
    echo "  Search the log manually for the first plugin/dependency resolution failure."
    return
  fi

  printf "%-45s %-10s %-10s\n" "artifact" "local_m2" "home_m2"
  printf "%-45s %-10s %-10s\n" "---------------------------------------------" "----------" "----------"

  local coord local_path home_path local_status home_status home_repo
  home_repo="$HOME/.m2/repository"
  for coord in "${artifacts[@]}"; do
    local_path="$(artifact_path_from_coord "$(local_m2)" "$coord")"
    home_path="$(artifact_path_from_coord "$home_repo" "$coord")"
    if [[ -f "$local_path" ]]; then
      local_status="present"
    else
      local_status="missing"
    fi
    if [[ -f "$home_path" ]]; then
      home_status="present"
    else
      home_status="missing"
    fi
    printf "%-45s %-10s %-10s\n" "$coord" "$local_status" "$home_status"
  done

  local missing_in_local_present_in_home=0
  local missing_everywhere=0
  for coord in "${artifacts[@]}"; do
    local_path="$(artifact_path_from_coord "$(local_m2)" "$coord")"
    home_path="$(artifact_path_from_coord "$home_repo" "$coord")"
    if [[ ! -f "$local_path" && -f "$home_path" ]]; then
      missing_in_local_present_in_home=$((missing_in_local_present_in_home + 1))
    fi
    if [[ ! -f "$local_path" && ! -f "$home_path" ]]; then
      missing_everywhere=$((missing_everywhere + 1))
    fi
  done

  echo
  echo "Summary:"
  echo "  missing in local cache but present in ~/.m2: $missing_in_local_present_in_home"
  echo "  missing in both local cache and ~/.m2: $missing_everywhere"
  if (( missing_in_local_present_in_home > 0 )); then
    echo "  Recommendation: retry with focused or full seed-home-m2."
  fi
  if (( missing_everywhere > 0 )); then
    echo "  Recommendation: full seed will not be enough for all artifacts; network or another source cache may still be required."
  fi
}

maven_diagnose() {
  set +e
  maven_build_impl
  local build_exit=$?
  set -e
  echo
  echo "Maven build exit code: $build_exit"
  diagnose_log
  return 0
}

gradle_build() {
  if [[ -z "$DEFAULT_GRADLE_BIN" ]]; then
    echo "Gradle executable not found."
    exit 1
  fi
  mkdir -p "$(local_gradle_home)"
  echo "Building Gollek Jupyter kernel with Gradle"
  echo "  gradle user home: $(local_gradle_home)"
  (
    cd "$REPO_ROOT"
    GRADLE_USER_HOME="$(local_gradle_home)" \
      "$DEFAULT_GRADLE_BIN" \
      --no-daemon \
      :integration:jupyter:gollek-jupyter-kernel:build
  )
}

gradle_fat_build() {
  if [[ -z "$DEFAULT_GRADLE_BIN" ]]; then
    echo "Gradle executable not found."
    exit 1
  fi
  mkdir -p "$(local_gradle_home)"
  echo "Building standalone Gollek Jupyter fat jar with Gradle"
  echo "  gradle user home: $(local_gradle_home)"
  (
    cd "$REPO_ROOT"
    GRADLE_USER_HOME="$(local_gradle_home)" \
      "$DEFAULT_GRADLE_BIN" \
      --no-daemon \
      :integration:jupyter:gollek-jupyter-kernel:fatJar
  )
}

local_fat_build() {
  echo "Building standalone Gollek Jupyter fat jar with local javac/jar fallback"
  "$MODULE_DIR/build_local_standalone.sh"
}

COMMAND="${1:-doctor}"

case "$COMMAND" in
  doctor)
    doctor
    ;;
  seed-home-m2)
    seed_home_m2_command
    ;;
  maven)
    maven_build
    ;;
  maven-diagnose)
    maven_diagnose
    ;;
  gradle)
    gradle_build
    ;;
  gradle-fat)
    gradle_fat_build
    ;;
  local-fat)
    local_fat_build
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo
    usage
    exit 1
    ;;
esac
