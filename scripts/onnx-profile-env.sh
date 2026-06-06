#!/usr/bin/env bash

gollek_onnx_profile_env_safe_tsv_field() {
  local value="${1:-}"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  printf '%s' "$value"
}

gollek_onnx_profile_env_uname() {
  local flag="$1"
  local fallback="${2:-unknown}"
  local value=""
  if command -v uname >/dev/null 2>&1; then
    value="$(uname "$flag" 2>/dev/null | awk 'NR == 1 { print; exit }')"
  fi
  printf '%s' "${value:-$fallback}"
}

gollek_onnx_profile_env_java_version() {
  local value=""
  if command -v java >/dev/null 2>&1; then
    value="$(java -version 2>&1 | awk 'NR == 1 { print; exit }')"
  fi
  printf '%s' "${value:-unknown}"
}

gollek_onnx_profile_env_root_dir() {
  local root=""
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
  printf '%s' "$root"
}

gollek_onnx_profile_env_gradle_value() {
  local root="$1"
  local key="$2"
  local file="${root%/}/build.gradle.kts"
  local value=""
  if [[ -f "$file" ]]; then
    value="$(awk -v key="$key" '
      index($0, "extra[\"" key "\"]") {
        value = $0
        sub(/^.*= *"/, "", value)
        sub(/".*$/, "", value)
        print value
        exit
      }
    ' "$file")"
  fi
  printf '%s' "${value:-unknown}"
}

gollek_onnx_profile_env_git_value() {
  local root="$1"
  local field="$2"
  local value=""
  if [[ -z "$root" || ! -d "$root" ]] || ! command -v git >/dev/null 2>&1; then
    printf 'unavailable'
    return 0
  fi
  if ! git -C "$root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'unavailable'
    return 0
  fi
  case "$field" in
    commit)
      value="$(git -C "$root" rev-parse --short=12 HEAD 2>/dev/null || true)"
      ;;
    commitFull)
      value="$(git -C "$root" rev-parse HEAD 2>/dev/null || true)"
      ;;
    branch)
      value="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
      ;;
    dirty)
      if [[ -n "$(git -C "$root" status --porcelain 2>/dev/null)" ]]; then
        value="dirty"
      else
        value="clean"
      fi
      ;;
    *)
      value="unknown"
      ;;
  esac
  printf '%s' "${value:-unknown}"
}

gollek_onnx_profile_env_write_row() {
  local key="$1"
  local value="${2:-}"
  printf '%s\t%s\n' \
    "$(gollek_onnx_profile_env_safe_tsv_field "$key")" \
    "$(gollek_onnx_profile_env_safe_tsv_field "$value")"
}

gollek_onnx_profile_env_write() {
  local out="$1"
  local gollek_bin="${2:-}"
  local root_dir="${3:-}"
  if [[ -z "$root_dir" ]]; then
    root_dir="$(gollek_onnx_profile_env_root_dir)"
  fi
  {
    printf 'key\tvalue\n'
    gollek_onnx_profile_env_write_row generatedAt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    gollek_onnx_profile_env_write_row rootDir "$root_dir"
    gollek_onnx_profile_env_write_row gollekVersion "$(gollek_onnx_profile_env_gradle_value "$root_dir" gollekVersion)"
    gollek_onnx_profile_env_write_row quarkusVersion "$(gollek_onnx_profile_env_gradle_value "$root_dir" quarkusVersion)"
    gollek_onnx_profile_env_write_row gitCommit "$(gollek_onnx_profile_env_git_value "$root_dir" commit)"
    gollek_onnx_profile_env_write_row gitCommitFull "$(gollek_onnx_profile_env_git_value "$root_dir" commitFull)"
    gollek_onnx_profile_env_write_row gitBranch "$(gollek_onnx_profile_env_git_value "$root_dir" branch)"
    gollek_onnx_profile_env_write_row gitDirty "$(gollek_onnx_profile_env_git_value "$root_dir" dirty)"
    gollek_onnx_profile_env_write_row hostOs "$(gollek_onnx_profile_env_uname -s)"
    gollek_onnx_profile_env_write_row hostArch "$(gollek_onnx_profile_env_uname -m)"
    gollek_onnx_profile_env_write_row kernelRelease "$(gollek_onnx_profile_env_uname -r)"
    gollek_onnx_profile_env_write_row javaVersion "$(gollek_onnx_profile_env_java_version)"
    gollek_onnx_profile_env_write_row javaHome "${JAVA_HOME:-}"
    gollek_onnx_profile_env_write_row gollekJavaOpts "${GOLLEK_JAVA_OPTS:-}"
    gollek_onnx_profile_env_write_row gollekBin "$gollek_bin"
  } > "$out"
}
