#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${MODULE_DIR}/../.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-report-slice.XXXXXX")"
SOURCE_MANIFEST="training/gollek-train-api/src/test/resources/smoke/report-slice-sources.txt"
CLASS_MANIFEST="training/gollek-train-api/src/test/resources/smoke/report-slice-classes.txt"

cleanup() {
  rm -rf "${OUT_DIR}"
}
trap cleanup EXIT

cd "${REPO_DIR}"

require_manifest() {
  local manifest="$1"
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing smoke manifest: ${manifest}" >&2
    exit 1
  fi
}

normalize_manifest_entry() {
  local entry="$1"
  entry="${entry#"${entry%%[![:space:]]*}"}"
  entry="${entry%"${entry##*[![:space:]]}"}"
  printf '%s\n' "${entry}"
}

manifest_entries() {
  local manifest="$1"
  local entry
  while IFS= read -r entry || [[ -n "${entry}" ]]; do
    entry="$(normalize_manifest_entry "${entry}")"
    [[ -z "${entry}" || "${entry}" == \#* ]] && continue
    printf '%s\n' "${entry}"
  done < "${manifest}"
}

validate_unique_entries() {
  local manifest="$1"
  local duplicates
  duplicates="$(manifest_entries "${manifest}" | sort | uniq -d || true)"
  if [[ -n "${duplicates}" ]]; then
    echo "Duplicate entries in ${manifest}:" >&2
    echo "${duplicates}" >&2
    exit 1
  fi
}

validate_source_manifest() {
  local manifest="$1"
  require_manifest "${manifest}"
  validate_unique_entries "${manifest}"
  while IFS= read -r source_file; do
    if [[ "${source_file}" != training/gollek-train-api/src/*/java/*.java ]]; then
      echo "Smoke source must be a Java source under training/gollek-train-api: ${source_file}" >&2
      exit 1
    fi
    if [[ ! -f "${source_file}" ]]; then
      echo "Missing smoke source file from ${manifest}: ${source_file}" >&2
      exit 1
    fi
  done < <(manifest_entries "${manifest}")
}

class_file_for() {
  local smoke_class="$1"
  printf '%s/%s.class\n' "${OUT_DIR}" "${smoke_class//.//}"
}

validate_compiled_smoke_classes() {
  local manifest="$1"
  local smoke_class
  local class_file
  while IFS= read -r smoke_class; do
    class_file="$(class_file_for "${smoke_class}")"
    if [[ ! -f "${class_file}" ]]; then
      echo "Smoke class did not compile from ${manifest}: ${smoke_class}" >&2
      echo "Expected class file: ${class_file}" >&2
      exit 1
    fi
  done < <(manifest_entries "${manifest}")
}

validate_class_manifest() {
  local manifest="$1"
  require_manifest "${manifest}"
  validate_unique_entries "${manifest}"
  while IFS= read -r smoke_class; do
    if [[ ! "${smoke_class}" =~ ^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$ ]]; then
      echo "Invalid smoke class entry in ${manifest}: ${smoke_class}" >&2
      exit 1
    fi
  done < <(manifest_entries "${manifest}")
}

validate_source_manifest "${SOURCE_MANIFEST}"
validate_class_manifest "${CLASS_MANIFEST}"

javac \
  -d "${OUT_DIR}" \
  -sourcepath training/gollek-train-api/src/main/java:training/gollek-train-api/src/test/java \
  "@${SOURCE_MANIFEST}"

validate_compiled_smoke_classes "${CLASS_MANIFEST}"

while IFS= read -r smoke_class; do
  java -cp "${OUT_DIR}" "${smoke_class}"
done < <(manifest_entries "${CLASS_MANIFEST}")

echo "OK: report/profiler slice compiles and smoke checks pass"
