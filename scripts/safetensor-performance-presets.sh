#!/usr/bin/env bash

gollek_safetensor_performance_preset_names() {
  printf '%s\n' m4-smoke m4-greedy-10
}

gollek_safetensor_performance_preset_names_csv() {
  printf 'm4-smoke,m4-greedy-10'
}

gollek_safetensor_performance_preset_description() {
  case "$1" in
    m4-smoke)
      printf 'M4 Metal smoke gate with real Metal path proof, full coverage, fallback rejection, and bounded safetensor profile stages'
      ;;
    m4-greedy-10)
      printf 'M4 greedy 10-token Jakarta smoke with real Metal path proof, full coverage, fallback rejection, and bounded safetensor profile stages'
      ;;
    *)
      return 1
      ;;
  esac
}

gollek_safetensor_performance_preset_validate() {
  case "$1" in
    ""|m4-smoke|m4-greedy-10) return 0 ;;
    *) return 1 ;;
  esac
}

gollek_safetensor_emit_m4_metal_path_gates() {
  printf '%s\t%s\n' profile true
  printf '%s\t%s\n' requireProfile true
  printf '%s\t%s\n' requireMetal true
  printf '%s\t%s\n' requireMetalPaths true
  printf '%s\t%s\n' rejectFallbackPaths true
  printf '%s\t%s\n' minCoreMetalCoverage 1.0
  printf '%s\t%s\n' minLinearMetalCoverage 1.0
  printf '%s\t%s\n' minLogitsMetalCoverage 1.0
  printf '%s\t%s\n' minFfnMetalCoverage 1.0
  printf '%s\t%s\n' minAttentionMetalCoverage 1.0
  printf '%s\t%s\n' minArgmaxMetalCoverage 1.0
}

gollek_safetensor_emit_m4_smoke_thresholds() {
  printf '%s\t%s\n' minSpeedTps 1.0
  printf '%s\t%s\n' topStageThresholdMs 5000
  printf '%s\t%s\n' prefillThresholdMs 5000
  printf '%s\t%s\n' decodeThresholdMs 5000
  printf '%s\t%s\n' tpotThresholdMs 2000
  printf '%s\t%s\n' samplingThresholdMs 1000
  printf '%s\t%s\n' argmaxThresholdMs 500
  printf '%s\t%s\n' attentionThresholdMs 2500
  printf '%s\t%s\n' ffnThresholdMs 2500
  printf '%s\t%s\n' logitsThresholdMs 1500
}

gollek_safetensor_performance_preset_defaults() {
  case "$1" in
    m4-smoke)
      gollek_safetensor_emit_m4_metal_path_gates
      printf '%s\t%s\n' maxTokens 3
      gollek_safetensor_emit_m4_smoke_thresholds
      ;;
    m4-greedy-10)
      gollek_safetensor_emit_m4_metal_path_gates
      printf '%s\t%s\n' maxTokens 10
      printf '%s\t%s\n' requireAnswerRegex 'jakarta.*indonesia|indonesia.*jakarta'
      printf '%s\t%s\n' maxRepeatedTokenRun 2
      printf '%s\t%s\n' minChunks 2
      gollek_safetensor_emit_m4_smoke_thresholds
      ;;
    "")
      ;;
    *)
      return 1
      ;;
  esac
}

gollek_safetensor_performance_preset_list_tsv() {
  local name
  printf 'name\tdescription\n'
  while IFS= read -r name; do
    printf '%s\t%s\n' "$name" "$(gollek_safetensor_performance_preset_description "$name")"
  done < <(gollek_safetensor_performance_preset_names)
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  case "${1:-list}" in
    list|--list)
      gollek_safetensor_performance_preset_list_tsv
      ;;
    names)
      gollek_safetensor_performance_preset_names
      ;;
    defaults)
      if [[ $# -ne 2 ]]; then
        echo "Usage: safetensor-performance-presets.sh defaults PRESET" >&2
        exit 2
      fi
      if ! gollek_safetensor_performance_preset_defaults "$2"; then
        echo "Unknown preset: $2" >&2
        exit 2
      fi
      ;;
    validate)
      if [[ $# -ne 2 ]]; then
        echo "Usage: safetensor-performance-presets.sh validate PRESET" >&2
        exit 2
      fi
      gollek_safetensor_performance_preset_validate "$2"
      ;;
    *)
      echo "Usage: safetensor-performance-presets.sh [list|names|defaults PRESET|validate PRESET]" >&2
      exit 2
      ;;
  esac
fi
