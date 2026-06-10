#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATTERN='tech\.kayys\.wayang|WAYANG_[A-Z0-9_]*|wayang\.(inference|multitenancy|home)|(^|[^[:alnum:]_])\.wayang(/|$)|Wayang-Gollek|Wayang/Gollek|Wayang API|wayang-inference'

SCAN_PATHS=(
  core
  plugins
  runner
  sdk
  ui/gollek-cli/src/main
  scripts/prepare-llama-source.sh
  scripts/prepare-libtorch-source.sh
)

if rg -n --color never \
  --glob '!**/target/**' \
  --glob '!**/build/**' \
  --glob '!**/.gradle/**' \
  --glob '!**/src/test/**' \
  --glob '!**/*.md' \
  --glob '!**/README*' \
  -e "$PATTERN" \
  "${SCAN_PATHS[@]}"; then
  echo
  echo "Gollek product boundary violation: Gollek production code must not depend on Wayang names."
  echo "Use Gollek-owned APIs/config such as gollek.* properties and GOLLEK_* environment variables."
  exit 1
fi

echo "Gollek product boundary OK."
