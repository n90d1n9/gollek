#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_INSTALLER="$SCRIPT_DIR/../gollek-jupyter-kernel/install.py"

if [[ ! -f "$PY_INSTALLER" ]]; then
  echo "❌ Standalone kernel installer not found: $PY_INSTALLER"
  exit 1
fi

echo "Gollek Jupyter legacy installer shim"
echo "  legacy IJava kernelspec flow has been replaced by the standalone Gollek kernel."
echo "  delegating to: $PY_INSTALLER"
echo

python3 "$PY_INSTALLER" "$@"
