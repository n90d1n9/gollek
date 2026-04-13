#!/usr/bin/env bash
set -euo pipefail

resolve_gollek_home() {
  if [ -n "${GOLLEK_HOME:-}" ]; then
    echo "$GOLLEK_HOME"
    return
  fi
  if [ -n "${WAYANG_HOME:-}" ]; then
    echo "$WAYANG_HOME/gollek"
    return
  fi
  if [ -d "$HOME/.wayang/gollek" ] || [ ! -d "$HOME/.gollek" ]; then
    echo "$HOME/.wayang/gollek"
    return
  fi
  echo "$HOME/.gollek"
}

GOLLEK_HOME_RESOLVED="$(resolve_gollek_home)"
LIBTORCH_SRC="${GOLLEK_LIBTORCH_SOURCE_DIR:-$GOLLEK_HOME_RESOLVED/source/vendor/libtorch}"

mkdir -p "$(dirname "$LIBTORCH_SRC")"

if [ -d "$LIBTORCH_SRC" ]; then
  echo "Using existing libtorch source at: $LIBTORCH_SRC"
  exit 0
fi

if [ -n "${GOLLEK_LIBTORCH_ARCHIVE_URL:-}" ]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  ARCHIVE_PATH="$TMP_DIR/libtorch.zip"
  curl -fsSL "$GOLLEK_LIBTORCH_ARCHIVE_URL" -o "$ARCHIVE_PATH"
  unzip -q "$ARCHIVE_PATH" -d "$TMP_DIR"
  if [ -d "$TMP_DIR/libtorch" ]; then
    mv "$TMP_DIR/libtorch" "$LIBTORCH_SRC"
    echo "Prepared libtorch source from archive at: $LIBTORCH_SRC"
    exit 0
  fi
  echo "Archive did not contain a top-level 'libtorch' directory: $GOLLEK_LIBTORCH_ARCHIVE_URL"
  exit 1
fi

echo "No libtorch source prepared. Expected existing directory at: $LIBTORCH_SRC"
echo "Set GOLLEK_LIBTORCH_ARCHIVE_URL to auto-download in CI when needed."
echo "Continuing without libtorch source; build may skip libtorch-dependent paths."
mkdir -p "$LIBTORCH_SRC/lib"
exit 0
