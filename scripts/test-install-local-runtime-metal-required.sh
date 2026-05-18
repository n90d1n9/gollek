#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(uname -s)" != "Darwin" ]; then
    printf 'install-local-runtime metal-required test skipped on non-Darwin\n'
    exit 0
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gollek-install-metal-required.XXXXXX")"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

FAKE_BIN="$TMP_DIR/bin"
mkdir -p "$FAKE_BIN" "$TMP_DIR/home" "$TMP_DIR/gollek-bin" "$TMP_DIR/libs" "$TMP_DIR/runtime"

cat > "$FAKE_BIN/make" <<'SH'
#!/usr/bin/env bash
printf 'fake make invoked: %s\n' "$*" >> "$GOLLEK_FAKE_MAKE_LOG"
exit 42
SH
chmod +x "$FAKE_BIN/make"

OUTPUT="$TMP_DIR/output.log"
export GOLLEK_FAKE_MAKE_LOG="$TMP_DIR/make.log"

set +e
HOME="$TMP_DIR/home" \
PATH="$FAKE_BIN:$PATH" \
GOLLEK_BIN_DIR="$TMP_DIR/gollek-bin" \
GOLLEK_LIB_DIR="$TMP_DIR/libs" \
GOLLEK_RUNTIME_DIR="$TMP_DIR/runtime" \
GOLLEK_SKIP_BUILD=true \
    bash "$ROOT_DIR/scripts/install-local-runtime.sh" >"$OUTPUT" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "Expected install-local-runtime to fail when Metal bridge make fails" >&2
    cat "$OUTPUT" >&2
    exit 1
fi
if ! grep -q 'Metal native bridge build/install failed' "$OUTPUT"; then
    echo "Expected fatal Metal build failure message" >&2
    cat "$OUTPUT" >&2
    exit 1
fi
if ! grep -q 'GOLLEK_ALLOW_METAL_BUILD_FAILURE=true' "$OUTPUT"; then
    echo "Expected diagnostic escape-hatch guidance" >&2
    cat "$OUTPUT" >&2
    exit 1
fi
if ! grep -q 'fake make invoked: -C .*/backend/metal/gollek-backend-metal/src/main/cpp/metal install INSTALL_DIR=' "$GOLLEK_FAKE_MAKE_LOG"; then
    echo "Expected installer to invoke Metal Makefile through make -C" >&2
    cat "$GOLLEK_FAKE_MAKE_LOG" >&2
    exit 1
fi

printf 'install-local-runtime metal-required test passed\n'
