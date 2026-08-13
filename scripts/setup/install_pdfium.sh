#!/bin/bash
# Ensure third_party/pdfium holds a PDFium matching the build OS + architecture.
#
# The repo VENDORS two binaries next to one shared include/ tree:
#   lib/libpdfium.so     linux-x64
#   lib/libpdfium.dylib  mac-arm64
# Anything else (linux-arm64, mac-x64) is fetched from bblanchon/pdfium-binaries
# and unpacked alongside them — installing one platform never removes another,
# so a cross-built tree keeps working.
#
# The release is PINNED (PDFIUM_RELEASE default below) and the downloaded
# tarball's SHA256 is verified against a recorded per-target hash; an empty or
# mismatched hash is a hard failure unless ALLOW_UNVERIFIED=1.
#
# Caveat: bblanchon's linux-aarch64 build aborts at startup on kernels whose
# page size is neither 4 KiB nor 16 KiB (e.g. some RHEL/CentOS aarch64
# configured at 64 KiB). Standard Ubuntu/Debian aarch64 and NVIDIA Jetson/L4T
# use 4 KiB pages and work. See
# https://github.com/bblanchon/pdfium-binaries/issues/148
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PDFIUM_DIR="$ROOT/third_party/pdfium"

# Prefer Docker's TARGETOS/TARGETARCH when set; else the host.
OS="${TARGETOS:-$(uname -s)}"
case "$OS" in
  [Ll]inux)        PDFIUM_OS="linux"; LIB_NAME="libpdfium.so" ;;
  [Dd]arwin|macos) PDFIUM_OS="mac";   LIB_NAME="libpdfium.dylib" ;;
  *) echo "install_pdfium: unsupported OS '$OS'" >&2; exit 1 ;;
esac

ARCH="${TARGETARCH:-$(uname -m)}"
case "$ARCH" in
  x86_64|amd64|x64) PDFIUM_ARCH="x64" ;;
  aarch64|arm64)    PDFIUM_ARCH="arm64" ;;
  *) echo "install_pdfium: unsupported arch '$ARCH'" >&2; exit 1 ;;
esac

TARGET="${PDFIUM_OS}-${PDFIUM_ARCH}"
LIB_PATH="$PDFIUM_DIR/lib/$LIB_NAME"

# Idempotent: if the installed library already matches the target, do nothing.
# Falls back to trusting the vendored copies when `file` is unavailable.
if [ -f "$LIB_PATH" ]; then
  if command -v file >/dev/null 2>&1; then
    cur="$(file -b "$LIB_PATH" 2>/dev/null || echo '')"
    case "$TARGET" in
      linux-x64)   echo "$cur" | grep -q "x86-64"  && { echo "install_pdfium: already $TARGET"; exit 0; } ;;
      linux-arm64) echo "$cur" | grep -q "aarch64" && { echo "install_pdfium: already $TARGET"; exit 0; } ;;
      mac-arm64)   echo "$cur" | grep -q "arm64"   && { echo "install_pdfium: already $TARGET"; exit 0; } ;;
      mac-x64)     echo "$cur" | grep -q "x86_64"  && { echo "install_pdfium: already $TARGET"; exit 0; } ;;
    esac
  else
    case "$TARGET" in
      linux-x64|mac-arm64)
        echo "install_pdfium: $TARGET — using vendored third_party/pdfium"
        exit 0 ;;
    esac
  fi
fi

# Pinned to a concrete bblanchon tag (not the moving `latest`) so a clean-room
# build is reproducible and the SHA256 below stays valid. Override PDFIUM_RELEASE
# to bump; if you do, also update the matching hash or pass PDFIUM_SHA256.
PDFIUM_RELEASE="${PDFIUM_RELEASE:-chromium/7857}"

# Per-target SHA256 of pdfium-<target>.tgz for the pinned release above.
# An explicit PDFIUM_SHA256 env overrides these (e.g. when bumping the tag).
PDFIUM_SHA256_LINUX_X64="2ad1fd4237cd491201ac74a72388199b9dcf546c5cb02d8fea700725a1b80541"
PDFIUM_SHA256_LINUX_ARM64="0e24373e73c50759136196c0078db8656860c8d03a10b2cb4a2e7b72d8068e35"
PDFIUM_SHA256_MAC_ARM64="65a4a6b0028675113cac99cad61469eb6a482d7283e21a1faefc6e63587109c5"
PDFIUM_SHA256_MAC_X64=""

if [ "$PDFIUM_RELEASE" = "latest" ]; then
  URL="https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-${TARGET}.tgz"
else
  URL="https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_RELEASE}/pdfium-${TARGET}.tgz"
fi

# Resolve the expected hash: explicit env wins, else the per-target pinned value.
EXPECTED_SHA256="${PDFIUM_SHA256:-}"
if [ -z "$EXPECTED_SHA256" ]; then
  case "$TARGET" in
    linux-x64)   EXPECTED_SHA256="$PDFIUM_SHA256_LINUX_X64" ;;
    linux-arm64) EXPECTED_SHA256="$PDFIUM_SHA256_LINUX_ARM64" ;;
    mac-arm64)   EXPECTED_SHA256="$PDFIUM_SHA256_MAC_ARM64" ;;
    mac-x64)     EXPECTED_SHA256="$PDFIUM_SHA256_MAC_X64" ;;
  esac
fi

echo "install_pdfium: fetching ${TARGET} PDFium from $URL"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TMP/pdfium.tgz"
else
  wget -q "$URL" -O "$TMP/pdfium.tgz"
fi

# Integrity check is REQUIRED: a known-good hash must exist and match, or we
# refuse to install. Set ALLOW_UNVERIFIED=1 only for throwaway experiments
# (e.g. pinning a brand-new tag before its hash is recorded here).
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1   # macOS ships shasum, not sha256sum
  fi
}
if [ "${ALLOW_UNVERIFIED:-0}" = "1" ]; then
  echo "install_pdfium: WARNING — ALLOW_UNVERIFIED=1, skipping SHA256 verification of $URL" >&2
elif [ -z "$EXPECTED_SHA256" ]; then
  echo "install_pdfium: no SHA256 known for target '$TARGET' at release '$PDFIUM_RELEASE'." >&2
  echo "install_pdfium: pin one via PDFIUM_SHA256=<hash> or set ALLOW_UNVERIFIED=1 to override." >&2
  exit 1
else
  actual="$(sha256_of "$TMP/pdfium.tgz")"
  if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "install_pdfium: SHA256 mismatch for $URL — refusing to install." >&2
    echo "install_pdfium:   expected $EXPECTED_SHA256" >&2
    echo "install_pdfium:   actual   $actual" >&2
    exit 1
  fi
fi

# bblanchon tarballs unpack to {include/, lib/libpdfium.*} with no top dir.
# Unpack to a staging dir and copy in ONLY this target's library plus the
# headers, so the other platform's vendored binary survives.
tar -xzf "$TMP/pdfium.tgz" -C "$TMP"
if [ ! -f "$TMP/lib/$LIB_NAME" ]; then
  echo "install_pdfium: $LIB_NAME missing after unpack" >&2
  exit 1
fi
mkdir -p "$PDFIUM_DIR/lib"
rm -rf "$PDFIUM_DIR/include"
cp -R "$TMP/include" "$PDFIUM_DIR/include"
cp "$TMP/lib/$LIB_NAME" "$LIB_PATH"

# bblanchon's dylib ships with an install name of "./libpdfium.dylib", which
# only resolves when the binary runs from the library's own directory. Rewrite
# it to @rpath and let the rpath CMake sets do the work. The vendored copy is
# stored already fixed; this keeps a freshly downloaded one consistent.
if [ "$PDFIUM_OS" = "mac" ] && command -v install_name_tool >/dev/null 2>&1; then
  install_name_tool -id "@rpath/$LIB_NAME" "$LIB_PATH"
  # Rewriting the load command invalidates the signature, and arm64 macOS
  # refuses to load an unsigned dylib — re-sign ad-hoc.
  codesign -s - -f "$LIB_PATH" >/dev/null 2>&1 || true
fi

echo "install_pdfium: installed ${TARGET} PDFium into $PDFIUM_DIR"
