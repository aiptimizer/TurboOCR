#!/bin/bash
# Ensure third_party/pdfium holds the PDFium SDK matching the build architecture.
#
# The repo vendors an x86_64 PDFium in third_party/pdfium. On aarch64 that copy
# is the wrong arch, so this script fetches pdfium-linux-arm64 from
# bblanchon/pdfium-binaries and unpacks it over third_party/pdfium. On x86_64
# the vendored copy is used as-is (no network needed).
#
# The release is PINNED (PDFIUM_RELEASE default below) and the downloaded
# tarball's SHA256 is verified against a recorded per-arch hash; an empty or
# mismatched hash is a hard failure unless ALLOW_UNVERIFIED=1.
#
# Caveat: bblanchon's aarch64 build aborts at startup on kernels whose page size
# is neither 4 KiB nor 16 KiB (e.g. some RHEL/CentOS aarch64 configured at
# 64 KiB). Standard Ubuntu/Debian aarch64 and NVIDIA Jetson/L4T use 4 KiB pages
# and work. See https://github.com/bblanchon/pdfium-binaries/issues/148
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PDFIUM_DIR="$ROOT/third_party/pdfium"

# Prefer Docker's TARGETARCH (amd64/arm64) when set; else the host machine.
ARCH="${TARGETARCH:-$(uname -m)}"
case "$ARCH" in
  x86_64|amd64|x64) PDFIUM_ARCH="x64" ;;
  aarch64|arm64)    PDFIUM_ARCH="arm64" ;;
  *) echo "install_pdfium: unsupported arch '$ARCH'" >&2; exit 1 ;;
esac

# Architecture an ELF file was built for, read from bytes 18-19 of its header
# (e_machine, little-endian): "x64", "arm64", "other" or "" when missing. Uses
# od so a minimal container without `file` gets the same answer.
elf_arch() {
  [ -f "$1" ] || { echo ""; return; }
  case "$(od -An -tx1 -j18 -N2 "$1" 2>/dev/null | tr -d ' \n')" in
    3e00) echo "x64" ;;
    b700) echo "arm64" ;;
    *)    echo "other" ;;
  esac
}

# Idempotent: if the installed PDFium is already built for the target arch, do
# nothing. Any other answer (wrong arch, missing, unreadable) means install.
cur="$(elf_arch "$PDFIUM_DIR/lib/libpdfium.so")"
if [ "$cur" = "$PDFIUM_ARCH" ]; then
  echo "install_pdfium: third_party/pdfium already ${PDFIUM_ARCH}"
  exit 0
fi
[ -n "$cur" ] && echo "install_pdfium: third_party/pdfium is ${cur}, target is ${PDFIUM_ARCH}; replacing it"

# Pinned to a concrete bblanchon tag (not the moving `latest`) so a clean-room
# build is reproducible and the SHA256 below stays valid. Override PDFIUM_RELEASE
# to bump; if you do, also update PDFIUM_SHA256_<ARCH> or pass PDFIUM_SHA256.
PDFIUM_RELEASE="${PDFIUM_RELEASE:-chromium/7857}"

# Per-arch SHA256 of pdfium-linux-<arch>.tgz for the pinned release above.
# An explicit PDFIUM_SHA256 env overrides these (e.g. when bumping the tag).
PDFIUM_SHA256_X64="2ad1fd4237cd491201ac74a72388199b9dcf546c5cb02d8fea700725a1b80541"
PDFIUM_SHA256_ARM64="0e24373e73c50759136196c0078db8656860c8d03a10b2cb4a2e7b72d8068e35"

if [ "$PDFIUM_RELEASE" = "latest" ]; then
  URL="https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-${PDFIUM_ARCH}.tgz"
else
  URL="https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_RELEASE}/pdfium-linux-${PDFIUM_ARCH}.tgz"
fi

# Resolve the expected hash: explicit env wins, else the per-arch pinned value.
EXPECTED_SHA256="${PDFIUM_SHA256:-}"
if [ -z "$EXPECTED_SHA256" ]; then
  case "$PDFIUM_ARCH" in
    x64)   EXPECTED_SHA256="$PDFIUM_SHA256_X64" ;;
    arm64) EXPECTED_SHA256="$PDFIUM_SHA256_ARM64" ;;
  esac
fi

echo "install_pdfium: fetching ${PDFIUM_ARCH} PDFium from $URL"
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
if [ "${ALLOW_UNVERIFIED:-0}" = "1" ]; then
  echo "install_pdfium: WARNING — ALLOW_UNVERIFIED=1, skipping SHA256 verification of $URL" >&2
elif [ -z "$EXPECTED_SHA256" ]; then
  echo "install_pdfium: no SHA256 known for arch '$PDFIUM_ARCH' at release '$PDFIUM_RELEASE'." >&2
  echo "install_pdfium: pin one via PDFIUM_SHA256=<hash> or set ALLOW_UNVERIFIED=1 to override." >&2
  exit 1
else
  echo "${EXPECTED_SHA256}  $TMP/pdfium.tgz" | sha256sum -c - || {
    echo "install_pdfium: SHA256 mismatch for $URL — refusing to install." >&2
    exit 1
  }
fi

rm -rf "$PDFIUM_DIR"
mkdir -p "$PDFIUM_DIR"
# bblanchon tarballs unpack to {include/, lib/libpdfium.so, ...} with no top dir.
tar -xzf "$TMP/pdfium.tgz" -C "$PDFIUM_DIR"

if [ ! -f "$PDFIUM_DIR/lib/libpdfium.so" ]; then
  echo "install_pdfium: libpdfium.so missing after unpack" >&2
  exit 1
fi
echo "install_pdfium: installed ${PDFIUM_ARCH} PDFium into $PDFIUM_DIR"
