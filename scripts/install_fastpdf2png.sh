#!/bin/bash
# Install fastpdf2png v2.0 (PDF renderer for the OCR server).
# Clones, builds, and copies the binary + library to bin/.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$ROOT/bin"
REPO="https://github.com/aiptimizer/fastpdf2png.git"
# Pinned commit on main. To refresh, run:
#   git ls-remote https://github.com/aiptimizer/fastpdf2png.git refs/heads/main
# and update FASTPDF2PNG_COMMIT below to the resulting 40-char SHA.
FASTPDF2PNG_COMMIT="8358bdc14378c1b33ada057f24aa43f81075dbf7"
TMP_DIR="/tmp/fastpdf2png_build_$$"

echo "=== Installing fastpdf2png ==="

# Target architecture: Docker's TARGETARCH (amd64/arm64) when set, else the host.
case "${TARGETARCH:-$(uname -m)}" in
  x86_64|amd64|x64) TARGET_ELF="x64" ;;
  aarch64|arm64)    TARGET_ELF="arm64" ;;
  *) echo "install_fastpdf2png: unsupported arch '${TARGETARCH:-$(uname -m)}'" >&2; exit 1 ;;
esac

# Architecture an ELF file was built for ("x64", "arm64", "other", "" if missing).
elf_arch() {
  [ -f "$1" ] || { echo ""; return; }
  case "$(od -An -tx1 -j18 -N2 "$1" 2>/dev/null | tr -d ' \n')" in
    3e00) echo "x64" ;;
    b700) echo "arm64" ;;
    *)    echo "other" ;;
  esac
}

# Skip only when the installed binary AND its PDFium are built for the target.
# The repo ships x86-64 copies in bin/, so on any other machine "the file is
# there and executable" says nothing — it would fail exec() with ENOEXEC.
have_bin="$(elf_arch "$BIN_DIR/fastpdf2png")"
have_lib="$(elf_arch "$BIN_DIR/libpdfium.so")"
if [ "$have_bin" = "$TARGET_ELF" ] && [ "$have_lib" = "$TARGET_ELF" ]; then
  echo "fastpdf2png already installed in $BIN_DIR (${TARGET_ELF})"
  echo "  Delete bin/fastpdf2png to force reinstall."
  exit 0
fi
if [ -n "$have_bin" ] && [ "$have_bin" != "$TARGET_ELF" ]; then
  echo "bin/fastpdf2png is ${have_bin}, target is ${TARGET_ELF}; rebuilding"
fi

mkdir -p "$BIN_DIR"

# Clone at pinned commit (full history needed so we can checkout an exact SHA).
echo "Cloning $REPO @ $FASTPDF2PNG_COMMIT..."
git clone "$REPO" "$TMP_DIR"
git -C "$TMP_DIR" checkout --quiet "$FASTPDF2PNG_COMMIT"
ACTUAL_SHA="$(git -C "$TMP_DIR" rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$FASTPDF2PNG_COMMIT" ]; then
  echo "fastpdf2png commit mismatch: expected $FASTPDF2PNG_COMMIT, got $ACTUAL_SHA" >&2
  exit 1
fi

# Make sure third_party/pdfium matches the build arch (no-op on x86_64; fetches
# the arm64 PDFium on aarch64) before we seed it into the fastpdf2png build.
bash "$SCRIPT_DIR/install_pdfium.sh"

# Build against the vendored PDFium (made arch-correct by install_pdfium.sh
# above) so the build needs no network access to
# github.com/bblanchon/pdfium-binaries (rate-limit-flaky from inside Docker
# builds). Never build against a mismatching SDK: the link would fail later
# with a far less readable error.
VENDORED_PDFIUM="$ROOT/third_party/pdfium"
vend_arch="$(elf_arch "$VENDORED_PDFIUM/lib/libpdfium.so")"
if [ "$vend_arch" != "$TARGET_ELF" ]; then
  echo "install_fastpdf2png: third_party/pdfium is '${vend_arch:-missing}' after install_pdfium.sh, target is ${TARGET_ELF}; refusing to build against it." >&2
  exit 1
fi

# Build. Configured explicitly rather than through the upstream preset: no
# -mcpu=native (the result may run on a different CPU than it was built on,
# e.g. an arm64 image built under emulation), static core library (one binary
# to ship), no tests.
echo "Building against ${VENDORED_PDFIUM} (${TARGET_ELF})..."
cd "$TMP_DIR"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DPDFium_DIR="$VENDORED_PDFIUM" \
  -DFP2P_NATIVE_ARCH=OFF \
  -DFP2P_BUILD_SHARED_LIB=OFF \
  -DFP2P_BUILD_TESTS=OFF \
  -DFP2P_BUILD_BENCHMARKS=OFF
cmake --build build --parallel

# Replace whatever bin/ held (possibly another architecture's copies).
rm -f "$BIN_DIR/fastpdf2png" "$BIN_DIR"/libfastpdf2png.so* "$BIN_DIR/libpdfium.so"
cp build/fastpdf2png "$BIN_DIR/"
cp "$VENDORED_PDFIUM/lib/libpdfium.so" "$BIN_DIR/"
chmod +x "$BIN_DIR/fastpdf2png"
built="$(elf_arch "$BIN_DIR/fastpdf2png")"
if [ "$built" != "$TARGET_ELF" ]; then
  echo "install_fastpdf2png: built binary is ${built}, expected ${TARGET_ELF}" >&2
  exit 1
fi

# Cleanup
rm -rf "$TMP_DIR"

echo ""
echo "=== Installed ==="
echo "  Binary:  $BIN_DIR/fastpdf2png"
echo "  Library: $BIN_DIR/libpdfium.so"
echo "  Arch:    ${TARGET_ELF}"
echo ""
echo "  Verify: $BIN_DIR/fastpdf2png --info some.pdf"
