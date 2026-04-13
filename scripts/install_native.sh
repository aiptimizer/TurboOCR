#!/usr/bin/env bash
set -euo pipefail

# Native install script for Turbo OCR on Arch Linux
# Tested: CUDA 13.1, GCC 15.2, RTX 5090
#
# Usage: bash scripts/install_native.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TENSORRT_INSTALL_DIR="/usr/local"
TENSORRT_LINK="/usr/local/tensorrt"

# ─── Check CUDA ──────────────────────────────────────────────────────────────

if ! command -v nvcc &>/dev/null; then
    error "nvcc not found. Install CUDA toolkit first: sudo pacman -S cuda"
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

info "CUDA version: $CUDA_VERSION"

if [[ "$CUDA_MAJOR" -lt 13 ]]; then
    error "CUDA $CUDA_VERSION detected. This project requires CUDA 13.x+"
fi

# ─── Check GPU ───────────────────────────────────────────────────────────────

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
info "GPU: $GPU_NAME"

# ─── Step 1: System packages ────────────────────────────────────────────────

info "Installing system packages (cmake, opencv, protobuf, grpc, nginx, drogon deps)..."
sudo pacman -S --needed --noconfirm cmake opencv protobuf grpc jsoncpp openssl c-ares nginx

# ─── Step 1b: Drogon HTTP framework ────────────────────────────────────────

DROGON_VERSION="v1.9.12"
if pkg-config --exists drogon 2>/dev/null; then
    info "Drogon already installed"
else
    info "Building Drogon ${DROGON_VERSION} from source..."
    DROGON_TMP=$(mktemp -d)
    git clone --depth 1 --branch "$DROGON_VERSION" https://github.com/drogonframework/drogon.git "$DROGON_TMP"
    cd "$DROGON_TMP"
    git submodule update --init
    cmake -B build -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
          -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
          -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF
    cmake --build build -j"$(nproc)"
    sudo cmake --install build
    rm -rf "$DROGON_TMP"
    cd "$PROJECT_DIR"
    info "Drogon installed"
fi

# ─── Step 2: TensorRT ───────────────────────────────────────────────────────

if [[ -f "$TENSORRT_LINK/include/NvInfer.h" ]]; then
    TRT_EXISTING=$(grep -oP '#define NV_TENSORRT_MAJOR\s+\K\d+' "$TENSORRT_LINK/include/NvInfer.h" 2>/dev/null || echo "?")
    TRT_MINOR=$(grep -oP '#define NV_TENSORRT_MINOR\s+\K\d+' "$TENSORRT_LINK/include/NvInfer.h" 2>/dev/null || echo "?")
    TRT_PATCH=$(grep -oP '#define NV_TENSORRT_PATCH\s+\K\d+' "$TENSORRT_LINK/include/NvInfer.h" 2>/dev/null || echo "?")
    info "TensorRT already installed: ${TRT_EXISTING}.${TRT_MINOR}.${TRT_PATCH} at $TENSORRT_LINK"
    read -rp "  Reinstall TensorRT? [y/N] " reinstall
    if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
        info "Keeping existing TensorRT"
        SKIP_TRT=1
    fi
fi

if [[ "${SKIP_TRT:-}" != "1" ]]; then
    # Determine correct TensorRT version for this CUDA
    # TensorRT tar must match CUDA major version — CUDA 12 builds won't work on CUDA 13
    case "$CUDA_VERSION" in
        13.2) TRT_VERSION="10.16.0.72"; TRT_CUDA="13.2" ;;
        13.1) TRT_VERSION="10.15.1.29"; TRT_CUDA="13.1" ;;
        13.0) TRT_VERSION="10.14.1.16"; TRT_CUDA="13.0" ;;
        *)
            warn "No known TensorRT mapping for CUDA $CUDA_VERSION"
            warn "Check https://developer.nvidia.com/tensorrt for the latest tar"
            warn "Look for TensorRT 10.x with cuda-${CUDA_VERSION} in the filename"
            read -rp "  Enter TensorRT version (e.g. 10.16.0.20): " TRT_VERSION
            read -rp "  Enter CUDA suffix (e.g. 13.1): " TRT_CUDA
            ;;
    esac

    TRT_TAR="TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA}.tar.gz"
    TRT_URL="https://developer.download.nvidia.com/compute/machine-learning/tensorrt/${TRT_VERSION%.*}/tars/${TRT_TAR}"
    TRT_DIR="TensorRT-${TRT_VERSION}"

    info "TensorRT $TRT_VERSION for CUDA $TRT_CUDA"

    # Check if already downloaded
    if [[ -f "/tmp/$TRT_TAR" ]]; then
        info "Found /tmp/$TRT_TAR (already downloaded)"
    else
        info "Downloading TensorRT (~7.5 GB)..."
        info "URL: $TRT_URL"

        # Verify URL exists
        HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" "$TRT_URL" 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" != "200" ]]; then
            warn "URL returned HTTP $HTTP_CODE"
            warn "The version mapping might be wrong. Check manually:"
            warn "  https://developer.nvidia.com/tensorrt"
            warn ""
            warn "Or enter a direct URL below."
            read -rp "  Download URL (or press Enter to abort): " MANUAL_URL
            if [[ -z "$MANUAL_URL" ]]; then
                error "TensorRT download aborted"
            fi
            TRT_URL="$MANUAL_URL"
            # Try to extract dir name from URL
            TRT_TAR=$(basename "$TRT_URL")
            TRT_DIR="${TRT_TAR%.tar.gz}"
            TRT_DIR="${TRT_DIR%.Linux.x86_64-gnu*}"
            # Re-expand: the tar usually extracts to the full name
            TRT_DIR=$(echo "$TRT_TAR" | sed 's/\.Linux\.x86_64-gnu\.cuda-.*\.tar\.gz//')
        fi

        wget -O "/tmp/$TRT_TAR" "$TRT_URL"
    fi

    info "Extracting to $TENSORRT_INSTALL_DIR..."
    sudo tar -xzf "/tmp/$TRT_TAR" -C "$TENSORRT_INSTALL_DIR"

    # Find the extracted directory (name may vary)
    EXTRACTED=$(ls -d ${TENSORRT_INSTALL_DIR}/TensorRT-* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$EXTRACTED" || ! -d "$EXTRACTED" ]]; then
        error "Could not find extracted TensorRT directory"
    fi

    sudo ln -sfn "$EXTRACTED" "$TENSORRT_LINK"
    info "Linked $TENSORRT_LINK -> $EXTRACTED"

    # Add to LD_LIBRARY_PATH in shell rc
    LD_LINE='export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}'
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && ! grep -qF '/usr/local/tensorrt/lib' "$rc"; then
            echo "$LD_LINE" >> "$rc"
            info "Added LD_LIBRARY_PATH to $rc"
        fi
    done
    export LD_LIBRARY_PATH="/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}"
fi

# ─── Step 3: Build ──────────────────────────────────────────────────────────

info "Building turbo-ocr..."
cd "$PROJECT_DIR"
cmake -B build -DTENSORRT_DIR="$TENSORRT_LINK"
cmake --build build -j"$(nproc)"

info "Build complete:"
ls -lh build/paddle_highspeed_cpp 2>/dev/null

# ─── Step 4: Verify ─────────────────────────────────────────────────────────

info "Checking linked libraries..."
if ldd build/paddle_highspeed_cpp | grep -q "not found"; then
    warn "Some libraries not found:"
    ldd build/paddle_highspeed_cpp | grep "not found"
    warn "You may need to run: export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:\$LD_LIBRARY_PATH"
else
    info "All libraries found"
fi

# ─── Done ────────────────────────────────────────────────────────────────────

echo ""
info "Installation complete!"
echo ""
echo "  Run the server:"
echo "    cd $PROJECT_DIR"
echo "    ./build/paddle_highspeed_cpp"
echo ""
echo "  First startup will build TensorRT engines from ONNX models (~2-3 min)."
echo "  Engines are cached in models/ for subsequent runs."
echo ""
echo "  Test:"
echo "    curl -X POST http://localhost:8000/ocr/raw --data-binary @image.png -H 'Content-Type: image/png'"
