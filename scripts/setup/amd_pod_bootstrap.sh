#!/usr/bin/env bash
# One-command AMD bring-up: fresh ROCm pod/container -> built tree (-> gates).
#
#   bash scripts/setup/amd_pod_bootstrap.sh [--with-ort] [--gates] [--bench]
#
# Idempotent by REAL checks (version probes, file existence), not marker
# files, so re-running after any failure resumes at the first unfinished
# stage — and on an image that already has ROCm >= 6.4 / cmake >= 3.24 /
# GCC >= 13, the corresponding stages cost nothing. Run from the repo root.
# Every non-obvious step carries the trap it dodges at the site; the
# narrative version lives in src/backends/amd/BRINGUP.md.
set -euo pipefail

ROCM_VERSION=7.1.1        # parity with the compile-verified toolchain
ROCM_MIN=6.4              # oldest acceptable (API drift confined to migraphx_engine.cpp)
DROGON_VERSION=v1.9.12
ORT_VERSION=v1.28.0
FUNSD_CACHE="${FUNSD_CACHE:-$HOME/funsd_cache}"
JOBS="$(nproc)"

WITH_ORT=0; RUN_GATES=0; RUN_BENCH=0
for a in "$@"; do case "$a" in
  --with-ort) WITH_ORT=1;; --gates) RUN_GATES=1;; --bench) RUN_BENCH=1;;
  *) echo "unknown flag: $a" >&2; exit 2;;
esac; done

[ -f CMakeLists.txt ] && [ -d src/backends/amd ] || {
  echo "FATAL: run from the TurboOCR repo root" >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive

# RunPod images put /opt/cache/bin (sccache) first in PATH, and it serves
# STALE objects for rsynced mtime-preserved sources — a "successful" build
# with new symbols silently missing.
export SCCACHE_DISABLE=1 CCACHE_DISABLE=1
grep -q SCCACHE_DISABLE ~/.bashrc 2>/dev/null || \
  echo 'export SCCACHE_DISABLE=1 CCACHE_DISABLE=1' >> ~/.bashrc

stage() { echo; echo "==== $* ===="; }

# ---------------------------------------------------------------- base tools
stage "base packages"
apt-get update -q || true   # tolerate one stale third-party list on pod images
apt-get install -y -q rsync wget gnupg ca-certificates curl unzip git \
  software-properties-common libturbojpeg0-dev >/dev/null

# ------------------------------------------------------------------- ROCm
rocm_ok() {
  [ -x /opt/rocm/bin/hipcc ] || return 1
  local v; v="$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1)"
  [ -n "$v" ] && [ "$(printf '%s\n' "$ROCM_MIN" "$v" | sort -V | head -1)" = "$ROCM_MIN" ]
}
if rocm_ok; then
  stage "ROCm $(cat /opt/rocm/.info/version) already present — skipping install"
else
  stage "installing ROCm $ROCM_VERSION (the default RunPod image ships 5.7 — too old)"
  mkdir -p /etc/apt/keyrings
  wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$ROCM_VERSION/ubuntu jammy main" > /etc/apt/sources.list.d/amdgpu.list
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION jammy main" > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
  apt-get update -q || true
  apt-get install -y -q --allow-downgrades -o Dpkg::Options::=--force-confnew \
    hipcc rocm-hip-runtime-dev rocminfo migraphx migraphx-dev half \
    miopen-hip-dev rocblas-dev hipblas-dev >/dev/null
  [ -L /opt/rocm ] && ln -sfn "/opt/rocm-$ROCM_VERSION" /opt/rocm
  rocm_ok || { echo "FATAL: ROCm install did not produce a usable /opt/rocm" >&2; exit 1; }
fi

# ----------------------------------------------------------- build toolchain
# Pod images ship cmake 3.18 (tree needs >= 3.24). pip's cmake/ninja land in
# a bin dir that precedes /usr/local in PATH on conda images; export
# explicitly so this also holds on non-conda images.
stage "cmake / ninja"
cmake_ok() { cmake --version 2>/dev/null | awk 'NR==1{split($3,v,"."); exit !(v[1]>3 || (v[1]==3 && v[2]>=24))}'; }
if ! cmake_ok; then
  python3 -m pip install -q cmake ninja
  export PATH="$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  cmake_ok || { echo "FATAL: cmake >= 3.24 still not first in PATH" >&2; exit 1; }
fi

# The tree uses <format> => GCC 13 libstdc++. Ubuntu 22.04 images ship g++ 11;
# 24.04 images pass this probe and skip the PPA entirely.
stage "host C++ compiler with <format>"
CXX_HOST=""
for c in g++ g++-14 g++-13; do
  command -v "$c" >/dev/null 2>&1 || continue
  if echo '#include <format>
int main(){}' | "$c" -std=c++20 -x c++ - -fsyntax-only 2>/dev/null; then CXX_HOST="$c"; break; fi
done
if [ -z "$CXX_HOST" ]; then
  add-apt-repository -y ppa:ubuntu-toolchain-r/test >/dev/null 2>&1
  apt-get install -y -q g++-13 >/dev/null
  CXX_HOST=g++-13
fi
CC_HOST="${CXX_HOST/g++/gcc}"
echo "host compiler: $CXX_HOST"

apt-get install -y -q libopencv-dev libgrpc++-dev protobuf-compiler-grpc \
  libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libcurl4-openssl-dev \
  gettext-base >/dev/null

# ------------------------------------------------------------------ Drogon
stage "Drogon $DROGON_VERSION"
if [ ! -f /usr/local/lib/cmake/Drogon/DrogonConfig.cmake ]; then
  rm -rf /tmp/drogon
  git clone -q --depth 1 --branch "$DROGON_VERSION" --recurse-submodules \
    https://github.com/drogonframework/drogon /tmp/drogon
  cmake -S /tmp/drogon -B /tmp/drogon/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF >/dev/null
  ninja -C /tmp/drogon/build install >/dev/null
  ldconfig
fi

# ------------------------------------------------- pdfium + PDF render daemon
stage "pdfium + fastpdf2png"
[ -f /usr/lib/libpdfium.so ] || {
  TARGETARCH=amd64 bash scripts/setup/install_pdfium.sh
  cp third_party/pdfium/lib/libpdfium.so /usr/lib/ && ldconfig
}
TARGETARCH=amd64 bash scripts/setup/install_fastpdf2png.sh

# ------------------------------------------------------- models + test data
stage "models + FUNSD cache (both fetchers are no-ops when already present)"
OUT=models bash scripts/models/fetch/fetch_release_models.sh
bash scripts/models/fetch/fetch_funsd_cache.sh "$FUNSD_CACHE"

# -------------------------------------------------------------------- build
stage "configure + build (backends: cpu;amd)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON \
  -DCMAKE_C_COMPILER="$CC_HOST" -DCMAKE_CXX_COMPILER="$CXX_HOST" \
  -DFETCH_MODELS=OFF -DTURBO_BACKENDS="cpu;amd" \
  -DCMAKE_HIP_ARCHITECTURES="$(/opt/rocm/bin/rocminfo | grep -om1 'gfx[0-9a-f]*')" \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DTURBO_FUNSD_CACHE="$FUNSD_CACHE"
ninja -C build turbo_backend_probe turbo_golden turbo_bench turbo_conformance \
  turboocr-server turbo_ocr_tests

# ---------------------------------------------------- ORT for onnx-mode (opt)
if [ "$WITH_ORT" = 1 ]; then
  stage "ONNX Runtime $ORT_VERSION + MIGraphX EP"
  [ -d "$HOME/onnxruntime" ] || git clone -q --depth 1 --branch "$ORT_VERSION" \
    https://github.com/microsoft/onnxruntime "$HOME/onnxruntime"
  # amdclang: jammy binutils 2.38 cannot assemble ORT MLAS AVX-NE-CONVERT.
  # --compile_no_warning_as_error: clang 20 trips -Werror in ORT's MIGraphX EP.
  ( cd "$HOME/onnxruntime" && \
    CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ \
    ./build.sh --config Release --build_shared_lib --parallel "$JOBS" \
      --skip_tests --compile_no_warning_as_error --use_migraphx \
      --rocm_home /opt/rocm --migraphx_home /opt/rocm --allow_running_as_root )
fi

# ------------------------------------------------------------- gates (opt)
if [ "$RUN_GATES" = 1 ]; then
  stage "gates: probe + unit + goldens + conformance + FUNSD"
  ./build/turbo_backend_probe --list
  ctest --test-dir build -j8 -R 'test_|backend_probe' --output-on-failure
  # Goldens run SEQUENTIALLY, never -jN: concurrent first-runs serialize on
  # the single GPU during MIGraphX compilation (~10x per-variant slowdown
  # measured on MI300X, 2026-08-26).
  for g in det cls rec layout; do
    ctest --test-dir build -R "golden_amd_$g" --output-on-failure
  done
  ctest --test-dir build -R 'backend_conformance' --output-on-failure
  ctest --test-dir build -R 'funsd_amd_tiny' --output-on-failure
fi
if [ "$RUN_BENCH" = 1 ]; then
  stage "bench (>= 15 s windows via --repeat; --count clamps to the 50 images)"
  # --threads 5 = the backend's recommended pool; single-threaded submission
  # under-reports an MI300X severely. turbo_bench fails loudly on short or
  # skewed windows, so a too-small --repeat is an error, not a bad number.
  for spec in "tiny 40" "small 20" "medium 10"; do
    set -- $spec
    ./build/turbo_bench --backend amd --tier "$1" --images "$FUNSD_CACHE" \
      --threads 5 --repeat "$2" \
      --words "/tmp/amd_$1.words.json" --out "/tmp/amd_$1.metrics.json"
    python3 tools/bench/score_funsd.py "/tmp/amd_$1.words.json" \
      --metrics "/tmp/amd_$1.metrics.json"
  done
fi

stage "DONE — backend 'amd' built$( [ "$RUN_GATES" = 1 ] && echo ', gates green' )"
