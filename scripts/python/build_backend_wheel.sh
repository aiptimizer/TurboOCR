#!/usr/bin/env bash
# Build a per-backend Python wheel (the onnxruntime / onnxruntime-gpu pattern):
#
#   scripts/python/build_backend_wheel.sh <cpu|cuda|openvino|rocm> [outdir]
#
# Every variant installs the SAME import package (`turboocr_engine`) under a
# DIFFERENT distribution name, so users install exactly one:
#
#   turboocr-engine-cpu       portable: CPU everywhere, + Metal/ANE on macOS
#   turboocr-engine-cuda      native TensorRT + the ONNX Runtime CUDA EP
#                      (requires a CUDA-enabled ORT in third_party/onnxruntime —
#                      the gpu_cuda tarball; the plain tarball is CPU-only)
#   turboocr-engine-openvino  + the native Intel OpenVINO backend compiled in
#                      (requires the OpenVINO runtime on CMAKE_PREFIX_PATH)
#   turboocr-engine-rocm      + the native AMD ROCm backend compiled in
#                      (requires ROCm; not yet hardware-tested)
#
# MECHANISM — STAGING. PEP 621 forbids a dynamic `name`, so each variant has its
# own complete config at python/wheels/<variant>/pyproject.toml. Those configs
# are written as if they lived at python/pyproject.toml (cmake.source-dir "..",
# wheel.packages ["turboocr_engine"]), because that is where this script puts them: the
# base config is backed up, the variant's is COPIED OVER python/pyproject.toml,
# python/ is built, and the base is restored on exit (including on failure and
# on ^C). Nothing is ever built from inside python/wheels/<variant>/ — a
# cibuildwheel container only mounts the package directory, so a config with
# ../../.. escapes out of it resolves to nothing there. CI stages the same way.
#
# The wheel is then repaired to be self-contained — delocate on macOS,
# auditwheel on Linux — with the CUDA / ROCm driver+toolkit sonames EXCLUDED
# (they must come from the host, exactly like onnxruntime-gpu).
set -euo pipefail

VARIANT="${1:?usage: build_backend_wheel.sh <cpu|cuda|openvino|rocm> [outdir]}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${2:-$REPO/build-wheels/$VARIANT}"
PY="${PYTHON:-python3}"

PYPROJECT="$REPO/python/pyproject.toml"
# The backup lives in build-wheels/ (gitignored) so a build in flight cannot be
# mistaken for a source edit, and so `git status` stays clean once restored.
BACKUP="$REPO/build-wheels/pyproject.base.toml"

case "$VARIANT" in
  cpu)      DIST="turboocr-engine-cpu";      STAGE_SRC="" ;;
  cuda)     DIST="turboocr-engine-cuda";     STAGE_SRC="$REPO/python/wheels/cuda/pyproject.toml" ;;
  openvino) DIST="turboocr-engine-openvino"; STAGE_SRC="$REPO/python/wheels/openvino/pyproject.toml" ;;
  rocm)     DIST="turboocr-engine-rocm";     STAGE_SRC="$REPO/python/wheels/rocm/pyproject.toml" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

# The vendor backends are Linux-only builds; on macOS only the base wheel (whose
# default TURBO_BACKENDS is cpu;apple) can configure at all. Fail here instead
# of several minutes into a CMake hunt for a toolkit this machine cannot have.
if [ "$(uname -s)" = Darwin ] && [ "$VARIANT" != cpu ]; then
  echo "FATAL: variant '$VARIANT' cannot be built on macOS — build 'cpu' here." >&2
  exit 2
fi

# CUDA needs a CUDA-enabled ORT (the gpu_cuda tarball); the plain tarball is
# CPU-only and would ship a lying wheel. TURBO_ORT_DIR overrides the location.
# ORT_DIR is also used by the repair step below to re-inject the provider libs.
EXTRA_CMAKE=""
if [ "$VARIANT" = cuda ]; then
  ORT_DIR="${TURBO_ORT_DIR:-$REPO/third_party/onnxruntime-gpu-cuda13}"
  if ! ls "$ORT_DIR"/lib/libonnxruntime_providers_cuda.so* >/dev/null 2>&1; then
    echo "FATAL: $ORT_DIR has no libonnxruntime_providers_cuda.so —" >&2
    echo "       point TURBO_ORT_DIR at a gpu_cuda ORT tarball." >&2
    exit 1
  fi
  EXTRA_CMAKE="-DONNXRUNTIME_LIB=$ORT_DIR/lib/libonnxruntime.so -DONNXRUNTIME_INCLUDE_DIR=$ORT_DIR/include"
fi

mkdir -p "$OUT" "$(dirname "$BACKUP")"

# ---- stage the variant config over python/pyproject.toml ---------------------
if [ -n "$STAGE_SRC" ]; then
  [ -f "$STAGE_SRC" ] || { echo "FATAL: no such config: $STAGE_SRC" >&2; exit 1; }

  # A leftover backup means an earlier run died without restoring: the base
  # config is in the BACKUP and python/pyproject.toml still holds some variant.
  # Backing up again would overwrite the only copy of the base with a variant
  # config and lose it for good, so stop and let a human do the one-line
  # recovery instead of guessing.
  if [ -e "$BACKUP" ]; then
    echo "FATAL: $BACKUP already exists — a previous build did not restore the" >&2
    echo "       base config (or another build is running). Recover with:" >&2
    echo "         mv '$BACKUP' '$PYPROJECT'" >&2
    exit 1
  fi

  # Refuse to stage over something that is not the base config — the guard that
  # makes the recovery above the only way to get into a bad state.
  base_name="$("$PY" - "$PYPROJECT" <<'EOF'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print(tomllib.load(f)["project"]["name"])
EOF
)"
  if [ "$base_name" != turboocr-engine-cpu ]; then
    echo "FATAL: $PYPROJECT is '$base_name', not the base 'turboocr-engine-cpu' config." >&2
    echo "       It looks like a staged variant was left behind; restore it first." >&2
    exit 1
  fi

  cp "$PYPROJECT" "$BACKUP"
  # Restore on ANY exit: success, failure, or ^C. Without this the repo is left
  # with a variant config committed-looking at python/pyproject.toml.
  trap 'cp "$BACKUP" "$PYPROJECT" && rm -f "$BACKUP"' EXIT
  cp "$STAGE_SRC" "$PYPROJECT"
  echo "== staged $STAGE_SRC -> $PYPROJECT (base backed up at $BACKUP)"
fi

# The staged config must name the right distribution AND resolve its paths from
# python/. A config still carrying ../../.. escapes builds an empty wheel (or
# fails opaquely inside a cibuildwheel container), so check before spending the
# compile.
"$PY" - "$PYPROJECT" "$DIST" <<'EOF'
import sys, tomllib
path, dist = sys.argv[1:3]
with open(path, "rb") as f:
    cfg = tomllib.load(f)
name = cfg["project"]["name"]
assert name == dist, f"staged config is '{name}', expected '{dist}'"
sb = cfg["tool"]["scikit-build"]
assert sb["cmake"]["source-dir"] == "..", f"cmake.source-dir must be '..', got {sb['cmake']['source-dir']!r}"
assert sb["wheel"]["packages"] == ["turboocr_engine"], f"wheel.packages must be ['turboocr_engine'], got {sb['wheel']['packages']!r}"
assert sb["metadata"]["version"]["input"] == "turboocr_engine/_version.py", "version input must be turboocr_engine/_version.py"
EOF

# A dirty checkout is no longer a hazard: the four configs carry an sdist.exclude
# that drops these from the package copy (see python/pyproject.toml). Say so
# rather than deleting a developer's working build.
if ls "$REPO"/python/turboocr_engine/_turboocr*.so "$REPO"/python/turboocr_engine/_turboocr*.pyd \
      "$REPO"/python/turboocr_engine/*.metallib >/dev/null 2>&1; then
  echo "== note: local build artifacts in python/turboocr_engine are excluded from the wheel"
fi

# ---- build -------------------------------------------------------------------
RAW="$OUT/raw"
rm -rf "$RAW"; mkdir -p "$RAW"
echo "== building $DIST"
# CMAKE_ARGS is appended by scikit-build-core AFTER the config's cmake.args, so
# the caller's own -D flags (TENSORRT_DIR, CMAKE_CUDA_ARCHITECTURES, ...) win.
CMAKE_ARGS="$EXTRA_CMAKE ${CMAKE_ARGS:-}" "$PY" -m pip wheel "$REPO/python" --no-deps -w "$RAW"

WHEEL="$(ls -t "$RAW"/*.whl | head -1)"
echo "== built $WHEEL"

# ---- repair to self-contained ------------------------------------------------
# A raw pip wheel is NOT self-contained: libonnxruntime / libpdfium / OpenCV are
# linked from wherever they were found at build time. delocate/auditwheel vendor
# them. CUDA/ROCm toolkit + driver libs are excluded: they are host-provided
# (driver) or too large/license-bound to vendor, exactly as onnxruntime-gpu ships.
CUDA_EXCLUDES=(libcuda.so.1 'libcudart.so.*' 'libcublas.so.*' 'libcublasLt.so.*'
               'libcudnn*.so.*' 'libcurand.so.*' 'libcufft.so.*' 'libnvinfer*.so.*'
               'libnvonnxparser*.so.*' 'libnccl.so.*' 'libcupti.so.*')
ROCM_EXCLUDES=('libamdhip64.so.*' 'libmigraphx*.so.*' 'librocblas.so.*'
               'libMIOpen.so.*' 'libhsa-runtime64.so.*' 'librocm_smi64.so.*')

mkdir -p "$OUT/fixed"
case "$(uname -s)" in
  Darwin)
    "$PY" -c 'import delocate' 2>/dev/null || "$PY" -m pip install --quiet delocate
    "$PY" -m delocate.cmd.delocate_wheel -w "$OUT/fixed" -v "$WHEEL"
    ;;
  Linux)
    "$PY" -m pip show wheel auditwheel >/dev/null 2>&1 || "$PY" -m pip install --quiet wheel auditwheel
    if [ "$VARIANT" = cuda ]; then
      # Two-step repair. auditwheel vendors the DT_NEEDED graph (pdfium,
      # OpenCV, libonnxruntime, ...) with correct rpaths, excluding the CUDA
      # toolkit/driver sonames. But the CUDA/TensorRT provider libs are
      # dlopen'd by ORT (not DT_NEEDED), so auditwheel silently drops them —
      # step 2 injects them, ORIGINAL names intact, into the same libs dir
      # (ORT dlopens providers from the directory containing libonnxruntime;
      # the mangled main-lib name does not matter, the directory does).
      command -v patchelf >/dev/null || { echo "FATAL: patchelf required" >&2; exit 1; }
      EX=(); for e in "${CUDA_EXCLUDES[@]}"; do EX+=(--exclude "$e"); done
      AW=$(mktemp -d)
      "$PY" -m auditwheel repair "${EX[@]}" -w "$AW" "$WHEEL"
      RW="$(ls "$AW"/*.whl | head -1)"
      W=$(mktemp -d); "$PY" -m wheel unpack -d "$W" "$RW"
      D=$(ls -d "$W"/*/)
      LIBS_DIR=$(ls -d "$D"/*.libs 2>/dev/null || echo "$D/turboocr_engine/.libs")
      mkdir -p "$LIBS_DIR"
      cp -P "$ORT_DIR"/lib/libonnxruntime_providers_*.so "$LIBS_DIR/"
      "$PY" -m wheel pack -d "$OUT/fixed" "$D"
      rm -rf "$W" "$AW"
    else
      EX=()
      if [ "$VARIANT" = rocm ]; then
        for e in "${ROCM_EXCLUDES[@]}"; do EX+=(--exclude "$e"); done
      fi
      # ${EX[@]+...} keeps an empty array from tripping `set -u` on bash < 4.4.
      if "$PY" -m auditwheel repair ${EX[@]+"${EX[@]}"} -w "$OUT/fixed" "$WHEEL"; then
        :
      else
        # auditwheel refuses when the build host's glibc is newer than any
        # manylinux policy it knows. The unrepaired wheel is still usable on
        # hosts with the same or newer glibc + the excluded runtimes present.
        echo "WARN: auditwheel repair failed — keeping the unrepaired wheel" >&2
        cp "$WHEEL" "$OUT/fixed/"
      fi
    fi
    ;;
esac
echo "== done:"; ls -lh "$OUT/fixed/"
