#!/usr/bin/env bash
# Type-check the vendor sources this machine cannot compile. See README.md.
#
#   tools/syntax_shims/check.sh                 # all of sources.txt
#   tools/syntax_shims/check.sh a.cpp b.cpp     # just these
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT" || exit 1

CXX="${CXX:-c++}"
# Generated protobuf headers live in the build tree; point at whichever exists.
BUILD="${TURBO_BUILD_DIR:-build-unified}"

INC=(
  -I "$BUILD/proto_gen"
  -I third_party/cli11
  -I third_party/onnxruntime/include
  -I include
  -I third_party
  -I third_party/simdutf
  -I third_party/pdfium/include
  -I src/backends
  -I third_party/catch2           # the GPU-only test TUs in sources.txt
  -I "$HERE"                      # the stubs — LAST so a real SDK wins if present
)
# Homebrew locations differ per machine; skip any that are absent rather than
# failing with a confusing "file not found" from deep inside a system header.
for d in /opt/homebrew/opt/jpeg-turbo/include /opt/homebrew/include \
         /opt/homebrew/opt/jsoncpp/include; do
  [ -d "$d" ] && INC+=(-isystem "$d")
done
for d in /opt/homebrew/Cellar/opencv/*/include/opencv4; do
  [ -d "$d" ] && INC+=(-isystem "$d")
done
# Linux distro / CI location for the same headers (libopencv-dev). Without this
# the list only found OpenCV on a Mac, so any source that includes opencv2/ was
# a guaranteed FAIL everywhere else.
# jsoncpp is here for the same reason the Homebrew loop above names it: Debian
# and Ubuntu install it as /usr/include/jsoncpp/json/json.h, so a source that
# reaches <json/json.h> (drogon's HttpResponse.h does) cannot find it on the
# default include path the way it can on a Mac.
for d in /usr/include/opencv4 /usr/local/include/opencv4 /usr/include/jsoncpp; do
  [ -d "$d" ] && INC+=(-isystem "$d")
done

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  mapfile -t FILES < <(grep -v '^[[:space:]]*\(#\|$\)' "$HERE/sources.txt")
fi

# A missing/broken toolchain must be a hard error, not an all-pass. This gate
# once decided pass/fail by grepping the compiler's output for "error:" — with
# $CXX unset to a nonexistent binary, "command not found" contains no "error:"
# and every file printed OK. The gate is the ONLY compile signal for the AMD
# and Intel arms, so failing open here means those arms are unguarded.
if ! "$CXX" --version >/dev/null 2>&1; then
  echo "FATAL: CXX='$CXX' is not a working compiler" >&2
  exit 2
fi

fail=0
for f in "${FILES[@]}"; do
  printf '%-52s ' "$f"
  # Pass/fail is the COMPILER'S EXIT STATUS. The output is captured only for
  # display; never grep it to decide the verdict (localized diagnostics, bad
  # flags rejected before parsing, and shell errors all evade a text match).
  if out=$("$CXX" -fsyntax-only -DPROTOBUF_USE_DLLS "${INC[@]}" -std=gnu++20 -w "$f" 2>&1); then
    echo OK
  else
    echo FAIL
    echo "$out" | grep 'error' | head -5 | sed 's/^/    /'
    [ -z "$(echo "$out" | grep 'error')" ] && echo "$out" | head -5 | sed 's/^/    /'
    fail=1
  fi
done
exit $fail
