#!/usr/bin/env bash
# Build and run the decode fuzzers under libFuzzer. Linux + clang only —
# macOS/Apple clang ships without the libFuzzer runtime, and this is the
# deployment platform anyway.
#
#   sudo apt-get install clang libclang-rt-<ver>-dev   # the -dev has the runtime;
#                                                        # `clang` alone does NOT
#   bash tests/fuzz/run.sh [seconds]                   # default 240 per target
#
# Each target's outcome is printed; a crash writes a crash-<hash> reproducer in
# the working directory and the run stops non-zero. No crash = the target ran to
# its time budget clean.
set -u
SECS="${1:-240}"
cd "$(dirname "$0")/../.."
OCV_C=$(pkg-config --cflags opencv4)
OCV_L=$(pkg-config --libs opencv4)
mkdir -p /tmp/tocr_fuzz/{ppm,img,pdf}
printf 'P6\n2 2\n255\n\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\xff' > /tmp/tocr_fuzz/ppm/seed
head -c 512 tests/fixtures/images/png/mixed_fonts.png > /tmp/tocr_fuzz/img/seed 2>/dev/null || true
head -c 1024 tests/fixtures/pdf/academic_paper.pdf > /tmp/tocr_fuzz/pdf/seed 2>/dev/null || true
rc=0

echo "=== PPM header (first-party parser, +ASan) ==="
clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address -fno-omit-frame-pointer \
  -I include -I src/pdf/render $OCV_C \
  tests/fuzz/fuzz_ppm_header.cpp src/pdf/render/pdf_ppm.cpp $OCV_L -o /tmp/tocr_fuzz/ppm.bin \
  && /tmp/tocr_fuzz/ppm.bin -max_total_time="$SECS" -rss_limit_mb=4096 /tmp/tocr_fuzz/ppm || rc=1

echo "=== image decode (Wuffs PNG + cv::imdecode, +ASan) ==="
clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address -fno-omit-frame-pointer \
  -I include $OCV_C \
  tests/fuzz/fuzz_image_decode.cpp src/image/fast_png_decoder.cpp $OCV_L -o /tmp/tocr_fuzz/img.bin \
  && /tmp/tocr_fuzz/img.bin -max_total_time="$SECS" -rss_limit_mb=4096 /tmp/tocr_fuzz/img || rc=1

echo "=== PDF open (in-process PDFium; fuzzer only — the vendored lib is not ASan-built) ==="
clang++ -std=c++20 -g -O1 -fsanitize=fuzzer -fno-omit-frame-pointer \
  -I include -I third_party/pdfium/include \
  tests/fuzz/fuzz_pdf_open.cpp -Lthird_party/pdfium/lib -lpdfium -o /tmp/tocr_fuzz/pdf.bin \
  && cp third_party/pdfium/lib/libpdfium.so /tmp/tocr_fuzz/ \
  && ( cd /tmp/tocr_fuzz && LD_LIBRARY_PATH=. ./pdf.bin -max_total_time="$SECS" -rss_limit_mb=4096 pdf ) || rc=1

echo ""
[ $rc -eq 0 ] && echo "ALL FUZZERS CLEAN (ran $SECS s each, no crash)" || echo "A FUZZER FOUND A CRASH — see crash-* reproducer above"
exit $rc
