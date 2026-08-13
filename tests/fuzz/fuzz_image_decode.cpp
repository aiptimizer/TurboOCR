// libFuzzer target — the image decode the HTTP/gRPC handlers run on the raw
// request body. decode_cpu_fallback is exactly what /ocr/raw reaches: our
// Wuffs-based PNG fast path (is_png + FastPngDecoder::decode) for PNG, the
// INT_MAX size guard, and cv::imdecode for everything else.
//
// A crash here matters most on the CPU and Apple builds, where decode runs in
// the server process. The Wuffs wrapper is first-party; a fault there is a bug
// to fix. A fault inside cv::imdecode is a third-party finding — triage it, and
// if the fix is a guard we own (a bound before the hand-off), add it here.
//
// Build (from repo root):
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address \
//     -I include $(pkg-config --cflags opencv4) \
//     tests/fuzz/fuzz_image_decode.cpp \
//     $(pkg-config --libs opencv4) -o fuzz_image
//   ./fuzz_image -max_total_time=120
#include <cstddef>
#include <cstdint>

#include "turbo_ocr/image/cpu_image_decode.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Exactly the server's call. The result is used for its side effects only:
  // ASan/UBSan watch for an over-read or overflow inside the decode, and a
  // returned Mat that is non-empty but malformed would surface downstream, not
  // here — this target owns the decode step itself.
  cv::Mat m = turbo_ocr::decode::decode_cpu_fallback(data, size);
  (void)m;
  return 0;
}
