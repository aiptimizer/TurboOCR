// libFuzzer target — the first-party PPM header parser.
//
// parse_ppm_header is pure and takes (bytes, len) over entirely
// attacker-influenced input: on the Linux daemon path a wedged fastpdf2png
// child can hand back arbitrary bytes, and the parser's contract is that ANY
// input yields a PpmHeader without a throw, an over-read, or an offset past the
// buffer. This target asserts that contract; ASan enforces the memory half.
//
// Build (from repo root):
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address \
//     -I include -I src/pdf/render \
//     tests/fuzz/fuzz_ppm_header.cpp src/pdf/render/pdf_ppm.cpp \
//     $(pkg-config --cflags --libs opencv4) -o fuzz_ppm
//   ./fuzz_ppm -max_total_time=120
#include <cstddef>
#include <cstdint>

#include "pdf_renderer_internal.h"

using turbo_ocr::render::pdfrdetail::PpmHeader;
using turbo_ocr::render::pdfrdetail::parse_ppm_header;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const PpmHeader h = parse_ppm_header(data, size);
  if (!h.valid)
    return 0;

  // The invariants every downstream reader relies on. A valid header that
  // violated any of these would let decode_ppm compute an out-of-bounds slice
  // of the mmap'd file, which is the bug this target exists to catch.
  if (h.payload_offset > size)
    __builtin_trap();                       // header claims data past the buffer
  if (h.w <= 0 || h.h <= 0 || h.w > 16384 || h.h > 16384)
    __builtin_trap();                       // bounds the decoder assumes
  const uint64_t expect =
      static_cast<uint64_t>(h.w) * static_cast<uint64_t>(h.h) * (h.gray ? 1u : 3u);
  if (h.payload_bytes != expect)
    __builtin_trap();                       // declared length must match w*h*ch
  return 0;
}
