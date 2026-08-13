#pragma once

// Internals shared by the pdf_*.cpp TUs in this directory.

#include <cstddef>
#include <cstdint>
#include <string>

namespace turbo_ocr::render::pdfrdetail {

// Parsed PPM (P5/P6) header. `valid` is false for a malformed or bomb-sized
// header; `payload_offset` is the byte index where pixel data begins and
// `payload_bytes` its exact declared length, so a complete file is exactly
// payload_offset + payload_bytes long. One parser shared by decode_ppm and the
// streamed completeness check — and, because it takes (bytes, len) over
// entirely attacker-controlled input, the fuzz target in tests/fuzz.
struct PpmHeader {
  bool   valid = false;
  bool   gray = false;
  int    w = 0, h = 0;
  size_t payload_offset = 0;
  size_t payload_bytes = 0;
};

// Parse a PPM header from a raw byte span. Pure and total: any input yields a
// PpmHeader, never a throw, an over-read, or an out-of-bounds offset. Exposed
// (rather than left in an anonymous namespace) so it can be fuzzed and
// unit-tested directly instead of only through a temp file.
[[nodiscard]] PpmHeader parse_ppm_header(const unsigned char *base, size_t len);

// Max rendered pixels per page (width*height), MAX_PDF_PAGE_PIXELS_MP.
// Defined in pdf_ppm.cpp; shared here so the darwin in-process renderer and
// the PPM decode path read ONE constant — the darwin path used to have no
// area cap at all, and /MediaBox is attacker-declared, so a huge page at the
// capped DPI drove a multi-GB FPDFBitmap_Create inside the SERVER process
// (Linux renders in a disposable subprocess; macOS does not).
[[nodiscard]] int64_t ppm_max_pixels();

// Upper bound on any page number the fastpdf2png daemon can legitimately
// produce. Shared by the "OK <n>" reply parser and the inotify filename
// parser so neither can be driven past it into a huge allocation or (in the
// filename case) signed-int overflow while accumulating digits.
constexpr int kMaxDaemonPages = 100000;

// Parse the page count out of a daemon "OK <n>" reply. A wedged or killed
// child can hand back arbitrary bytes; that must surface as PdfRenderError,
// not as std::invalid_argument escaping the error taxonomy.
[[nodiscard]] int parse_daemon_page_count(const std::string &resp);

// True when `path` is a fully-written PPM: a parseable header plus the entire
// declared pixel payload present on disk. Used by the streamed safety-net so a
// file still being flushed by a forked worker is retried, not delivered
// truncated. Reads only the header prefix, then stats the size.
[[nodiscard]] bool ppm_is_complete(const std::string &path);

} // namespace turbo_ocr::render::pdfrdetail
