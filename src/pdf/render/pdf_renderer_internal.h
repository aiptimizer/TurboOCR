#pragma once

// Internals shared by the pdf_*.cpp TUs in this directory.

#include <string>

namespace turbo_ocr::render::pdfrdetail {

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
