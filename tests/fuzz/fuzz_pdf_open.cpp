// libFuzzer target — opening a PDF from raw bytes via PDFium.
//
// On macOS and Windows the in-process renderer means FPDF_LoadMemDocument runs
// inside the server, so a malformed PDF that faults PDFium takes the whole
// process down — the single worst outcome on the untrusted surface. (On Linux
// the daemon isolates it in a subprocess; this target models the mac/win case.)
//
// PDFium is third-party and memory-safe by intent, so the goal here is a
// regression net, not first-party bug-hunting: confirm that neither the load
// nor a minimal render walks off the end of an attacker-sized document. Any
// crash is a PDFium finding to report upstream and, if possible, guard against
// with a bound we own before the hand-off.
//
// Build (from repo root, with a PDFium that has ASan-compatible symbols):
//   clang++ -std=c++20 -g -O1 -fsanitize=fuzzer,address \
//     -I include -I third_party/pdfium/include \
//     tests/fuzz/fuzz_pdf_open.cpp \
//     -Lthird_party/pdfium/lib -lpdfium -o fuzz_pdf
//   ./fuzz_pdf -max_total_time=120 -rss_limit_mb=4096
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <fpdfview.h>

namespace {
// PDFium requires one process-wide init. Guard it so libFuzzer's in-process
// re-entry does not re-initialize per iteration.
void ensure_init() {
  static std::once_flag once;
  std::call_once(once, [] {
    FPDF_LIBRARY_CONFIG cfg{};
    cfg.version = 2;
    FPDF_InitLibraryWithConfig(&cfg);
  });
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  ensure_init();
  FPDF_DOCUMENT doc = FPDF_LoadMemDocument(data, static_cast<int>(size), nullptr);
  if (!doc)
    return 0;

  // Touch the structure the renderer touches: page count, then load and size
  // the first page. This is where a malformed object graph actually gets
  // walked — a load that succeeds but a page that faults is the interesting
  // case, and it is invisible unless a page is opened.
  const int pages = FPDF_GetPageCount(doc);
  if (pages > 0) {
    if (FPDF_PAGE page = FPDF_LoadPage(doc, 0)) {
      (void)FPDF_GetPageWidthF(page);
      (void)FPDF_GetPageHeightF(page);
      FPDF_ClosePage(page);
    }
  }
  FPDF_CloseDocument(doc);
  return 0;
}
