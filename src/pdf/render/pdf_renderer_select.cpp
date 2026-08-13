// PdfRenderer lifecycle + arm selection — the ONE place that decides whether a
// render call goes to the fastpdf2png daemon pool or to in-process PDFium.
//
// WHY THIS FILE EXISTS. There are two renderer implementations with the same
// public contract:
//
//   pdf_renderer.cpp + pdf_daemon.cpp   fans pages out to fastpdf2png worker
//                                       PROCESSES and collects the PPMs via
//                                       inotify. Needs pipe2/sigtimedwait/
//                                       inotify, so it builds on Linux only,
//                                       and needs the fastpdf2png BINARY at
//                                       run time.
//   pdf_renderer_inprocess.cpp          rasterizes in this process. Needs
//                                       nothing beyond PDFium, builds
//                                       everywhere, and is serial: every
//                                       FPDF_* call in the tree already runs
//                                       under one library-wide lock.
//
// They used to be mutually exclusive at COMPILE time, chosen by whether
// <sys/inotify.h> existed. That made Linux the only platform whose PDF support
// depended on an external binary — and the daemon constructor THREW when the
// binary was absent, from a PdfRenderer built unconditionally in server_main
// before anything else. So a plain `cmake` build on Linux (which, unlike
// docker/Dockerfile, never runs scripts/setup/install_fastpdf2png.sh) produced
// a server that refused to start at all, including for pure OCR that never
// touches a PDF.
//
// Now both arms compile on Linux and the choice is made HERE, once, at
// construction: daemon when its binary is present and its pool came up,
// in-process otherwise. Where the binary IS installed the daemon is selected
// exactly as before, so nothing that works today gets slower.

#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

namespace turbo_ocr::render {

PdfRenderer::PdfRenderer(int pool_size, int workers_per_render)
    : pool_size_(pool_size), workers_per_render_(workers_per_render) {
  // PDFium is initialized for BOTH arms: the daemon arm still reads text
  // layers in-process, and the in-process arm rasterizes with it.
  pdf::ensure_pdfium_initialized();

#ifdef TURBO_PDF_DAEMON
  use_daemon_ = try_init_daemons();
#endif

  if (use_daemon_) {
    TOCR_LOG_INFO("PDF renderer: fastpdf2png daemon pool", "pool_size",
                  pool_size_, "workers", workers_per_render_, "binary",
                  binary_path_);
  } else {
    // Not a warning on macOS/Windows, where this is the only arm there is.
    // On Linux it means the helper binary is missing: say so, and say how to
    // get the faster arm back, because the difference is real for concurrent
    // PDF load (the daemon renders several documents in parallel processes;
    // this one serializes on PDFium's library-wide lock).
#ifdef TURBO_PDF_DAEMON
    TOCR_LOG_INFO("PDF renderer: in-process PDFium — fastpdf2png not found, "
                  "so concurrent PDFs serialize. Run "
                  "scripts/setup/install_fastpdf2png.sh for the daemon pool.");
#else
    TOCR_LOG_INFO("PDF renderer: in-process PDFium");
#endif
  }
}

PdfRenderer::~PdfRenderer() noexcept {
#ifdef TURBO_PDF_DAEMON
  if (use_daemon_) shutdown_daemons();
#endif
}

// Both arms are real renderers, so this is true wherever either is compiled.
// The no-PDF build overrides it with `false` from pdf_unavailable.cpp.
bool PdfRenderer::can_render() noexcept { return true; }

std::vector<cv::Mat> PdfRenderer::render(const uint8_t *data, size_t len,
                                         int dpi) {
#ifdef TURBO_PDF_DAEMON
  if (use_daemon_) return render_daemon(data, len, dpi);
#endif
  return render_inprocess(data, len, dpi);
}

PdfRenderer::StreamHandle PdfRenderer::render_streamed(const uint8_t *data,
                                                       size_t len, int dpi,
                                                       PageCallback on_page) {
#ifdef TURBO_PDF_DAEMON
  if (use_daemon_)
    return render_streamed_daemon(data, len, dpi, std::move(on_page));
#endif
  return render_streamed_inprocess(data, len, dpi, std::move(on_page));
}

// Shared by both arms: the daemon arm writes a scratch PDF to disk and the
// in-process arm leaves pdf_tmpfile empty, but the tmpdir of PPMs is removed
// the same way either way. std::filesystem rather than ::unlink so this TU
// stays free of <unistd.h> and builds on Windows.
void PdfRenderer::StreamHandle::cleanup() noexcept {
  // Best-effort from a noexcept path: a failed remove only leaks a temp file
  // the OS reclaims later, and we must not throw here.
  try {
    if (!pdf_tmpfile.empty()) std::filesystem::remove(pdf_tmpfile);
    if (!ppm_tmpdir.empty()) std::filesystem::remove_all(ppm_tmpdir);
  } catch (...) { /* noexcept cleanup: leaked temp is reclaimed by the OS */ }
  pdf_tmpfile.clear();
  ppm_tmpdir.clear();
  num_pages = 0;
}

} // namespace turbo_ocr::render
