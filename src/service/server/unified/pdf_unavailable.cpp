// pdf_unavailable.cpp — the "no PDF page RENDERER" arm of the unified server.
//
// WHY THIS FILE EXISTS
// --------------------
// The tree used to guard its entire PDF subsystem with ONE predicate,
// `if(NOT APPLE)`. That predicate conflated two unrelated facts:
//
//   1. VENDORING — pdfium was treated as Linux-only, so there might be no
//      libpdfium.so to link (third_party/pdfium/…).
//   2. POSIX     — src/pdf/render/pdf_daemon.cpp uses inotify + sigtimedwait +
//      pipe2, none of which darwin has.
//
// Because both facts presented as "APPLE", the whole server was unlinkable on
// macOS: `turbo_ocr::render::PdfRenderer::*` and `turbo_ocr::pdf::*` were
// simply undefined.
//
// Decomposed, only ONE of those facts is really platform-bound. pdfium ships
// for mac-arm64 too (scripts/setup/install_pdfium.sh), so the text layer,
// mode=auto_verified and the searchable-PDF writer all compile and run there —
// they are REAL symbols in turbo_ocr_cpu and are deliberately NOT stubbed
// below. What a platform without inotify genuinely cannot build is the
// DAEMON-based page RENDERER, so this TU supplies exactly that and nothing else.
//
// The consequence for a request: anything that must RASTERIZE a page (a scanned
// PDF, i.e. the OCR path) answers 501; a PDF that already HAS a text layer is
// served normally, including back out as a searchable PDF.
//
// SCOPE — WHEN THIS TU IS COMPILED AT ALL. src/service/server/unified/unified_server.cmake adds
// it only when TURBO_PDF_RENDER_AVAILABLE is OFF, which now means one thing:
// the build passed -DTURBO_ENABLE_PDF=OFF. A missing <sys/inotify.h> no longer
// reaches here — every platform without it compiles
// src/pdf/render/pdf_renderer_inprocess.cpp (in-process PDFium, serial by
// design) and gets a REAL renderer, macOS and Windows alike. So this file is
// never compiled alongside a working PdfRenderer and can never shadow the real
// implementations.

#include <stdexcept>
#include <string>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <drogon/HttpResponse.h>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/service/server/error_codes.h"

namespace {

constexpr const char *kWhy =
    "this build has no PDF page RENDERER (PDF support was disabled at build "
    "time with -DTURBO_ENABLE_PDF=OFF), so a page cannot be rasterized for "
    "OCR. PDFs that already carry a text layer are still extracted, verified "
    "and can be written back as searchable PDFs";

[[noreturn]] void pdf_unavailable(const char *what) {
  throw std::runtime_error(std::string(what) + ": " + kWhy);
}

bool g_advice_registered = false;

// 501 on every PDF endpoint. Registered from the PdfRenderer constructor —
// the unified server_main builds the renderer as startup step 2, well before
// drogon::app().run(), which is exactly when a pre-routing advice must be
// installed. Doing it here keeps the whole "no PDF" arm inside this one file:
// server_main.cpp stays vendor- and platform-neutral.
void register_pdf_501_advice() {
  if (g_advice_registered) return;
  g_advice_registered = true;
  drogon::app().registerPreRoutingAdvice(
      [](const drogon::HttpRequestPtr &req,
         drogon::AdviceCallback &&stop,
         drogon::AdviceChainCallback &&pass) {
        // Only the RENDER-requiring endpoints. A text-layer extraction does
        // not touch PdfRenderer, so blanket-501'ing every /pdf path would
        // refuse work this build can genuinely do.
        // pdf::PdfMode has four values and only THREE of them rasterize:
        //   ocr            — always renders pages                     -> 501
        //   auto           — renders when the text layer is unusable  -> 501
        //   auto_verified  — renders to verify the text layer         -> 501
        //   geometric      — pure PDFium text extraction, no raster   -> ALLOW
        // Blanket-501'ing every /pdf path would refuse `geometric`, which this
        // build can genuinely serve (pdfium is present; only the daemon-based
        // renderer is missing).
        const std::string &path = req->path();
        // /ocr/stream serves PDFs too (it content-sniffs %PDF and runs the
        // same run_pdf_job); without this clause a streamed PDF fell through
        // to the throwing render_streamed stub below and came back as a
        // generic INFERENCE_ERROR instead of the honest 501 — the ONLY route
        // that named the real cause was /ocr/pdf. Image bodies on
        // /ocr/stream must keep working, hence the body sniff.
        const bool is_stream_pdf =
            path.rfind("/ocr/stream", 0) == 0 && req->body().size() >= 4 &&
            req->body().compare(0, 4, "%PDF") == 0;
        const bool is_pdf_route =
            path.rfind("/ocr/pdf", 0) == 0 || path.rfind("/pdf", 0) == 0;
        if ((is_stream_pdf || is_pdf_route) &&
            req->getParameter("mode") != "geometric") {
          auto resp = drogon::HttpResponse::newHttpResponse();
          // From the shared table, not a literal: PDF_NOT_AVAILABLE is the same
          // condition gRPC answers UNIMPLEMENTED for, and one row moves both.
          resp->setStatusCode(static_cast<drogon::HttpStatusCode>(
              turbo_ocr::server::error_http_status(
                  turbo_ocr::server::ErrorCode::kPdfNotAvailable)));
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          resp->setBody(std::string("{\"error\":\"pdf_unavailable\",\"detail\":\"") +
                        kWhy + "\"}");
          stop(resp);
          return;
        }
        pass();
      });
}

} // namespace

namespace turbo_ocr::render {

// The pool constructor normally fork()s fastpdf2png daemons. Here it must NOT
// throw: server_main builds it unconditionally at startup, and a throw would
// take down a server that is otherwise fully functional.
PdfRenderer::PdfRenderer(int pool_size, int workers_per_render)
    : pool_size_(pool_size), workers_per_render_(workers_per_render) {
  TOCR_LOG_WARN("PDF subsystem not built; PDF routes will answer 501",
                "reason", std::string_view(kWhy));
  register_pdf_501_advice();
}

PdfRenderer::~PdfRenderer() noexcept = default;

bool PdfRenderer::can_render() noexcept { return false; }

std::vector<cv::Mat> PdfRenderer::render(const uint8_t *, size_t, int) {
  pdf_unavailable("PdfRenderer::render");
}

PdfRenderer::StreamHandle
PdfRenderer::render_streamed(const uint8_t *, size_t, int, PageCallback) {
  pdf_unavailable("PdfRenderer::render_streamed");
}

cv::Mat PdfRenderer::decode_ppm(const std::string &) { return {}; }

void PdfRenderer::StreamHandle::cleanup() noexcept {}

// Private members are never reached (every public entry point above either
// throws or returns empty), but they are declared in the frozen header, so
// define them rather than leave a latent undefined symbol.
int PdfRenderer::acquire_daemon() { return -1; }
std::string PdfRenderer::send_cmd(Daemon &, const std::string &) {
  pdf_unavailable("PdfRenderer::send_cmd");
}
bool PdfRenderer::send_cmd_once(Daemon &, const std::string &, std::string &) {
  return false;
}
void PdfRenderer::spawn_daemon(Daemon &) { pdf_unavailable("PdfRenderer::spawn_daemon"); }
bool PdfRenderer::respawn_daemon(Daemon &) { return false; }

} // namespace turbo_ocr::render
