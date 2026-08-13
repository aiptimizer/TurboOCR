// In-process PDFium page renderer — the darwin arm of PdfRenderer.
//
// The Linux renderer fans pages out to fastpdf2png daemon processes and picks
// the PPMs up via inotify. Neither inotify nor pipe2/sigtimedwait exists on
// darwin, so this arm rasterizes in-process instead: same public contract,
// same PPM-on-disk handoff, no daemon pool.
//
// It is deliberately serial. Every FPDF_* call in this repo runs under one
// library-wide lock (PDFium is not thread-safe), so a worker pool would only
// contend on that lock. Pages are still handed to `on_page` as each one lands,
// so OCR overlaps rendering exactly as it does on Linux.

#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <mutex>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <fpdfview.h>

#include "../text/pdf_text_internal.h"
#include "pdf_renderer_internal.h" // pdfrdetail::ppm_max_pixels
#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

namespace turbo_ocr::render {
namespace {

// Owns one FPDF_DOCUMENT for the duration of a render call.
//
// Teardown is a PDFium call like any other, so it takes the library-wide lock
// — same as PdfDocument::~PdfDocument in pdf_text_layer.cpp. Closing outside
// the lock would run FPDF_CloseDocument concurrently with another request's
// FPDF_LoadPage/FPDF_RenderPageBitmap and corrupt PDFium's shared font/char
// caches, which is exactly the failure documented in pdf_text_layer.cpp.
//
// INVARIANT: a DocGuard must NOT be destroyed while the caller already holds
// pdfium_lock() — it is a plain std::mutex, not recursive, so that would
// self-deadlock. Every guard here is a function-scope local and both helpers
// (open_document, render_page) release the lock before returning, so the
// destructor always runs unlocked.
struct DocGuard {
  FPDF_DOCUMENT doc = nullptr;
  ~DocGuard() {
    if (!doc) return;
    std::lock_guard<std::mutex> guard(pdf::detail::pdfium_lock());
    FPDF_CloseDocument(doc);
    doc = nullptr;
  }
};

std::string make_temp_dir() {
  // mkdtemp is POSIX with no Windows equivalent, and it was the ONLY thing in
  // this file that was — which is why the renderer was excluded on Windows
  // rather than genuinely being unable to run there.
  //
  // create_directory returns false (no error) when the path already exists, and
  // the underlying mkdir/CreateDirectory is atomic, so the first caller to win a
  // name owns it exclusively — the same guarantee mkdtemp gives.
  //
  // The permissions call is not decoration: mkdtemp creates 0700, whereas
  // create_directory creates 0777 & ~umask. These directories hold rendered
  // pages of whatever document was uploaded, so leaving them group/world
  // readable on a shared host would be a real disclosure. Windows has no umask
  // and its per-user temp directory is already restricted; permissions() is a
  // documented no-op there.
  static std::mutex rng_mu;
  static std::mt19937_64 rng{std::random_device{}()};
  const auto base = std::filesystem::temp_directory_path();
  for (int attempt = 0; attempt < 64; ++attempt) {
    std::uint64_t v;
    {
      std::lock_guard<std::mutex> lk(rng_mu);
      v = rng();
    }
    const auto dir = base / std::format("turbo_pdf_{:016x}", v);
    std::error_code ec;
    if (std::filesystem::create_directory(dir, ec)) {
      std::error_code pec;
      std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                   std::filesystem::perm_options::replace, pec);
      return dir.string();
    }
    if (ec) break; // a real filesystem error, not a name collision
  }
  throw PdfRenderError("could not create a temp dir for rendered pages");
}

// Rasterize one page to BGR. Caller must NOT hold the PDFium lock.
cv::Mat render_page(FPDF_DOCUMENT doc, int index, int dpi) {
  std::lock_guard<std::mutex> guard(pdf::detail::pdfium_lock());

  FPDF_PAGE page = FPDF_LoadPage(doc, index);
  if (!page) return {};

  const double scale = dpi / 72.0;
  const int w = static_cast<int>(std::lround(FPDF_GetPageWidthF(page) * scale));
  const int h = static_cast<int>(std::lround(FPDF_GetPageHeightF(page) * scale));
  if (w <= 0 || h <= 0) {
    FPDF_ClosePage(page);
    return {};
  }
  // AREA CAP — same constant as the Linux PPM path (pdf_renderer_internal.h).
  // /MediaBox is attacker-declared, and unlike Linux (which renders in a
  // disposable fastpdf2png subprocess) this rasterizes IN the server process:
  // without the cap a huge declared page at the capped DPI drives a multi-GB
  // FPDFBitmap_Create right here. Empty return = the route's normal
  // page-decode-failure path, identical to how the PPM decoder reports it.
  if (static_cast<int64_t>(w) * h > pdfrdetail::ppm_max_pixels()) {
    TOCR_LOG_WARN("page exceeds MAX_PDF_PAGE_PIXELS_MP; refusing to render",
                  "w", w, "h", h);
    FPDF_ClosePage(page);
    return {};
  }

  FPDF_BITMAP bitmap = FPDFBitmap_Create(w, h, /*alpha=*/0);
  if (!bitmap) {
    FPDF_ClosePage(page);
    return {};
  }
  // Scanned pages carry no background of their own; white matches what the
  // daemon renderer produces, and OCR preprocessing assumes it.
  FPDFBitmap_FillRect(bitmap, 0, 0, w, h, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bitmap, page, 0, 0, w, h, /*rotate=*/0, FPDF_ANNOT);

  // PDFium hands back 4 bytes per pixel, blue first; copy out before the
  // bitmap is destroyed.
  cv::Mat wrapped(h, w, CV_8UC4, FPDFBitmap_GetBuffer(bitmap),
                  static_cast<size_t>(FPDFBitmap_GetStride(bitmap)));
  cv::Mat bgr;
  cv::cvtColor(wrapped, bgr, cv::COLOR_BGRA2BGR);

  FPDFBitmap_Destroy(bitmap);
  FPDF_ClosePage(page);
  return bgr;
}

// Binary P6, which is what decode_ppm() on the consuming side expects.
void write_ppm(const std::string &path, const cv::Mat &bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) throw PdfRenderError(std::format("could not open {} for writing", path));
  std::fprintf(f, "P6\n%d %d\n255\n", rgb.cols, rgb.rows);
  for (int y = 0; y < rgb.rows; ++y) {
    if (std::fwrite(rgb.ptr(y), 1, static_cast<size_t>(rgb.cols) * 3, f) !=
        static_cast<size_t>(rgb.cols) * 3) {
      std::fclose(f);
      throw PdfRenderError(std::format("short write to {}", path));
    }
  }
  std::fclose(f);
}

// Loads the document. The buffer must outlive `out` — PDFium does not copy it.
int open_document(const uint8_t *data, size_t len, DocGuard &out) {
  std::lock_guard<std::mutex> guard(pdf::detail::pdfium_lock());
  out.doc = FPDF_LoadMemDocument(data, static_cast<int>(len), nullptr);
  if (!out.doc) throw PdfRenderError("PDFium could not open the document");
  return FPDF_GetPageCount(out.doc);
}

} // namespace

std::vector<cv::Mat> PdfRenderer::render_inprocess(const uint8_t *data,
                                                   size_t len, int dpi) {
  DocGuard doc;
  const int num_pages = open_document(data, len, doc);

  std::vector<cv::Mat> pages;
  pages.reserve(static_cast<size_t>(num_pages));
  for (int i = 0; i < num_pages; ++i) pages.push_back(render_page(doc.doc, i, dpi));
  return pages;
}

PdfRenderer::StreamHandle
PdfRenderer::render_streamed_inprocess(const uint8_t *data, size_t len, int dpi,
                                       PageCallback on_page) {
  DocGuard doc;
  const int num_pages = open_document(data, len, doc);

  StreamHandle handle;
  handle.ppm_tmpdir = make_temp_dir();
  handle.num_pages = num_pages;

  for (int i = 0; i < num_pages; ++i) {
    cv::Mat page = render_page(doc.doc, i, dpi);
    if (page.empty()) {
      TOCR_LOG_WARN("PDF renderer: page produced no bitmap", "page_index", i);
      continue;
    }
    // Page files are 1-based on disk, the callback index is 0-based — the
    // same convention the daemon renderer uses.
    std::string path = std::format("{}/p_{:04d}.ppm", handle.ppm_tmpdir, i + 1);
    write_ppm(path, page);
    on_page(i, std::move(path));
  }
  return handle;
}

} // namespace turbo_ocr::render
