#include "turbo_ocr/pdf/pdf_text_layer.h"

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/log/logger.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

// PDFium headers from third_party/pdfium/include
#include <fpdf_edit.h>   // FPDFPage_GetRotation
#include <fpdf_text.h>
#include <fpdfview.h>

#include "pdf_text_internal.h"

namespace turbo_ocr::pdf {

// ── PDFium process-wide initialization + lock ──────────────────────────
//
// PDFium is NOT thread-safe. The API docs make this explicit:
// https://pdfium.googlesource.com/pdfium/+/refs/heads/main/README.md
//
//   "PDFium is not thread-safe. If you use it in a multi-threaded
//    environment, you have to serialize all PDFium function calls."
//
// Under concurrent /ocr/pdf load this bit us for real: multiple
// simultaneous `FPDF_LoadMemDocument` calls began returning err=3
// (FORMAT) on completely valid PDFs because internal font/char caches
// were corrupted. The safe fix is one big library-wide mutex taken
// around every FPDF_* / FPDFText_* / FPDFPage_* call in this file.
//
// Throughput impact: with ~5–10 ms of PDFium work per page, a single
// mutex serializes text-extraction across requests. In practice we
// cache page handles per document via PdfDocument::Impl and most
// geometric-mode extraction finishes in a few ms per page, so the
// lock window is short. Image decode, render, layout, and OCR all
// stay parallel — only the text-layer lookups serialize.

namespace detail {

// Library-wide lock. Held around every PDFium call this file makes.
// Must NOT be held while waiting on CUDA, I/O, or anything else that
// can block — it's a straight mutex, not a condition variable.
std::mutex &pdfium_lock() {
  static std::mutex m;
  return m;
}
// Convert a UTF-16LE code unit array (what FPDFText_GetText / GetBoundedText
// return) to a UTF-8 std::string. PDFium always emits UTF-16LE regardless
// of host endianness. `n` is the number of UTF-16 code units actually
// copied, possibly including a trailing NUL terminator which we strip.
std::string utf16le_to_utf8(const unsigned short *buf, int n) {
  while (n > 0 && buf[n - 1] == 0) --n; // drop trailing NULs
  std::string out;
  out.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ) {
    uint32_t cp = buf[i++];
    if (cp >= 0xD800 && cp <= 0xDBFF) {
      // High surrogate: pair only with a valid low surrogate; a malformed or
      // unpaired half becomes U+FFFD rather than an out-of-range/WTF-8 code
      // point (a crafted ToUnicode map could otherwise emit invalid UTF-8).
      if (i < n && buf[i] >= 0xDC00 && buf[i] <= 0xDFFF) {
        uint32_t lo = buf[i++];
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
      } else {
        cp = 0xFFFD;
      }
    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
      cp = 0xFFFD;  // lone low surrogate
    }
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  return out;
}
} // namespace detail

using detail::pdfium_lock;
using detail::utf16le_to_utf8;

namespace {

std::once_flag g_pdfium_init_flag;

void do_init() {
  // FPDF_InitLibraryWithConfig is itself not reentrant; std::call_once
  // guarantees it runs exactly once before any other FPDF_* call.
  FPDF_LIBRARY_CONFIG cfg{};
  cfg.version = 2;
  cfg.m_pUserFontPaths = nullptr;
  cfg.m_pIsolate = nullptr;
  cfg.m_v8EmbedderSlot = 0;
  FPDF_InitLibraryWithConfig(&cfg);
}
} // namespace

void ensure_pdfium_initialized() {
  std::call_once(g_pdfium_init_flag, do_init);
}

// ── PdfDocument ─────────────────────────────────────────────────────────

PdfDocument::PdfDocument(const uint8_t *data, size_t len)
    : impl_(std::make_unique<Impl>()) {
  ensure_pdfium_initialized();
  // FPDF_LoadMemDocument takes an int length; a >2 GiB body (possible when
  // MAX_BODY_MB is raised above 2048) would wrap negative and fail-open the
  // page-count guard. Refuse up front instead. PDFs that large are not a
  // real workload for in-process text-layer extraction.
  if (len > static_cast<size_t>(INT_MAX)) {
    TOCR_LOG_ERROR("pdf_text PDF body exceeds 2 GiB FPDF limit; rejecting",
                   "bytes", len);
    return;  // doc_ stays null -> ok() == false
  }
  // PDFium is not thread-safe: hold the library-wide lock around any
  // FPDF_* call (load, close, page ops, text ops).
  std::lock_guard<std::mutex> gl(pdfium_lock());
  doc_ = FPDF_LoadMemDocument(data, static_cast<int>(len), /*password=*/nullptr);
  if (!doc_) {
    TOCR_LOG_ERROR_RL("pdf_text FPDF_LoadMemDocument failed",
                      "err", static_cast<long>(FPDF_GetLastError()));
  }
}

PdfDocument::~PdfDocument() noexcept {
  std::lock_guard<std::mutex> gl(pdfium_lock());
  // Tear down page handles and document under the global lock so any
  // FPDFText_ClosePage / FPDF_ClosePage calls happen while serialized
  // against other threads. Resetting impl_ explicitly here (rather than
  // letting the unique_ptr member destroy it after the destructor body
  // exits) keeps those close calls inside the lock scope.
  if (impl_) impl_->pages.clear();
  impl_.reset();
  if (doc_) {
    FPDF_CloseDocument(static_cast<FPDF_DOCUMENT>(doc_));
    doc_ = nullptr;
  }
}

PdfDocument::PdfDocument(PdfDocument &&o) noexcept
    : impl_(std::move(o.impl_)), doc_(o.doc_) {
  o.doc_ = nullptr;
}
PdfDocument &PdfDocument::operator=(PdfDocument &&o) noexcept {
  if (this != &o) {
    std::lock_guard<std::mutex> gl(pdfium_lock());
    if (impl_) impl_->pages.clear();
    if (doc_) FPDF_CloseDocument(static_cast<FPDF_DOCUMENT>(doc_));
    impl_ = std::move(o.impl_);
    doc_ = o.doc_;
    o.doc_ = nullptr;
  }
  return *this;
}

int PdfDocument::page_count() const noexcept {
  if (!doc_) return 0;
  std::lock_guard<std::mutex> gl(pdfium_lock());
  return FPDF_GetPageCount(static_cast<FPDF_DOCUMENT>(doc_));
}

std::string
PdfDocument::text_in_rect_pt(int page_index,
                             float x0_pt, float y0_pt,
                             float x1_pt, float y1_pt) const {
  if (!doc_ || !impl_) return {};

  std::lock_guard<std::mutex> lock(impl_->mtx);
  std::lock_guard<std::mutex> gl(pdfium_lock());
  auto *ph = impl_->get_locked(static_cast<FPDF_DOCUMENT>(doc_), page_index);
  if (!ph || !ph->textpage) return {};

  // Common path: no rotation, no cropbox offset.
  double left, right, top, bottom;
  if (ph->rotation_deg == 0 && ph->origin_x_pt == 0.0f &&
      ph->origin_y_pt == 0.0f) [[likely]] {
    const float page_h = ph->visual_h_pt;
    left   = x0_pt;
    right  = x1_pt;
    top    = page_h - y0_pt;
    bottom = page_h - y1_pt;
  } else {
    // Rotated / trimmed page: transform all 4 visual corners back to
    // pre-rotation space and take the bbox.
    const float vx[4] = {x0_pt, x1_pt, x1_pt, x0_pt};
    const float vy[4] = {y0_pt, y0_pt, y1_pt, y1_pt};
    float px, py;
    visual_to_pre(*ph, vx[0], vy[0], px, py);
    left = px; right = px; top = py; bottom = py;
    for (int k = 1; k < 4; ++k) {
      visual_to_pre(*ph, vx[k], vy[k], px, py);
      if (px < left)  left  = px;
      if (px > right) right = px;
      if (py > top)   top   = py;
      if (py < bottom) bottom = py;
    }
  }
  if (left > right) std::swap(left, right);
  if (top < bottom) std::swap(top, bottom);

  int need = FPDFText_GetBoundedText(ph->textpage, left, top, right, bottom,
                                     nullptr, 0);
  if (need <= 0) return {};
  std::vector<unsigned short> buf(static_cast<size_t>(need) + 1, 0);
  int got = FPDFText_GetBoundedText(ph->textpage, left, top, right, bottom,
                                    buf.data(), static_cast<int>(buf.size()));
  if (got <= 0) return {};
  std::string utf8 = utf16le_to_utf8(buf.data(), got);
  detail::strip_trailing_ws(utf8);
  return utf8;
}
} // namespace turbo_ocr::pdf
