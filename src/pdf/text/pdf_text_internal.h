#pragma once

// Internal shared pieces of the pdf_text_* TUs: the PDFium page-handle cache,
// the pre/post-rotation coordinate transforms, and the process-wide PDFium
// lock. PDFium is NOT thread-safe — every FPDF_* call in these TUs must run
// under detail::pdfium_lock() (full rationale in pdf_text_layer.cpp).

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"

namespace turbo_ocr::pdf {

namespace detail {

// Library-wide lock. Held around every PDFium call. Must NOT be held while
// waiting on CUDA, I/O, or anything else that can block.
std::mutex &pdfium_lock();

// UTF-16LE code units (what PDFium emits regardless of host endianness) to
// UTF-8. `n` may include a trailing NUL terminator, which is stripped.
std::string utf16le_to_utf8(const unsigned short *buf, int n);

// Strip trailing whitespace — makes equality tests and sanity checks behave
// consistently across PDFium rect padding / generated line breaks.
inline void strip_trailing_ws(std::string &s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                        s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
}

} // namespace detail

// Per-page cache entry: keeps FPDF_PAGE + FPDF_TEXTPAGE alive for the
// lifetime of the PdfDocument so repeated lookups on the same page are
// O(1) PDFium calls instead of reloading.
struct PageHandle {
  FPDF_PAGE     page     = nullptr;
  FPDF_TEXTPAGE textpage = nullptr;
  // PRE-rotation page extents (MediaBox width/height as PDFium reports them).
  float pre_w_pt  = 0.0f;
  float pre_h_pt  = 0.0f;
  // VISUAL extents after applying /Rotate — these are what PdfRenderer
  // rasterizes to and what the rest of the pipeline expects to see.
  float visual_w_pt = 0.0f;
  float visual_h_pt = 0.0f;
  int   rotation_deg = 0; // 0 / 90 / 180 / 270 clockwise
  // Optional MediaBox/CropBox origin offset. Most PDFs start at (0,0).
  float origin_x_pt = 0.0f;
  float origin_y_pt = 0.0f;

  ~PageHandle() {
    if (textpage) FPDFText_ClosePage(textpage);
    if (page)     FPDF_ClosePage(page);
  }
};

// Transform a point from PDFium's pre-rotation space (y-up, origin bottom-
// left of MediaBox after `origin` subtraction) to the visual top-left space
// used by the rest of the pipeline (y-down, origin visual top-left, size
// visual_w × visual_h). Handles rotation ∈ {0, 90, 180, 270}. Wrapped as
// two separate functions so extract_page and text_in_rect_pt don't have to
// recompute the branch logic inline.
inline void pre_to_visual(const PageHandle &ph, float x_pre, float y_pre,
                          float &x_vis, float &y_vis) {
  // Strip cropbox offset first so the math below is origin-(0,0).
  x_pre -= ph.origin_x_pt;
  y_pre -= ph.origin_y_pt;
  const float Wp = ph.pre_w_pt;
  const float Hp = ph.pre_h_pt;
  switch (ph.rotation_deg) {
    case 90:
      x_vis = Hp - y_pre;
      y_vis = Wp - x_pre;
      break;
    case 180:
      x_vis = Wp - x_pre;
      y_vis = y_pre;
      break;
    case 270:
      x_vis = y_pre;
      y_vis = x_pre;
      break;
    case 0:
    default:
      x_vis = x_pre;
      y_vis = Hp - y_pre;
      break;
  }
}

inline void visual_to_pre(const PageHandle &ph, float x_vis, float y_vis,
                          float &x_pre, float &y_pre) {
  const float Wp = ph.pre_w_pt;
  const float Hp = ph.pre_h_pt;
  switch (ph.rotation_deg) {
    case 90:
      x_pre = Wp - y_vis;
      y_pre = Hp - x_vis;
      break;
    case 180:
      x_pre = Wp - x_vis;
      y_pre = y_vis;
      break;
    case 270:
      x_pre = y_vis;
      y_pre = x_vis;
      break;
    case 0:
    default:
      x_pre = x_vis;
      y_pre = Hp - y_vis;
      break;
  }
  x_pre += ph.origin_x_pt;
  y_pre += ph.origin_y_pt;
}

struct PdfDocument::Impl {
  mutable std::mutex mtx;
  mutable std::unordered_map<int, std::unique_ptr<PageHandle>> pages;

  // Fetch / lazily open the PageHandle for page_index. Returns nullptr on
  // failure. Called under `mtx`.
  PageHandle *get_locked(FPDF_DOCUMENT doc, int page_index) const {
    auto it = pages.find(page_index);
    if (it != pages.end()) return it->second.get();

    FPDF_PAGE page = FPDF_LoadPage(doc, page_index);
    if (!page) {
      TOCR_LOG_ERROR_RL("pdf_text FPDF_LoadPage failed", "page_index", page_index);
      return nullptr;
    }
    FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
    if (!tp) {
      FPDF_ClosePage(page);
      return nullptr;
    }
    auto ph = std::make_unique<PageHandle>();
    ph->page = page;
    ph->textpage = tp;
    ph->pre_w_pt = FPDF_GetPageWidthF(page);
    ph->pre_h_pt = FPDF_GetPageHeightF(page);
    ph->rotation_deg = FPDFPage_GetRotation(page) * 90;
    if (ph->rotation_deg % 180 == 0) {
      ph->visual_w_pt = ph->pre_w_pt;
      ph->visual_h_pt = ph->pre_h_pt;
    } else {
      ph->visual_w_pt = ph->pre_h_pt;
      ph->visual_h_pt = ph->pre_w_pt;
    }
    // Cropbox/MediaBox origin offset for trimmed PDFs. Most files start
    // at (0, 0) — in that case this is a no-op throughout pre_to_visual /
    // visual_to_pre.
    FS_RECTF bbox{};
    if (FPDF_GetPageBoundingBox(page, &bbox)) {
      ph->origin_x_pt = bbox.left;
      ph->origin_y_pt = bbox.bottom;
    }
    PageHandle *raw = ph.get();
    pages.emplace(page_index, std::move(ph));
    return raw;
  }
};

} // namespace turbo_ocr::pdf
