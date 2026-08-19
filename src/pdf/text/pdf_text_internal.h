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

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

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
  // Named for their CONTENTS (the old pre_*/visual_* names were inverted —
  // FPDF_GetPageWidthF/HeightF report POST-/Rotate extents, verified against
  // the vendored fpdfview.h:713-748, and against rendered ink):
  //   visual_*_pt — the POST-/Rotate page size: what PdfRenderer rasterizes
  //                 to, and what page width/height reports must use.
  //   media_*_pt  — the PRE-rotation MediaBox/CropBox extents: the space
  //                 FPDFText_GetCharBox coordinates live in, which the
  //                 transforms below need.
  float visual_w_pt = 0.0f;
  float visual_h_pt = 0.0f;
  float media_w_pt  = 0.0f;
  float media_h_pt  = 0.0f;
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
// visual_w × visual_h). Handles rotation ∈ {0, 90, 180, 270} — every case
// verified against RENDERED INK, including the cropped+rotated combination
// (the old 90/270 cases were transposed AND reached for the wrong dimension
// pair; unreachable in production only because the quality gate rejected
// rotated pages). /Rotate is CLOCKWISE display rotation, so for 90 the
// pre-space bottom-left corner becomes the visual top-left: (x,y)->(y,x).
inline void pre_to_visual(const PageHandle &ph, float x_pre, float y_pre,
                          float &x_vis, float &y_vis) {
  // Strip cropbox offset first so the math below is origin-(0,0).
  x_pre -= ph.origin_x_pt;
  y_pre -= ph.origin_y_pt;
  const float Wm = ph.media_w_pt;
  const float Hm = ph.media_h_pt;
  switch (ph.rotation_deg) {
    case 90:
      x_vis = y_pre;
      y_vis = x_pre;
      break;
    case 180:
      x_vis = Wm - x_pre;
      y_vis = y_pre;
      break;
    case 270:
      x_vis = Hm - y_pre;
      y_vis = Wm - x_pre;
      break;
    case 0:
    default:
      x_vis = x_pre;
      y_vis = Hm - y_pre;
      break;
  }
}

inline void visual_to_pre(const PageHandle &ph, float x_vis, float y_vis,
                          float &x_pre, float &y_pre) {
  const float Wm = ph.media_w_pt;
  const float Hm = ph.media_h_pt;
  switch (ph.rotation_deg) {
    case 90:
      x_pre = y_vis;
      y_pre = x_vis;
      break;
    case 180:
      x_pre = Wm - x_vis;
      y_pre = y_vis;
      break;
    case 270:
      x_pre = Wm - y_vis;
      y_pre = Hm - x_vis;
      break;
    case 0:
    default:
      x_pre = x_vis;
      y_pre = Hm - y_vis;
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
    ph->visual_w_pt = FPDF_GetPageWidthF(page);   // POST-/Rotate extents
    ph->visual_h_pt = FPDF_GetPageHeightF(page);
    ph->rotation_deg = FPDFPage_GetRotation(page) * 90;
    if (ph->rotation_deg % 180 == 0) {
      ph->media_w_pt = ph->visual_w_pt;
      ph->media_h_pt = ph->visual_h_pt;
    } else {
      ph->media_w_pt = ph->visual_h_pt;             // swap back to MediaBox
      ph->media_h_pt = ph->visual_w_pt;
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
