#include "turbo_ocr/pdf/pdf_text_layer.h"

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/logger.h"

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

namespace {

std::once_flag g_pdfium_init_flag;

// Library-wide lock. Held around every PDFium call this file makes.
// Must NOT be held while waiting on CUDA, I/O, or anything else that
// can block — it's a straight mutex, not a condition variable.
std::mutex &pdfium_lock() {
  static std::mutex m;
  return m;
}

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

} // namespace

void ensure_pdfium_initialized() {
  std::call_once(g_pdfium_init_flag, do_init);
}

// ── PdfDocument::Impl ───────────────────────────────────────────────────

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

PdfPageText PdfDocument::extract_page(int page_index) const {
  PdfPageText out;
  if (!doc_ || !impl_) return out;

  // Per-document cache mutex first (cheap, protects the unordered_map),
  // then the global PDFium lock for the actual API calls.
  std::lock_guard<std::mutex> lock(impl_->mtx);
  std::lock_guard<std::mutex> gl(pdfium_lock());
  auto *ph = impl_->get_locked(static_cast<FPDF_DOCUMENT>(doc_), page_index);
  if (!ph) return out;

  // Report visual dimensions so downstream code (PdfRenderer, layout,
  // client coord conversions) sees a single post-rotation coordinate
  // system regardless of /Rotate.
  out.page_width_pt  = ph->visual_w_pt;
  out.page_height_pt = ph->visual_h_pt;
  out.rotation_deg   = ph->rotation_deg;

  const int n_chars = FPDFText_CountChars(ph->textpage);
  out.char_count = std::max(0, n_chars);
  if (n_chars <= 0) return out;

  // Pull the full page text once and scan it for U+FFFD / non-printable
  // counts. This is cheap — PDFium returns UTF-16 that we transcode below.
  {
    std::vector<unsigned short> buf(static_cast<size_t>(n_chars) + 1);
    int written = FPDFText_GetText(ph->textpage, 0, n_chars, buf.data());
    for (int i = 0; i < written; ++i) {
      uint32_t cp = buf[static_cast<size_t>(i)];
      if (cp == 0) continue;
      if (cp == 0xFFFD) ++out.fffd_count;
      else if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r')
        ++out.nonprint_count;
    }
  }

  // Line grouping: walk chars and split on PDFium's OWN layout line breaks
  // (the generated \r\n in the char stream). NOT FPDFText_CountRects/GetRect:
  // those rects are per same-font/style RUN, so a mid-line font-size change
  // (small-caps headings: "M"+"AMBA") fragments one visual line into several
  // out-of-order boxes. PDFium's char-flow segmentation handles those
  // correctly; the per-char box union gives the true line bbox.
  const int n_units = FPDFText_CountChars(ph->textpage);
  std::vector<unsigned short> line_units;
  line_units.reserve(128);
  double lleft = 0, ltop = 0, lright = 0, lbottom = 0;
  bool have_box = false;

  auto flush_line = [&]() {
    if (line_units.empty() || !have_box) {
      line_units.clear();
      have_box = false;
      return;
    }
    std::string utf8 =
        utf16le_to_utf8(line_units.data(), static_cast<int>(line_units.size()));
    while (!utf8.empty() &&
           (utf8.back() == '\n' || utf8.back() == '\r' ||
            utf8.back() == ' '  || utf8.back() == '\t'))
      utf8.pop_back();
    line_units.clear();
    have_box = false;
    if (utf8.empty()) return;

    PdfTextLine line;
    line.text = std::move(utf8);
    const double left = lleft, top = ltop, right = lright, bottom = lbottom;
    // Common path: no rotation, no cropbox offset. Single subtract + flip,
    // no 4-corner transform. This is the shape of ~99% of real PDFs.
    if (ph->rotation_deg == 0 && ph->origin_x_pt == 0.0f &&
        ph->origin_y_pt == 0.0f) [[likely]] {
      const float page_h = ph->visual_h_pt;
      line.x0_pt = static_cast<float>(left);
      line.x1_pt = static_cast<float>(right);
      line.y0_pt = page_h - static_cast<float>(top);
      line.y1_pt = page_h - static_cast<float>(bottom);
    } else {
      // Rotated / trimmed page: transform all 4 corners through
      // pre_to_visual and take the AABB.
      const float pre_x[4] = {
          static_cast<float>(left), static_cast<float>(right),
          static_cast<float>(right), static_cast<float>(left)};
      const float pre_y[4] = {
          static_cast<float>(top), static_cast<float>(top),
          static_cast<float>(bottom), static_cast<float>(bottom)};
      float vx, vy;
      pre_to_visual(*ph, pre_x[0], pre_y[0], vx, vy);
      float vx0 = vx, vx1 = vx, vy0 = vy, vy1 = vy;
      for (int k = 1; k < 4; ++k) {
        pre_to_visual(*ph, pre_x[k], pre_y[k], vx, vy);
        if (vx < vx0) vx0 = vx; else if (vx > vx1) vx1 = vx;
        if (vy < vy0) vy0 = vy; else if (vy > vy1) vy1 = vy;
      }
      line.x0_pt = vx0;
      line.y0_pt = vy0;
      line.x1_pt = vx1;
      line.y1_pt = vy1;
    }
    out.lines.push_back(std::move(line));
  };

  for (int idx = 0; idx < n_units; ++idx) {
    const unsigned int u = FPDFText_GetUnicode(ph->textpage, idx);
    if (u == '\r' || u == '\n') {
      flush_line();
      continue;
    }
    // Skip control characters (U+0000–U+001F except the \r/\n handled above and
    // \t). Glyphs with no proper ToUnicode mapping — soft/discretionary hyphens
    // at line-wrap points, ligature components — report a low control code
    // (U+0000, or e.g. U+0002 for a wrap hyphen: "de\x02ployment"). Dropping
    // them rejoins the word ("deployment") and avoids embedding a NUL that would
    // truncate the line, or a control glyph that renders as tofu (□).
    if (u < 0x20 && u != '\t') continue;
    // FPDFText_GetUnicode returns a full code point (uint32). Re-encode astral
    // (>U+FFFF) code points as a UTF-16 surrogate pair so utf16le_to_utf8
    // reconstructs them; a bare cast to unsigned short would truncate to 16
    // bits and corrupt emoji / SMP CJK / math-alphanumeric glyphs.
    if (u > 0xFFFF) {
      const unsigned int v = u - 0x10000u;
      line_units.push_back(static_cast<unsigned short>(0xD800u + (v >> 10)));
      line_units.push_back(static_cast<unsigned short>(0xDC00u + (v & 0x3FFu)));
    } else {
      line_units.push_back(static_cast<unsigned short>(u));
    }
    double cl = 0, cr = 0, cb = 0, ct = 0;
    // Generated chars (inserted spaces) return degenerate boxes — keep their
    // text, skip them in the bbox union.
    if (FPDFText_GetCharBox(ph->textpage, idx, &cl, &cr, &cb, &ct) &&
        cr > cl && ct > cb) {
      if (!have_box) {
        lleft = cl; lright = cr; lbottom = cb; ltop = ct;
        have_box = true;
      } else {
        lleft = std::min(lleft, cl);
        lright = std::max(lright, cr);
        lbottom = std::min(lbottom, cb);
        ltop = std::max(ltop, ct);
      }
    }
  }
  flush_line();

  return out;
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
  // Strip trailing whitespace — makes equality tests and sanity checks
  // behave consistently across PDFium rect padding.
  while (!utf8.empty() &&
         (utf8.back() == '\n' || utf8.back() == '\r' ||
          utf8.back() == ' '  || utf8.back() == '\t'))
    utf8.pop_back();
  return utf8;
}

// ── Sanity check ────────────────────────────────────────────────────────

SanityVerdict passes_sanity_check(const std::string &text,
                                  float box_width_pt,
                                  float box_height_pt) {
  if (text.empty())
    return {false, "no native text in box"};

  int n = 0, fffd = 0, nonprint = 0;
  for (size_t i = 0; i < text.size(); ) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    uint32_t cp = 0;
    int step = 1;
    if (c < 0x80) { cp = c; step = 1; }
    else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp = (c & 0x1F) << 6 | (static_cast<unsigned char>(text[i+1]) & 0x3F);
      step = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = (c & 0x0F) << 12
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+2]) & 0x3F);
      step = 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = (c & 0x07) << 18
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 12
         | (static_cast<unsigned char>(text[i+2]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+3]) & 0x3F);
      step = 4;
    }
    if (cp == 0xFFFD) ++fffd;
    else if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r') ++nonprint;
    ++n;
    i += step;
  }
  if (n == 0) return {false, "empty after decode"};
  if (fffd * 20 > n) return {false, "too many U+FFFD replacement chars"};
  if (nonprint * 10 > n) return {false, "too many non-printable chars"};

  if (box_width_pt > 0 && box_height_pt > 0) {
    float min_expected = box_width_pt / 30.0f;
    float max_expected = box_width_pt / 2.0f;
    if (static_cast<float>(n) < min_expected * 0.5f ||
        static_cast<float>(n) > max_expected * 2.0f)
      return {false, "char count implausible for box width"};
  }

  return {true, "trusted"};
}

void verify_results_with_text_layer(std::vector<OCRResultItem> &results,
                                    const PdfDocument &doc, int page_index,
                                    int dpi) {
  // dpi is caller-supplied; a zero/negative value would make px_to_pt inf/nan
  // and silently blank the text-layer coords for every item (mis-verifying the
  // whole page). Callers validate DPI, but this guard keeps the invariant local.
  if (dpi <= 0) return;
  const float px_to_pt = 72.0f / static_cast<float>(dpi);
  for (auto &item : results) {
    auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
    float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
    float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
    std::string native = doc.text_in_rect_pt(page_index, x0, y0, x1, y1);
    auto verdict = passes_sanity_check(native, x1 - x0, y1 - y0);
    if (verdict.accept) {
      item.text = std::move(native);
      item.source = "pdf";
      item.confidence = 1.0f;
    }
  }
}

} // namespace turbo_ocr::pdf
