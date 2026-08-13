#pragma once

// pdf_searchable_detail.h — the internal model shared by the four TUs that write
// a searchable PDF.
//
// pdf_searchable.cpp had reached 1469 lines, well past the 900-line ceiling
// tools/checks/architecture.sh enforces. It was split along the phase boundaries
// the file already documented and banner-numbered for itself:
//
//   pdf_searchable_encoding.cpp  codepoints, encodings and size statistics —
//                                pure values, no PDFium handle anywhere
//   pdf_searchable_style.cpp     phases 1-3: decode to CIDs, vote on the
//                                document's typeface, snap the sizes
//   pdf_searchable_shapes.cpp    the non-text page furniture: figures, blocks,
//                                rules, layout annotations
//   pdf_searchable.cpp           phases 4-6 (placement, covers, text runs) and
//                                write_searchable_pdf itself
//
// The helpers used to share one anonymous namespace. They now share the NAMED
// namespace declared here, which is what makes the split possible at all:
// internal linkage cannot cross a translation unit. None of this is public — the
// header sits beside its sources rather than under include/ for that reason.
//
// The structs below are the interfaces BETWEEN the phases, and they carry
// exactly what the next phase needs and no more. The split is by what each phase
// KNOWS: the first three see the whole document and no page, the emitters see
// one page and no document.

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <fpdf_edit.h>
#include <fpdfview.h>

#include "turbo_ocr/pdf/text/pdf_searchable.h"

namespace turbo_ocr::pdf {
namespace searchable_detail {

// Font metrics of kGlyphlessFont, as fractions of the em.
inline constexpr float kEmHeight = 1.0f;      // ascender 800 - descender -200
inline constexpr float kDescent = 0.2f;       // -descender
inline constexpr float kAdvance = 0.5f;       // every glyph, uniformly
inline constexpr float kMinExtentPt = 0.5f;   // below this a box is noise

struct Mat {
  float a = 1, b = 0, c = 0, d = 1, e = 0, f = 0;
};

// Unrotated page box, in PDFium's own user space.
struct PageGeom {
  float pre_w = 0, pre_h = 0;      // extents of the crop box
  float origin_x = 0, origin_y = 0; // its lower-left corner
  float visual_w = 0, visual_h = 0; // extents as the page displays
  int rotation = 0;
};

// Create + colour + place + insert ONE filled rectangle. The same five PDFium
// calls appear at four sites in this file (the figure patch, the rule patch, the
// rule shape, the text covers) and differ only in the rect, the BGR fill and the
// matrix. `bgr` is anything indexable — uint8_t[3] or cv::Vec3b, both of which
// this file has. Returns nullptr WITHOUT inserting anything if PDFium refused to
// make the object, so a caller can decide before it has changed the page.
template <class Bgr>
FPDF_PAGEOBJECT insert_filled_rect(FPDF_PAGE page, float x, float y, float w,
                                   float h, const Bgr &bgr, const Mat &m) {
  FPDF_PAGEOBJECT rect = FPDFPageObj_CreateNewRect(x, y, w, h);
  if (rect == nullptr) return nullptr;
  FPDFPageObj_SetFillColor(rect, bgr[2], bgr[1], bgr[0], 255);
  FPDFPath_SetDrawMode(rect, FPDF_FILLMODE_WINDING, 0);
  const FS_MATRIX fs{m.a, m.b, m.c, m.d, m.e, m.f};
  FPDFPageObj_SetMatrix(rect, &fs);
  FPDFPage_InsertObject(page, rect);
  return rect;
}

// How far past the ink the cover rectangle reaches, as a fraction of the box
// height. The detection box hugs the glyphs, and a scan's antialiased fringe
// sits just outside it; without a margin that fringe survives as a grey ghost
// around every word.
inline constexpr float kCoverMargin = 0.18F;

// Standard-14 fonts, loaded on demand and shared for the whole document. There
// are at most twelve, and a document uses one or two.
class FontCache {
public:
  explicit FontCache(FPDF_DOCUMENT doc) : doc_(doc) {}
  FontCache(const FontCache &) = delete;
  FontCache &operator=(const FontCache &) = delete;
  FontCache(FontCache &&) = delete;
  FontCache &operator=(FontCache &&) = delete;

  ~FontCache() { close(); }

  // Fonts belong to the document and must be released BEFORE it is. Left to
  // the destructor this runs at the end of the enclosing scope, which is after
  // FPDF_CloseDocument — and FPDFFont_Close then dereferences a document that
  // is gone. Idempotent, so the destructor stays correct on every path.
  void close() {
    for (auto &[name, font] : fonts_)
      if (font) FPDFFont_Close(font);
    fonts_.clear();
  }

  FPDF_FONT get(const char *name) {
    auto [it, inserted] = fonts_.try_emplace(name, nullptr);
    if (inserted) it->second = FPDFText_LoadStandardFont(doc_, name);
    return it->second;
  }

private:
  FPDF_DOCUMENT doc_;
  std::map<std::string, FPDF_FONT> fonts_;
};

// Any positive size will do; big enough that PDFium's bounds keep their
// precision, and the same one everywhere so the ratios below are comparable.
inline constexpr float kTrialSize = 100.0F;

// One kept word, decoded once.
//
// `index` is its position in the page's RESULTS, not in the run list: runs are
// compacted (a word below the confidence floor, or one that would not decode,
// leaves no entry), and the LineStyle arrays are not. Everything downstream
// that needs a style, a font or a size looks it up by `index`.
struct Run {
  const OCRResultItem *item = nullptr;
  size_t index = 0;
  std::vector<uint32_t> cps;
};

// Every kept word of the document, decoded, plus the CID each distinct
// codepoint was given.
struct DocumentRuns {
  std::vector<std::vector<Run>> page_runs;  // index-aligned with `pages`
  std::map<unsigned, unsigned> cid_of;      // codepoint -> CID, 1-based
  int dropped = 0;                          // words that would not decode
};

// What the document-wide passes decided, before any page is drawn.
//
// Everything here is indexed the same way: page p's line i lives at
// style_base[p] + i. That single indexing rule is what ties a page's LineStyle
// to the font and the size the DOCUMENT chose for it, which is the whole point
// of resolving them once rather than per page.
struct DocumentStyle {
  // Per page: where its styles start in the flattened arrays below, or
  // SIZE_MAX when the page has no usable ones.
  std::vector<size_t> style_base;
  std::vector<LineStyle> styles;  // every usable page's styles, end to end
  std::vector<FontChoice> fonts;  // one per entry of `styles`
  // Size as em-per-raster-pixel, snapped across the document; filled by
  // snap_document_sizes once PDFium is up. Same indexing again, and 0 means
  // this line cannot be drawn visibly.
  std::vector<float> em_px;
};

// The frame every emitter below works in: one page's geometry, the raster-to-
// points scale it implies, and the matrix onto PDFium's user space.
struct PageCanvas {
  PageGeom geom;
  float sx = 0, sy = 0;  // raster pixel -> point, per axis
  Mat to_user;
};

// Where each run sits on the page, and whether it can be drawn as real type.
// Worked out for the whole page before anything is inserted, because the
// covering rectangles all have to go down before the first glyph does —
// otherwise one word's cover paints over its neighbour's text wherever two
// detection boxes overlap.
struct Placed {
  const Run *run = nullptr;
  float x0 = 0, y0 = 0, cos_a = 0, sin_a = 0, width = 0, height = 0;
  bool visible = false;
  const LineStyle *style = nullptr;
  FontChoice font{};
  float em_px = 0;  // snapped across the document, not fitted per line
  // The ink's own rectangle inside the box, in points.
  float ink_left_pt = 0, ink_top_pt = 0, ink_w_pt = 0, ink_h_pt = 0;
  // Built before anything is covered; inserted after. EMPTY means this run
  // could not be set and keeps its scan. More than one element means the run
  // was set word by word to spread its slack — see prepare_visible_run.
  std::vector<FPDF_PAGEOBJECT> objs{};
};

// ── defined in pdf_searchable_encoding.cpp ────────────────────────────────
bool decode_utf8(const std::string &s, std::vector<unsigned> &out);
void append_hex4(std::string &out, unsigned v);
std::string build_to_unicode(const std::vector<std::pair<unsigned, unsigned>> &entries);
bool keep(const OCRResultItem &r, float min_confidence);
bool spellable_in_standard14(const std::vector<uint32_t> &cps);
std::vector<unsigned short> to_utf16(const std::vector<uint32_t> &cps);
std::vector<std::vector<uint32_t>> split_words(const std::vector<uint32_t> &cps);
std::vector<std::vector<uint32_t>> split_glyphs(const std::vector<uint32_t> &cps);
float median_of(std::vector<float> &v);
std::vector<float> snap_line_sizes(const std::vector<float> &sizes);

// ── defined in pdf_searchable_shapes.cpp ──────────────────────────────────
bool is_visual_region(int class_id);
std::vector<unsigned short> widen(std::string_view s);
bool emit_movable_regions(FPDF_DOCUMENT doc, FPDF_PAGE page,
                          const std::vector<RegionImage> &regions,
                          const PageCanvas &c, int &movable_count);
bool emit_blocks(FPDF_PAGE page, const std::vector<BlockShape> &blocks,
                 const PageCanvas &c, int &block_count);
bool emit_rules(FPDF_PAGE page, const std::vector<RuleShape> &rules,
                const PageCanvas &c, int &rule_count);
void emit_layout_annotations(FPDF_PAGE page,
                             const std::vector<layout::LayoutBox> &layout,
                             const PageCanvas &c, int &region_count);

// ── defined in pdf_searchable_style.cpp ───────────────────────────────────
DocumentRuns collect_runs(const std::vector<SearchablePage> &pages,
                          float min_confidence);
DocumentStyle resolve_document_style(const std::vector<SearchablePage> &pages,
                                     TextLayerMode mode);
void snap_document_sizes(FPDF_DOCUMENT doc, FontCache &fonts,
                         const std::vector<SearchablePage> &pages,
                         const DocumentRuns &runs, TextLayerMode mode,
                         DocumentStyle &style);

// ── defined in pdf_searchable.cpp ─────────────────────────────────────────
Mat concat(const Mat &t, const Mat &r);
float natural_ink_height(FPDF_DOCUMENT doc, FPDF_FONT font,
                         const std::vector<uint32_t> &cps);

} // namespace searchable_detail
} // namespace turbo_ocr::pdf
