// pdf_searchable_shapes.cpp — the non-text page furniture.
//
// Figures re-placed as their own objects, block and rule shapes, and the
// layout annotations.
//
// Insertion ORDER is the contract, and it is the order the functions appear
// in: figures, then rules, then — back in pdf_searchable.cpp — the covers and
// the glyphs. A PDF content stream paints in the order objects were added, and
// each of these has to end up on top of the one before it.
//
// See pdf_searchable_detail.h for the whole split.

#include "pdf_searchable_detail.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <fpdf_annot.h>

#include "turbo_ocr/base/log/logger.h"
#include "pdf_text_internal.h"

namespace turbo_ocr::pdf {
namespace searchable_detail {

// Layout classes worth marking: the ones a reader would want to select or a
// consumer would want to crop. Text-bearing regions are already selectable
// through the text layer itself.
bool is_visual_region(int class_id) {
  const std::string_view label = layout::label_name(class_id);
  return label == "image" || label == "chart" || label == "table" ||
         label == "seal" || label == "header_image" || label == "footer_image";
}

// ASCII label to the UTF-16LE PDFium wants for a string value.
std::vector<unsigned short> widen(std::string_view s) {
  std::vector<unsigned short> out(s.size() + 1, 0);
  for (size_t i = 0; i < s.size(); ++i)
    out[i] = static_cast<unsigned char>(s[i]);
  return out;
}

// Feeds PDFium an in-memory JPEG through the file-reader interface its inline
// image loader expects.
struct JpegSource {
  FPDF_FILEACCESS access{};
  const std::vector<uint8_t> *bytes = nullptr;
};

// INTERNAL LINKAGE, deliberately — every `static` below.
//
// The four-way split replaced this file's anonymous namespace with the NAMED
// `searchable_detail`, because internal linkage cannot cross a translation unit
// and ~15 helpers now have to. That handed EXTERNAL linkage to every helper
// that does NOT cross, so a same-named function added to this namespace by
// another TU would be a duplicate-symbol link error at best and, at worst, an
// ODR violation the linker settles silently by picking one body. `write_block`,
// `prepare_page`, `skip_page` and `decode_to_bitmap` are exactly the generic
// names a future PDF helper reuses. `static` restores what the anonymous
// namespace gave them, without moving a line.
static int jpeg_read_block(void *param, unsigned long position, unsigned char *buf,
                    unsigned long size) {
  auto *src = static_cast<JpegSource *>(param);
  if (src == nullptr || src->bytes == nullptr) return 0;
  const size_t end = static_cast<size_t>(position) + size;
  if (end > src->bytes->size()) return 0;
  std::memcpy(buf, src->bytes->data() + position, size);
  return 1;
}

// Decodes an encoded region into a fresh PDFium bitmap. Caller owns it.
static FPDF_BITMAP decode_to_bitmap(const std::vector<uint8_t> &bytes) {
  const cv::Mat decoded = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (decoded.empty()) return nullptr;
  FPDF_BITMAP bmp = FPDFBitmap_Create(decoded.cols, decoded.rows, 0);
  if (bmp == nullptr) return nullptr;
  auto *dst = static_cast<uint8_t *>(FPDFBitmap_GetBuffer(bmp));
  const int stride = FPDFBitmap_GetStride(bmp);
  for (int y = 0; y < decoded.rows; ++y) {
    const auto *row = decoded.ptr<cv::Vec3b>(y);
    uint8_t *out_row =
        dst + static_cast<size_t>(y) * static_cast<size_t>(stride);
    for (int x = 0; x < decoded.cols; ++x) {
      out_row[4 * x + 0] = row[x][0];
      out_row[4 * x + 1] = row[x][1];
      out_row[4 * x + 2] = row[x][2];
      out_row[4 * x + 3] = 255;
    }
  }
  return bmp;
}

// Each figure is re-placed as its own object over a patch of paper. Done
// FIRST, so a text line that happens to fall inside a figure is drawn on
// top of it rather than being buried by it.
bool emit_movable_regions(FPDF_DOCUMENT doc, FPDF_PAGE page,
                          const std::vector<RegionImage> &regions,
                          const PageCanvas &c, int &movable_count) {
  bool any = false;
  for (const RegionImage &rg : regions) {
    if (rg.w <= 0 || rg.h <= 0 || rg.bytes.empty()) continue;
    const float rx = static_cast<float>(rg.x) * c.sx;
    const float rw = static_cast<float>(rg.w) * c.sx;
    const float rh = static_cast<float>(rg.h) * c.sy;
    const float ry = c.geom.visual_h - static_cast<float>(rg.y + rg.h) * c.sy;
    if (rw < kMinExtentPt || rh < kMinExtentPt) continue;

    // BUILD BEFORE COVERING — the same rule the visible-text path states at
    // "Build every replacement line BEFORE anything is covered". The paper patch
    // ERASES the figure; if it went down first and the replacement image then
    // failed to load, the page would keep an orphan rectangle of flat paper
    // where the figure used to be, and FPDFPage_GenerateContent would commit it
    // as soon as anything else on the page succeeded. So the image object is
    // built and proved loadable first, and only then is the hole punched.
    FPDF_PAGEOBJECT obj = FPDFPageObj_NewImageObj(doc);
    if (obj == nullptr) continue;

    // Embed the JPEG as it stands where PDFium will take it — a photograph
    // that is already compressed should not be decoded and compressed
    // again. It refuses some perfectly good files, so a decoded bitmap is
    // the fallback: bigger, but it always works, and a figure that fails to
    // embed is a figure that cannot be moved.
    // The inline path stores the bytes AS a DCTDecode stream and reports
    // success without ever decoding them, so it "succeeds" on a payload no
    // viewer can draw — and then the hole below erases a figure that has been
    // replaced by nothing. Gate it on the JPEG SOI marker so anything that is
    // not actually a JPEG falls through to decode_to_bitmap, which decodes for
    // real (cv::imdecode) and therefore fails honestly.
    const bool is_jpeg = rg.bytes.size() > 3 && rg.bytes[0] == 0xFF &&
                         rg.bytes[1] == 0xD8 && rg.bytes[2] == 0xFF;
    JpegSource src{{}, &rg.bytes};
    src.access.m_FileLen = static_cast<unsigned long>(rg.bytes.size());
    src.access.m_GetBlock = &jpeg_read_block;
    src.access.m_Param = &src;
    bool loaded = is_jpeg &&
                  FPDFImageObj_LoadJpegFileInline(&page, 1, obj, &src.access) != 0;
    FPDF_BITMAP bmp = nullptr;
    if (!loaded) {
      bmp = decode_to_bitmap(rg.bytes);
      if (bmp != nullptr) loaded = FPDFImageObj_SetBitmap(&page, 1, obj, bmp) != 0;
    }
    if (!loaded) {
      if (bmp != nullptr) FPDFBitmap_Destroy(bmp);
      FPDFPageObj_Destroy(obj);
      continue; // nothing has been inserted: the figure is left exactly as it was
    }

    // The hole. Without it the original pixels stay on the page and show
    // through the moment the copy above them is moved — which is the whole
    // reason the region is being lifted out. Inserted immediately before the
    // image, so paint order (patch under figure) is unchanged.
    //
    // A refused patch is BUILD BEFORE COVERING in reverse, and it is not
    // harmless: the figure would go down as a movable object over scan that was
    // never erased, and `movable_count` would report a figure that uncovers its
    // own original the first time a viewer drags it. Nothing has been inserted
    // yet, so drop the image object and take the same exit as a figure that
    // failed to load — the region keeps its scan, which is at least honest.
    if (insert_filled_rect(page, rx, ry, rw, rh, rg.paper, c.to_user) ==
        nullptr) {
      if (bmp != nullptr) FPDFBitmap_Destroy(bmp);
      FPDFPageObj_Destroy(obj);
      continue;
    }

    // An image object's own space is the unit square, so this matrix IS its
    // position and size — and it is the single thing a viewer changes to
    // move the figure.
    const Mat place{rw, 0, 0, rh, rx, ry};
    const Mat m = concat(place, c.to_user);
    const FS_MATRIX fs{m.a, m.b, m.c, m.d, m.e, m.f};
    FPDFPageObj_SetMatrix(obj, &fs);
    FPDFPage_InsertObject(page, obj);
    // SetBitmap copies, so the source bitmap is ours to release.
    if (bmp != nullptr) FPDFBitmap_Destroy(bmp);
    ++movable_count;
    any = true;
  }
  return any;
}

// Flat colour blocks, redrawn as real filled rectangles.
//
// Emitted BEFORE the figures, the rules and the text, because a block is the
// ground those sit on and a PDF content stream paints in insertion order.
//
// Unlike a figure or a rule, a block is NOT painted over a patch of paper
// first. The patch exists to stop the original showing through when the copy is
// dragged away, and for a block that trade is the wrong way round: a block is
// usually the background of a whole panel of content, so erasing it would take
// with it every word inside it that stayed as scan — the lines the font matcher
// could not spell, or could not safely cover. Drawing the block in its own
// colour exactly where it already was changes nothing to look at, and the page
// gains a selectable, movable, re-colourable object. Move it and the flat
// colour it came from stays behind; that is a far smaller wrong than deleting
// somebody's paragraph.
bool emit_blocks(FPDF_PAGE page, const std::vector<BlockShape> &blocks,
                 const PageCanvas &c, int &block_count) {
  bool any = false;
  for (const BlockShape &bl : blocks) {
    if (bl.w <= 0 || bl.h <= 0) continue;
    const float bx = static_cast<float>(bl.x) * c.sx;
    const float bw = static_cast<float>(bl.w) * c.sx;
    const float bh = static_cast<float>(bl.h) * c.sy;
    const float by = c.geom.visual_h - static_cast<float>(bl.y + bl.h) * c.sy;
    if (bw < kMinExtentPt || bh < kMinExtentPt) continue;
    if (insert_filled_rect(page, bx, by, bw, bh, bl.fill, c.to_user) == nullptr)
      continue;
    ++block_count;
    any = true;
  }
  return any;
}

// Printed rules, redrawn as real filled rectangles over a patch of paper. A
// rule that is ink in the page image can be looked at and nothing else; the
// same rule as a path can be selected, moved, recoloured and deleted — which
// is what "select everything on the page" actually requires.
bool emit_rules(FPDF_PAGE page, const std::vector<RuleShape> &rules,
                const PageCanvas &c, int &rule_count) {
  bool any = false;
  for (const RuleShape &rl : rules) {
    if (rl.w <= 0 || rl.h <= 0) continue;
    const float rx = static_cast<float>(rl.x) * c.sx;
    const float rw = std::max(0.4f, static_cast<float>(rl.w) * c.sx);
    const float rh = std::max(0.4f, static_cast<float>(rl.h) * c.sy);
    const float ry = c.geom.visual_h - static_cast<float>(rl.y + rl.h) * c.sy;

    // BUILD BEFORE COVERING, same rule as the figures and the visible text: the
    // patch is deliberately OVERSIZED (see below) and therefore ERASES the
    // printed rule. Prove the replacement shape exists before punching the hole,
    // or a null CreateNewRect leaves an orphan patch that wipes the rule out
    // with nothing drawn back — and `rule_count`/`any` would not even record
    // that the page had been changed.
    FPDF_PAGEOBJECT shape = FPDFPageObj_CreateNewRect(rx, ry, rw, rh);
    if (shape == nullptr) continue;

    // Patch first, then the shape on top, at the same place. A hair of
    // margin, because a scanned rule has a soft edge that would otherwise
    // survive as a grey ghost either side of the crisp new one.
    const float m = std::max(0.5f, (rl.horizontal ? rh : rw) * 0.6f);
    // No patch means no erasure, so the crisp shape would be laid straight over
    // the scanned rule it exists to replace — soft edge and all — while
    // `rule_count` claimed a redrawn rule that was only doubled. The shape has
    // not been inserted yet, so dropping it leaves the rule exactly as it was.
    if (insert_filled_rect(
            page, rx - (rl.horizontal ? 0 : m), ry - (rl.horizontal ? m : 0),
            rw + (rl.horizontal ? 0 : 2 * m), rh + (rl.horizontal ? 2 * m : 0),
            rl.paper, c.to_user) == nullptr) {
      FPDFPageObj_Destroy(shape);
      continue;
    }

    FPDFPageObj_SetFillColor(shape, rl.ink[2], rl.ink[1], rl.ink[0], 255);
    FPDFPath_SetDrawMode(shape, FPDF_FILLMODE_WINDING, 0);
    const FS_MATRIX sm{c.to_user.a, c.to_user.b, c.to_user.c,
                       c.to_user.d, c.to_user.e, c.to_user.f};
    FPDFPageObj_SetMatrix(shape, &sm);
    FPDFPage_InsertObject(page, shape);
    ++rule_count;
    any = true;
  }
  return any;
}

// An annotation per figure/chart/table, so the area is selectable in a reader
// and croppable by a consumer that never saw the JSON.
//
// Deliberately does NOT report "the page changed": annotations live outside the
// content stream, so a page that gained nothing but these needs no
// FPDFPage_GenerateContent and is not counted as a stamped page.
void emit_layout_annotations(FPDF_PAGE page,
                             const std::vector<layout::LayoutBox> &layout,
                             const PageCanvas &c, int &region_count) {
  for (const auto &region : layout) {
    if (!is_visual_region(region.class_id)) continue;
    float lo_x = 0, lo_y = 0, hi_x = 0, hi_y = 0;
    for (int k = 0; k < 4; ++k) {
      const float vx = region.box[k][0] * c.sx;
      const float vy = c.geom.visual_h - region.box[k][1] * c.sy;
      const float ux = vx * c.to_user.a + vy * c.to_user.c + c.to_user.e;
      const float uy = vx * c.to_user.b + vy * c.to_user.d + c.to_user.f;
      if (k == 0) {
        lo_x = hi_x = ux;
        lo_y = hi_y = uy;
      } else {
        lo_x = std::min(lo_x, ux); hi_x = std::max(hi_x, ux);
        lo_y = std::min(lo_y, uy); hi_y = std::max(hi_y, uy);
      }
    }
    if (hi_x - lo_x < kMinExtentPt || hi_y - lo_y < kMinExtentPt) continue;

    FPDF_ANNOTATION annot = FPDFPage_CreateAnnot(page, FPDF_ANNOT_SQUARE);
    if (!annot) continue;
    const FS_RECTF rect{lo_x, hi_y, hi_x, lo_y};
    FPDFAnnot_SetRect(annot, &rect);
    // A thin outline, no fill. Regions are only stamped when the caller
    // asked for layout, so an invisible marker would be a feature nobody
    // can find: a reader needs to SEE the region to click it. The interior
    // stays transparent so the page content is never obscured.
    // NOTE: the alpha argument writes /CA, which is the annotation's SINGLE
    // constant-alpha entry — setting an interior colour afterwards would
    // reset /CA to 0 and make the outline invisible again. Leave the
    // interior unset instead: no /IC means no fill, which is what we want.
    FPDFAnnot_SetColor(annot, FPDFANNOT_COLORTYPE_Color, 37, 99, 235, 255);
    FPDFAnnot_SetBorder(annot, 0.0f, 0.0f, 2.0f);
    const auto label = widen(layout::label_name(region.class_id));
    FPDFAnnot_SetStringValue(annot, "Contents",
                             reinterpret_cast<FPDF_WIDESTRING>(label.data()));
    const auto title = widen("TurboOCR");
    FPDFAnnot_SetStringValue(annot, "T",
                             reinterpret_cast<FPDF_WIDESTRING>(title.data()));
    FPDFPage_CloseAnnot(annot);
    ++region_count;
  }
}

} // namespace searchable_detail
} // namespace turbo_ocr::pdf
