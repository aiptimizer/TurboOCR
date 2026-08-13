#include "turbo_ocr/pdf/text/pdf_searchable.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <fpdf_annot.h>
#include <fpdf_save.h>

#include "turbo_ocr/base/log/logger.h"
#include "pdf_text_internal.h"
#include "pdf_searchable_detail.h"
#include "glyphless_font.inc"

namespace turbo_ocr::pdf {
namespace searchable_detail {






struct Writer : FPDF_FILEWRITE {
  // FPDF_FILEWRITE is a plain C struct with no initialiser of its own, so a
  // default-constructed Writer would carry indeterminate `version` and
  // `WriteBlock` until both are assigned below. Both always are — this zeroes
  // them anyway so a field PDFium adds later cannot arrive as garbage, and so
  // the class has no uninitialised member to warn about.
  Writer() : FPDF_FILEWRITE{} {}
  std::string buf;
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
static int write_block(FPDF_FILEWRITE *self, const void *data, unsigned long size) {
  static_cast<Writer *>(self)->buf.append(static_cast<const char *>(data), size);
  return 1;
}






// Maps the upright visual page (y up, origin at the visual bottom-left) onto
// user space, as a PDF matrix [a b c d e f]. This undoes the /Rotate a viewer
// applies, which is what puts the always-upright OCR raster back onto the
// page's own coordinates.
static Mat visual_to_user(const PageGeom &g) {
  const float ox = g.origin_x, oy = g.origin_y;
  switch (g.rotation) {
    case 90:  return {0, 1, -1, 0, g.pre_w + ox, oy};
    case 180: return {-1, 0, 0, -1, g.pre_w + ox, g.pre_h + oy};
    case 270: return {0, -1, 1, 0, ox, g.pre_h + oy};
    default:  return {1, 0, 0, 1, ox, oy};
  }
}

// PDF matrix product (row-vector convention): apply `t`, then `r`.
Mat concat(const Mat &t, const Mat &r) {
  return {t.a * r.a + t.b * r.c,
          t.a * r.b + t.b * r.d,
          t.c * r.a + t.d * r.c,
          t.c * r.b + t.d * r.d,
          t.e * r.a + t.f * r.c + r.e,
          t.e * r.b + t.f * r.d + r.f};
}




// ── visible mode ──────────────────────────────────────────────────────────





// Draws one run as real type where the print used to be. Returns false when it
// cannot be placed, and the caller falls back to the invisible layer — which
// leaves the scan showing exactly as it was.

// The height of the ink THIS string makes in THIS face, at kTrialSize.
//
// This is what turns a detected box back into a point size correctly, and the
// measured x-height is not. A line's ink box depends on which letters it holds
// — "PLZ / Ort:" is capital-height, "worn" is x-height, "quality" reaches from
// ascender to descender — and the same dependence applies to the type we are
// about to set it in. Dividing one by the other cancels it out. Going through
// the x-height instead does not: it read the capitals of "PLZ / Ort:" as an
// x-height and asked for type a third too large.
float natural_ink_height(FPDF_DOCUMENT doc, FPDF_FONT font,
                         const std::vector<uint32_t> &cps) {
  FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, kTrialSize);
  if (!obj) return 0.0F;
  const auto wide = to_utf16(cps);
  float h = 0.0F;
  if (FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()))) {
    float l = 0;
    float b = 0;
    float r = 0;
    float t = 0;
    if (FPDFPageObj_GetBounds(obj, &l, &b, &r, &t) && t > b) h = t - b;
  }
  FPDFPageObj_Destroy(obj);
  return h;
}

// ── fitting recognised text to the print it replaces ──────────────────────
//
// How much the letterforms themselves may be distorted to make a line span the
// same piece of page as the print underneath it.
//
// Horizontal scaling is the ONLY width lever PDFium's public API offers: there
// is no character-spacing setter (fpdf_edit.h has SetFontSize and the object
// matrix, and no equivalent of the PDF `Tc` operator), so everything has to be
// expressed as an x-scale on the text matrix or as a change of size. And an
// x-scale distorts glyphs — a stretched face has stems visibly heavier than its
// bowls — so it is only usable in the band where nobody can see it. Typographic
// practice puts that band at 85–115% for body text.
//
// The band here used to be 40–250%, which is where "the editable text does not
// match the length in the document" came from. Measured on the output for a real
// scanned form, 39% of replacement lines fell outside 85–115%, the worst
// stretched to 1.56x and squeezed to 0.67x. Outside the band the answer is never
// more scaling:
//
//   too wide    the TYPE IS SHRUNK until it fits. This is what enforces the
//               property the whole feature needs: a replacement line is never
//               larger than the print it covers.
//   too narrow  the slack is distributed BETWEEN THE WORDS, which is what a
//               typesetter does and, on a form or a tracked heading, what the
//               original actually was. Glyph shapes are left alone entirely.
constexpr float kMinScaleX = 0.85F;
constexpr float kMaxScaleX = 1.15F;
// A line needing to shrink below this to fit has diverged from the print too
// far to be a likeness of it — OCR read something materially different — and
// type this much smaller than its neighbours reads as an error. Keep the scan.
constexpr float kMaxShrink = 0.6F;
// Ceiling on the gap letter spacing may open, as a fraction of the em. A space
// is 0.278 em in Helvetica, and a gap approaching that is read as a word break
// by every text extractor — which would leave the searchable layer this feature
// exists to provide spelling words out letter by letter.
constexpr float kMaxTrackEm = 0.25F;

// One built text object and the inked extent it actually occupies.
//
// The extent is asked of PDFium rather than computed from font metrics, because
// what has to be matched is the INK: the detection box is inked extent too, so
// the two are comparable without either side agreeing about metrics, encodings
// or units.
struct BuiltText {
  FPDF_PAGEOBJECT obj = nullptr;
  float left = 0;   // where the ink starts, in the object's own space
  float width = 0;  // how far it runs
};

static BuiltText build_text(FPDF_DOCUMENT doc, FPDF_FONT font, float size_pt,
                     const std::vector<uint32_t> &cps) {
  BuiltText out;
  if (font == nullptr || size_pt <= 0.0F || cps.empty()) return out;
  FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, size_pt);
  if (obj == nullptr) return out;
  const auto wide = to_utf16(cps);
  if (!FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()))) {
    FPDFPageObj_Destroy(obj);
    return out;
  }
  float l = 0;
  float b = 0;
  float r = 0;
  float t = 0;
  if (!FPDFPageObj_GetBounds(obj, &l, &b, &r, &t) || r <= l) {
    FPDFPageObj_Destroy(obj);
    return out;
  }
  out.obj = obj;
  out.left = l;
  out.width = r - l;
  return out;
}

// Colour, render mode and final placement for one piece of a run.
//
// `dx` is where this piece's ink starts along the baseline, already in the
// run's own space — the caller has subtracted the piece's own `left`, so a
// glyph whose outline starts left of its origin still lands where intended.
static void place_text_obj(FPDF_PAGEOBJECT obj, const LineStyle &style, float scale_x,
                    float dx, float x0, float y0, float cos_a, float sin_a,
                    float baseline_up, const Mat &to_user) {
  // The ink colour read off the scan, not black: a form printed in navy or a
  // stamp in red stays the colour it was.
  FPDFPageObj_SetFillColor(obj, style.ink[2], style.ink[1], style.ink[0], 255);
  FPDFTextObj_SetTextRenderMode(obj, FPDF_TEXTRENDERMODE_FILL);
  const Mat fit{scale_x, 0, 0, 1, dx, 0};
  const Mat place{cos_a,          sin_a, -sin_a,
                  cos_a,          x0 - sin_a * baseline_up,
                  y0 + cos_a * baseline_up};
  const Mat m = concat(concat(fit, place), to_user);
  const FS_MATRIX fs{m.a, m.b, m.c, m.d, m.e, m.f};
  FPDFPageObj_SetMatrix(obj, &fs);
}



// Set `pieces` across exactly `width`, each at its natural shape, with the slack
// shared equally between them. Appends to `out` and returns true on success;
// on failure nothing is appended and nothing is left allocated.
//
// `max_gap` is the ceiling on the space BETWEEN pieces, and it is what keeps
// letter spacing honest. Track a word wider than about a space and every PDF
// text extractor starts reading the gaps as word breaks, so the searchable layer
// this feature exists to provide comes back as "H e l l o". Word spacing has no
// such ceiling — a wide gap between words is a wide gap between words.
static bool distribute_pieces(FPDF_DOCUMENT doc, FPDF_FONT font, float size_pt,
                       const std::vector<std::vector<uint32_t>> &pieces,
                       float width, float max_gap, const LineStyle &style,
                       float x0, float y0, float cos_a, float sin_a,
                       float baseline_up, const Mat &to_user,
                       std::vector<FPDF_PAGEOBJECT> &out) {
  if (pieces.size() < 2) return false;
  std::vector<BuiltText> built;
  built.reserve(pieces.size());
  float sum = 0.0F;
  for (const auto &piece : pieces) {
    BuiltText bt = build_text(doc, font, size_pt, piece);
    if (bt.obj == nullptr) break;
    sum += bt.width;
    built.push_back(bt);
  }
  const float gap =
      built.size() < 2 ? 0.0F
                       : (width - sum) / static_cast<float>(built.size() - 1);
  if (built.size() != pieces.size() || sum <= 0.0F || sum >= width ||
      gap > max_gap) {
    for (const BuiltText &bt : built)
      if (bt.obj != nullptr) FPDFPageObj_Destroy(bt.obj);
    return false;
  }
  float cursor = 0.0F;
  for (const BuiltText &bt : built) {
    place_text_obj(bt.obj, style, 1.0F, cursor - bt.left, x0, y0, cos_a, sin_a,
                   baseline_up, to_user);
    out.push_back(bt.obj);
    cursor += bt.width + gap;
  }
  return true;
}

// Builds the objects for one run, fully placed and coloured, but does NOT put
// them on the page. Returns empty when the run cannot be set.
//
// Separating build from insert is what makes covering safe. The covering
// rectangles have to go down before any glyph, or a later word's cover paints
// over an earlier word's text — but if a run is covered and then fails to
// draw, the page is left with a blank rectangle where the print used to be.
// Building every run first means a failure is known before anything is
// painted, and that run simply keeps its scan.
//
// Usually ONE object. A run whose recognised text is too narrow for the space
// the print occupied comes back as one object PER WORD instead, so the line can
// span the right extent through its word gaps rather than through stretched
// letters — see the band comment above.
static std::vector<FPDF_PAGEOBJECT>
prepare_visible_run(FPDF_DOCUMENT doc, FontCache &fonts,
                    const FontChoice &choice, const LineStyle &style,
                    float size_pt, const std::vector<uint32_t> &cps, float x0,
                    float y0, float cos_a, float sin_a, float width,
                    float height, float py, const Mat &to_user) {
  std::vector<FPDF_PAGEOBJECT> out;
  FPDF_FONT f = fonts.get(choice.standard_name());
  if (f == nullptr || size_pt <= 0.0F) return out;

  // Seat the type on the baseline that was MEASURED on the scan, rather than on
  // the bottom of the detection box. They are not the same line: the box stops
  // at the lowest ink, which for a line with a descender is well below the
  // baseline and for a line without one is exactly on it. Following the box
  // would make every line carrying a 'g' or a 'p' ride high against its
  // neighbours.
  const float baseline_up =
      height * (1.0F - (py > 0.0F ? style.baseline_px / py : 1.0F));

  BuiltText whole = build_text(doc, f, size_pt, cps);
  if (whole.obj == nullptr) return out;
  float scale_x = width / whole.width;

  if (scale_x > kMaxScaleX) {
    // Too narrow for the space the print filled. Spread the slack between the
    // pieces instead of stretching the letters — nothing is distorted, every
    // piece is set at its natural width and scale 1, and the line still ends
    // exactly where the print ended.
    //
    // WORD gaps first: widening those is what the original almost always was
    // (a form's aligned columns, a tracked heading), and it leaves the text
    // extracting word for word. LETTER gaps second, for the single word that
    // has no word gap to widen — capped, because tracking wide enough to read
    // as a space would corrupt the searchable layer.
    if (distribute_pieces(doc, f, size_pt, split_words(cps), width,
                          std::numeric_limits<float>::max(), style, x0, y0,
                          cos_a, sin_a, baseline_up, to_user, out) ||
        distribute_pieces(doc, f, size_pt, split_glyphs(cps), width,
                          kMaxTrackEm * size_pt, style, x0, y0, cos_a, sin_a,
                          baseline_up, to_user, out)) {
      FPDFPageObj_Destroy(whole.obj);
      return out;
    }
    // Neither worked — a single character, or slack too large to absorb
    // honestly. Leave it at the widest scale that is still invisible: the line
    // comes out a little short of the print, which is the safe direction.
    scale_x = kMaxScaleX;
  } else if (scale_x < kMinScaleX) {
    // Too wide for its box. Shrink the TYPE rather than squeezing the letters
    // further: this is what guarantees a replacement line never overflows the
    // print it replaces.
    const float shrink = scale_x / kMinScaleX;
    FPDFPageObj_Destroy(whole.obj);
    if (shrink < kMaxShrink) return out;
    whole = build_text(doc, f, size_pt * shrink, cps);
    if (whole.obj == nullptr) return out;
    // Re-measured at the new size, so this lands on kMinScaleX up to the
    // nonlinearity of hinting rather than by assumption.
    scale_x = std::clamp(width / whole.width, kMinScaleX, kMaxScaleX);
  }

  place_text_obj(whole.obj, style, scale_x, -whole.left * scale_x, x0, y0,
                 cos_a, sin_a, baseline_up, to_user);
  out.push_back(whole.obj);
  return out;
}



// ── the phases ────────────────────────────────────────────────────────────
//
// write_searchable_pdf runs SIX of them, laid out below in the order it calls
// them and banner-numbered to match: (1) decode UTF-8 to CIDs, (2) vote on a
// typeface, (3) snap the sizes, (4) place, (5) emit the page objects,
// (6) emit the text runs. Serialisation is still INLINE at the end of
// write_searchable_pdf, not a seventh phase — an earlier version of this note
// promised one and it does not exist, which sent readers looking for a function
// that was never written.
//
// Two small predicates sit after phase 6 under their own banner ("the two
// questions the page loop asks"): has_non_text_work and skip_page.
//
// The split is by what each phase KNOWS — the first three see the whole document
// and no page, the emitters see one page and no document — and the structs
// between them carry exactly that much and no more, which is what keeps a change
// to one phase from reaching into the next.

// ── phase 1: UTF-8 / CID ──────────────────────────────────────────────────




// ── phase 2: the document's own type ──────────────────────────────────────



// ── phase 3: size snapping ────────────────────────────────────────────────


// ── phase 4: placement and transform math ─────────────────────────────────


// Reads the page's box and rotation. NOT a pure measurement: it also turns the
// page, because autorotate turned the raster upright before detection and the
// page has to match for the boxes to describe what is shown.
static PageCanvas prepare_page(FPDF_PAGE page, const SearchablePage &in) {
  PageCanvas c;
  PageGeom &g = c.geom;

  // FPDF_GetPageWidthF is post-/Rotate; the bounding box is not, so it is
  // the one that yields the unrotated extents the content stream uses.
  FS_RECTF bbox{};
  if (FPDF_GetPageBoundingBox(page, &bbox) && bbox.right > bbox.left &&
      bbox.top > bbox.bottom) {
    g.pre_w = bbox.right - bbox.left;
    g.pre_h = bbox.top - bbox.bottom;
    g.origin_x = bbox.left;
    g.origin_y = bbox.bottom;
  } else {
    const bool turned = FPDFPage_GetRotation(page) % 2 != 0;
    g.pre_w = turned ? FPDF_GetPageHeightF(page) : FPDF_GetPageWidthF(page);
    g.pre_h = turned ? FPDF_GetPageWidthF(page) : FPDF_GetPageHeightF(page);
  }
  // autorotate turned the raster upright before detection; turn the page to
  // match, so it displays upright too and the boxes describe what is shown.
  g.rotation = (FPDFPage_GetRotation(page) * 90 + in.orientation_deg) % 360;
  if (g.rotation < 0) g.rotation += 360;
  if (in.orientation_deg) FPDFPage_SetRotation(page, g.rotation / 90);
  const bool swapped = g.rotation % 180 != 0;
  g.visual_w = swapped ? g.pre_h : g.pre_w;
  g.visual_h = swapped ? g.pre_w : g.pre_h;

  c.sx = g.visual_w / static_cast<float>(in.raster_w);
  c.sy = g.visual_h / static_cast<float>(in.raster_h);
  c.to_user = visual_to_user(g);
  return c;
}


static std::vector<Placed> place_page_runs(const std::vector<Run> &runs,
                                    const SearchablePage &in,
                                    const DocumentStyle &style, size_t p,
                                    const PageCanvas &c, TextLayerMode mode,
                                    int &dropped) {
  std::vector<Placed> placed;
  placed.reserve(runs.size());

  for (const Run &run : runs) {
    const Box &b = run.item->box;
    // Corners arrive clockwise from top-left, so bl->br is the baseline —
    // following it keeps skewed scans aligned instead of forcing horizontal.
    const float x0 = b[3][0] * c.sx, y0 = c.geom.visual_h - b[3][1] * c.sy;
    const float bx = b[2][0] * c.sx - x0,
                by = (c.geom.visual_h - b[2][1] * c.sy) - y0;
    const float ux = b[0][0] * c.sx - x0,
                uy = (c.geom.visual_h - b[0][1] * c.sy) - y0;
    const float width = std::hypot(bx, by);
    const float height = std::hypot(ux, uy);
    if (width < kMinExtentPt || height < kMinExtentPt) {
      ++dropped;
      continue;
    }

    Placed pl{&run, x0, y0, bx / width, by / width, width, height};
    if (mode == TextLayerMode::Visible && style.style_base[p] != SIZE_MAX) {
      const size_t k = style.style_base[p] + run.index;
      const LineStyle &st = (*in.styles)[run.index];
      // All three have to hold. A line the estimator could not read has no
      // colour to paint with; a line on patterned ground cannot be covered
      // without destroying the pattern; and a line the standard-14 fonts
      // cannot spell would come out as the wrong glyphs entirely.
      if (st.measured && st.flat_paper && spellable_in_standard14(run.cps) &&
          k < style.fonts.size()) {
        pl.visible = true;
        pl.style = &st;
        pl.font = style.fonts[k];
        pl.em_px = k < style.em_px.size() ? style.em_px[k] : 0.0f;
        if (pl.em_px <= 0.0f) pl.visible = false;
      }
    }
    placed.push_back(pl);
  }
  return placed;
}

// ── phase 5: content-stream emission ──────────────────────────────────────
//
// Insertion ORDER is the whole contract of this section, and it is the order
// the functions appear in: figures, then rules, then the covers, then the
// glyphs, because a PDF content stream paints in the order objects were added
// and each of these has to end up on top of the one before it.





// Build every replacement line BEFORE anything is covered. A run that
// cannot be set must keep its scan, and the only way to know that in time
// is to have built it already.
static void build_visible_runs(FPDF_DOCUMENT doc, FontCache &fonts,
                        std::vector<Placed> &placed, const PageCanvas &c) {
  for (Placed &pl : placed) {
    if (!pl.visible) continue;
    // ink_* live in the RECTIFIED LINE's own pixel space (font_style.h measures
    // them off the deskewed crop), so ONE scalar converts all four — not sx for
    // the horizontal pair and sy for the vertical one, which is what place_page_
    // runs correctly does for the page-space box corners. That scalar is c.sy.
    //
    // (This used to be spelled `box_px = pl.height / c.sy; px_to_pt =
    // pl.height / box_px`, which is algebraically just `c.sy` with the round
    // trip through pl.height cancelling — it derived nothing and read as if the
    // two axes were being handled separately. Same value, no false derivation.)
    const float box_px = c.sy > 0.0f ? pl.height / c.sy : 0.0f;
    const float px_to_pt = box_px > 0.0f ? c.sy : 0.0f;
    // Placed against the INK the estimator found inside the box, not the
    // box: detection unclips its polygons, so the box stands off the
    // glyphs and type fitted to it comes out too large and offset left.
    pl.ink_left_pt = static_cast<float>(pl.style->ink_x) * px_to_pt;
    pl.ink_w_pt = static_cast<float>(pl.style->ink_w) * px_to_pt;
    pl.ink_top_pt = static_cast<float>(pl.style->ink_y) * px_to_pt;
    pl.ink_h_pt = static_cast<float>(pl.style->ink_h) * px_to_pt;
    if (pl.ink_w_pt <= 0.0f || pl.ink_h_pt <= 0.0f) {
      pl.visible = false;
      continue;
    }
    pl.objs = prepare_visible_run(
        doc, fonts, pl.font, *pl.style, pl.em_px * c.sy, pl.run->cps,
        pl.x0 + pl.cos_a * pl.ink_left_pt, pl.y0 + pl.sin_a * pl.ink_left_pt,
        pl.cos_a, pl.sin_a, pl.ink_w_pt, pl.height, box_px, c.to_user);
    if (pl.objs.empty()) pl.visible = false;
  }
}

// The covers, for the runs that really do have a replacement.
static bool emit_covers(FPDF_PAGE page, const std::vector<Placed> &placed,
                 const PageCanvas &c) {
  bool any = false;
  for (const Placed &pl : placed) {
    if (!pl.visible) continue;
    // Sized to the INK, not to the detection box, and with a margin small
    // enough to stay inside normal leading. Covering the padded box plus a
    // fifth of its height on every side reaches a third of a line beyond
    // the words, which is further than lines are apart — it erased the
    // neighbours that had deliberately been left as scan.
    const float m = kCoverMargin * pl.ink_h_pt;
    const float top = pl.height - pl.ink_top_pt - pl.ink_h_pt;
    const Mat place{pl.cos_a, pl.sin_a, -pl.sin_a, pl.cos_a, pl.x0, pl.y0};
    if (insert_filled_rect(page, pl.ink_left_pt - m, top - m,
                           pl.ink_w_pt + 2 * m, pl.ink_h_pt + 2 * m,
                           pl.style->paper, concat(place, c.to_user)) == nullptr)
      continue;
    any = true;
  }
  return any;
}

// Puts every run's text on the page: the real type built above where there is
// some, an invisible glyphless run where there is not.
//
// `cid_of` is CONST. The font's ToUnicode CMap and CIDToGIDMap are built from
// this map BEFORE the document is opened, so an insertion here would be a
// silently-late charcode 0 — notdef, absent from ToUnicode, a word that extracts
// as nothing. `operator[]` could insert; `find` cannot. Output is identical for
// every reachable input (every codepoint here was put in the map by collect_runs)
// and the phase can no longer corrupt frozen font state.
static bool emit_text_runs(FPDF_DOCUMENT doc, FPDF_PAGE page, FPDF_FONT font,
                    const std::vector<Placed> &placed,
                    const std::map<unsigned, unsigned> &cid_of, const PageCanvas &c,
                    TextLayerMode mode, SearchableStats &tally) {
  bool any = false;
  std::vector<uint32_t> codes;

  for (const Placed &pl : placed) {
    const Run &run = *pl.run;
    // The measured x-height, snapped to the document's own set of sizes,
    // turned back into a point size through the chosen face's own metrics.
    if (!pl.objs.empty()) {
      for (FPDF_PAGEOBJECT o : pl.objs) FPDFPage_InsertObject(page, o);
      // Counted once per RUN, not once per object: a run set word by word is
      // still one replaced line, and tallying its pieces would make the same
      // page report more words the worse its type fitted.
      ++tally.words;
      ++tally.visible;
      any = true;
      continue;
    }
    // Either invisible mode, or a run visible mode could not take on. Both
    // land here, and both leave the page showing exactly what it showed
    // before: the scan, with the words findable behind it.
    //
    // Counted against the ASK, not against the attempt: a caller who turned
    // visible mode on wants to know how much of the document stayed a
    // picture, and most of those never reach the drawing at all — they are
    // ruled out earlier, for patterned ground or unspellable text.
    if (mode == TextLayerMode::Visible) ++tally.uncovered;

    codes.clear();
    codes.reserve(run.cps.size());
    for (uint32_t cp : run.cps) {
      const auto it = cid_of.find(cp);
      codes.push_back(it == cid_of.end() ? 0u : it->second);
    }

    const float size = pl.height / kEmHeight;
    const float natural = kAdvance * static_cast<float>(codes.size()) * size;
    if (natural <= 0.0f) {
      ++tally.dropped;
      continue;
    }
    // Stretch along the baseline so the invisible run spans exactly the
    // detected box: selection highlights then land on the visible glyphs.
    const float stretch = pl.width / natural;
    // Seat the baseline off the box's bottom edge by the descender.
    const float lift = kDescent * size;
    const Mat text{stretch * pl.cos_a, stretch * pl.sin_a, -pl.sin_a, pl.cos_a,
                   pl.x0 - pl.sin_a * lift, pl.y0 + pl.cos_a * lift};
    const Mat m = concat(text, c.to_user);

    FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, size);
    if (!obj) {
      ++tally.dropped;
      continue;
    }
    if (!FPDFText_SetCharcodes(obj, codes.data(), codes.size())) {
      FPDFPageObj_Destroy(obj);
      ++tally.dropped;
      continue;
    }
    FPDFTextObj_SetTextRenderMode(obj, FPDF_TEXTRENDERMODE_INVISIBLE);
    const FS_MATRIX fs{m.a, m.b, m.c, m.d, m.e, m.f};
    FPDFPageObj_SetMatrix(obj, &fs);
    FPDFPage_InsertObject(page, obj);
    ++tally.words;
    any = true;
  }
  return any;
}

// ── phase 6: layout annotations ───────────────────────────────────────────


// ── the two questions the page loop asks ──────────────────────────────────

// Anything at all to add beyond text? Layout annotations OR figures to lift
// out. Leaving the last one out of this test meant a page of nothing but
// pictures — a diagram sheet, a full-page photograph — was handed back
// untouched, and the regions were silently never lifted.
static bool has_non_text_work(const std::vector<SearchablePage> &pages,
                          bool movable_regions) {
  for (const auto &page : pages)
    if (page.layout && page.mark_regions)
      for (const auto &region : *page.layout)
        if (is_visual_region(region.class_id)) return true;
  if (movable_regions)
    for (const auto &page : pages)
      if ((page.regions != nullptr && !page.regions->empty()) ||
          (page.rules != nullptr && !page.rules->empty()))
        return true;
  return false;
}

// Nothing to add to this page, or nothing that can be added to? Skip it.
// Regions count as something to add: a full-page photograph or a diagram sheet
// has no recognised words and no layout ANNOTATIONS requested, and skipping it
// here meant its figures were never lifted out — the one kind of page where
// lifting them matters most.
static bool skip_page(const SearchablePage &in, const std::vector<Run> &runs,
               int doc_pages) {
  const bool has_regions = (in.regions != nullptr && !in.regions->empty()) ||
                           (in.rules != nullptr && !in.rules->empty());
  return (runs.empty() && !in.layout && !has_regions) || in.page_index < 0 ||
         in.page_index >= doc_pages || in.raster_w <= 0 || in.raster_h <= 0;
}

} // namespace searchable_detail

using namespace searchable_detail;

std::string write_searchable_pdf(const uint8_t *pdf, size_t len,
                                 const std::vector<SearchablePage> &pages,
                                 float min_confidence, SearchableStats *stats,
                                 std::string &err, TextLayerMode mode,
                                 bool movable_regions) {
  err.clear();
  if (!pdf || len == 0) {
    err = "empty PDF";
    return {};
  }

  DocumentRuns runs = collect_runs(pages, min_confidence);
  DocumentStyle style = resolve_document_style(pages, mode);

  if (runs.cid_of.empty() && !has_non_text_work(pages, movable_regions)) {
    // Nothing to stamp (an all-native-text document, or every page empty):
    // hand the original back untouched rather than rewriting it. `pages` is
    // pages STAMPED, and none were — reporting pages.size() here made an
    // all-native-text document log "pages=N words=0", which reads identically
    // to "stamped N pages, found no words".
    if (stats) {
      SearchableStats none;
      none.dropped = runs.dropped;
      *stats = none;
    }
    return std::string(reinterpret_cast<const char *>(pdf), len);
  }
  if (runs.cid_of.size() > 0xFFFE) {
    err = "document uses more than 65534 distinct characters";
    return {};
  }

  const std::vector<std::pair<unsigned, unsigned>> entries(runs.cid_of.begin(),
                                                           runs.cid_of.end());
  const std::string to_unicode = build_to_unicode(entries);
  // Every CID resolves to glyph 0: the font's only glyph, blank and uniform.
  const std::vector<uint8_t> cid_to_gid((runs.cid_of.size() + 1) * 2, 0);

  ensure_pdfium_initialized();
  std::lock_guard<std::mutex> guard(detail::pdfium_lock());

  FPDF_DOCUMENT doc = FPDF_LoadMemDocument(pdf, static_cast<int>(len), nullptr);
  if (!doc) {
    err = "PDFium could not open the document";
    return {};
  }
  FPDF_FONT font = nullptr;
  if (!runs.cid_of.empty()) {
    font = FPDFText_LoadCidType2Font(
        doc, kGlyphlessFont, static_cast<uint32_t>(sizeof(kGlyphlessFont)),
        to_unicode.c_str(), cid_to_gid.data(),
        static_cast<uint32_t>(cid_to_gid.size()));
    if (!font) {
      FPDF_CloseDocument(doc);
      err = "PDFium rejected the text-layer font";
      return {};
    }
  }

  FontCache font_cache(doc);
  snap_document_sizes(doc, font_cache, pages, runs, mode, style);

  const int doc_pages = FPDF_GetPageCount(doc);
  SearchableStats tally;
  tally.dropped = runs.dropped;

  for (size_t p = 0; p < pages.size(); ++p) {
    const SearchablePage &in = pages[p];
    if (skip_page(in, runs.page_runs[p], doc_pages)) continue;

    FPDF_PAGE page = FPDF_LoadPage(doc, in.page_index);
    if (!page) {
      // This page HAD work queued and will get none. Silently skipping it made
      // a partially-stamped PDF indistinguishable from a complete one.
      ++tally.pages_failed;
      TOCR_LOG_WARN("searchable PDF: page could not be loaded; it keeps no text "
                    "layer", "page", in.page_index);
      continue;
    }

    const PageCanvas canvas = prepare_page(page, in);
    std::vector<Placed> placed = place_page_runs(
        runs.page_runs[p], in, style, p, canvas, mode, tally.dropped);

    // `any` is "this page's content stream changed", which is exactly the
    // condition FPDFPage_GenerateContent has to be called on.
    bool any = false;
    if (movable_regions && in.blocks != nullptr)
      any |= emit_blocks(page, *in.blocks, canvas, tally.blocks);
    if (movable_regions && in.regions != nullptr)
      any |= emit_movable_regions(doc, page, *in.regions, canvas, tally.movable);
    if (movable_regions && in.rules != nullptr)
      any |= emit_rules(page, *in.rules, canvas, tally.rules);
    if (mode == TextLayerMode::Visible) {
      build_visible_runs(doc, font_cache, placed, canvas);
      any |= emit_covers(page, placed, canvas);
    }
    any |= emit_text_runs(doc, page, font, placed, runs.cid_of, canvas, mode,
                          tally);
    if (in.layout && in.mark_regions)
      emit_layout_annotations(page, *in.layout, canvas, tally.regions);

    if (any) {
      if (FPDFPage_GenerateContent(page)) {
        ++tally.pages;
      } else {
        // GenerateContent DISCARDS every object just inserted into this page —
        // covers, glyph runs, movable figures, rules. The page comes back
        // unchanged and the caller must be able to tell.
        ++tally.pages_failed;
        TOCR_LOG_WARN("searchable PDF: FPDFPage_GenerateContent failed; this "
                      "page keeps no text layer", "page", in.page_index);
      }
    }
    FPDF_ClosePage(page);
  }
  if (tally.pages_failed > 0)
    TOCR_LOG_WARN("searchable PDF is only partially stamped",
                  "pages_stamped", tally.pages,
                  "pages_failed", tally.pages_failed);

  Writer out;
  out.version = 1;
  out.WriteBlock = write_block;
  // Incremental: the original bytes are appended to, not rewritten, so cost
  // scales with what we added rather than with document size.
  const bool saved = FPDF_SaveAsCopy(doc, &out, FPDF_INCREMENTAL);
  font_cache.close();
  if (font) FPDFFont_Close(font);
  FPDF_CloseDocument(doc);

  if (!saved || out.buf.empty()) {
    err = "PDFium could not serialise the searchable PDF";
    return {};
  }
  if (stats) *stats = tally;
  return std::move(out.buf);
}

} // namespace turbo_ocr::pdf
