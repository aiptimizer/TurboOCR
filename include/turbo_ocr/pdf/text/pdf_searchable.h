#pragma once

// Searchable-PDF writer: stamps recognised words back onto the source PDF as an
// invisible text layer, so /ocr/pdf?output=pdf answers with the finished
// document instead of geometry a caller has to render itself.
//
// The visible pages are untouched — vector art, images, bookmarks and metadata
// all survive; only a text-rendering-mode-3 content stream is appended.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pdf/text/font_style.h"
#include "turbo_ocr/pdf/text/region_extract.h"

namespace turbo_ocr::pdf {

// What to do with the recognised text.
enum class TextLayerMode : uint8_t {
  // Render mode 3 in a glyphless font, over an untouched page. Search and
  // selection find the words; there is nothing to see and nothing to edit,
  // because what you are looking at is still the scan.
  Invisible,
  // Real type, drawn in place of the print it was read from. The words become
  // editable — a viewer can retype them, because they are text rather than
  // pixels — and they LOOK edited afterwards, because the printed original
  // underneath has been covered rather than left showing through.
  //
  // Needs SearchablePage::styles, and falls back to Invisible per line wherever
  // it cannot be done safely. Two cases, both deliberate:
  //
  //   - the background is not plain. Covering a ruled line, a shaded cell or a
  //     logo would destroy page content that is not text, so those lines are
  //     left exactly as they were printed.
  //   - the text needs glyphs the standard-14 fonts do not have. Those cover
  //     Latin-1 only; a line of Greek, Cyrillic or CJK cannot be drawn in them,
  //     and drawing it wrong is worse than leaving the scan to speak for
  //     itself. Such lines stay searchable, just not editable.
  Visible,
};

// One page of OCR output to stamp. Boxes are pixel coordinates in the raster
// the page was recognised on: origin top-left, y down, `raster_w` × `raster_h`.
struct SearchablePage {
  int page_index = 0;
  int raster_w = 0;
  int raster_h = 0;
  // Rotation autorotate applied before detection (0/90/180/270, clockwise).
  // The output page is turned to match, so it also arrives upright.
  int orientation_deg = 0;
  const std::vector<OCRResultItem> *results = nullptr;
  // Optional (?layout=1): figure/chart/table regions are marked with an
  // invisible annotation each, so a reader can select one and a consumer can
  // crop it without re-running detection.
  const std::vector<layout::LayoutBox> *layout = nullptr;
  // Required by TextLayerMode::Visible, ignored otherwise. Index-aligned with
  // `results`. Null (or the wrong length) means no line on this page can be
  // drawn visibly, and the page falls back to an invisible layer.
  const std::vector<LineStyle> *styles = nullptr;
  // What this page's type looked like to the shape matcher. The writer votes
  // over the pages, so one bad page cannot change the document's typeface.
  PageFontMatch font_match;
  // Figures, charts, tables and seals cut out of the raster, to be re-placed as
  // objects in their own right. Null unless the caller asked for movable
  // regions; see write_searchable_pdf's `movable_regions`.
  const std::vector<RegionImage> *regions = nullptr;
  // Draw the blue outline annotation for each figure/chart/table on this page.
  //
  // Separate from `layout` being present, because wanting to KNOW where the
  // figures are and wanting them RINGED are different wishes — and lifting
  // figures out needs layout detection, so tying the outlines to it meant
  // asking for movable figures silently drew a box round each one.
  bool mark_regions = true;
  // Printed rules recovered from this page, to be redrawn as real shapes.
  // Null unless the caller asked for movable elements.
  const std::vector<RuleShape> *rules = nullptr;
  // Flat colour blocks — header bars, shaded panels — to be redrawn as real
  // shapes. Null unless the caller asked for movable elements.
  const std::vector<BlockShape> *blocks = nullptr;
};

struct SearchableStats {
  // Pages whose content stream was successfully REGENERATED. Not "pages seen":
  // on the nothing-to-stamp early return this is 0, because nothing was stamped
  // and the original bytes are handed straight back.
  int pages = 0;
  // Pages that HAD work to do and did not get it: PDFium refused to load the
  // page, or FPDFPage_GenerateContent failed and discarded every object just
  // inserted into it. Non-zero means the returned PDF is PARTIALLY stamped —
  // indistinguishable from a fully stamped one without this counter.
  int pages_failed = 0;
  int words = 0;
  int dropped = 0;  // undecodable text or degenerate geometry
  int regions = 0;  // layout regions marked
  // Visible mode only: words drawn as real type over covered print, and words
  // that had to stay invisible because covering them was unsafe or the font
  // could not spell them. Together with `words` these say how much of the
  // document actually became editable.
  int visible = 0;
  int uncovered = 0;
  // Regions lifted off the page and re-placed as separate, movable objects.
  int movable = 0;
  // Printed rules turned into selectable vector shapes.
  int rules = 0;
  // Flat colour blocks turned into selectable vector shapes.
  int blocks = 0;
};

// Returns the new PDF bytes, or an empty string with `err` set.
//
// Words already sourced from the document's own text layer (source == "pdf")
// are skipped: they are visible to search already and overlaying them would
// duplicate every hit. `pdf` must stay alive for the duration of the call.
// `movable_regions` re-places each figure, chart, table and seal as its OWN
// image object, over a patch of the page's own paper colour. The page looks
// exactly as it did; the difference is that those things are now objects, so a
// viewer can move one and the page underneath is clean rather than showing the
// original through the gap. Requires SearchablePage::regions.
[[nodiscard]] std::string write_searchable_pdf(
    const uint8_t *pdf, size_t len, const std::vector<SearchablePage> &pages,
    float min_confidence, SearchableStats *stats, std::string &err,
    TextLayerMode mode = TextLayerMode::Invisible,
    bool movable_regions = false);

} // namespace turbo_ocr::pdf
