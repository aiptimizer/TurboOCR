#pragma once

// Reads what a recognised line LOOKS like off the page raster, so a text layer
// can be written in type that resembles the scan instead of a glyphless
// placeholder that only search can see.
//
// Every measurement here is per line, cheap, and individually unreliable — a
// single line of a scan carries very little evidence about a typeface. That is
// deliberate: resolve_document_fonts() votes across the whole document before
// any font is chosen, so the estimator only has to be right on average, and the
// answer is stable for the document as a whole rather than drifting line to
// line. Two consequences worth knowing:
//
//   - the family (serif / sans / mono) is decided ONCE per document;
//   - bold and italic are decided per line but RELATIVE to that document's own
//     median, because absolute stroke width moves with scan resolution and
//     binarisation, while the gap between a document's regular and its bold
//     does not. The same subtraction removes global scan skew from the slant,
//     so a page that leans 3° reads as upright rather than as all-italic.

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/types.h"

namespace turbo_ocr::pdf {

// What one line's pixels say about the type it was set in. Lengths are in the
// pixels of the raster the line was recognised on.
struct LineStyle {
  // False when the crop was too small, too faint, or carried no usable ink.
  // Callers must fall back rather than trust the other fields.
  bool measured = false;

  // Higher means more serif-like. Only meaningful compared against other lines
  // of the same document — see resolve_document_fonts().
  float serif = 0.0f;
  // Stroke width as a fraction of x-height. Around 0.09-0.12 for regular text,
  // 0.16-0.22 for bold, but the absolute value moves with scan quality.
  float weight = 0.0f;
  // Degrees the stems lean from vertical; positive leans right, as italics do.
  float slant_deg = 0.0f;
  // Mean advance per character over the line, as a multiple of x-height. Zero
  // when the caller did not say how many characters the line holds.
  //
  // One line's value says nothing. Across a document it says everything about
  // pitch: a monospaced face puts every character on the same advance, so this
  // comes back identical line after line whatever they happen to say, while
  // proportional text swings with content — a line of "Illinois" is far
  // narrower per character than one of "WAXWORK".
  float advance_ratio = 0.0f;

  // The ink's own bounds inside the detection box, in that box's pixels.
  //
  // Not the same as the box, and the difference matters: text detection unclips
  // its polygons, so the box stands off the glyphs by a tenth or so on every
  // side. Fitting replacement type to the BOX therefore sets it about a third
  // too large. Fitting it to what was actually inked gets the size right.
  int ink_x = 0, ink_y = 0, ink_w = 0, ink_h = 0;

  float x_height_px = 0.0f;
  // Where the glyphs actually sit inside the detected box, measured down from
  // its top edge. The box hugs the ink, so this is what puts replacement text
  // on the same baseline as the print it replaces.
  float baseline_px = 0.0f;
  float ascent_px = 0.0f;

  cv::Vec3b ink{0, 0, 0};          // BGR, median over the glyph pixels
  cv::Vec3b paper{255, 255, 255};  // BGR, median over what surrounds them
  // The background is plain enough that painting `paper` over the line's box
  // hides the printed glyphs without erasing anything else. False over a rule,
  // a shaded cell, a logo or a photo, where covering the area would destroy
  // page content that is not text.
  bool flat_paper = false;
};

enum class FontFamily : uint8_t { Sans, Serif, Mono };

struct FontChoice {
  FontFamily family = FontFamily::Sans;
  bool bold = false;
  bool italic = false;

  // The PDF standard-14 name for this combination: "Helvetica-BoldOblique",
  // "Times-Roman", "Courier-Bold", and so on. Standard-14 needs no embedded
  // font file and is metrically compatible with the faces scanned business
  // documents are actually set in (Arial/Helvetica, Times New Roman/Times,
  // Courier New/Courier), so the common case costs no bytes and no licence.
  [[nodiscard]] const char *standard_name() const noexcept;

  [[nodiscard]] bool operator==(const FontChoice &) const noexcept = default;
};

// Measures one line. `box` is in `page`'s pixel space, corners clockwise from
// top-left. `page` may be BGR or grayscale. Returns measured=false rather than
// guessing when the crop is too small or carries no ink.
//
// `char_count` is how many characters the line was recognised as holding; pass
// 0 when that is not known, which only costs the pitch measurement.
[[nodiscard]] LineStyle measure_line_style(const cv::Mat &page, const Box &box,
                                           int char_count = 0);

// What the shape matcher made of one page, carried to the writer.
//
// Matching happens per page and on the pipeline worker, because that is the
// only place the raster still exists — and per page rather than per document
// because holding crops from every page until write time would mean holding a
// scanned document twice over in memory. The document-wide answer is a vote
// over these, taken where the pages meet again.
struct PageFontMatch {
  FontFamily family = FontFamily::Sans;
  // Mean shape agreement of the winner, 0 when nothing could be matched. Used
  // to weight this page's vote.
  float score = 0.0f;
};

// Straightens one line into an upright, ink-tight crop. Shared so the font
// matcher rectifies exactly the way the estimator does — an axis-aligned
// bounding box is NOT the same thing on a skewed page, where a line 600 px
// long leaning 2 degrees gives a box some 20 px taller than the ink in it, and
// anything that normalises by that height is working at the wrong scale.
// Returns false when the box is too small to be worth measuring.
[[nodiscard]] bool rectify_line(const cv::Mat &page, const Box &box,
                                cv::Mat &out);

// Measures every recognised line on one page. Returns a vector index-aligned
// with `results` — lines the estimator cannot read come back with
// measured == false rather than being dropped, so the alignment always holds.
// An empty `page` yields an empty vector.
[[nodiscard]] std::vector<LineStyle>
measure_page_line_styles(const std::vector<OCRResultItem> &results,
                         const cv::Mat &page);

// Turns per-line measurements into one font per line, voting across the whole
// document so that lines which look alike land on exactly the same font.
// Output is index-aligned with `lines`; unmeasured lines get the document's
// majority choice, which is a better guess than any per-line default.
//
// `family_override` replaces the measured family decision while leaving bold
// and italic to the measurements. That split is where each method is strongest:
// over 90 installed text faces the shape matcher in font_match.h reads the
// family correctly 100% of the time given a rich catalogue, while weight and
// slant are per-line properties a document-wide match cannot see at all.
[[nodiscard]] std::vector<FontChoice>
resolve_document_fonts(const std::vector<LineStyle> &lines,
                       const FontFamily *family_override = nullptr);

} // namespace turbo_ocr::pdf
