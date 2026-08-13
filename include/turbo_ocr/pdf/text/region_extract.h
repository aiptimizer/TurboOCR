#pragma once

// Cuts the figures, charts, tables and seals a layout model found OUT of the
// page raster, so each becomes an object in its own right instead of pixels
// inside one flat picture.
//
// This is the difference between a viewer that can outline a chart and one that
// can move it. Marking a region with an annotation tells a reader where the
// chart is; dragging that annotation drags an outline, because the chart is
// still part of the single scanned image underneath it. To move a chart, the
// chart has to BE something — its own image, with its own placement matrix —
// and the page underneath has to stop showing it, or the original stays behind
// as a ghost when the copy moves away.
//
// So each region is lifted out and re-placed exactly where it was, over a patch
// of the page's own paper colour. Nothing changes to look at. Everything
// changes about what can be done with it.
//
// Extraction happens on the pipeline worker, where the raster is still alive;
// the writer is handed encoded bytes and a rectangle.

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/image/page_image_encoder.h"

namespace turbo_ocr::pdf {

// One region lifted off the page.
struct RegionImage {
  // Where it came from, in the pixels of the raster it was cut from. The writer
  // maps this onto the page, so the object lands exactly over its own hole.
  int x = 0, y = 0, w = 0, h = 0;
  // What the layout model called it — "chart", "table", "image", "seal".
  std::string label;
  // The cut-out, encoded. JPEG for photographs and charts, which are what these
  // regions mostly are; a region of flat line-art would be smaller as PNG, but
  // one format keeps the writer simple and the difference is not worth a second
  // decode path.
  std::vector<uint8_t> bytes;
  // The colour to paint over the hole, taken from the page around the region so
  // a cream or grey form does not gain a white rectangle.
  uint8_t paper[3] = {255, 255, 255};
};

// One printed rule — a table border, an underline, the box around a panel —
// recovered from the page as a shape rather than as pixels.
//
// Deliberately NOT an image crop like RegionImage. A rule is a filled
// rectangle, and saying so makes it a few dozen bytes instead of a few
// thousand, selectable and re-colourable in any editor, and crisp at every
// zoom. Lifting it as a picture of a line would be all of those things worse.
struct RuleShape {
  int x = 0, y = 0, w = 0, h = 0;
  // The rule's own colour, read off the page; forms are not all printed black.
  uint8_t ink[3] = {0, 0, 0};
  // The paper it sits on, to patch the hole it leaves.
  uint8_t paper[3] = {255, 255, 255};
  bool horizontal = true;
};

// One solid block of colour — a header bar, a shaded table body, a sidebar
// panel, a coloured callout — recovered as a shape rather than as pixels.
//
// The third and last class of thing a printed page is made of. Figures come
// back as images and rules as thin rectangles; what was left baked into the
// raster was the flat colour BEHIND everything, which on a designed document is
// most of what you see. A block is a rule that stopped being thin, so it is
// stored the same way and for the same reasons: a rectangle and a colour, a few
// dozen bytes, crisp at any zoom and selectable in any editor.
//
// Blocks are drawn FIRST of everything, because that is what they are — the
// ground the rest of the page sits on.
struct BlockShape {
  int x = 0, y = 0, w = 0, h = 0;
  // The block's own fill, the median of the flat colour that defines it.
  uint8_t fill[3] = {255, 255, 255};
  // The colour around it, to patch the hole when the block is lifted out.
  uint8_t paper[3] = {255, 255, 255};
};

struct RegionExtractOptions {
  // Below this a "region" is not worth making movable, and is more likely a
  // detection artefact than a figure.
  int min_side_px = 24;
  // A region covering this much of the page is the page — lifting it out would
  // replace the scan with a copy of itself and gain nothing.
  float max_page_fraction = 0.9f;
  // Ceiling on how much encoded image one page may produce. A dense page of
  // figures should not turn a scan into something several times its size.
  size_t max_bytes_per_page = 12u * 1024u * 1024u;
  int jpeg_quality = 88;

  // A rule must run at least this fraction of the page's width or height. Below
  // it, a "line" is more likely an underscore, a dash, or the stem of a letter.
  float rule_min_run = 0.06f;
  // And be no thicker than this fraction of the page, or it is a filled panel.
  float rule_max_thickness = 0.012f;
  // How many times longer than it is thick a rule has to be.
  //
  // Length alone is not enough to tell a rule from a letter: a column of tall
  // lowercase l's runs as far down the page as a short table border does, and
  // was being lifted out as one. A printed rule is enormously longer than it is
  // thick — fifty to one and up — while even the thinnest letter stroke is
  // nearer ten.
  float rule_min_aspect = 20.0f;

  // ── colour blocks ──
  // How far a pixel's colour may sit from the page background before it counts
  // as something drawn. Per channel, 0-255. Low enough to catch a pale grey
  // panel, high enough that scanner noise on white paper is not a block.
  int block_min_contrast = 18;
  // A block must cover at least this fraction of the page. Smaller flat patches
  // are icons, logo marks and the insides of glyphs — things the figure and
  // text paths already own.
  float block_min_area = 0.004f;
  // And no more than this, or it is the sheet itself.
  float block_max_area = 0.85f;
  // The fraction of a candidate's pixels that must lie within tolerance of its
  // median colour for it to be one flat block. Well below 1 because a block is
  // usually full of text: a teal header bar with white lettering runs about
  // 0.85, while a photograph never comes close.
  float block_flatness = 0.70f;
  // And how rectangular it has to be — filled area over bounding-box area. A
  // block is a rectangle; an L-shaped union of two panels is two blocks.
  float block_rectangularity = 0.88f;
  // How far a pixel may sit from the candidate's OWN median and still count
  // towards flatness.
  int block_tolerance = 26;
};

// Cuts every movable region out of `page`. Returns them in reading order.
// Regions the layout model marked as text are left alone: text is handled by
// the text layer, and lifting it would fight with it.
[[nodiscard]] std::vector<RegionImage>
extract_movable_regions(const cv::Mat &page,
                        const std::vector<layout::LayoutBox> &layout,
                        const RegionExtractOptions &opt = {});

// Finds every printed rule on the page: table borders, underlines, the lines
// around a panel. These are the last things on a scan that cannot be selected,
// because they are ink in the image rather than objects — the same complaint
// figures had before they were lifted, and the same answer.
[[nodiscard]] std::vector<RuleShape>
extract_rules(const cv::Mat &page, const RegionExtractOptions &opt = {});

// Finds the flat colour blocks: header bars, shaded panels, coloured table
// bodies. Returns them LARGEST FIRST, which is the order they have to be drawn
// in — a panel inside a panel has to land on top of the one that contains it.
//
// A block is allowed to hold text; most do. What makes it a block is that the
// great majority of it is ONE colour, so the test is on the fraction of its
// pixels near the median rather than on the variance, which a caption would
// wreck. The page's own paper is excluded by construction: a candidate whose
// fill is the background colour is the background.
[[nodiscard]] std::vector<BlockShape>
extract_blocks(const cv::Mat &page, const RegionExtractOptions &opt = {});

} // namespace turbo_ocr::pdf
