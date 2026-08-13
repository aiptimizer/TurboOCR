#include "turbo_ocr/pdf/text/region_extract.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/analysis/forms/field_detector.h"

namespace turbo_ocr::pdf {
namespace {

// The classes worth making movable: things a person would drag. Text is not
// among them — it is handled by the text layer, and lifting it out as a picture
// would put a picture of words on top of the words.
bool is_movable(int class_id) {
  const std::string_view label = layout::label_name(class_id);
  return label == "image" || label == "chart" || label == "table" ||
         label == "seal" || label == "header_image" || label == "footer_image";
}

// The page colour just outside a region, which is what the hole gets painted.
//
// Sampled from a ring around the region rather than from the whole page: a form
// may be white at the top and grey in a shaded panel, and the patch has to match
// the neighbourhood it sits in, not the average of the sheet.
cv::Vec3b surrounding_paper(const cv::Mat &page, const cv::Rect &r) {
  const int pad = std::max(4, std::min(r.width, r.height) / 8);
  const cv::Rect outer =
      cv::Rect(r.x - pad, r.y - pad, r.width + 2 * pad, r.height + 2 * pad) &
      cv::Rect(0, 0, page.cols, page.rows);
  if (outer.width <= 0 || outer.height <= 0) return {255, 255, 255};

  std::vector<uint8_t> ch[3];
  const cv::Mat crop = page(outer);
  const cv::Rect inner(r.x - outer.x, r.y - outer.y, r.width, r.height);
  for (int y = 0; y < crop.rows; ++y) {
    const auto *row = crop.ptr<cv::Vec3b>(y);
    for (int x = 0; x < crop.cols; ++x) {
      if (inner.contains(cv::Point(x, y))) continue;  // that is the region
      for (int c = 0; c < 3; ++c) ch[c].push_back(row[x][c]);
    }
  }
  if (ch[0].size() < 16) return {255, 255, 255};

  cv::Vec3b out;
  for (int c = 0; c < 3; ++c) {
    // The BRIGHTEST quartile, not the median. A ring around a chart still
    // catches its axis labels and frame; taking the middle of that would paint
    // the hole a dirty grey. What is wanted is the paper the ink sits on.
    auto &v = ch[c];
    const size_t k = v.size() * 3 / 4;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    out[c] = v[k];
  }
  return out;
}

} // namespace

std::vector<RegionImage>
extract_movable_regions(const cv::Mat &page,
                        const std::vector<layout::LayoutBox> &layout,
                        const RegionExtractOptions &opt) {
  std::vector<RegionImage> out;
  if (page.empty() || layout.empty()) return out;

  cv::Mat bgr;
  if (page.channels() == 1) cv::cvtColor(page, bgr, cv::COLOR_GRAY2BGR);
  else if (page.channels() == 4) cv::cvtColor(page, bgr, cv::COLOR_BGRA2BGR);
  else bgr = page;

  const double page_area = static_cast<double>(bgr.cols) * bgr.rows;
  size_t budget = opt.max_bytes_per_page;

  // Biggest first, so that when the budget runs out it is the small decorative
  // regions that are dropped rather than the figure someone actually wants.
  std::vector<const layout::LayoutBox *> picked;
  for (const auto &region : layout)
    if (is_movable(region.class_id)) picked.push_back(&region);
  std::ranges::sort(picked, [](const layout::LayoutBox *a, const layout::LayoutBox *b) {
    const auto area = [](const layout::LayoutBox *r) {
      int lo_x = r->box[0][0], hi_x = r->box[0][0];
      int lo_y = r->box[0][1], hi_y = r->box[0][1];
      for (int k = 1; k < 4; ++k) {
        lo_x = std::min(lo_x, r->box[k][0]);
        hi_x = std::max(hi_x, r->box[k][0]);
        lo_y = std::min(lo_y, r->box[k][1]);
        hi_y = std::max(hi_y, r->box[k][1]);
      }
      return static_cast<long long>(hi_x - lo_x) * (hi_y - lo_y);
    };
    return area(a) > area(b);
  });

  EncodeOptions enc;
  enc.format = PageImageFormat::Jpeg;
  enc.quality = opt.jpeg_quality;

  for (const layout::LayoutBox *region : picked) {
    int lo_x = region->box[0][0], hi_x = region->box[0][0];
    int lo_y = region->box[0][1], hi_y = region->box[0][1];
    for (int k = 1; k < 4; ++k) {
      lo_x = std::min(lo_x, region->box[k][0]);
      hi_x = std::max(hi_x, region->box[k][0]);
      lo_y = std::min(lo_y, region->box[k][1]);
      hi_y = std::max(hi_y, region->box[k][1]);
    }
    const cv::Rect r =
        cv::Rect(lo_x, lo_y, hi_x - lo_x, hi_y - lo_y) &
        cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (r.width < opt.min_side_px || r.height < opt.min_side_px) continue;
    if (static_cast<double>(r.width) * r.height >
        opt.max_page_fraction * page_area)
      continue;

    RegionImage img;
    img.bytes = encode_page_image(bgr(r), enc);
    if (img.bytes.empty()) continue;
    if (img.bytes.size() > budget) continue;
    budget -= img.bytes.size();

    img.x = r.x;
    img.y = r.y;
    img.w = r.width;
    img.h = r.height;
    img.label = layout::label_name(region->class_id);
    const cv::Vec3b paper = surrounding_paper(bgr, r);
    img.paper[0] = paper[0];
    img.paper[1] = paper[1];
    img.paper[2] = paper[2];
    out.push_back(std::move(img));
  }

  // Back into reading order for the caller.
  std::ranges::sort(out, [](const RegionImage &a, const RegionImage &b) {
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
  });
  return out;
}

} // namespace turbo_ocr::pdf

namespace turbo_ocr::pdf {
namespace {

// Median colour of the ink a rule is drawn in, and of the paper beside it.
void rule_colours(const cv::Mat &bgr, const cv::Rect &r, bool horizontal,
                  uint8_t ink[3], uint8_t paper[3]) {
  std::vector<uint8_t> ic[3];
  std::vector<uint8_t> pc[3];
  // The rule itself, and a band on either side of it.
  const int pad = std::max(2, (horizontal ? r.height : r.width) * 3);
  const cv::Rect outer =
      cv::Rect(r.x - (horizontal ? 0 : pad), r.y - (horizontal ? pad : 0),
               r.width + (horizontal ? 0 : 2 * pad),
               r.height + (horizontal ? 2 * pad : 0)) &
      cv::Rect(0, 0, bgr.cols, bgr.rows);
  if (outer.width <= 0 || outer.height <= 0) return;

  for (int y = outer.y; y < outer.y + outer.height; ++y) {
    const auto *row = bgr.ptr<cv::Vec3b>(y);
    for (int x = outer.x; x < outer.x + outer.width; ++x) {
      const bool on_rule = r.contains(cv::Point(x, y));
      for (int c = 0; c < 3; ++c)
        (on_rule ? ic[c] : pc[c]).push_back(row[x][c]);
    }
  }
  for (int c = 0; c < 3; ++c) {
    if (ic[c].size() >= 4) {
      std::nth_element(ic[c].begin(), ic[c].begin() + static_cast<long>(ic[c].size() / 2),
                       ic[c].end());
      ink[c] = ic[c][ic[c].size() / 2];
    }
    if (pc[c].size() >= 4) {
      // Brightest quartile again: the band beside a table border still catches
      // the next cell's text.
      const size_t k = pc[c].size() * 3 / 4;
      std::nth_element(pc[c].begin(), pc[c].begin() + static_cast<long>(k), pc[c].end());
      paper[c] = pc[c][k];
    }
  }
}

} // namespace

std::vector<RuleShape> extract_rules(const cv::Mat &page,
                                     const RegionExtractOptions &opt) {
  std::vector<RuleShape> out;
  if (page.empty() || page.cols < 40 || page.rows < 40) return out;

  cv::Mat bgr;
  if (page.channels() == 1) cv::cvtColor(page, bgr, cv::COLOR_GRAY2BGR);
  else if (page.channels() == 4) cv::cvtColor(page, bgr, cv::COLOR_BGRA2BGR);
  else bgr = page;

  // Binarisation and the morphological line pass are BORROWED, not repeated.
  //
  // Field detection has needed to find ruled lines since long before this did,
  // and had a tuned implementation of exactly that in models/forms. Writing a
  // second one here left two morphology passes that would drift apart the first
  // time either was corrected — the same mistake as fixing a bug per backend
  // instead of once in the shared policy.
  const cv::Mat binary = forms::binarize_page(bgr);

  const int min_h = std::max(12, static_cast<int>(opt.rule_min_run * bgr.cols));
  const int min_v = std::max(12, static_cast<int>(opt.rule_min_run * bgr.rows));
  const int max_thick =
      std::max(2, static_cast<int>(opt.rule_max_thickness *
                                   std::max(bgr.cols, bgr.rows)));

  const forms::LineMasks masks = forms::extract_line_masks(binary, min_h, min_v);

  for (int pass = 0; pass < 2; ++pass) {
    const bool horizontal = pass == 0;
    const cv::Mat &lines = horizontal ? masks.horizontal : masks.vertical;
    if (lines.empty()) continue;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(lines, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
      const cv::Rect r = cv::boundingRect(contour);
      const int run = horizontal ? r.width : r.height;
      const int thick = horizontal ? r.height : r.width;
      if (run < (horizontal ? min_h : min_v)) continue;
      if (thick > max_thick || thick <= 0) continue;
      if (static_cast<float>(run) < opt.rule_min_aspect * static_cast<float>(thick))
        continue;

      RuleShape rule;
      rule.x = r.x;
      rule.y = r.y;
      rule.w = r.width;
      rule.h = r.height;
      rule.horizontal = horizontal;
      rule_colours(bgr, r, horizontal, rule.ink, rule.paper);
      out.push_back(rule);
    }
  }

  std::ranges::sort(out, [](const RuleShape &a, const RuleShape &b) {
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
  });
  return out;
}

} // namespace turbo_ocr::pdf

namespace turbo_ocr::pdf {
namespace {

// Colour distance as the worst single channel. Euclidean distance in BGR would
// let a large error in one channel hide behind two small ones, which is exactly
// the case that matters here: a teal bar and a grey panel of the same lightness
// differ in one channel and are not the same block.
int chan_dist(const cv::Vec3b &a, const cv::Vec3b &b) {
  return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]),
                   std::abs(a[2] - b[2])});
}

// The colours the page is actually made of, most common first.
//
// Quantised to 4 bits per channel before counting. Scanning introduces a spread
// of a few levels around every flat fill, so counting exact BGR triples would
// split one header bar across hundreds of near-identical colours and none of
// them would look common. 16 levels per channel is coarse enough to collapse
// that spread and fine enough to keep two design colours apart.
std::vector<cv::Vec3b> dominant_colours(const cv::Mat &bgr, int want) {
  constexpr int kBins = 16 * 16 * 16;
  std::vector<long> count(kBins, 0);
  // Subsampled: a page has millions of pixels and this only needs the shape of
  // the histogram, not an exact census.
  const int step = std::max(1, std::min(bgr.cols, bgr.rows) / 400);
  for (int y = 0; y < bgr.rows; y += step) {
    const auto *row = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < bgr.cols; x += step) {
      const cv::Vec3b &p = row[x];
      count[((p[0] >> 4) << 8) | ((p[1] >> 4) << 4) | (p[2] >> 4)] += 1;
    }
  }
  std::vector<int> order(kBins);
  for (int i = 0; i < kBins; ++i) order[i] = i;
  std::ranges::partial_sort(
      order, order.begin() + std::min<int>(want, kBins),
      [&](int a, int b) { return count[a] > count[b]; });

  std::vector<cv::Vec3b> out;
  for (int i = 0; i < want && i < kBins; ++i) {
    if (count[order[i]] == 0) continue;
    const int bin = order[i];
    // The centre of the bin, not its corner.
    out.emplace_back(static_cast<uint8_t>((((bin >> 8) & 0xF) << 4) | 0x8),
                     static_cast<uint8_t>((((bin >> 4) & 0xF) << 4) | 0x8),
                     static_cast<uint8_t>(((bin & 0xF) << 4) | 0x8));
  }
  return out;
}

} // namespace

std::vector<BlockShape> extract_blocks(const cv::Mat &page,
                                       const RegionExtractOptions &opt) {
  std::vector<BlockShape> out;
  if (page.empty() || page.cols < 40 || page.rows < 40) return out;

  cv::Mat bgr;
  if (page.channels() == 1) cv::cvtColor(page, bgr, cv::COLOR_GRAY2BGR);
  else if (page.channels() == 4) cv::cvtColor(page, bgr, cv::COLOR_BGRA2BGR);
  else bgr = page;

  // Candidate fills, and the page's own ground. The most common colour on a
  // printed page is the paper; everything is measured against it.
  const std::vector<cv::Vec3b> palette = dominant_colours(bgr, 12);
  if (palette.empty()) return out;
  const cv::Vec3b background = palette.front();

  const double page_area = static_cast<double>(bgr.cols) * bgr.rows;
  // Sized off the page, so one set of numbers holds at 100 and 300 dpi: the
  // open has to erase text strokes and the close has to bridge the gaps BETWEEN
  // letters, which are several times wider than a stroke.
  const int k = std::max(3, std::min(bgr.cols, bgr.rows) / 200);
  const cv::Mat open_k =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
  const cv::Mat close_k =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k * 4, k * 4));

  for (const cv::Vec3b &colour : palette) {
    // The paper is not a block drawn ON the paper.
    if (chan_dist(colour, background) < opt.block_min_contrast) continue;

    const cv::Scalar lo(std::max(0, colour[0] - opt.block_tolerance),
                        std::max(0, colour[1] - opt.block_tolerance),
                        std::max(0, colour[2] - opt.block_tolerance));
    const cv::Scalar hi(std::min(255, colour[0] + opt.block_tolerance),
                        std::min(255, colour[1] + opt.block_tolerance),
                        std::min(255, colour[2] + opt.block_tolerance));
    cv::Mat raw;
    cv::inRange(bgr, lo, hi, raw);
    // OPEN first to erase the strokes of text set IN this colour, then CLOSE to
    // fill the lettering knocked out of a block that is this colour. Both are
    // needed and the order matters: closing first would weld a line of text into
    // a solid bar and then present it as a block.
    cv::Mat mask;
    cv::morphologyEx(raw, mask, cv::MORPH_OPEN, open_k);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
      const cv::Rect r =
          cv::boundingRect(contour) & cv::Rect(0, 0, bgr.cols, bgr.rows);
      if (r.width < opt.min_side_px || r.height < opt.min_side_px) continue;
      const double area = static_cast<double>(r.width) * r.height;
      if (area < opt.block_min_area * page_area) continue;
      if (area > opt.block_max_area * page_area) continue;

      // Rectangular: measured on the CLOSED mask, so the lettering punched out
      // of a header bar does not count against the bar being a rectangle.
      if (static_cast<double>(cv::countNonZero(mask(r))) / area <
          opt.block_rectangularity)
        continue;

      // Flat: measured on the ORIGINAL pixels, which is what decides whether
      // this is a fill or a picture. A photograph can pass the shape tests; it
      // cannot pass this one.
      // Subsampled: this is a proportion, and a grid of a few thousand samples
      // pins one far tighter than the tolerance it is compared against. Reading
      // every pixel of every candidate made the whole block pass cost about as
      // much as the rest of the page put together.
      const int fs = std::max(1, std::min(r.width, r.height) / 64);
      long near = 0;
      long seen = 0;
      for (int y = r.y; y < r.y + r.height; y += fs) {
        const auto *row = bgr.ptr<cv::Vec3b>(y);
        for (int x = r.x; x < r.x + r.width; x += fs) {
          ++seen;
          if (chan_dist(row[x], colour) <= opt.block_tolerance) ++near;
        }
      }
      if (seen == 0) continue;
      if (static_cast<double>(near) / static_cast<double>(seen) <
          opt.block_flatness)
        continue;

      BlockShape block;
      block.x = r.x;
      block.y = r.y;
      block.w = r.width;
      block.h = r.height;
      for (int c = 0; c < 3; ++c) block.fill[c] = colour[c];
      const cv::Vec3b paper = surrounding_paper(bgr, r);
      for (int c = 0; c < 3; ++c) block.paper[c] = paper[c];
      out.push_back(block);
    }
  }

  // Largest first: a block drawn inside another has to land on top of it, and
  // that is the order the writer emits in.
  std::ranges::sort(out, [](const BlockShape &a, const BlockShape &b) {
    return static_cast<long long>(a.w) * a.h >
           static_cast<long long>(b.w) * b.h;
  });

  // Two neighbouring quantiser bins can both cover one fill and return the same
  // rectangle twice. Keep the first (the larger, and the more common colour).
  std::vector<BlockShape> kept;
  for (const BlockShape &b : out) {
    const bool dup = std::ranges::any_of(kept, [&](const BlockShape &k) {
      const int ix = std::max(0, std::min(k.x + k.w, b.x + b.w) - std::max(k.x, b.x));
      const int iy = std::max(0, std::min(k.y + k.h, b.y + b.h) - std::max(k.y, b.y));
      const long long inter = static_cast<long long>(ix) * iy;
      const long long small = std::min(static_cast<long long>(k.w) * k.h,
                                       static_cast<long long>(b.w) * b.h);
      return small > 0 && inter * 10 > small * 8;
    });
    if (!dup) kept.push_back(b);
  }
  return kept;
}

} // namespace turbo_ocr::pdf
