#include "turbo_ocr/layout/blocks/text_line_cluster.h"

#include <algorithm>
#include <climits>
#include <cstddef>

#include "turbo_ocr/common/geometry/box.h"

namespace turbo_ocr::layout {

namespace {

// 1D projection overlap ratio along [a0,a1) vs [b0,b1) using "small"
// mode — intersection divided by the smaller of the two extents.
// Mirrors paddlex.layout_parsing.utils.calculate_projection_overlap_ratio
// with mode="small" used by group_boxes_into_lines.
inline float projection_overlap_small(int a0, int a1, int b0, int b1) noexcept {
  const int inter = std::max(0, std::min(a1, b1) - std::max(a0, b0));
  if (inter <= 0) return 0.0f;
  const int len_a = std::max(0, a1 - a0);
  const int len_b = std::max(0, b1 - b0);
  const int small = std::min(len_a, len_b);
  return small > 0 ? float(inter) / float(small) : 0.0f;
}

// Per-result AABB (axis-aligned bbox of the 4-corner detection quad).
struct ResAabb {
  int x0, y0, x1, y1;
  int idx;  // index into results vector
};

// Direction vote for a single bbox using PaddleX's text-line ratio
// (width * 1.5 >= height ⇒ horizontal). Used both per-result and per-
// cell majority.
inline bool box_is_horizontal(const ResAabb &b) noexcept {
  const int w = b.x1 - b.x0;
  const int h = b.y1 - b.y0;
  return float(w) * kTextLineDirectionRatio >= float(h);
}

// A clustered text line: union bbox of the spans assigned to it.
struct TextLineAcc {
  int x0 = INT_MAX, y0 = INT_MAX, x1 = INT_MIN, y1 = INT_MIN;
  int height() const noexcept { return std::max(0, y1 - y0); }
  int width() const noexcept { return std::max(0, x1 - x0); }
  void add(const ResAabb &r) noexcept {
    x0 = std::min(x0, r.x0);
    y0 = std::min(y0, r.y0);
    x1 = std::max(x1, r.x1);
    y1 = std::max(y1, r.y1);
  }
};

} // namespace

void cluster_text_lines(const std::vector<OCRResultItem> &results,
                        std::vector<LayoutBox> &layout) {
  const std::size_t ncells = layout.size();
  if (ncells == 0) return;

  // Bucket the result AABBs by layout_id into a single flat buffer laid out
  // in CSR form (offsets[li]..offsets[li+1) is cell li's span range). This is
  // a stable counting sort: iterating results in index order keeps each cell's
  // spans in ascending result-index order — byte-identical to the previous
  // per-cell push_back order, which the (unstable) std::sort below depends on
  // for a deterministic, regression-free result. One buffer replaces N
  // per-cell vectors: no per-cell allocation, contiguous, cache-friendly.
  std::vector<int> offsets(ncells + 1, 0);
  for (const auto &r : results) {
    const int lid = r.layout_id;
    if (lid >= 0 && static_cast<std::size_t>(lid) < ncells)
      ++offsets[static_cast<std::size_t>(lid) + 1];
  }
  for (std::size_t i = 0; i < ncells; ++i) offsets[i + 1] += offsets[i];

  std::vector<ResAabb> spans(static_cast<std::size_t>(offsets[ncells]));
  std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
  for (std::size_t i = 0; i < results.size(); ++i) {
    const int lid = results[i].layout_id;
    if (lid < 0 || static_cast<std::size_t>(lid) >= ncells) continue;
    const auto [x0, y0, x1, y1] = turbo_ocr::aabb(results[i].box);
    spans[static_cast<std::size_t>(cursor[static_cast<std::size_t>(lid)]++)] =
        {x0, y0, x1, y1, static_cast<int>(i)};
  }

  for (std::size_t li = 0; li < ncells; ++li) {
    auto &cell = layout[li];
    const auto beg = spans.begin() + offsets[li];
    const auto end = spans.begin() + offsets[li + 1];
    if (beg == end) {
      cell.direction = Direction::kHorizontal;
      cell.num_of_lines = 0;
      cell.text_line_height = 0;
      cell.text_line_width = 0;
      cell.seg_start_coordinate = 0;
      cell.seg_end_coordinate = 0;
      continue;
    }
    const std::size_t n = static_cast<std::size_t>(end - beg);

    // 1. Per-cell direction vote (horizontal wins ties: PaddleX declares
    //    horizontal when horizontal_count >= vertical_count).
    int horizontal_votes = 0;
    for (auto it = beg; it != end; ++it)
      if (box_is_horizontal(*it)) ++horizontal_votes;
    const bool horizontal = horizontal_votes >= static_cast<int>((n + 1) / 2);
    cell.direction = horizontal ? Direction::kHorizontal : Direction::kVertical;

    // 2. Sort spans by primary axis. group_boxes_into_lines uses ascending y
    //    for horizontal and DESCENDING x for vertical (right-to-left CJK).
    if (horizontal)
      std::sort(beg, end, [](const ResAabb &a, const ResAabb &b) noexcept {
        return a.y0 < b.y0;
      });
    else
      std::sort(beg, end, [](const ResAabb &a, const ResAabb &b) noexcept {
        return a.x0 > b.x0;
      });

    // 3. Walk the sorted spans, growing the current line's union bbox and
    //    splitting when a span's perpendicular projection no longer overlaps
    //    it (vertical band for horizontal lines, horizontal band otherwise).
    //    Line stats (count, height/width sums, first/last line bbox) are
    //    accumulated in this single streaming pass — no per-line container is
    //    allocated. The direction branch is hoisted out of the inner loop.
    TextLineAcc current;
    current.add(*beg);
    TextLineAcc first_line;  // union bbox of the first completed line
    bool have_first = false;
    long long sum_h = 0, sum_w = 0;
    int num_lines = 0;

    const auto close_line = [&](const TextLineAcc &ln) noexcept {
      sum_h += ln.height();
      sum_w += ln.width();
      ++num_lines;
      if (!have_first) {
        first_line = ln;
        have_first = true;
      }
    };

    for (auto it = beg + 1; it != end; ++it) {
      const ResAabb &s = *it;
      const float overlap =
          horizontal
              ? projection_overlap_small(current.y0, current.y1, s.y0, s.y1)
              : projection_overlap_small(current.x0, current.x1, s.x0, s.x1);
      if (overlap >= kLineHeightIouThreshold) {
        current.add(s);
      } else {
        close_line(current);
        current = TextLineAcc{};
        current.add(s);
      }
    }
    close_line(current);  // final line; `current` is now the last line's union

    // 4. Populate cell metadata. num_lines >= 1 here, so the means are safe.
    cell.num_of_lines = num_lines;
    cell.text_line_height = static_cast<int>(sum_h / num_lines);
    cell.text_line_width = static_cast<int>(sum_w / num_lines);

    // 5. Segment start/end along the primary axis: left/top edge of the FIRST
    //    line and right/bottom edge of the LAST line.
    const TextLineAcc &last_line = current;
    if (horizontal) {
      cell.seg_start_coordinate = first_line.x0;
      cell.seg_end_coordinate = last_line.x1;
    } else {
      cell.seg_start_coordinate = first_line.y0;
      cell.seg_end_coordinate = last_line.y1;
    }
  }
}

Direction infer_page_direction(const std::vector<LayoutBox> &layout) {
  int horizontal = 0;
  int total = 0;
  for (const auto &lb : layout) {
    // Only `text`-class cells (class_id 22) participate in page-level
    // direction vote — PaddleX's update_region_label only counts
    // normal_text_block_idxes, which excludes header/footer/title/etc.
    if (lb.class_id != 22) continue;
    ++total;
    if (lb.direction == Direction::kHorizontal) ++horizontal;
  }
  if (total == 0) return Direction::kHorizontal;
  return horizontal >= (total + 1) / 2 ? Direction::kHorizontal
                                       : Direction::kVertical;
}

} // namespace turbo_ocr::layout
