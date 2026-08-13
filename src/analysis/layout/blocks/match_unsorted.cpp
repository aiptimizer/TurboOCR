#include "turbo_ocr/analysis/layout/blocks/match_unsorted.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace turbo_ocr::layout {

OrderLabel order_label_for(int class_id) noexcept {
  switch (class_id) {
    case 6:  // doc_title
      return OrderLabel::kDocTitle;
    case 17: // paragraph_title
      return OrderLabel::kParagraphTitle;
    case 7:  // figure_title
      return OrderLabel::kVisionTitle;
    case 14: // image
    case 21: // table
    case 3:  // chart
      return OrderLabel::kVision;
    case 18: // reference
    case 19: // reference_content
    case 10: // footnote
    case 24: // vision_footnote
      return OrderLabel::kCrossReference;
    case 2:  // aside_text
    case 16: // number (page number)
    case 11: // formula_number
    case 20: // seal
      return OrderLabel::kUnordered;
    default:
      return OrderLabel::kBody;
  }
}

namespace {

// 1D projection overlap ratio along an axis. Mirrors
// paddlex.layout_parsing.utils.calculate_projection_overlap_ratio with
// mode="iou": intersection / union of the projected segments.
inline float projection_overlap(int a0, int a1, int b0, int b1) noexcept {
  const int inter = std::max(0, std::min(a1, b1) - std::max(a0, b0));
  if (inter <= 0) return 0.0f;
  const int uni = std::max(a1, b1) - std::min(a0, b0);
  return uni > 0 ? float(inter) / float(uni) : 0.0f;
}

// get_nearest_edge_distance(bbox1, bbox2, weight)
// weight = [left, right, up, down] — applied to the corresponding gap.
//
// If the boxes overlap on BOTH axes (horizontal_iou>0 AND vertical_iou>0)
// the distance is 0. Otherwise we take the gap on each non-overlapping
// axis (multiplied by the directional weight) and sum.
inline float nearest_edge_distance(const std::array<int, 4> &a,
                                   const std::array<int, 4> &b,
                                   const std::array<float, 4> &w) noexcept {
  const auto [x1, y1, x2, y2] = a;
  const auto [x1p, y1p, x2p, y2p] = b;
  const float h_iou = projection_overlap(x1, x2, x1p, x2p);
  const float v_iou = projection_overlap(y1, y2, y1p, y2p);
  if (h_iou > 0 && v_iou > 0) return 0.0f;

  float dx = 0.0f, dy = 0.0f;
  if (h_iou == 0.0f) {
    const int gap = std::min(std::abs(x1 - x2p), std::abs(x2 - x1p));
    dx = static_cast<float>(gap) * (x2 < x1p ? w[0] : w[1]);
  }
  if (v_iou == 0.0f) {
    const int gap = std::min(std::abs(y1 - y2p), std::abs(y2 - y1p));
    dy = static_cast<float>(gap) * (y2 < y1p ? w[2] : w[3]);
  }
  return dx + dy;
}

// Manhattan distance between bbox top-left corners.
inline float manhattan_dist(const std::array<int, 4> &a,
                            const std::array<int, 4> &b) noexcept {
  return float(std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]));
}

// Squared distance from origin to the centroid of a bbox.
inline double centroid_l2_sq(const std::array<int, 4> &a) noexcept {
  const double cx = 0.5 * (a[0] + a[2]);
  const double cy = 0.5 * (a[1] + a[3]);
  return cx * cx + cy * cy;
}

// _get_weights(label, direction) for the four insertion gap-direction
// weights. Mirrors xycut_enhanced/utils.py:_get_weights.
inline std::array<float, 4> weights_for(OrderLabel ol,
                                         Direction dir) noexcept {
  switch (ol) {
    case OrderLabel::kDocTitle:
      return dir == Direction::kHorizontal
                 ? std::array<float, 4>{1.0f, 0.1f, 0.1f, 1.0f}
                 : std::array<float, 4>{0.2f, 0.1f, 1.0f, 1.0f};
    case OrderLabel::kParagraphTitle:
    case OrderLabel::kVisionTitle:
    case OrderLabel::kVision:
      return {1.0f, 1.0f, 0.1f, 1.0f};
    default:
      return {1.0f, 1.0f, 1.0f, 0.1f};
  }
}

constexpr float kEdgeDistanceTolerance = 2.0f;
constexpr float kEdgeWeight = 1e4f;
constexpr float kUpEdgeWeight = 1.0f;
constexpr float kLeftEdgeWeight = 1e-4f;

// A layout_idx is a valid subscript into `layout` iff it is non-negative
// and in range. Orphan/synthetic entries carry negative sentinels.
inline bool in_layout_range(int idx, const std::vector<LayoutBox> &layout) noexcept {
  return idx >= 0 && static_cast<size_t>(idx) < layout.size();
}

// reference_insert: pick the highest sorted block that is fully ABOVE
// `block` (sorted_block.bbox[3] <= block.bbox[1]) and insert AFTER it.
// Used for cross-references / footnotes that should appear at the END
// of whatever section they textually belong to.
void reference_insert(const UnsortedBlock &block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }
  float min_distance = std::numeric_limits<float>::infinity();
  size_t nearest = 0;
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const auto &sb = sorted_blocks[i];
    if (sb.aabb[3] > block.aabb[1]) continue;
    const float distance = -float(sb.aabb[2]) * 10.0f - float(sb.aabb[3]);
    if (distance < min_distance) {
      min_distance = distance;
      nearest = i;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + nearest + 1, block);
}

// manhattan_insert: nearest by Manhattan-distance between top-left
// corners. Used for `unordered` (aside_text, page numbers, seals).
void manhattan_insert(const UnsortedBlock &block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }
  float min_distance = std::numeric_limits<float>::infinity();
  size_t nearest = 0;
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const float d = manhattan_dist(block.aabb, sorted_blocks[i].aabb);
    if (d < min_distance) {
      min_distance = d;
      nearest = i;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + nearest + 1, block);
}

// euclidean_insert: place by ascending centroid distance from origin.
// Used for whole-region blocks where simple radial ordering is enough.
void euclidean_insert(const UnsortedBlock &block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  const double block_d = centroid_l2_sq(block.aabb);
  size_t pos = sorted_blocks.size();
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    if (centroid_l2_sq(sorted_blocks[i].aabb) > block_d) {
      pos = i;
      break;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + pos, block);
}

// weighted_distance_insert: the heavy one. Combines:
//   - edge distance with per-label directional weights
//   - up-edge distance (vertical position)
//   - left-edge distance (horizontal position)
// Each block iteration tracks the minimum aggregate; the chosen
// nearest block determines the insertion point, with a "below-or-above"
// secondary heuristic comparing the candidate's y1/x1 to the sorted
// block's. Used for titles, captions, and vision blocks.
//
// `text_line_width` is the median width of `text`-class blocks on the
// page, used to scale the doc_title disperse tolerance — without it the
// disperse term collapses to 0 and a doc_title's edge_distance ties
// with the wrong neighbour, landing it mid-paragraph.
//
// `direction` flips the up-edge / left-edge axis assignments so the
// algorithm handles vertical-direction (CJK tategaki) layouts. For
// vertical pages, the "primary" axis is x (right-to-left columns,
// top-to-bottom within each).
//
// `layout` is consulted via get_seg_flag for the vision step-back
// branch — when an adjacent neighbour's last line continues into the
// current vision block (paragraph wrap), step back so the figure
// doesn't split the paragraph.
void weighted_distance_insert(const UnsortedBlock &block,
                              std::vector<UnsortedBlock> &sorted_blocks,
                              int text_line_width,
                              Direction direction,
                              const std::vector<LayoutBox> &layout) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }

  // Everything about `block` is constant across the scan, so its geometry,
  // its label class, and its origin-centroid are computed exactly once
  // rather than re-derived for every candidate comparison.
  const auto weights = weights_for(block.order_label, direction);
  const bool horizontal = direction == Direction::kHorizontal;
  const int x1 = block.aabb[0], y1 = block.aabb[1], x2 = block.aabb[2];
  const double block_centroid = centroid_l2_sq(block.aabb);
  const bool label_flips_below =
      block.order_label == OrderLabel::kDocTitle ||
      block.order_label == OrderLabel::kParagraphTitle ||
      block.order_label == OrderLabel::kVisionTitle ||
      block.order_label == OrderLabel::kVision;
  const bool label_is_vision =
      block.order_label == OrderLabel::kVision ||
      block.order_label == OrderLabel::kVisionTitle;

  // Disperse term: doc_title gets a tolerance of max(2, text_line_width)
  // px so up-edge distance ties cleanly across same-row neighbours
  // (mirrors PaddleX disperse = max(1, region.text_line_width)).
  float tolerance_len = kEdgeDistanceTolerance;
  if (block.order_label == OrderLabel::kDocTitle && text_line_width > 0) {
    tolerance_len = std::max(tolerance_len, float(text_line_width));
  }

  float min_weighted = std::numeric_limits<float>::infinity();
  float min_up_edge = std::numeric_limits<float>::infinity();
  size_t nearest = 0;

  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const auto &sb = sorted_blocks[i];
    const auto [x1p, y1p, x2p, y2p] = sb.aabb;

    const float edge_distance =
        nearest_edge_distance(block.aabb, sb.aabb, weights);
    // PaddleX's primary "up_edge_distance" is y1' for horizontal,
    // -x2' for vertical (a column closer to the right edge has a
    // SMALLER -x2 → reads earlier).
    float up_edge = horizontal ? float(y1p) : -float(x2p);
    float left_edge = horizontal ? float(x1p) : float(y1p);
    const bool is_below_sorted = horizontal ? y2p < y1 : x1p > x2;

    if (label_flips_below && is_below_sorted) {
      up_edge = -up_edge;
      left_edge = -left_edge;
    }

    if (std::abs(min_up_edge - up_edge) <= tolerance_len) {
      up_edge = min_up_edge;
    }

    const float weighted = edge_distance * kEdgeWeight +
                           up_edge * kUpEdgeWeight +
                           left_edge * kLeftEdgeWeight;
    if (up_edge < min_up_edge) min_up_edge = up_edge;

    if (weighted < min_weighted) {
      min_weighted = weighted;
      nearest = i;

      int sorted_distance, block_distance;
      if (std::abs(y1 / 2 - y1p / 2) > 0) {
        sorted_distance = y1p;
        block_distance = y1;
      // The x-gates compare the BLOCK against the SORTED block (mirroring the
      // y-gate above): left edges for horizontal text, right edges for
      // vertical. They used to compare the block's own left edge to its own
      // right edge — true for any block wider than a pixel — which made both
      // gates vacuous and the centroid fallback unreachable on y-ties.
      } else if (horizontal && std::abs(x1 / 2 - x1p / 2) > 0) {
        sorted_distance = x1p;
        block_distance = x1;
      } else if (!horizontal && std::abs(x2 - x2p) > 0) {
        sorted_distance = -x2p;
        block_distance = -x2;
      } else {
        sorted_distance = static_cast<int>(centroid_l2_sq(sb.aabb));
        block_distance  = static_cast<int>(block_centroid);
      }
      if (block_distance > sorted_distance) {
        nearest = i + 1;
        // Vision look-ahead: if the block we're inserting AFTER (sb)
        // and the next block in sorted (i+1) form a paragraph
        // continuation, bump past the next block so the figure
        // doesn't split mid-paragraph.
        if (label_is_vision && i + 1 < sorted_blocks.size()) {
          const int next_idx = sorted_blocks[i + 1].layout_idx;
          if (in_layout_range(next_idx, layout) &&
              in_layout_range(sb.layout_idx, layout)) {
            const SegFlag sf = get_seg_flag(
                layout[static_cast<size_t>(next_idx)],
                layout[static_cast<size_t>(sb.layout_idx)],
                direction);
            if (!sf.seg_start_flag) ++nearest;
          }
        }
      } else if (label_is_vision && i > 0) {
        // Vision look-behind: only step into the previous slot when
        // the current sorted block is mid-paragraph relative to its
        // predecessor (i.e. not a clean paragraph start).
        const int curr_idx = sb.layout_idx;
        const int prev_idx = sorted_blocks[i - 1].layout_idx;
        if (in_layout_range(curr_idx, layout) &&
            in_layout_range(prev_idx, layout)) {
          const SegFlag sf = get_seg_flag(
              layout[static_cast<size_t>(curr_idx)],
              layout[static_cast<size_t>(prev_idx)],
              direction);
          if (!sf.seg_start_flag) nearest = i - 1;
        }
      }
    }
  }
  if (nearest > sorted_blocks.size()) nearest = sorted_blocks.size();
  sorted_blocks.insert(sorted_blocks.begin() +
                           static_cast<std::ptrdiff_t>(nearest),
                       block);
}

} // namespace

SegFlag get_seg_flag(const LayoutBox &current,
                     const LayoutBox &prev,
                     Direction direction) {
  // Default: every block is its own paragraph (start AND end). We
  // flip the flags only when the cluster pre-pass has populated
  // text-line metadata — without it (no detection results inside the
  // cell) we can't infer continuation.
  SegFlag out;

  // For paragraph-continuation we need the prev block to actually
  // BE a multi-line paragraph and to have populated its seg_*_
  // coordinate fields.
  if (prev.num_of_lines <= 1 || current.num_of_lines == 0) return out;
  if (prev.text_line_height <= 0) return out;

  // Use the same direction's primary axis for both. Mismatch ⇒ keep
  // defaults (we can't reason about cross-direction continuation).
  if (prev.direction != direction || current.direction != direction) {
    return out;
  }

  // Tolerance: PaddleX uses 10 pixels but scales loosely with line
  // height. Stick with the absolute 10-px threshold — that's what
  // appears as `< 10` literals in get_seg_flag.
  constexpr int kEdgeTolerance = 10;

  if (direction == Direction::kHorizontal) {
    auto [prev_x0, prev_y0, prev_x1, prev_y1] = turbo_ocr::aabb(prev.box);
    auto [cur_x0,  cur_y0,  cur_x1,  cur_y1]  = turbo_ocr::aabb(current.box);
    // prev's last line ends within `tol` of its right margin?
    const bool prev_full_right =
        std::abs(prev.seg_end_coordinate - prev_x1) < kEdgeTolerance;
    // current's first line starts within `tol` of its left margin?
    const bool cur_full_left =
        std::abs(current.seg_start_coordinate - cur_x0) < kEdgeTolerance;
    if (prev_full_right && cur_full_left) {
      out.seg_start_flag = false;  // current continues prev's paragraph
    }
    // current itself ends near its right margin → mid-paragraph at end.
    const bool cur_full_right =
        std::abs(current.seg_end_coordinate - cur_x1) < kEdgeTolerance;
    if (cur_full_right && current.num_of_lines > 1) {
      out.seg_end_flag = false;
    }
  } else {
    auto [prev_x0, prev_y0, prev_x1, prev_y1] = turbo_ocr::aabb(prev.box);
    auto [cur_x0,  cur_y0,  cur_x1,  cur_y1]  = turbo_ocr::aabb(current.box);
    const bool prev_full_bottom =
        std::abs(prev.seg_end_coordinate - prev_y1) < kEdgeTolerance;
    const bool cur_full_top =
        std::abs(current.seg_start_coordinate - cur_y0) < kEdgeTolerance;
    if (prev_full_bottom && cur_full_top) {
      out.seg_start_flag = false;
    }
    const bool cur_full_bottom =
        std::abs(current.seg_end_coordinate - cur_y1) < kEdgeTolerance;
    if (cur_full_bottom && current.num_of_lines > 1) {
      out.seg_end_flag = false;
    }
  }
  return out;
}

void match_unsorted_block(const UnsortedBlock &block,
                          std::vector<UnsortedBlock> &sorted_blocks,
                          int text_line_width,
                          Direction direction,
                          const std::vector<LayoutBox> &layout) {
  switch (block.order_label) {
    case OrderLabel::kCrossReference:
      reference_insert(block, sorted_blocks);
      break;
    case OrderLabel::kUnordered:
      manhattan_insert(block, sorted_blocks);
      break;
    case OrderLabel::kDocTitle:
    case OrderLabel::kParagraphTitle:
    case OrderLabel::kVisionTitle:
    case OrderLabel::kVision:
      weighted_distance_insert(block, sorted_blocks, text_line_width,
                                direction, layout);
      break;
    case OrderLabel::kBody:
    default:
      euclidean_insert(block, sorted_blocks);
      break;
  }
}

void match_unsorted_blocks(std::vector<UnsortedBlock> &sorted,
                           std::vector<UnsortedBlock> &unsorted,
                           int text_line_width,
                           Direction direction,
                           const std::vector<LayoutBox> &layout) {
  // Two-pass to honour the "doc_title pinned to 0 if first" rule: any
  // doc_title in `unsorted` goes through weighted_distance_insert first
  // (the algorithm naturally lands the first one at the top-left).
  std::stable_partition(unsorted.begin(), unsorted.end(),
                        [](const UnsortedBlock &b) {
                          return b.order_label == OrderLabel::kDocTitle;
                        });
  for (auto &b : unsorted) {
    match_unsorted_block(b, sorted, text_line_width, direction, layout);
  }
  unsorted.clear();
}

} // namespace turbo_ocr::layout
