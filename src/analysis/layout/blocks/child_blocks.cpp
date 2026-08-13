#include "turbo_ocr/analysis/layout/blocks/child_blocks.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::layout {

namespace {

// Class IDs for the labels we need to test. Mirrors the PaddleX
// BLOCK_LABEL_MAP partitioning used in update_*_child_blocks.
constexpr int kClassDocTitle        = 6;
constexpr int kClassParagraphTitle  = 17;
constexpr int kClassFigureTitle     = 7;
constexpr int kClassImage           = 14;
constexpr int kClassTable           = 21;
constexpr int kClassChart           = 3;
constexpr int kClassText            = 22;

inline bool is_text(int class_id) noexcept {
  return class_id == kClassText;
}
inline bool is_vision(int class_id) noexcept {
  return class_id == kClassImage || class_id == kClassTable ||
         class_id == kClassChart;
}
inline bool is_vision_title(int class_id) noexcept {
  return class_id == kClassFigureTitle;
}
inline bool is_paragraph_title(int class_id) noexcept {
  return class_id == kClassParagraphTitle;
}

struct AABB { int x0, y0, x1, y1; };

inline AABB aabb_of(const LayoutBox &lb) noexcept {
  auto [x0, y0, x1, y1] = turbo_ocr::aabb(lb.box);
  return {x0, y0, x1, y1};
}

inline int area_of(const AABB &b) noexcept {
  return std::max(0, b.x1 - b.x0) * std::max(0, b.y1 - b.y0);
}

inline int short_side(const AABB &b) noexcept {
  return std::min(std::max(0, b.x1 - b.x0), std::max(0, b.y1 - b.y0));
}

inline int long_side(const AABB &b) noexcept {
  return std::max(std::max(0, b.x1 - b.x0), std::max(0, b.y1 - b.y0));
}

inline std::array<int, 2> centroid(const AABB &b) noexcept {
  return {(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2};
}

// Nearest-edge distance between two AABBs along the union of axes
// they don't overlap on. Mirrors PaddleX's get_nearest_edge_distance with
// uniform weights — renamed _unweighted because match_unsorted.cpp carries the
// weighted variant of the same reference helper under the plain name.
// uniform weights = 1.
inline int nearest_edge_distance_unweighted(const AABB &a, const AABB &b) noexcept {
  const bool h_overlap = std::min(a.x1, b.x1) > std::max(a.x0, b.x0);
  const bool v_overlap = std::min(a.y1, b.y1) > std::max(a.y0, b.y0);
  if (h_overlap && v_overlap) return 0;
  int dx = 0, dy = 0;
  if (!h_overlap) dx = std::min(std::abs(a.x0 - b.x1), std::abs(a.x1 - b.x0));
  if (!v_overlap) dy = std::min(std::abs(a.y0 - b.y1), std::abs(a.y1 - b.y0));
  return dx + dy;
}

// IoU over the smaller of the two areas — matches calculate_overlap_ratio
// with mode="small". Returns 0 when either area is zero.
inline float overlap_small(const AABB &a, const AABB &b) noexcept {
  const int ix0 = std::max(a.x0, b.x0);
  const int iy0 = std::max(a.y0, b.y0);
  const int ix1 = std::min(a.x1, b.x1);
  const int iy1 = std::min(a.y1, b.y1);
  if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
  const int inter = (ix1 - ix0) * (iy1 - iy0);
  const int small = std::min(area_of(a), area_of(b));
  return small > 0 ? float(inter) / float(small) : 0.0f;
}

// Read the cluster pre-pass output for a layout cell. When the cell
// has no clustered text lines (no OCR detection landed in it) we
// fall back to the height-over-text_line_height approximation so
// non-text layout cells (image, table, chart) still produce
// sensible line counts for the proximity gates.
inline int num_lines_for(const LayoutBox &lb, int page_text_line_height) noexcept {
  if (lb.num_of_lines > 0) return lb.num_of_lines;
  const int h = std::max(1, lb.box[3][1] - lb.box[0][1]);
  const int tlh = std::max(1, page_text_line_height);
  return std::max(1, h / tlh);
}

// Per-cell text-line-height. Falls back to the page-level mean when
// the cell has no clustered lines (vision blocks etc.).
inline int line_height_for(const LayoutBox &lb,
                            int page_text_line_height) noexcept {
  return lb.text_line_height > 0 ? lb.text_line_height
                                  : std::max(1, page_text_line_height);
}

// A "vertical neighbour" relation — used to pick the prev/post block
// of a parent. Two blocks neighbour each other vertically when their
// horizontal projections overlap by more than the small-area
// threshold (0.1 in PaddleX's XYCUT_SETTINGS).
inline bool horizontally_overlaps(const AABB &a, const AABB &b,
                                   float thresh = 0.1f) noexcept {
  const int ix0 = std::max(a.x0, b.x0);
  const int ix1 = std::min(a.x1, b.x1);
  if (ix1 <= ix0) return false;
  const int inter = ix1 - ix0;
  const int small = std::min(a.x1 - a.x0, b.x1 - b.x0);
  return small > 0 && float(inter) / float(small) > thresh;
}

// Membership set for claimed/child indices. `std::vector<char>` keyed by
// layout index beats a hashed set here: O(1) branch-free probe, contiguous
// (cache-friendly), and no per-page allocator churn. Only membership is ever
// queried, so ordering is irrelevant.
using IndexSet = std::vector<char>;
inline bool contains(const IndexSet &s, int idx) noexcept {
  return idx >= 0 && static_cast<size_t>(idx) < s.size() &&
         s[static_cast<size_t>(idx)] != 0;
}
inline void mark(IndexSet &s, int idx) noexcept {
  if (idx >= 0 && static_cast<size_t>(idx) < s.size())
    s[static_cast<size_t>(idx)] = 1;
}

// Return indices into `candidates` of blocks that sit ABOVE `parent`
// (sorted by descending y2 — closest first) and BELOW `parent`
// (sorted by ascending y0 — closest first), restricted to those that
// horizontally overlap parent.
//
// `boxes` holds the page's AABBs computed once up front, so neither the
// filter nor the two sorts ever reconstruct a box from its 4 corners.
struct PrevPost {
  std::vector<int> prev;
  std::vector<int> post;
};

PrevPost get_nearest_neighbours(const AABB &parent,
                                const std::vector<AABB> &boxes,
                                const std::vector<int> &candidates) {
  PrevPost out;
  for (int idx : candidates) {
    const AABB &cand = boxes[static_cast<size_t>(idx)];
    if (!horizontally_overlaps(parent, cand)) continue;
    if (cand.y1 <= parent.y0) out.prev.push_back(idx);
    else if (cand.y0 >= parent.y1) out.post.push_back(idx);
  }
  std::sort(out.prev.begin(), out.prev.end(), [&](int a, int b) {
    return boxes[static_cast<size_t>(a)].y1 > boxes[static_cast<size_t>(b)].y1;
  });
  std::sort(out.post.begin(), out.post.end(), [&](int a, int b) {
    return boxes[static_cast<size_t>(a)].y0 < boxes[static_cast<size_t>(b)].y0;
  });
  return out;
}

// Mirrors update_doc_title_child_blocks. Children: small adjacent text
// blocks (sub-titles, author lines) and any text fully overlapping the
// title bbox.
void detect_doc_title_children(int parent_idx,
                               const std::vector<LayoutBox> &layout,
                               const std::vector<AABB> &boxes,
                               const std::vector<int> &text_indices,
                               int page_text_line_height,
                               std::vector<int> &children,
                               IndexSet &claimed) {
  const AABB parent = boxes[static_cast<size_t>(parent_idx)];
  const int parent_short = short_side(parent);
  const int parent_long = long_side(parent);
  PrevPost neigh = get_nearest_neighbours(parent, boxes, text_indices);

  auto try_attach = [&](int idx) {
    if (contains(claimed, idx)) return;
    const AABB &cand = boxes[static_cast<size_t>(idx)];
    const int short_s = short_side(cand);
    const int long_s = long_side(cand);
    if (short_s >= parent_short * 4 / 5) return;
    if (long_s >= parent_long && long_s <= parent_long * 3 / 2) return;
    if (num_lines_for(layout[idx], page_text_line_height) >= 3) return;
    const int tlh = line_height_for(layout[idx], page_text_line_height);
    if (nearest_edge_distance_unweighted(parent, cand) >= tlh * 2) return;
    children.push_back(idx);
    mark(claimed, idx);
  };
  if (!neigh.prev.empty()) try_attach(neigh.prev.front());
  if (!neigh.post.empty()) try_attach(neigh.post.front());

  for (int idx : text_indices) {
    if (contains(claimed, idx)) continue;
    if (overlap_small(parent, boxes[static_cast<size_t>(idx)]) > 0.9f) {
      children.push_back(idx);
      mark(claimed, idx);
    }
  }
}

// Mirrors update_paragraph_title_child_blocks. Children: same-class
// adjacent paragraph_titles with similar left edge (sub-headings).
// Uses the MIN of parent and candidate text_line_heights as PaddleX
// does (`min_text_line_height` in the reference) so a tall caption
// next to a short subtitle still passes the proximity gate.
void detect_paragraph_title_children(int parent_idx,
                                     const std::vector<LayoutBox> &layout,
                                     const std::vector<AABB> &boxes,
                                     const std::vector<int> &paragraph_title_indices,
                                     int page_text_line_height,
                                     std::vector<int> &children,
                                     IndexSet &claimed) {
  const AABB parent = boxes[static_cast<size_t>(parent_idx)];
  const int parent_tlh = line_height_for(layout[parent_idx],
                                          page_text_line_height);
  PrevPost neigh =
      get_nearest_neighbours(parent, boxes, paragraph_title_indices);

  auto try_attach_run = [&](const std::vector<int> &run) {
    for (int idx : run) {
      if (idx == parent_idx) continue;
      if (contains(claimed, idx)) break;
      const AABB &cand = boxes[static_cast<size_t>(idx)];
      const int min_tlh = std::min(
          parent_tlh, line_height_for(layout[idx], page_text_line_height));
      if (std::abs(cand.x0 - parent.x0) >= min_tlh * 2) break;
      if (nearest_edge_distance_unweighted(parent, cand) > min_tlh * 3 / 2) break;
      children.push_back(idx);
      mark(claimed, idx);
    }
  };
  try_attach_run(neigh.prev);
  try_attach_run(neigh.post);
}

// Mirrors update_vision_child_blocks. Children: nearby vision_title
// blocks and small adjacent text caption blocks (vision_footnote).
void detect_vision_children(int parent_idx,
                            const std::vector<LayoutBox> &layout,
                            const std::vector<AABB> &boxes,
                            const std::vector<int> &text_indices,
                            const std::vector<int> &vision_title_indices,
                            int page_text_line_height,
                            std::vector<int> &children,
                            IndexSet &claimed) {
  const AABB parent = boxes[static_cast<size_t>(parent_idx)];
  const int parent_short = short_side(parent);
  const int parent_long = long_side(parent);
  const auto parent_c = centroid(parent);

  bool has_vision_footnote = false;

  std::vector<int> ref_indices;
  ref_indices.reserve(text_indices.size() + vision_title_indices.size());
  ref_indices.insert(ref_indices.end(), text_indices.begin(), text_indices.end());
  ref_indices.insert(ref_indices.end(), vision_title_indices.begin(),
                     vision_title_indices.end());

  PrevPost neigh = get_nearest_neighbours(parent, boxes, ref_indices);

  auto consider = [&](int idx) {
    if (contains(claimed, idx)) return false;
    const AABB &cand = boxes[static_cast<size_t>(idx)];
    const int dist = nearest_edge_distance_unweighted(parent, cand);
    const int cls = layout[idx].class_id;
    const int cand_tlh = line_height_for(layout[idx], page_text_line_height);
    if (is_vision_title(cls) && dist <= cand_tlh * 2) {
      children.push_back(idx);
      mark(claimed, idx);
      return true;
    }
    if (is_text(cls) && !has_vision_footnote &&
        long_side(cand) < parent_long &&
        dist <= cand_tlh * 2) {
      const auto cand_c = centroid(cand);
      const int cand_lines = num_lines_for(layout[idx], page_text_line_height);
      const bool tight_caption =
          short_side(cand) < parent_short &&
          long_side(cand) < parent_long / 2 &&
          std::abs(parent_c[0] - cand_c[0]) < 10;
      const bool aligned_left =
          std::abs(parent.x0 - cand.x0) < 10 && cand_lines == 1;
      const bool aligned_right =
          std::abs(parent.x1 - cand.x1) < 10 && cand_lines == 1;
      if (tight_caption || aligned_left || aligned_right) {
        has_vision_footnote = true;
        children.push_back(idx);
        mark(claimed, idx);
        return true;
      }
    }
    return false;
  };

  for (int idx : neigh.prev) { if (!consider(idx)) break; }
  for (int idx : neigh.post) { if (!consider(idx)) break; }

  // Fully-overlapping text ⇒ vision_footnote unconditionally.
  for (int idx : text_indices) {
    if (contains(claimed, idx)) continue;
    if (overlap_small(parent, boxes[static_cast<size_t>(idx)]) > 0.9f) {
      children.push_back(idx);
      mark(claimed, idx);
    }
  }
}

} // namespace

std::vector<ChildLinks>
detect_child_blocks(const std::vector<LayoutBox> &layout,
                    int text_line_height) {
  std::vector<ChildLinks> out(layout.size());
  if (layout.empty() || text_line_height <= 0) return out;

  // Hoist every region's AABB once. Detection then indexes this table in
  // the O(k log k) neighbour sorts and every proximity/overlap test instead
  // of rebuilding a box from its 4 corners on each comparison.
  std::vector<AABB> boxes(layout.size());
  for (size_t i = 0; i < layout.size(); ++i) boxes[i] = aabb_of(layout[i]);

  std::vector<int> text_indices, paragraph_title_indices,
                   vision_title_indices, vision_indices, doc_title_indices;
  for (size_t i = 0; i < layout.size(); ++i) {
    const int cls = layout[i].class_id;
    if (is_text(cls)) text_indices.push_back(int(i));
    else if (is_paragraph_title(cls)) paragraph_title_indices.push_back(int(i));
    else if (is_vision_title(cls)) vision_title_indices.push_back(int(i));
    else if (is_vision(cls)) vision_indices.push_back(int(i));
    else if (cls == kClassDocTitle) doc_title_indices.push_back(int(i));
  }

  IndexSet claimed(layout.size(), 0);

  // Vision parents claim children first — vision_footnote text/captions
  // are the most consequential gluing case in PaddleX's pipeline.
  for (int idx : vision_indices) {
    if (contains(claimed, idx)) continue;
    detect_vision_children(idx, layout, boxes, text_indices,
                           vision_title_indices, text_line_height,
                           out[idx].child_indices, claimed);
  }
  // Paragraph titles next — they may pull other paragraph_titles as
  // sub-headings. A paragraph_title that has already been claimed (e.g.
  // as another title's sub-heading) cannot itself become a parent —
  // mirrors PaddleX's order_label == "sub_paragraph_title" early
  // return.
  for (int idx : paragraph_title_indices) {
    if (contains(claimed, idx)) continue;
    detect_paragraph_title_children(idx, layout, boxes, paragraph_title_indices,
                                    text_line_height, out[idx].child_indices,
                                    claimed);
  }
  // Doc title last — accrues remaining adjacent text (subtitle / author).
  for (int idx : doc_title_indices) {
    if (contains(claimed, idx)) continue;
    detect_doc_title_children(idx, layout, boxes, text_indices,
                              text_line_height, out[idx].child_indices,
                              claimed);
  }

  return out;
}

std::vector<int>
flatten_descendants(int parent_idx,
                    const std::vector<ChildLinks> &links,
                    const std::vector<LayoutBox> &layout) {
  std::vector<int> out;
  if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= layout.size())
    return out;
  if (links.size() != layout.size()) return out;

  // O(1) membership keyed by layout index — no hashing, no rehash growth.
  std::vector<char> visited(layout.size(), 0);
  visited[static_cast<size_t>(parent_idx)] = 1;

  // Iterative DFS: each frame keeps its top-then-left sorted children and a
  // resume cursor so a child's whole sub-tree emits before the next sibling.
  struct Frame {
    int idx = 0;
    std::vector<int> kids;
    size_t cursor = 0;
  };
  auto sorted_kids = [&](int idx) {
    // Decorate each child with its AABB top-left ONCE, then sort the small
    // decorated array. The sort never reconstructs a box in its comparator,
    // and invalid indices are dropped up front so layout[k] is never read
    // out of bounds (the walk below still bounds-checks every child).
    struct Kid { int y0, x0, idx; };
    std::vector<Kid> decorated;
    if (idx >= 0 && static_cast<size_t>(idx) < links.size()) {
      const auto &ci = links[static_cast<size_t>(idx)].child_indices;
      decorated.reserve(ci.size());
      for (int k : ci) {
        if (k < 0 || static_cast<size_t>(k) >= layout.size()) continue;
        auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[k].box);
        decorated.push_back({y0, x0, k});
      }
    }
    std::sort(decorated.begin(), decorated.end(),
              [](const Kid &a, const Kid &b) {
                if (a.y0 != b.y0) return a.y0 < b.y0;
                return a.x0 < b.x0;
              });
    std::vector<int> kids;
    kids.reserve(decorated.size());
    for (const Kid &k : decorated) kids.push_back(k.idx);
    return kids;
  };

  std::vector<Frame> stack;
  stack.push_back({parent_idx, sorted_kids(parent_idx), 0});
  // A valid acyclic walk never stacks more than one frame per region;
  // `visited` already breaks cycles, so this only backstops a bug where it
  // somehow doesn't.
  const size_t depth_limit = layout.size() + 1;
  while (!stack.empty()) {
    if (stack.size() > depth_limit) break;
    Frame &top = stack.back();
    if (top.cursor >= top.kids.size()) {
      stack.pop_back();
      continue;
    }
    const int ci = top.kids[top.cursor++];
    if (ci < 0 || static_cast<size_t>(ci) >= layout.size()) continue;
    if (visited[static_cast<size_t>(ci)]) continue;
    visited[static_cast<size_t>(ci)] = 1;
    out.push_back(ci);
    stack.push_back({ci, sorted_kids(ci), 0});
  }
  return out;
}

void splice_child_blocks(std::vector<UnsortedBlock> &sorted,
                         const std::vector<ChildLinks> &links,
                         const std::vector<LayoutBox> &layout) {
  if (sorted.empty() || links.empty()) return;

  // 1. Mark every index that is a child of some parent — these must NOT
  //    emit standalone in `sorted`. Keyed by layout index for O(1) tests.
  std::vector<char> is_child(layout.size(), 0);
  for (const auto &cl : links) {
    for (int ci : cl.child_indices) {
      if (ci >= 0 && static_cast<size_t>(ci) < is_child.size())
        is_child[static_cast<size_t>(ci)] = 1;
    }
  }

  // 2/3. Emit each surviving (non-child) entry followed by its whole
  //      descendant sub-tree in DFS order (each level top-then-left),
  //      matching flatten_descendants and PaddleX's insert_child_blocks.
  std::vector<UnsortedBlock> out;
  out.reserve(sorted.size());
  std::vector<char> emitted(layout.size(), 0);
  for (const auto &b : sorted) {
    if (b.layout_idx >= 0 && static_cast<size_t>(b.layout_idx) < is_child.size() &&
        is_child[static_cast<size_t>(b.layout_idx)])
      continue;  // standalone child slot — re-emitted under its parent below
    out.push_back(b);
    if (b.layout_idx >= 0 && static_cast<size_t>(b.layout_idx) < layout.size())
      emitted[static_cast<size_t>(b.layout_idx)] = 1;
    if (b.layout_idx < 0 ||
        static_cast<size_t>(b.layout_idx) >= links.size()) continue;
    for (int ci : flatten_descendants(b.layout_idx, links, layout)) {
      auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[ci].box);
      out.push_back({ci, {x0, y0, x1, y1}, OrderLabel::kBody,
                     layout[ci].class_id});
      emitted[static_cast<size_t>(ci)] = 1;
    }
  }

  // Safety net (H1 data-loss): a child removed in step 2 whose parent slot
  // was absent from `sorted` would otherwise silently VANISH. Emit any
  // still-unaccounted child standalone rather than dropping detected content.
  for (size_t ci = 0; ci < is_child.size(); ++ci) {
    if (!is_child[ci] || emitted[ci]) continue;
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[ci].box);
    out.push_back({static_cast<int>(ci), {x0, y0, x1, y1}, OrderLabel::kBody,
                   layout[ci].class_id});
  }
  sorted = std::move(out);
}

} // namespace turbo_ocr::layout
