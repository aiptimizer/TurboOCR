#include "turbo_ocr/analysis/table/cell_matcher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "turbo_ocr/base/env_utils.h"

#ifdef __has_include
#  if __has_include(<boost/geometry.hpp>)
#    define TURBO_OCR_HAS_BOOST_GEOMETRY 1
#  endif
#endif

#ifdef TURBO_OCR_HAS_BOOST_GEOMETRY
#  include <boost/geometry.hpp>
#  include <boost/geometry/index/rtree.hpp>
#endif

namespace turbo_ocr::table {

namespace {
// Row-band height for within-cell reading order. Quantizing y into bands of
// this size yields a TOTAL order over (band, x), which std::sort requires —
// the raw tolerance comparison is non-transitive (strict-weak-ordering UB).
constexpr float READING_ORDER_BAND_PX = 8.0f;

inline float box_area(const std::array<float, 4>& b) {
    return std::max(b[2] - b[0], 0.0f) * std::max(b[3] - b[1], 0.0f);
}

// Fraction of `box` (whose area is passed in precomputed) lying inside `region`
// — PaddleX's inter_area / box_b_area. Split out so the box area, which is
// invariant across every cell a given line is tested against, is computed once
// per line instead of once per (cell, line) pair. Bit-identical to the inline
// form: same intersection edges, same divisor bits.
inline float overlap_fraction(const std::array<float, 4>& region,
                              const std::array<float, 4>& box,
                              float box_a) {
    const float x_left = std::max(region[0], box[0]);
    const float y_top = std::max(region[1], box[1]);
    const float x_right = std::min(region[2], box[2]);
    const float y_bottom = std::min(region[3], box[3]);
    const float iw = std::max(x_right - x_left, 0.0f);
    const float ih = std::max(y_bottom - y_top, 0.0f);
    return box_a > 0.0f ? (iw * ih) / box_a : 0.0f;
}
} // namespace

float compute_inter(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    return overlap_fraction(a, b, box_area(b));
}

std::array<float, 4> quad_to_bbox(const std::array<int, 8>& quad) {
    const int xs[4] = {quad[0], quad[2], quad[4], quad[6]};
    const int ys[4] = {quad[1], quad[3], quad[5], quad[7]};
    int xmin = xs[0], xmax = xs[0];
    int ymin = ys[0], ymax = ys[0];
    for (int i = 1; i < 4; ++i) {
        xmin = std::min(xmin, xs[i]);
        xmax = std::max(xmax, xs[i]);
        ymin = std::min(ymin, ys[i]);
        ymax = std::max(ymax, ys[i]);
    }
    return {static_cast<float>(xmin), static_cast<float>(ymin),
            static_cast<float>(xmax), static_cast<float>(ymax)};
}

namespace {
// Per-cell reading order: top-to-bottom by row band (8px tolerance), then
// left-to-right — so multi-line cell text joins correctly regardless of
// page-global OCR order.
inline bool reading_order_less(const std::vector<OcrLine>& ocr,
                               std::size_t a, std::size_t b) {
    const auto& ba = ocr[a].bbox; const auto& bb = ocr[b].bbox;
    const int ra = static_cast<int>(std::floor(ba[1] / READING_ORDER_BAND_PX));
    const int rb = static_cast<int>(std::floor(bb[1] / READING_ORDER_BAND_PX));
    if (ra != rb) return ra < rb;
    return ba[0] < bb[0];
}

// Overlap fraction a line must have inside a cell quad to be placed there. PaddleX
// uses 0.7, but SLANeXt quads are smaller than DB text-line boxes so 0.7 drops too
// many; 0.5 is the OmniDocBench-125 optimum. Env-tunable (TABLE_MATCH_INTER) for sweeps.
inline float match_threshold() {
    // 0 and below are rejected rather than clamped: an overlap threshold of
    // zero matches every line to every cell, which is not a setting anyone
    // means, so it keeps the default instead.
    static const float t = [] {
        const float v = env::env_float("TABLE_MATCH_INTER", MATCH_INTER_THRESHOLD, 0.0f, 1.0f);
        return v > 0.0f ? v : MATCH_INTER_THRESHOLD;
    }();
    return t;
}
// A line clearing no cell's threshold is rescued into its argmax cell when at least
// this fraction overlaps, so a line straddling internal boundaries is never silently
// dropped; 0 disables the rescue. Env-tunable (TABLE_MATCH_FALLBACK).
inline float fallback_floor() {
    static const float t = [] {
        // Malformed input keeps the default AND SAYS SO: parsing garbage to
        // 0.0f would silently hit the "0 disables the rescue" sentinel on a
        // typo, and a rescue that quietly stopped happening is invisible in the
        // output. env_or does the recording; the parse stays here because the
        // warning is the point.
        const std::string e = env::env_or("TABLE_MATCH_FALLBACK", "");
        if (!e.empty()) {
            char* end = nullptr;
            const float v = std::strtof(e.c_str(), &end);
            if (end != e.c_str() && *end == '\0' && !std::isnan(v)) return v;
            std::cerr << "[cell_matcher] ignoring malformed TABLE_MATCH_FALLBACK=\""
                      << e << "\"\n";
        }
        return 0.15f;
    }();
    return t;
}

// Precompute each line's own area once; it is the divisor in every overlap test
// (both the threshold pass and the argmax rescue), so hoisting it out of the
// O(cells × lines) comparisons is a strict work reduction with identical output.
inline std::vector<float> line_areas(const std::vector<OcrLine>& ocr) {
    std::vector<float> areas(ocr.size());
    for (std::size_t i = 0; i < ocr.size(); ++i) areas[i] = box_area(ocr[i].bbox);
    return areas;
}

// Rescue still-unassigned lines into their argmax cell (>= floor), then sort each
// cell's indices into reading order. PaddleX's match_table_and_ocr places a line in
// EVERY cell it clears the threshold for (one line may land in multiple cells) —
// empirically higher TEDS than forcing one cell. `assigned[i]` marks lines already
// placed during the per-cell threshold pass. Ties resolve to the lowest cell index
// (strict `>`), matching PaddleX's first-wins argmax.
inline void finalize_multicell(const std::vector<OcrLine>& ocr,
                               const std::vector<float>& areas,
                               std::vector<MatchedCell>& out,
                               std::vector<char>& assigned) {
    const float floor = fallback_floor();
    for (std::size_t i = 0; i < ocr.size(); ++i) {
        if (assigned[i]) continue;
        const auto& box = ocr[i].bbox;
        const float area = areas[i];
        float best = 0.0f;
        std::size_t bestc = out.size();
        for (std::size_t c = 0; c < out.size(); ++c) {
            const float r = overlap_fraction(out[c].bbox, box, area);
            if (r > best) { best = r; bestc = c; }
        }
        if (bestc < out.size() && best > floor) {
            out[bestc].ocr_indices.push_back(i);
            assigned[i] = 1;
        }
    }
    for (auto& cell : out) {
        std::sort(cell.ocr_indices.begin(), cell.ocr_indices.end(),
                  [&ocr](std::size_t a, std::size_t b) {
                      return reading_order_less(ocr, a, b);
                  });
    }
}

inline std::vector<MatchedCell> make_cells(
    const std::vector<std::array<int, 8>>& cells) {
    std::vector<MatchedCell> out;
    out.reserve(cells.size());
    for (const auto& quad : cells)
        out.push_back(MatchedCell{quad_to_bbox(quad), {}});
    return out;
}
} // namespace

#ifdef TURBO_OCR_HAS_BOOST_GEOMETRY

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using BPoint = bg::model::point<float, 2, bg::cs::cartesian>;
using BBox = bg::model::box<BPoint>;
using BValue = std::pair<BBox, std::size_t>;
using RTree = bgi::rtree<BValue, bgi::rstar<16>>;

std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr) {
    std::vector<MatchedCell> out = make_cells(cells);
    if (ocr.empty()) return out;

    std::vector<BValue> entries;
    entries.reserve(ocr.size());
    for (std::size_t i = 0; i < ocr.size(); ++i) {
        const auto& b = ocr[i].bbox;
        entries.emplace_back(BBox(BPoint(b[0], b[1]), BPoint(b[2], b[3])), i);
    }
    const RTree tree(entries.begin(), entries.end());
    const std::vector<float> areas = line_areas(ocr);

    // PaddleX match_table_and_ocr semantics: a line goes into EVERY cell where
    // >threshold of its area overlaps (one line may land in multiple cells — this
    // scores higher TEDS than forcing one cell). Unmatched lines are rescued in
    // finalize. The rtree query yields each line once per cell, so no per-cell dedup.
    const float thr = match_threshold();
    std::vector<char> assigned(ocr.size(), 0);
    for (auto& cell : out) {
        const auto& cb = cell.bbox;
        const BBox query(BPoint(cb[0], cb[1]), BPoint(cb[2], cb[3]));
        for (auto it = tree.qbegin(bgi::intersects(query)); it != tree.qend(); ++it) {
            const std::size_t i = it->second;
            if (overlap_fraction(cb, ocr[i].bbox, areas[i]) > thr) {
                cell.ocr_indices.push_back(i);
                assigned[i] = 1;
            }
        }
    }
    finalize_multicell(ocr, areas, out, assigned);
    return out;
}

#else

// Dependency-free uniform-grid spatial index used when Boost.Geometry is absent.
// A line box is registered into every grid bucket its extent overlaps; a cell
// query scans only the buckets its bbox covers. Because bucketing is monotone in
// each coordinate, two boxes whose extents overlap ALWAYS share a bucket, so this
// enumerates the exact same candidate set as an exhaustive scan (and the rtree
// path) — no positive-overlap pair is ever missed. Output is byte-identical to the
// rtree variant: finalize re-sorts each cell, so bucket iteration order is moot.
namespace {
constexpr int GRID_N = 16;
constexpr int GRID_CELLS = GRID_N * GRID_N;

struct Grid {
    float x0{0}, y0{0}, x1{0}, y1{0};
    // CSR layout: `items` holds line indices grouped by bucket; `start[b]..start[b+1]`
    // delimits bucket b. One allocation instead of GRID_CELLS separate vectors.
    std::array<int, GRID_CELLS + 1> start{};
    std::vector<std::size_t> items;

    void range(const std::array<float, 4>& b,
               int& bx0, int& by0, int& bx1, int& by1) const {
        const float dx = std::max(x1 - x0, 1.0f);
        const float dy = std::max(y1 - y0, 1.0f);
        auto bx = [&](float v) {
            return std::clamp(static_cast<int>(((v - x0) / dx) * GRID_N), 0, GRID_N - 1);
        };
        auto by = [&](float v) {
            return std::clamp(static_cast<int>(((v - y0) / dy) * GRID_N), 0, GRID_N - 1);
        };
        bx0 = bx(b[0]); by0 = by(b[1]); bx1 = bx(b[2]); by1 = by(b[3]);
    }

    void build(const std::vector<OcrLine>& ocr) {
        if (ocr.empty()) return;
        x0 = x1 = ocr[0].bbox[0]; y0 = y1 = ocr[0].bbox[1];
        for (const auto& l : ocr) {
            x0 = std::min(x0, l.bbox[0]); y0 = std::min(y0, l.bbox[1]);
            x1 = std::max(x1, l.bbox[2]); y1 = std::max(y1, l.bbox[3]);
        }
        // Counting sort into CSR: tally bucket occupancy, prefix-sum, then scatter.
        std::array<int, GRID_CELLS + 1> count{};
        auto touch = [&](const std::array<float, 4>& b, auto&& fn) {
            int bx0, by0, bx1, by1;
            range(b, bx0, by0, bx1, by1);
            for (int gy = by0; gy <= by1; ++gy)
                for (int gx = bx0; gx <= bx1; ++gx) fn(gy * GRID_N + gx);
        };
        for (const auto& l : ocr) touch(l.bbox, [&](int b) { ++count[b + 1]; });
        for (int i = 0; i < GRID_CELLS; ++i) count[i + 1] += count[i];
        start = count;
        items.resize(count[GRID_CELLS]);
        for (std::size_t i = 0; i < ocr.size(); ++i)
            touch(ocr[i].bbox, [&](int b) { items[count[b]++] = i; });
    }
};
} // namespace

std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr) {
    std::vector<MatchedCell> out = make_cells(cells);
    if (ocr.empty()) return out;

    Grid grid;
    grid.build(ocr);
    const std::vector<float> areas = line_areas(ocr);

    // A line spans multiple buckets, so a per-cell `seen` guard avoids adding it to
    // the SAME cell twice; across cells it may repeat (multi-cell, see rtree variant).
    const float thr = match_threshold();
    std::vector<char> assigned(ocr.size(), 0);
    std::vector<char> seen(ocr.size(), 0);
    for (auto& cell : out) {
        const auto& cb = cell.bbox;
        int bx0, by0, bx1, by1;
        grid.range(cb, bx0, by0, bx1, by1);
        std::fill(seen.begin(), seen.end(), 0);
        for (int gy = by0; gy <= by1; ++gy) {
            for (int gx = bx0; gx <= bx1; ++gx) {
                const int b = gy * GRID_N + gx;
                for (int k = grid.start[b]; k < grid.start[b + 1]; ++k) {
                    const std::size_t i = grid.items[k];
                    if (seen[i]) continue;
                    seen[i] = 1;
                    if (overlap_fraction(cb, ocr[i].bbox, areas[i]) > thr) {
                        cell.ocr_indices.push_back(i);
                        assigned[i] = 1;
                    }
                }
            }
        }
    }
    finalize_multicell(ocr, areas, out, assigned);
    return out;
}

#endif

} // namespace turbo_ocr::table
