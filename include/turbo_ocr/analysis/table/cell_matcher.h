#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace turbo_ocr::table {

// One OCR text-line on the original page.
struct OcrLine {
    // Axis-aligned bbox in original-image pixels: [x1, y1, x2, y2].
    std::array<float, 4> bbox{};
    std::string text;
};

// One cell quad with the OCR indices that fall inside it.
struct MatchedCell {
    // Axis-aligned bbox derived from the quad (min/max over x_even/y_odd of
    // the 8 coords) — matches PaddleX's quad→bbox conversion in
    // match_table_and_ocr.
    std::array<float, 4> bbox{};
    // Indices into the input OCR slice. Empty for cells with no match.
    std::vector<std::size_t> ocr_indices;
};

// PaddleX `match_table_and_ocr` uses `compute_inter > 0.7`, but SLANeXt cell
// quads are smaller than the DB text-line boxes, so a correctly-placed line
// often has <70% of its area inside the quad and gets silently dropped (≈35% of
// lines / 45% of cells came out empty on OmniDocBench); lines matching no cell
// fall back to argmax. 0.5 is the measured OmniDocBench-125 optimum: it assigns
// each line to its dominant cell (a line split across two non-overlapping cells
// can clear 0.5 in at most one), and the per-cell crop-OCR backfill recovers any
// cell the page detector under-segmented. Sweep: 0.3→TEDS 0.7605, 0.4→0.7691,
// 0.5→0.7737. Env-tunable via TABLE_MATCH_INTER.
inline constexpr float MATCH_INTER_THRESHOLD = 0.5f;

// PaddleX-equivalent intersect ratio: inter_area / box_b_area, using exclusive
// (x_right - x_left) width/height arithmetic.
float compute_inter(const std::array<float, 4>& a, const std::array<float, 4>& b);

// Quad → axis-aligned bbox over the 8 coords laid out as
// [x1, y1, x2, y2, x3, y3, x4, y4].
std::array<float, 4> quad_to_bbox(const std::array<int, 8>& quad);

// Match OCR lines to cell quads (PaddleX match_table_and_ocr semantics). For each
// cell returns the OCR indices whose box satisfies compute_inter(cell, ocr) >
// threshold (env TABLE_MATCH_INTER, default MATCH_INTER_THRESHOLD). A line may land
// in MULTIPLE cells it clears the threshold for — at the 0.5 default this happens
// only where cell quads OVERLAP (a line split across two non-overlapping cells can
// clear 0.5 in at most one); lines clearing no cell are rescued into their argmax.
std::vector<MatchedCell> match_cells_to_ocr(
    const std::vector<std::array<int, 8>>& cells,
    const std::vector<OcrLine>& ocr);

} // namespace turbo_ocr::table
