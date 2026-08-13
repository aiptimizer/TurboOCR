#include <catch_amalgamated.hpp>

#include "turbo_ocr/analysis/table/cell_matcher.h"

using turbo_ocr::table::match_cells_to_ocr;
using turbo_ocr::table::OcrLine;
using turbo_ocr::table::quad_to_bbox;

namespace {

OcrLine line(float x1, float y1, float x2, float y2, const char* t) {
    return OcrLine{{x1, y1, x2, y2}, t};
}

std::array<int, 8> quad(int x1, int y1, int x2, int y2) {
    return {x1, y1, x2, y1, x2, y2, x1, y2};
}

} // namespace

TEST_CASE("quad_to_bbox picks extremes", "[cell_matcher]") {
    std::array<int, 8> q = {10, 20, 100, 22, 110, 80, 8, 78};
    auto b = quad_to_bbox(q);
    REQUIRE(b[0] == 8.0f);
    REQUIRE(b[1] == 20.0f);
    REQUIRE(b[2] == 110.0f);
    REQUIRE(b[3] == 80.0f);
}

TEST_CASE("matches only when intersect exceeds threshold", "[cell_matcher]") {
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 100, 100)};
    std::vector<OcrLine> ocr = {
        line(10.0f, 10.0f, 90.0f, 90.0f, "inside"),
        line(95.0f, 95.0f, 195.0f, 195.0f, "barely"),
    };
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m.size() == 1);
    REQUIRE(m[0].ocr_indices == std::vector<std::size_t>{0});
}

TEST_CASE("unmatched cell returns empty indices", "[cell_matcher]") {
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 50, 50)};
    std::vector<OcrLine> ocr = {line(200.0f, 200.0f, 300.0f, 300.0f, "far")};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m[0].ocr_indices.empty());
}

TEST_CASE("a line below threshold in a neighbor lands in the dominant cell only",
          "[cell_matcher]") {
    // Two adjacent cells. The line sits MOSTLY in the left cell (~0.67 of its area)
    // and only ~0.33 in the right — below the 0.4 threshold — so it goes left only.
    std::vector<std::array<int, 8>> cells = {
        quad(0, 0, 100, 100), quad(100, 0, 200, 100)};
    std::vector<OcrLine> ocr = {line(60.0f, 10.0f, 120.0f, 90.0f, "straddle")};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m.size() == 2);
    REQUIRE(m[0].ocr_indices == std::vector<std::size_t>{0});
    REQUIRE(m[1].ocr_indices.empty());
}

TEST_CASE("a line clearing the threshold in two OVERLAPPING cells lands in both "
          "(multi-cell)", "[cell_matcher]") {
    // PaddleX match_table_and_ocr places a line in EVERY cell it clears the
    // threshold for. At the 0.5 default a line split across two NON-overlapping
    // cells can clear 0.5 in at most one (the two fractions sum to <=1), so
    // multi-cell only happens where the cell quads OVERLAP — the realistic
    // SLANeXt case. Here both cells cover the line by ~0.83, so it lands in both.
    std::vector<std::array<int, 8>> cells = {
        quad(0, 0, 120, 100), quad(80, 0, 200, 100)};  // overlap x:80..120
    std::vector<OcrLine> ocr = {line(70.0f, 10.0f, 130.0f, 90.0f, "shared")};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m.size() == 2);
    REQUIRE(m[0].ocr_indices == std::vector<std::size_t>{0});
    REQUIRE(m[1].ocr_indices == std::vector<std::size_t>{0});
}

TEST_CASE("at the 0.5 default a boundary-centred line over non-overlapping cells "
          "lands in the dominant cell only", "[cell_matcher]") {
    // A line exactly centred on the shared edge has ~0.5 of its area in each of
    // two non-overlapping cells; neither clears the strict >0.5 gate, so the
    // argmax fallback assigns it to a single cell rather than duplicating it.
    std::vector<std::array<int, 8>> cells = {
        quad(0, 0, 100, 100), quad(100, 0, 200, 100)};
    std::vector<OcrLine> ocr = {line(50.0f, 10.0f, 150.0f, 90.0f, "boundary")};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m.size() == 2);
    const std::size_t total =
        m[0].ocr_indices.size() + m[1].ocr_indices.size();
    REQUIRE(total == 1);  // assigned once, not duplicated
}

TEST_CASE("rtree handles many lines efficiently", "[cell_matcher]") {
    std::vector<OcrLine> ocr;
    ocr.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        const float x = static_cast<float>(i % 50) * 20.0f;
        const float y = static_cast<float>(i / 50) * 20.0f;
        ocr.push_back(line(x, y, x + 10.0f, y + 10.0f, "x"));
    }
    std::vector<std::array<int, 8>> cells = {quad(0, 0, 100, 100)};
    auto m = match_cells_to_ocr(cells, ocr);
    REQUIRE(m[0].ocr_indices.size() == 25);
}
