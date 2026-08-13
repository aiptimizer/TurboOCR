// slanext_postprocess_region — THE shared SLANeXt table policy (the body both
// the TRT and CPU recognizers were carrying as verbatim private copies until
// it was extracted; see slanext_postprocess.h). These tests pin the behaviors
// the extraction asserted were identical, so the next "small fix" to the
// policy is made once and verified here rather than re-forked per backend.

#include <catch_amalgamated.hpp>

#include <array>
#include <string>
#include <vector>

#include "turbo_ocr/core/types.h"
#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"
#include "turbo_ocr/analysis/table/table_types.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::table::slanext_postprocess_region;
using turbo_ocr::table::SlanextCellRecFn;
using turbo_ocr::table::StructureCell;
using turbo_ocr::table::StructureResult;

namespace {

Box make_box(int x1, int y1, int x2, int y2) {
  Box b;
  b.pts = {{{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}}};
  return b;
}

OCRResultItem ocr_item(const std::string &text, int x1, int y1, int x2, int y2) {
  OCRResultItem r{};
  r.text = text;
  r.confidence = 0.99f;
  r.box = make_box(x1, y1, x2, y2);
  return r;
}

// A minimal two-cell, one-row structure the way decode_structure would emit it:
// full wrap + one <tr> + two <td></td> tokens, one StructureCell per <td>,
// bboxes REGION-LOCAL (the function under test shifts them to page coords).
StructureResult two_cell_structure() {
  StructureResult sr;
  sr.structure = {"<html>", "<body>", "<table>", "<tr>",
                  "<td></td>", "<td></td>", "</tr>",
                  "</table>", "</body>", "</html>"};
  sr.cells.push_back(StructureCell{{0, 0, 50, 0, 50, 20, 0, 20}});
  sr.cells.push_back(StructureCell{{60, 0, 110, 0, 110, 20, 60, 20}});
  sr.structure_score = 0.9f;
  return sr;
}

// Same shape with THREE cells in the one row, so a crop-OCR result dropped in
// the middle of the batch can be told apart from one that shifted the rest of
// the batch onto the wrong cells.
StructureResult three_cell_structure() {
  StructureResult sr;
  sr.structure = {"<html>", "<body>", "<table>", "<tr>",
                  "<td></td>", "<td></td>", "<td></td>", "</tr>",
                  "</table>", "</body>", "</html>"};
  sr.cells.push_back(StructureCell{{0, 0, 50, 0, 50, 20, 0, 20}});
  sr.cells.push_back(StructureCell{{60, 0, 110, 0, 110, 20, 60, 20}});
  sr.cells.push_back(StructureCell{{120, 0, 170, 0, 170, 20, 120, 20}});
  sr.structure_score = 0.9f;
  return sr;
}

} // namespace

TEST_CASE("region postprocess: quad shift, in-region filter, substitution",
          "[table][slanext]") {
  // Region sits at a NON-ZERO page origin so the region-local -> page-coord
  // quad shift is actually exercised.
  const Box region = make_box(100, 200, 260, 240);
  const auto sr = two_cell_structure();

  const std::vector<OCRResultItem> page_ocr = {
      // centred inside cell 0 after the +100/+200 shift
      ocr_item("alpha", 105, 205, 145, 218),
      // centred inside cell 1
      ocr_item("beta", 165, 205, 205, 218),
      // OUTSIDE the region entirely — must be filtered before matching
      ocr_item("noise", 500, 500, 540, 515),
  };

  const auto tr = slanext_postprocess_region(sr, page_ocr, region, {});

  CHECK(tr.layout_id == -1); // stamped by the caller, never here
  CHECK(tr.score == sr.structure_score);
  CHECK(tr.html.find("alpha") != std::string::npos);
  CHECK(tr.html.find("beta") != std::string::npos);
  CHECK(tr.html.find("noise") == std::string::npos);
  // alpha is the left cell: it must come first in the walk order.
  CHECK(tr.html.find("alpha") < tr.html.find("beta"));
  // cells[] built from the same pools as the HTML
  REQUIRE(tr.cells.size() == 2);
  // page-coordinate quads: cell 0 shifted by the region origin
  CHECK(tr.cells[0].box[0][0] == 100);
  CHECK(tr.cells[0].box[0][1] == 200);
}

TEST_CASE("region postprocess: empty-cell crop-OCR backfill thresholds",
          "[table][slanext]") {
  const Box region = make_box(0, 0, 300, 40);
  StructureResult sr = two_cell_structure();
  // Make cell 1 degenerate (2px wide): even with a cell recognizer it must be
  // SKIPPED by the w/h >= 4 gate and never reach the recognizer.
  sr.cells[1].bbox = {60, 0, 62, 0, 62, 2, 60, 2};

  const std::vector<OCRResultItem> page_ocr = {}; // no page OCR at all

  std::size_t crops_requested = 0;
  SlanextCellRecFn cell_rec = [&](const std::vector<Box> &empty_cells) {
    crops_requested = empty_cells.size();
    std::vector<std::pair<std::string, float>> out;
    out.reserve(empty_cells.size());
    for (std::size_t i = 0; i < empty_cells.size(); ++i)
      out.emplace_back("filled", 0.9f);
    return out;
  };

  const auto tr = slanext_postprocess_region(sr, page_ocr, region, cell_rec);

  // Only the non-degenerate cell 0 was offered for crop OCR.
  CHECK(crops_requested == 1);
  CHECK(tr.html.find("filled") != std::string::npos);

  // The 0.5-confidence floor: a low-confidence crop result is dropped.
  SlanextCellRecFn low_conf = [](const std::vector<Box> &cells) {
    return std::vector<std::pair<std::string, float>>(
        cells.size(), {"garbage", 0.4f});
  };
  const auto tr2 =
      slanext_postprocess_region(two_cell_structure(), {}, region, low_conf);
  CHECK(tr2.html.find("garbage") == std::string::npos);
}

TEST_CASE("region postprocess: crop-OCR text lands in the cell it was cropped from",
          "[table][slanext]") {
  // Two NON-degenerate empty cells, so more than one crop is offered and the
  // empty-cell -> cell-index association is actually observable. The recognizer
  // derives its answer from the quad it was handed, so a swapped/reversed
  // mapping shows up as text in the wrong cell rather than as "some text".
  // Non-zero region origin: the crops must be offered in PAGE coordinates, so
  // cell 0 spans x 100-150 and cell 1 spans x 160-210.
  const Box region = make_box(100, 200, 400, 240);

  std::vector<Box> offered;
  SlanextCellRecFn cell_rec = [&](const std::vector<Box> &empty_cells) {
    offered = empty_cells;
    std::vector<std::pair<std::string, float>> out;
    out.reserve(empty_cells.size());
    for (const auto &b : empty_cells)
      out.emplace_back(b.pts[0][0] < 155 ? "LEFT" : "RIGHT", 0.9f);
    return out;
  };

  const auto tr = slanext_postprocess_region(two_cell_structure(),
                                             /*page_ocr=*/{}, region, cell_rec);

  // Both cells are empty and pass the 4px gate, so both are offered — in cell
  // order, already shifted to page coords (a region-local crop would OCR the
  // wrong pixels).
  REQUIRE(offered.size() == 2);
  CHECK(offered[0].pts[0][0] == 100);
  CHECK(offered[0].pts[0][1] == 200);
  CHECK(offered[1].pts[0][0] == 160);

  // Each recovered string comes back to the cell its crop came from.
  REQUIRE(tr.cells.size() == 2);
  CHECK(tr.cells[0].text == "LEFT");
  CHECK(tr.cells[1].text == "RIGHT");
  const std::size_t left = tr.html.find("LEFT");
  const std::size_t right = tr.html.find("RIGHT");
  REQUIRE(left != std::string::npos);
  REQUIRE(right != std::string::npos);
  CHECK(left < right);
}

TEST_CASE("region postprocess: crop results land on NON-CONTIGUOUS empty cells",
          "[table][slanext]") {
  // The case above has every cell empty, so the k-th crop belongs to cell k and
  // the empty-cell index list is indistinguishable from the loop counter. Here
  // the MIDDLE cell is already filled by page OCR, so the empty cells are 0 and
  // 2: the k-th crop result belongs to cell empty_ci[k], not to cell k. Walking
  // the results with the crop counter instead appends "RIGHT" to the middle
  // cell (which already has text) and leaves the right-hand cell blank.
  const Box region = make_box(100, 200, 400, 240);
  // Page coords after the +100/+200 shift: cell 0 x 100-150, cell 1 x 160-210,
  // cell 2 x 220-270, all y 200-220. This line sits wholly inside cell 1, so it
  // clears the matcher's overlap threshold there and nowhere else.
  const std::vector<OCRResultItem> page_ocr = {
      ocr_item("MIDDLE", 165, 204, 205, 216)};

  std::vector<Box> offered;
  SlanextCellRecFn cell_rec = [&](const std::vector<Box> &empty_cells) {
    offered = empty_cells;
    std::vector<std::pair<std::string, float>> out;
    out.reserve(empty_cells.size());
    for (const auto &b : empty_cells)
      out.emplace_back(b.pts[0][0] < 155 ? "LEFT" : "RIGHT", 0.9f);
    return out;
  };

  const auto tr =
      slanext_postprocess_region(three_cell_structure(), page_ocr, region, cell_rec);

  // The cell page OCR already filled is not offered for crop OCR at all.
  REQUIRE(offered.size() == 2);
  CHECK(offered[0].pts[0][0] == 100); // cell 0
  CHECK(offered[1].pts[0][0] == 220); // cell 2 — the index gap

  REQUIRE(tr.cells.size() == 3);
  CHECK(tr.cells[0].text == "LEFT");
  CHECK(tr.cells[1].text == "MIDDLE"); // page OCR text, not "MIDDLE RIGHT"
  CHECK(tr.cells[2].text == "RIGHT");
  const std::size_t left = tr.html.find("LEFT");
  const std::size_t middle = tr.html.find("MIDDLE");
  const std::size_t right = tr.html.find("RIGHT");
  REQUIRE(left != std::string::npos);
  REQUIRE(middle != std::string::npos);
  REQUIRE(right != std::string::npos);
  CHECK(left < middle);
  CHECK(middle < right);
}

TEST_CASE("region postprocess: a dropped crop result does not shift the others",
          "[table][slanext]") {
  // Three empty cells; the MIDDLE crop comes back below the 0.5 confidence
  // floor. The drop must consume that cell's slot too: if the accepted results
  // were walked with their own cursor, "gamma" would land in cell 1.
  const Box region = make_box(0, 0, 300, 40);
  std::size_t crops_requested = 0;
  SlanextCellRecFn cell_rec = [&](const std::vector<Box> &cells) {
    crops_requested = cells.size();
    std::vector<std::pair<std::string, float>> out;
    out.emplace_back("alpha", 0.9f);
    out.emplace_back("beta", 0.4f); // below the floor — dropped
    out.emplace_back("gamma", 0.9f);
    return out;
  };

  const auto tr = slanext_postprocess_region(three_cell_structure(),
                                             /*page_ocr=*/{}, region, cell_rec);

  REQUIRE(crops_requested == 3);
  REQUIRE(tr.cells.size() == 3);
  CHECK(tr.cells[0].text == "alpha");
  CHECK(tr.cells[1].text.empty());
  CHECK(tr.cells[2].text == "gamma");
  CHECK(tr.html.find("beta") == std::string::npos);
  CHECK(tr.html.find("alpha") < tr.html.find("gamma"));
}

TEST_CASE("region postprocess: no cell recognizer means no backfill",
          "[table][slanext]") {
  const auto tr = slanext_postprocess_region(
      two_cell_structure(), {}, make_box(0, 0, 300, 40), /*cell_rec=*/{});
  // Structure survives; the cells simply stay empty.
  CHECK(tr.html.find("<table>") != std::string::npos);
  CHECK(tr.cells.size() == 2);
}
