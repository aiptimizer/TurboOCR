#include <catch_amalgamated.hpp>

#include <array>
#include <string>
#include <vector>

#include "turbo_ocr/serialization/serialization_items.h"
#include "turbo_ocr/analysis/table/table_cells.h"

using turbo_ocr::router::TableCell;
using turbo_ocr::router::TableResult;
using turbo_ocr::table::build_table_cells;
using turbo_ocr::table::MatchedCell;

namespace {

// Minimal wrapped structure stream, matching what decode_structure emits.
std::vector<std::string> wrap(std::vector<std::string> body) {
  std::vector<std::string> s{"<html>", "<body>", "<table>"};
  for (auto &t : body) s.push_back(std::move(t));
  s.push_back("</table>");
  s.push_back("</body>");
  s.push_back("</html>");
  return s;
}

// A unit quad at (x, y) — only its identity matters for these tests.
std::array<int, 8> quad(int x, int y) {
  return {x, y, x + 10, y, x + 10, y + 5, x, y + 5};
}

MatchedCell matched_with(std::vector<std::size_t> idx) {
  MatchedCell c;
  c.ocr_indices = std::move(idx);
  return c;
}

} // namespace

TEST_CASE("build_table_cells emits one cell per td slot in order", "[table_cells]") {
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td></td>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(20, 0)},
      {matched_with({0}), matched_with({1})},
      {"Warengruppe", "Menge"});

  REQUIRE(cells.size() == 2);
  CHECK(cells[0].text == "Warengruppe");
  CHECK(cells[1].text == "Menge");
  CHECK(cells[0].box.pts[0] == std::array<int, 2>{0, 0});
  CHECK(cells[1].box.pts[0] == std::array<int, 2>{20, 0});
}

TEST_CASE("build_table_cells derives row and column from the token stream",
          "[table_cells]") {
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td></td>", "<td></td>", "</tr>",
            "<tr>", "<td></td>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(20, 0), quad(0, 10), quad(20, 10)},
      {matched_with({0}), matched_with({1}), matched_with({2}), matched_with({3})},
      {"a", "b", "c", "d"});

  REQUIRE(cells.size() == 4);
  CHECK(cells[0].row == 0); CHECK(cells[0].col == 0);
  CHECK(cells[1].row == 0); CHECK(cells[1].col == 1);
  CHECK(cells[2].row == 1); CHECK(cells[2].col == 0);
  CHECK(cells[3].row == 1); CHECK(cells[3].col == 1);
  for (const auto &c : cells) { CHECK(c.rowspan == 1); CHECK(c.colspan == 1); }
}

TEST_CASE("build_table_cells reads colspan off the attribute token", "[table_cells]") {
  // Row 0: one cell spanning both columns. Row 1: two ordinary cells.
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td", " colspan=\"2\"", ">", "</td>", "</tr>",
            "<tr>", "<td></td>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(0, 10), quad(20, 10)},
      {matched_with({0}), matched_with({1}), matched_with({2})},
      {"span", "a", "b"});

  REQUIRE(cells.size() == 3);
  CHECK(cells[0].row == 0); CHECK(cells[0].col == 0); CHECK(cells[0].colspan == 2);
  CHECK(cells[1].row == 1); CHECK(cells[1].col == 0);
  CHECK(cells[2].row == 1); CHECK(cells[2].col == 1);
}

TEST_CASE("build_table_cells keeps later rows off a rowspan'd column",
          "[table_cells]") {
  // Row 0 col 0 spans two rows, so row 1's first cell lands in column 1.
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td", " rowspan=\"2\"", ">", "</td>", "<td></td>", "</tr>",
            "<tr>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(20, 0), quad(20, 10)},
      {matched_with({0}), matched_with({1}), matched_with({2})},
      {"tall", "b", "c"});

  REQUIRE(cells.size() == 3);
  CHECK(cells[0].row == 0); CHECK(cells[0].col == 0); CHECK(cells[0].rowspan == 2);
  CHECK(cells[1].row == 0); CHECK(cells[1].col == 1);
  CHECK(cells[2].row == 1); CHECK(cells[2].col == 1);
}

TEST_CASE("build_table_cells shifts a colspan past an interior live rowspan",
          "[table_cells]") {
  // Row 0: A(rowspan=2) B C(rowspan=3) D. Row 1: one colspan=3 cell.
  // At row 1, col 0 is blocked by A and col 2 by C, so the colspan=3 cell's
  // first fully-free 3-wide span starts at column 3 — a browser shifts it
  // right past the blocker. The old placement checked only the FIRST column,
  // put the cell at col 1, and stamped over C's live rowspan: two cells then
  // claimed (row 1, col 2).
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td", " rowspan=\"2\"", ">", "</td>", "<td></td>",
            "<td", " rowspan=\"3\"", ">", "</td>", "<td></td>", "</tr>",
            "<tr>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>"}),
      {quad(0, 0), quad(20, 0), quad(40, 0), quad(60, 0), quad(20, 10)},
      {matched_with({0}), matched_with({1}), matched_with({2}),
       matched_with({3}), matched_with({4})},
      {"A", "B", "C", "D", "wide"});

  REQUIRE(cells.size() == 5);
  CHECK(cells[2].row == 0); CHECK(cells[2].col == 2); CHECK(cells[2].rowspan == 3);
  CHECK(cells[4].row == 1); CHECK(cells[4].col == 3); CHECK(cells[4].colspan == 3);
}

TEST_CASE("build_table_cells joins multi-fragment cell text with single spaces",
          "[table_cells]") {
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td></td>", "</tr>"}), {quad(0, 0)},
      {matched_with({0, 1})}, {"<b>Ware</b>", " gruppe"});

  REQUIRE(cells.size() == 1);
  // The <b> emphasis wrapper is HTML decoration, not cell text.
  CHECK(cells[0].text == "Ware gruppe");
}

TEST_CASE("build_table_cells leaves unmatched cells empty, not dropped",
          "[table_cells]") {
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td></td>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(20, 0)}, {matched_with({}), matched_with({0})}, {"only"});

  REQUIRE(cells.size() == 2);
  CHECK(cells[0].text.empty());
  CHECK(cells[1].text == "only");
  CHECK(cells[1].col == 1);  // index alignment with the <td> stream survives
}

TEST_CASE("build_table_cells reports unknown position for a td outside any tr",
          "[table_cells]") {
  const auto cells = build_table_cells(wrap({"<td></td>"}), {quad(0, 0)},
                                       {matched_with({0})}, {"orphan"});
  REQUIRE(cells.size() == 1);
  CHECK(cells[0].row == -1);
  CHECK(cells[0].col == -1);
}

TEST_CASE("build_table_cells tolerates ragged matched/text inputs", "[table_cells]") {
  // More quads than matched entries, and an out-of-range text index.
  const auto cells = build_table_cells(
      wrap({"<tr>", "<td></td>", "<td></td>", "</tr>"}),
      {quad(0, 0), quad(20, 0)}, {matched_with({9})}, {"present"});

  REQUIRE(cells.size() == 2);
  CHECK(cells[0].text.empty());
  CHECK(cells[1].text.empty());
}

// ---- serialization ----------------------------------------------------------

TEST_CASE("append_tables_array emits cells additively", "[table_cells][serialization]") {
  TableResult t;
  t.layout_id = 3;
  t.html = "<table></table>";
  t.score = 0.5f;
  TableCell c;
  c.box.pts = {{{1, 2}, {3, 2}, {3, 4}, {1, 4}}};
  c.text = "Umsatz";
  c.row = 0; c.col = 2; c.rowspan = 1; c.colspan = 1;
  t.cells.push_back(c);

  std::string j;
  turbo_ocr::detail::append_tables_array(j, {t});

  // The pre-existing fields keep their exact order and spelling.
  CHECK(j.find("\"layout_id\":3,\"html\":\"<table></table>\",\"confidence\":0.5,"
               "\"bounding_box\":") != std::string::npos);
  CHECK(j.find("\"cells\":[{\"text\":\"Umsatz\",\"bounding_box\":"
               "[[1,2],[3,2],[3,4],[1,4]],\"row\":0,\"col\":2,"
               "\"rowspan\":1,\"colspan\":1}]") != std::string::npos);
}

TEST_CASE("append_tables_array omits grid fields for an unplaced cell",
          "[table_cells][serialization]") {
  TableResult t;
  t.cells.emplace_back();  // row/col default to -1
  std::string j;
  turbo_ocr::detail::append_tables_array(j, {t});
  CHECK(j.find("\"row\":") == std::string::npos);
  CHECK(j.find("\"colspan\":") == std::string::npos);
}

TEST_CASE("append_tables_array emits an empty cells array for a VLM table",
          "[table_cells][serialization]") {
  TableResult t;
  t.html = "<table><tr><td>x</td></tr></table>";
  std::string j;
  turbo_ocr::detail::append_tables_array(j, {t});
  CHECK(j.find("\"cells\":[]") != std::string::npos);
}
