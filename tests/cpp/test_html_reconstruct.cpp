#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/table/html_reconstruct.h"

using turbo_ocr::table::MatchedCell;
using turbo_ocr::table::reconstruct_html;

namespace {

// Minimal wrapped structure stream: <html><body><table> + tds + closers.
std::vector<std::string> wrap(std::vector<std::string> tds) {
  std::vector<std::string> s{"<html>", "<body>", "<table>"};
  for (auto &t : tds) s.push_back(std::move(t));
  s.push_back("</table>");
  s.push_back("</body>");
  s.push_back("</html>");
  return s;
}

MatchedCell cell_with(std::vector<std::size_t> idx) {
  MatchedCell c;
  c.ocr_indices = std::move(idx);
  return c;
}

} // namespace

TEST_CASE("reconstruct_html substitutes cell text in td order", "[html_reconstruct]") {
  const auto html = reconstruct_html(wrap({"<td></td>", "<td></td>"}),
                                     {cell_with({0}), cell_with({1})},
                                     {"alpha", "beta"});
  CHECK(html == "<html><body><table><td>alpha</td><td>beta</td>"
                "</table></body></html>");
}

TEST_CASE("reconstruct_html escapes markup in OCR text", "[html_reconstruct]") {
  const auto html = reconstruct_html(
      wrap({"<td></td>"}), {cell_with({0})},
      {"<script>alert(1)</script> & <img src=x>"});
  CHECK(html.find("<script>") == std::string::npos);
  CHECK(html.find("&lt;script&gt;") != std::string::npos);
  CHECK(html.find("&amp;") != std::string::npos);
  CHECK(html.find("<img") == std::string::npos);
}

TEST_CASE("reconstruct_html preserves the <b> emphasis wrapper", "[html_reconstruct]") {
  const auto html = reconstruct_html(wrap({"<td></td>"}), {cell_with({0})},
                                     {"<b>Header</b>"});
  CHECK(html.find("<td><b>Header</b></td>") != std::string::npos);
}

TEST_CASE("reconstruct_html joins multi-fragment cells with single spaces",
          "[html_reconstruct]") {
  const auto html = reconstruct_html(wrap({"<td></td>"}), {cell_with({0, 1})},
                                     {"first", "second"});
  CHECK(html.find("<td>first second</td>") != std::string::npos);
}

TEST_CASE("reconstruct_html escapes every fragment of a joined cell",
          "[html_reconstruct]") {
  const auto html = reconstruct_html(wrap({"<td></td>"}), {cell_with({0, 1})},
                                     {"a<b", "c&d"});
  CHECK(html.find("a&lt;b") != std::string::npos);
  CHECK(html.find("c&amp;d") != std::string::npos);
}

TEST_CASE("reconstruct_html tolerates out-of-range ocr indices", "[html_reconstruct]") {
  const auto html = reconstruct_html(wrap({"<td></td>"}), {cell_with({7})},
                                     {"only"});
  CHECK(html.find("<td></td>") != std::string::npos);
}

TEST_CASE("reconstruct_html below the wrapped minimum falls back to concatenation",
          "[html_reconstruct]") {
  const auto html = reconstruct_html({"<table>", "</table>"}, {}, {});
  CHECK(html == "<table></table>");
}
