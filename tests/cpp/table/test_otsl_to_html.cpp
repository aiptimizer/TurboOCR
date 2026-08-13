#include <catch_amalgamated.hpp>

#include "nvidia/stages/vlm_table.h"

using turbo_ocr::table::otsl_to_html;

TEST_CASE("otsl_to_html simple 2x2 grid", "[otsl]") {
    // Two rows, two columns each.
    auto html = otsl_to_html("<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>");
    REQUIRE(html.find("<td>a</td>") != std::string::npos);
    REQUIRE(html.find("<td>d</td>") != std::string::npos);
    REQUIRE(html.find("colspan") == std::string::npos);
}

TEST_CASE("otsl_to_html colspan from lcel", "[otsl]") {
    // First row: one cell spanning two columns. Second row: two cells.
    auto html = otsl_to_html("<fcel>wide<lcel><nl><fcel>a<fcel>b<nl>");
    REQUIRE(html.find("colspan=\"2\"") != std::string::npos);
}

TEST_CASE("pad_rows: short row left-merges into last cell, preserving column count",
          "[otsl]") {
    // Irregular table: row 0 has a 2-col header, row 1 has THREE cells. Row 0 is
    // short (2 cols) vs the 3-col max. With the correct pad side, row 0's last
    // cell absorbs the missing column (colspan 2) instead of a phantom empty
    // column being inserted that would shift row 1's geometry.
    auto html = otsl_to_html("<fcel>H1<fcel>H2<nl><fcel>a<fcel>b<fcel>c<nl>");
    // Row 1 must keep all THREE cells intact.
    REQUIRE(html.find("<td>a</td>") != std::string::npos);
    REQUIRE(html.find("<td>b</td>") != std::string::npos);
    REQUIRE(html.find("<td>c</td>") != std::string::npos);
    // The short header row's last cell spans the missing column.
    REQUIRE(html.find("colspan=\"2\"") != std::string::npos);
    // The pad must NOT fabricate a standalone trailing empty <td></td> on the
    // header row (that is the wrong-side <ecel> bug).
    REQUIRE(html.find("<td>H2</td><td></td>") == std::string::npos);
}

TEST_CASE("pad_rows: a short row ending in <ucel> pads with <ecel>, not a "
          "malformed <lcel>", "[otsl]") {
    // Row 0 defines 3 columns. Row 1 has D then a <ucel> (rowspan continuation
    // of B), and is short (2 cols). A <ucel> cannot root a horizontal merge, so
    // padding it with <lcel> leaves column 2 of row 1 with no root cell — it
    // silently vanishes. The fix pads with a standalone <ecel> so the column is
    // preserved as an empty cell.
    auto html = otsl_to_html(
        "<fcel>A<fcel>B<fcel>C<nl><fcel>D<ucel><nl>");
    // B spans into row 1's <ucel>.
    REQUIRE(html.find("rowspan=\"2\"") != std::string::npos);
    // Row 1 keeps its third column as a standalone empty cell (the fix); the old
    // <lcel> pad would drop it, leaving "<td>D</td></tr>".
    REQUIRE(html.find("<td>D</td><td></td>") != std::string::npos);
    REQUIRE(html.find("<td>D</td></tr>") == std::string::npos);
}
