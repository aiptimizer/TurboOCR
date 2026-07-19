#include <catch_amalgamated.hpp>

#include <string>

// extract_latex is the ONE shared implementation (formula/latex_extract.h)
// used by both the vLLM sidecar backend and the generic OpenAI endpoint;
// otsl_or_html is a thin shell over the reusable table::otsl_to_html. Both
// underlying transforms are tested directly here.
#include "turbo_ocr/formula/latex_extract.h"
#include "turbo_ocr/table/vlm_table.h"

using turbo_ocr::formula::extract_latex;
using turbo_ocr::table::otsl_to_html;

TEST_CASE("extract_latex prefers a latex fence", "[openai_parsers][latex]") {
  CHECK(extract_latex("```latex\nx^2 + y^2\n```") == "x^2 + y^2");
  CHECK(extract_latex("prose before ```tex\n\\frac{a}{b}``` prose after") ==
        "\\frac{a}{b}");
  CHECK(extract_latex("```\nbare fence\n```") == "bare fence");
}

TEST_CASE("extract_latex falls through the delimiter ladder", "[openai_parsers][latex]") {
  CHECK(extract_latex("here: $$E=mc^2$$ done") == "E=mc^2");
  CHECK(extract_latex("here: \\[a+b\\] done") == "a+b");
  CHECK(extract_latex("inline $x_i$ math") == "x_i");
}

TEST_CASE("extract_latex strips answer prefixes on bare replies", "[openai_parsers][latex]") {
  CHECK(extract_latex("LaTeX: x + y") == "x + y");
  CHECK(extract_latex("Answer:  z ") == "z");
  CHECK(extract_latex("  plain reply \n") == "plain reply");
  CHECK(extract_latex("") == "");
}

TEST_CASE("otsl_to_html builds a simple grid", "[openai_parsers][table]") {
  // 2x2 full cells -> two rows of two <td>.
  CHECK(otsl_to_html("<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>") ==
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>C</td><td>D</td></tr></table>");
}

TEST_CASE("otsl_to_html emits colspan for left-merge", "[openai_parsers][table]") {
  // <lcel> extends the previous cell one column to the right.
  CHECK(otsl_to_html("<fcel>A<lcel><nl>") ==
        "<table><tr><td colspan=\"2\">A</td></tr></table>");
}

TEST_CASE("otsl_to_html emits rowspan for up-merge", "[openai_parsers][table]") {
  // <ucel> in the row below extends the cell above downward.
  CHECK(otsl_to_html("<fcel>A<nl><ucel><nl>") ==
        "<table><tr><td rowspan=\"2\">A</td></tr><tr></tr></table>");
}

TEST_CASE("otsl_to_html emits row+col span for corner-merge",
          "[openai_parsers][table]") {
  // <xcel> merges both left and up: a 2x2 block collapses to one rowspan=2
  // colspan=2 cell; the trailing <lcel> on row 2 is absorbed by the merge.
  CHECK(otsl_to_html("<fcel>A<lcel><nl><ucel><xcel><nl>") ==
        "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td></tr>"
        "<tr></tr></table>");
}

TEST_CASE("otsl_to_html pads a short row by left-merging into the last cell",
          "[openai_parsers][table]") {
  // Row 2 has one cell but the table is two-wide. A short row means the decoder
  // truncated the trailing colspan tokens of its LAST cell, so we pad with
  // <lcel> (left-merge) -> the cell spans the missing column (colspan=2). The
  // old behavior padded with a phantom standalone <td></td>, which shifts every
  // <ucel> rowspan below it and corrupts colspan geometry on irregular tables.
  CHECK(otsl_to_html("<fcel>A<fcel>B<nl><fcel>C<nl>") ==
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td colspan=\"2\">C</td></tr></table>");
}

TEST_CASE("otsl_to_html escapes cell text", "[openai_parsers][table]") {
  CHECK(otsl_to_html("<fcel>a<b><nl>") ==
        "<table><tr><td>a&lt;b&gt;</td></tr></table>");
}

TEST_CASE("otsl_to_html treats a no-newline string as a single row",
          "[openai_parsers][table]") {
  // Missing <nl> is tolerated: the whole string is one row.
  CHECK(otsl_to_html("<fcel>X<fcel>Y") ==
        "<table><tr><td>X</td><td>Y</td></tr></table>");
}

TEST_CASE("otsl_to_html returns empty for degenerate input",
          "[openai_parsers][table]") {
  // Empty / whitespace-only / token-less input must yield an empty string,
  // never a crash or a malformed half-table.
  CHECK(otsl_to_html("").empty());
  CHECK(otsl_to_html("   ").empty());
  CHECK(otsl_to_html("plain text with no otsl tokens").empty());
}
