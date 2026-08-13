#include <catch_amalgamated.hpp>

#include <string>

// extract_latex is the ONE shared implementation (formula/latex_extract.cpp,
// in turbo_ocr_common) used by both the vLLM sidecar backend and the generic
// OpenAI endpoint. Lives in a CPU-linkable TU so this coverage runs in every
// build (the otsl_to_html tests in test_openai_parsers need the GPU lib and
// stay GPU-only).
#include "turbo_ocr/analysis/formula/latex_extract.h"

using turbo_ocr::formula::extract_latex;

TEST_CASE("extract_latex prefers a latex fence", "[latex]") {
  CHECK(extract_latex("```latex\nx^2 + y^2\n```") == "x^2 + y^2");
  CHECK(extract_latex("prose before ```tex\n\\frac{a}{b}``` prose after") ==
        "\\frac{a}{b}");
  CHECK(extract_latex("```\nbare fence\n```") == "bare fence");
}

TEST_CASE("extract_latex falls through the delimiter ladder", "[latex]") {
  CHECK(extract_latex("here: $$E=mc^2$$ done") == "E=mc^2");
  CHECK(extract_latex("here: \\[a+b\\] done") == "a+b");
  CHECK(extract_latex("inline $x_i$ math") == "x_i");
}

TEST_CASE("extract_latex strips answer prefixes on bare replies", "[latex]") {
  CHECK(extract_latex("LaTeX: x + y") == "x + y");
  CHECK(extract_latex("Answer:  z ") == "z");
  CHECK(extract_latex("  plain reply \n") == "plain reply");
  CHECK(extract_latex("") == "");
}
