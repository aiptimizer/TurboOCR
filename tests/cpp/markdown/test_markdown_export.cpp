// render_markdown — the ONLY faithful-Markdown renderer, and until now the only
// subsystem in src/ with zero unit coverage. It was just restructured into five
// phases (build_page_index / collect_payloads / fold_formula_numbers /
// bucket_order+apply_column_order / emit_region) under a "byte-identical output"
// mandate that nothing could check.
//
// It is a PURE function over CUDA-free result types (markdown_export.h says so:
// "Pure (no OpenCV)"), so every case below is a hand-built OcrPipelineResult and
// an exact-string assertion. Exact strings are the point — that is what makes
// the NEXT refactor safe.

#include <string>
#include <vector>

#include "catch_amalgamated.hpp"

#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/pipeline/pipeline_result.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::markdown::MarkdownOptions;
using turbo_ocr::markdown::render_markdown;
using turbo_ocr::pipeline::OcrPipelineResult;

namespace {

Box rect(int x0, int y0, int x1, int y1) {
  return Box{{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}}};
}

// One layout cell. `id` is set to its own index, which is what
// assign_layout_ids does and what the renderer's result->region mapping uses.
turbo_ocr::layout::LayoutBox cell(int class_id, int id, int x0, int y0, int x1,
                                  int y1) {
  turbo_ocr::layout::LayoutBox lb{};
  lb.class_id = class_id;
  lb.score = 0.9F;
  lb.box = rect(x0, y0, x1, y1);
  lb.id = id;
  return lb;
}

OCRResultItem word(const char *text, int layout_id, int x0, int y0, int x1,
                   int y1) {
  OCRResultItem it;
  it.text = text;
  it.confidence = 0.95F;
  it.box = rect(x0, y0, x1, y1);
  it.layout_id = layout_id;
  return it;
}

// Class ids, resolved from the SHARED label table rather than hardcoded, so a
// renumbering breaks the table (and markdown_export.cpp's own static_asserts)
// rather than quietly re-pointing this suite at the wrong classes.
int class_id_named(std::string_view want) {
  for (int i = 0; i < static_cast<int>(turbo_ocr::layout::kLayoutLabels.size());
       ++i)
    if (turbo_ocr::layout::kLayoutLabels[i] == want) return i;
  FAIL("no layout class named " << want);
  return 0;
}

const int kDisplayFormula = class_id_named("display_formula");
const int kFormulaNumber = class_id_named("formula_number");
const int kText = class_id_named("text");
const int kDocTitle = class_id_named("doc_title");
const int kTable = class_id_named("table");

} // namespace

TEST_CASE("markdown: no layout renders one paragraph per result", "[markdown]") {
  OcrPipelineResult res;
  res.results = {word("first line", -1, 0, 0, 100, 20),
                 word("second line", -1, 0, 30, 100, 50)};

  const std::string md = render_markdown(res);
  CHECK(md == "first line\n\nsecond line\n");
}

TEST_CASE("markdown: a display formula folds its formula_number into \\tag",
          "[markdown]") {
  OcrPipelineResult res;
  res.layout = {cell(kDisplayFormula, 0, 10, 100, 300, 160),
                cell(kFormulaNumber, 1, 320, 110, 380, 150)};
  // The number region's own OCR text — it must be CONSUMED, never emitted as a
  // bare "(1)" paragraph of its own.
  res.results = {word("(1)", 1, 320, 110, 380, 150)};
  turbo_ocr::router::FormulaResult f;
  f.layout_id = 0;
  f.latex = "x^2";
  f.score = 0.99F;
  res.formulas = {f};

  SECTION("folded by default") {
    const std::string md = render_markdown(res);
    CHECK(md == "$$\nx^2 \\tag{1}\n$$\n");
    CHECK(md.find("(1)") == std::string::npos);
  }
  SECTION("fold_formula_numbers=false still suppresses the bare number") {
    MarkdownOptions opts;
    opts.fold_formula_numbers = false;
    const std::string md = render_markdown(res, opts);
    CHECK(md == "$$\nx^2\n$$\n");
    CHECK(md.find("\\tag") == std::string::npos);
    CHECK(md.find("(1)") == std::string::npos);
  }
}

TEST_CASE("markdown: a display formula with no recognizer result is prose, "
          "never $$-wrapped",
          "[markdown]") {
  // gather() returns the region's RAW OCR characters, which are not LaTeX.
  // Wrapping those in $$ renders broken math in KaTeX/MathJax.
  OcrPipelineResult res;
  res.layout = {cell(kDisplayFormula, 0, 10, 100, 300, 160)};
  res.results = {word("E = mc2", 0, 12, 105, 290, 155)};

  const std::string md = render_markdown(res);
  CHECK(md == "E = mc2\n");
  CHECK(md.find("$$") == std::string::npos);
}

TEST_CASE("markdown: unbalanced LaTeX is demoted to a fenced listing",
          "[markdown]") {
  OcrPipelineResult res;
  res.layout = {cell(kDisplayFormula, 0, 10, 100, 300, 160)};
  turbo_ocr::router::FormulaResult f;
  f.layout_id = 0;
  f.latex = "\\frac{1}{2";  // brace never closed
  res.formulas = {f};

  MarkdownOptions opts;
  opts.safe_formula_fallback = true;
  const std::string md = render_markdown(res, opts);
  CHECK(md == "```latex\n\\frac{1}{2\n```\n");
  CHECK(md.rfind("$$", 0) == std::string::npos);
}

TEST_CASE("markdown: a garbage equation number costs only the tag, not the "
          "whole equation",
          "[markdown]") {
  // REGRESSION. The tag used to be appended BEFORE the render-safety check, so
  // one mis-OCR'd number ("1}") unbalanced the braces and demoted an otherwise
  // perfectly safe equation to a code fence. clean_tag now refuses anything
  // containing LaTeX syntax, and the tag is only kept if the TAGGED string is
  // still safe.
  OcrPipelineResult res;
  res.layout = {cell(kDisplayFormula, 0, 10, 100, 300, 160),
                cell(kFormulaNumber, 1, 320, 110, 380, 150)};
  res.results = {word("1}", 1, 320, 110, 380, 150)};
  turbo_ocr::router::FormulaResult f;
  f.layout_id = 0;
  f.latex = "x^2";
  res.formulas = {f};

  const std::string md = render_markdown(res);
  CHECK(md == "$$\nx^2\n$$\n");           // still real math
  CHECK(md.find("```") == std::string::npos);
  CHECK(md.find("1}") == std::string::npos);
}

TEST_CASE("markdown: a table with no router HTML falls back to its raw text",
          "[markdown]") {
  OcrPipelineResult res;
  res.layout = {cell(kTable, 0, 0, 0, 400, 200)};
  res.results = {word("Header Value", 0, 10, 10, 390, 40)};
  // res.tables deliberately empty: the table backend was off or failed.

  const std::string md = render_markdown(res);
  CHECK(md == "Header Value\n");
}

TEST_CASE("markdown: the orphan-length gate spares titles", "[markdown]") {
  MarkdownOptions opts;   // min_text_codepoints defaults to 2

  SECTION("a one-codepoint text region is dropped") {
    OcrPipelineResult res;
    res.layout = {cell(kText, 0, 0, 0, 100, 40)};
    res.results = {word("X", 0, 5, 5, 95, 35)};
    CHECK(render_markdown(res, opts).empty());
  }
  SECTION("a one-codepoint doc_title is kept") {
    OcrPipelineResult res;
    res.layout = {cell(kDocTitle, 0, 0, 0, 100, 40)};
    res.results = {word("X", 0, 5, 5, 95, 35)};
    CHECK(render_markdown(res, opts) == "# X\n");
  }
}
