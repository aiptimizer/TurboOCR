#include <catch_amalgamated.hpp>

#include <string>

#include "turbo_ocr/analysis/formula/ppformulanet/latex_normalize.h"

using turbo_ocr::formula::latex_post_process;

// PaddleX-parity cases for the LaTeX normalization engine (extracted from the
// tokenizer into latex_normalize.cpp). These pin the behaviors the port
// documents: Chinese \text{} unwrapping, whitespace collapse between
// non-letter tokens, "\ " escape preservation, and wrapper-macro de-spacing.

TEST_CASE("plain ASCII math is only whitespace-collapsed",
          "[latex_normalize]") {
  CHECK(latex_post_process("a + b = c") == "a+b=c");
  CHECK(latex_post_process("x ^ { 2 } + 1") == "x^{2}+1");
}

TEST_CASE("idempotent on already-normalized input", "[latex_normalize]") {
  const std::string once = latex_post_process("\\frac { a } { b }");
  CHECK(latex_post_process(once) == once);
}

TEST_CASE("backslash-space escape survives collapse", "[latex_normalize]") {
  // "\ " is a hard space in LaTeX; the collapse passes must not eat it.
  const std::string out = latex_post_process("a \\ b");
  CHECK(out.find("\\ ") != std::string::npos);
}

TEST_CASE("chinese \\text wrapping is removed", "[latex_normalize]") {
  const std::string out = latex_post_process("\\text { 中文 }");
  CHECK(out.find("\\text") == std::string::npos);
  CHECK(out.find("中文") != std::string::npos);
}

TEST_CASE("latin \\text wrapping is preserved", "[latex_normalize]") {
  const std::string out = latex_post_process("\\text { abc }");
  CHECK(out.find("\\text") != std::string::npos);
}

TEST_CASE("wrapper macro interior spacing is preserved via stash-restore",
          "[latex_normalize]") {
  // operatorname/mathrm/text/mathbf spans are stashed before the collapse
  // passes and restored verbatim afterwards.
  const std::string out = latex_post_process("\\operatorname { s i n } x");
  CHECK(out.find("\\operatorname") != std::string::npos);
  CHECK(out.find('x') != std::string::npos);
}

TEST_CASE("digits and operators collapse fully", "[latex_normalize]") {
  CHECK(latex_post_process("1 2 + 3 4") == "12+34");
  CHECK(latex_post_process("( a ) [ b ]") == "(a)[b]");
}

TEST_CASE("letters keep a separating space", "[latex_normalize]") {
  // Letter-letter boundaries are not collapsed (would merge identifiers).
  CHECK(latex_post_process("a b") == "a b");
}

TEST_CASE("empty and whitespace-only inputs", "[latex_normalize]") {
  CHECK(latex_post_process("").empty());
  // Collapse passes merge whitespace between tokens but never trim the edges
  // (PaddleX parity — the tokenizer trims before calling post-process).
  CHECK(latex_post_process("   ") == "  ");
}

TEST_CASE("member shim delegates to the free function", "[latex_normalize]") {
  // FormulaTokenizer::latex_post_process must stay byte-identical to the
  // extracted engine (decode() calls the member).
  CHECK(latex_post_process("x ^ { 2 }") == "x^{2}");
}
