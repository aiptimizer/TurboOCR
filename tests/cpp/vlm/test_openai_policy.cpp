// openai_policy — the SHARED remote-VLM endpoint policy that backs BOTH
// endpoint classes (turbo_ocr::vlm::OpenAIEndpoint on the old CUDA-typed seam,
// turbo_ocr::vlm::BackendOpenAIEndpoint on the backend:: seam). It had no tests
// at all: before the dedup it was file-local statics in a CUDA-only TU, which is
// why the nearest suite (test_openai_parsers.cpp) tests table::otsl_to_html
// directly instead. That constraint is gone — the header is device-free — so the
// policy is covered here, in the ALWAYS-built test target.
//
// Deliberately NOT covered here: for_each_crop and infer_crops. Both odr-use
// vlm::encode_png_bgr / the crop pool, which live in src/analysis/vlm/vlm_client.cpp and
// crop_pool.cpp — GPU-target sources. Testing them from the CPU configure means
// putting those TUs (and libcurl) on this target, which is a build-config change
// this file should not smuggle in.

#include <string>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/analysis/vlm/openai_policy.h"

namespace policy = turbo_ocr::vlm::openai_policy;
using turbo_ocr::backend_routing::Parser;
using policy::CropOut;

TEST_CASE("otsl_or_html passes model-emitted HTML through the sanitizer",
          "[vlm][policy]") {
  // The `<table` sniff runs on the TRIMMED string but sanitize_table_html is
  // handed the RAW one — that asymmetry is deliberate (leading whitespace must
  // not defeat the detection) and it is a SECURITY branch, so pin both halves.
  const std::string html =
      "  \n<table><tr><td>a<script>alert(1)</script></td></tr></table>";
  const std::string out = policy::otsl_or_html(html);
  CHECK(out.find("<table") != std::string::npos);
  CHECK(out.find("a") != std::string::npos);
  CHECK(out.find("<script") == std::string::npos);
  CHECK(out.find("alert(1)") == std::string::npos);
}

TEST_CASE("otsl_or_html converts OTSL and leaves empty input empty",
          "[vlm][policy]") {
  const std::string out = policy::otsl_or_html("<fcel>a<fcel>b<nl>");
  CHECK(out.find("<td>a</td>") != std::string::npos);
  CHECK(out.find("<td>b</td>") != std::string::npos);

  CHECK(policy::otsl_or_html("").empty());
}

TEST_CASE("parse_with dispatches on the parser enum only", "[vlm][policy]") {
  // Text trims; Raw is byte-exact (it is the escape hatch for a model whose
  // leading/trailing whitespace is meaningful).
  CHECK(policy::parse_with(Parser::Text, "  hi \n") == "hi");
  CHECK(policy::parse_with(Parser::Raw, "  hi \n") == "  hi \n");
  // Otsl takes the HTML passthrough branch rather than running otsl_to_html.
  const std::string t =
      policy::parse_with(Parser::Otsl, "<table><tr><td>x</td></tr></table>");
  CHECK(t.find("<td>x</td>") != std::string::npos);
}

TEST_CASE("to_table_results distinguishes a FAILED call from an empty table",
          "[vlm][policy]") {
  // score is the ONLY signal that separates the two — default-confident is 1.0,
  // so a failed endpoint call must come back 0.0 rather than a confident empty.
  std::vector<CropOut> crops{CropOut{"", false},
                             CropOut{"<table><tr><td>x</td></tr></table>", true}};
  const auto out = policy::to_table_results(std::move(crops), {});
  REQUIRE(out.size() == 2);
  CHECK(out[0].score == 0.0F);
  CHECK(out[0].html.empty());
  CHECK(out[0].layout_id == -1);
  CHECK(out[1].score == 1.0F);
  CHECK(out[1].html.find("<td>x</td>") != std::string::npos);
  CHECK(out[1].layout_id == -1);
}

TEST_CASE("to_table_results attaches the region box when one is supplied",
          "[vlm][policy]") {
  std::vector<CropOut> crops{CropOut{"<table/>", true}};
  const turbo_ocr::Box b{{{{1, 2}, {3, 2}, {3, 4}, {1, 4}}}};
  const auto out = policy::to_table_results(std::move(crops), {b});
  REQUIRE(out.size() == 1);
  CHECK(out[0].box.pts[0][0] == 1);
  CHECK(out[0].box.pts[2][1] == 4);
}

namespace {
// Structurally what BOTH turbo_ocr::formula::FormulaEngineResult and
// turbo_ocr::backend::FormulaEngineResult are — the two types this template is
// really instantiated with, declared in two different headers. Naming the
// members here is the point: the builder uses designated initialisers, so a
// field inserted before `hit_eos` in either real header is a compile error
// instead of a silent write into the wrong slot.
struct FakeFormulaResult {
  std::string latex;
  int token_count = 0;
  bool hit_eos = false;
  bool ok = true;
};
} // namespace

TEST_CASE("to_formula_results carries the transport status in hit_eos",
          "[vlm][policy]") {
  std::vector<CropOut> crops{CropOut{"", false}, CropOut{"x^2", true}};
  const auto out = policy::to_formula_results<FakeFormulaResult>(std::move(crops));
  REQUIRE(out.size() == 2);
  // A failed call is never stamped as a clean stop.
  CHECK_FALSE(out[0].hit_eos);
  CHECK(out[0].token_count == 0);
  CHECK(out[0].latex.empty());
  CHECK(out[1].hit_eos);
  CHECK(out[1].latex == "x^2");
  // `ok` stays at its default true on BOTH — see the comment on
  // to_formula_results: clearing it would suppress the CJK formula fallback
  // (auto_cjk_formula.cpp gates its re-run on `ok`), so consumers test
  // `!ok || latex.empty()` instead. Pinned so the change is deliberate.
  CHECK(out[0].ok);
  CHECK(out[1].ok);
}
