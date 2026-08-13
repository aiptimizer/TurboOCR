#include <catch_amalgamated.hpp>

#include <future>
#include <string>

#include "nvidia/stages/formula_recognizer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "nvidia/stages/table_recognizer.h"

using turbo_ocr::pipeline::finalize_deferred;
using turbo_ocr::pipeline::OcrPipelineResult;
using turbo_ocr::pipeline::PendingCrop;

namespace {

// Minimal IFormulaRecognizer whose parse_async_result is identity (default),
// so a future resolving to "" parses to empty LaTeX — the exact silent-failure
// shape the deferred path must surface.
struct StubFormula final : turbo_ocr::formula::IFormulaRecognizer {
  bool load_model_dir(const std::string &) override { return true; }
  bool load_tokenizer(const std::string &) override { return true; }
  std::vector<turbo_ocr::formula::FormulaEngineResult>
  run(const turbo_ocr::GpuImage &, const std::vector<turbo_ocr::Box> &,
      cudaStream_t) override {
    return {};
  }
  bool is_ready() const noexcept override { return true; }
  std::string_view backend_name() const noexcept override { return "stub"; }
};

struct StubTable final : turbo_ocr::table::ITableRecognizer {
  bool load() override { return true; }
  std::vector<turbo_ocr::router::TableResult>
  run(const turbo_ocr::GpuImage &, const std::vector<turbo_ocr::Box> &,
      const std::vector<turbo_ocr::OCRResultItem> &, cudaStream_t) override {
    return {};
  }
  bool is_ready() const noexcept override { return true; }
  std::string_view backend_name() const noexcept override { return "stub"; }
};

std::future<std::string> ready(std::string s) {
  std::promise<std::string> p;
  auto f = p.get_future();
  p.set_value(std::move(s));
  return f;
}

// A future whose promise is destroyed without set_value — .get() throws
// std::future_error(broken_promise). Models the pool being torn down mid-flight.
std::future<std::string> broken() {
  std::promise<std::string> p;
  return p.get_future();  // p destroyed at return -> broken promise
}

PendingCrop crop(int lid, std::future<std::string> f) {
  PendingCrop pc;
  pc.layout_id = lid;
  pc.fut = std::move(f);
  return pc;
}

} // namespace

TEST_CASE("finalize_deferred flags a formula region that resolved to empty",
          "[finalize_deferred]") {
  StubFormula stub;
  OcrPipelineResult out;
  out.pending.formula_parse = stub.async_result_parser();
  out.pending.formula.push_back(crop(0, ready("E = mc^2")));
  out.pending.formula.push_back(crop(1, ready("")));  // transport failure shape

  finalize_deferred(out);

  REQUIRE(out.formulas.size() == 2);
  CHECK(out.formulas[0].latex == "E = mc^2");
  CHECK(out.formulas[1].latex.empty());
  CHECK(out.formula_degraded);
  CHECK(out.formula_warning.find("1 of 2") != std::string::npos);
  CHECK(out.pending.empty());  // cleared after finalize
}

TEST_CASE("finalize_deferred flags a table region that resolved to empty",
          "[finalize_deferred]") {
  StubTable stub;
  OcrPipelineResult out;
  out.pending.table_parse = stub.async_result_parser();
  out.pending.table.push_back(crop(0, ready("<table><tr><td>a</td></tr></table>")));
  out.pending.table.push_back(crop(1, ready("")));

  finalize_deferred(out);

  REQUIRE(out.tables.size() == 2);
  CHECK_FALSE(out.tables[0].html.empty());
  CHECK(out.tables[1].html.empty());
  CHECK(out.table_degraded);
  CHECK(out.table_warning.find("1 of 2") != std::string::npos);
}

TEST_CASE("finalize_deferred treats a broken-promise future as degraded",
          "[finalize_deferred]") {
  StubFormula sf;
  StubTable st;
  OcrPipelineResult out;
  out.pending.formula_parse = sf.async_result_parser();
  out.pending.formula.push_back(crop(0, broken()));  // .get() throws
  out.pending.table_parse = st.async_result_parser();
  out.pending.table.push_back(crop(0, broken()));

  finalize_deferred(out);  // must not propagate the exception

  REQUIRE(out.formulas.size() == 1);
  REQUIRE(out.tables.size() == 1);
  CHECK(out.formulas[0].latex.empty());
  CHECK(out.tables[0].html.empty());
  CHECK(out.formula_degraded);
  CHECK(out.table_degraded);
}

TEST_CASE("finalize_deferred leaves a clean async batch undegraded",
          "[finalize_deferred]") {
  StubFormula sf;
  StubTable st;
  OcrPipelineResult out;
  out.pending.formula_parse = sf.async_result_parser();
  out.pending.formula.push_back(crop(0, ready("x^2")));
  out.pending.table_parse = st.async_result_parser();
  out.pending.table.push_back(crop(0, ready("<table></table>")));

  finalize_deferred(out);

  CHECK_FALSE(out.formula_degraded);
  CHECK_FALSE(out.table_degraded);
  CHECK(out.formula_warning.empty());
  CHECK(out.table_warning.empty());
}

// Regression: the parser snapshot must be self-contained, so finalize_deferred
// stays valid even if the recognizer that produced it is destroyed first (the
// pipeline-recycle use-after-free the snapshot replaces a raw pointer to fix).
TEST_CASE("finalize_deferred parser snapshot outlives the recognizer",
          "[finalize_deferred]") {
  OcrPipelineResult out;
  {
    StubFormula sf;
    StubTable st;
    out.pending.formula_parse = sf.async_result_parser();
    out.pending.table_parse = st.async_result_parser();
  }  // recognizers destroyed here — snapshots must NOT dangle
  out.pending.formula.push_back(crop(0, ready("a+b")));
  out.pending.table.push_back(crop(0, ready("<table></table>")));

  finalize_deferred(out);  // must not touch the freed recognizers

  REQUIRE(out.formulas.size() == 1);
  REQUIRE(out.tables.size() == 1);
  CHECK(out.formulas[0].latex == "a+b");
  CHECK_FALSE(out.tables[0].html.empty());
}
