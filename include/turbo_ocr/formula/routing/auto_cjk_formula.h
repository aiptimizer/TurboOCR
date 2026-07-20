#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/common/cjk_stats.h"
#include "turbo_ocr/formula/formula_recognizer.h"

namespace turbo_ocr::formula {

class PPFormulaNetOrt;

// Opt-in composite formula backend (FORMULA_BACKEND=auto). Runs the fast
// EN/Latin PP-FormulaNet-S over every crop, then RE-RUNS only the crops whose
// -S output contains CJK codepoints on the Chinese-capable PP-FormulaNet_plus-M.
//
// Rationale (measured, OmniDocBench 5-doc): the
// entire formula accuracy gap is Chinese content -S structurally cannot decode
// (it emits garbage CJK glyphs, e.g. \frac{抎汎慎沎…}). plus-M cuts formula
// edit-distance 68% on CJK docs. -S's own CJK-glyph emission is a reliable
// per-crop escalation signal: pure-EN crops never contain CJK in -S output, so
// they never pay the plus-M cost — this AVOIDS the small pure-EN regression a
// blanket plus-M default would incur, and needs no page text context.
//
// GPU build only (both children are PPFormulaNetOrt, the CUDA in-process ORT
// backend). Sync path only; supports_async() stays false.
class AutoCjkFormula final : public IFormulaRecognizer {
public:
  AutoCjkFormula();
  ~AutoCjkFormula() noexcept override;

  // `model_dir` is the -S bundle (models/formula/ppformulanet_s). plus-M is
  // resolved as its sibling (…/ppformulanet_plus_m). Both must load.
  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  [[nodiscard]] bool is_ready() const noexcept override;
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "auto_cjk";
  }

  void set_context_hint(bool page_has_cjk) noexcept override {
    page_has_cjk_ = page_has_cjk;
  }
  [[nodiscard]] bool wants_context_hint() const noexcept override { return true; }

private:
  std::unique_ptr<PPFormulaNetOrt> fast_;  // ppformulanet_s (EN/Latin)
  std::unique_ptr<PPFormulaNetOrt> cjk_;   // ppformulanet_plus_m (Chinese)
  std::string cjk_model_dir_;
  bool page_has_cjk_ = false;
};

} // namespace turbo_ocr::formula
