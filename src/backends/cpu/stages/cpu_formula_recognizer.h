#pragma once

// CpuFormulaRecognizerAdapter — the backend-agnostic backend::IFormulaRecognizer
// (include/turbo_ocr/backend/formula_recognizer.h) for the CpuBackend,
// wrapping the proven formula::OrtFormulaRecognizer (fused PP-FormulaNet-S on
// ORT-CPU). ImageView -> host cv::Mat, DeviceQueue dropped. The recognizer's
// single load(model_dir, tokenizer_json) is exposed through the interface's
// two-step load_model_dir()/load_tokenizer(): the directory is stashed on the
// first call and the real load happens once the tokenizer path arrives.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/formula_recognizer.h" // new IFormulaRecognizer
#include "turbo_ocr/analysis/formula/ppformulanet/ort_formula_recognizer.h"

namespace turbo_ocr::cpu {

class CpuFormulaRecognizerAdapter final
    : public turbo_ocr::backend::IFormulaRecognizer {
public:
  CpuFormulaRecognizerAdapter() = default;
  ~CpuFormulaRecognizerAdapter() noexcept override = default;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;
  [[nodiscard]] std::vector<turbo_ocr::backend::FormulaEngineResult>
  run(const backend::ImageView &page, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "ppformulanet_s";
  }

private:
  formula::OrtFormulaRecognizer impl_;
  std::string model_dir_; // stashed until load_tokenizer() completes the load
};

} // namespace turbo_ocr::cpu
