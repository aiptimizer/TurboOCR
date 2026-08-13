#pragma once

// NvFormulaRecognizer — the NEW backend-agnostic backend::IFormulaRecognizer
// (include/turbo_ocr/backend/formula_recognizer.h) for NVIDIA, wrapping
// the proven PP-FormulaNet-S ORT-CUDA host-loop recognizer
// (formula::PPFormulaNetOrt) behind the NvFormulaImpl bridge. The two de-CUDA
// changes (ImageView, DeviceQueue) are absorbed at this seam; the split-graph
// FAST host AR loop with on-GPU argmax + KV ping-pong stays untouched.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/formula_recognizer.h" // new backend::IFormulaRecognizer

namespace turbo_ocr::nvidia {

class NvFormulaImpl; // nv_formula_bridge.h

class NvFormulaRecognizer final : public turbo_ocr::backend::IFormulaRecognizer {
public:
  // engine = the routed local key ("ppformulanet_s" default, plus-M, "auto"),
  // forwarded through the bridge to the ONE old-side factory.
  explicit NvFormulaRecognizer(std::string engine = "ppformulanet_s");
  ~NvFormulaRecognizer() noexcept override;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;
  [[nodiscard]] std::vector<turbo_ocr::backend::FormulaEngineResult>
  run(const backend::ImageView &page, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return engine_;
  }
  void set_context_hint(bool page_has_cjk) noexcept override;
  [[nodiscard]] bool wants_context_hint() const noexcept override;

private:
  std::string engine_;
  std::unique_ptr<NvFormulaImpl> impl_;
};

} // namespace turbo_ocr::nvidia
