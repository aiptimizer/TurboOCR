#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/formula/formula_recognizer.h"

namespace turbo_ocr::formula {

// VLM backend that talks to an OpenAI-compatible /v1/chat/completions endpoint
// (vllm-server, sglang, etc.) per formula crop. Crops are PNG-encoded,
// base64-wrapped into image_url content blocks, and either batched into a
// single multi-image chat request or sent in parallel one-per-request when
// the server rejects multi-image bodies.
//
// Configured entirely via env vars:
//   VLLM_BASE_URL            (default http://localhost:8000)
//   VLLM_MODEL               (default openbmb/MiniCPM-V-4.6 — auto-detected via /v1/models if unset and a model is loaded)
//   VLLM_FORMULA_BATCH       (default 8)
//   VLLM_FORMULA_TIMEOUT_S   (default 30)
//   VLLM_FORMULA_MAX_TOKENS  (default 512)
//   VLLM_FORMULA_PROMPT      (optional override of the LaTeX-extraction prompt)
class VLMFormula final : public IFormulaRecognizer {
public:
  VLMFormula();
  ~VLMFormula() noexcept override;

  VLMFormula(const VLMFormula &) = delete;
  VLMFormula &operator=(const VLMFormula &) = delete;

  // No on-disk model — load_model_dir health-checks the endpoint and resolves
  // VLLM_MODEL via /v1/models when unset. Returns false if the endpoint is
  // unreachable so the pipeline falls back cleanly to "formulas disabled".
  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  // Async variant: PNG-encode crops (D2H already done), submit to the
  // global VLMCropPool, and return futures immediately without blocking.
  // The caller is responsible for resolving futures and assembling results.
  // Only valid when VLM_BACKEND != "legacy".
  [[nodiscard]] std::vector<std::future<std::string>>
  submit_async(const std::vector<uint8_t> &host_page,
               const GpuImage &page,
               const std::vector<Box> &boxes);

  // IFormulaRecognizer async decouple: D2H the page on `stream` then delegate
  // to the host_page submit_async above. supports_async() is true only on the
  // pool backend (the legacy per-page HTTP path can't be decoupled cleanly).
  [[nodiscard]] bool supports_async() const noexcept override;
  [[nodiscard]] std::vector<std::future<std::string>>
  submit_async(const GpuImage &page, const std::vector<Box> &boxes,
               cudaStream_t stream) override;
  [[nodiscard]] std::string
  parse_async_result(const std::string &raw) const override;
  // Self-contained LaTeX parser snapshot (no `this` capture) so the deferred
  // finalize survives a pipeline recycle that frees this recognizer.
  [[nodiscard]] std::function<std::string(const std::string &)>
  async_result_parser() const override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "vlm";
  }

private:
  bool single_request(const std::vector<uint8_t> &crop_png,
                      std::string &out_latex);
  bool batched_request(const std::vector<std::vector<uint8_t>> &crops_png,
                       std::vector<std::string> &out_latex);

  // Pool backend: parallel PNG encode + global async submit.
  std::vector<FormulaEngineResult>
  run_pool(const std::vector<uint8_t> &host_page,
           const GpuImage &page, const std::vector<Box> &boxes);

  // Legacy backend: per-page batched/parallel HTTP (for VLM_BACKEND=legacy).
  std::vector<FormulaEngineResult>
  run_legacy(const std::vector<uint8_t> &host_page,
             const GpuImage &page, const std::vector<Box> &boxes);

  std::string base_url_;
  std::string model_;
  std::string prompt_;
  int         batch_       = 8;
  int         timeout_s_   = 30;
  int         max_tokens_  = 512;
  int         png_threads_ = 4;
  bool        ready_       = false;
};

} // namespace turbo_ocr::formula
