#pragma once

#include <functional>
#include <future>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/types.h"          // OCRResultItem
#include "nvidia/support/gpu_image.h"
#include "nvidia/stages/formula_recognizer.h"
#include "turbo_ocr/backend/routing_config.h"
#include "nvidia/stages/table_recognizer.h"

namespace turbo_ocr::vlm {

// One generic OpenAI-compatible endpoint backend implementing BOTH the formula
// and table recognizer interfaces. The only per-modality difference is the
// response parser (Otsl|Latex|Text|Raw, from the BackendSpec). Crops are
// PNG-encoded and submitted through the process-global VLMCropPool, so this
// shares vLLM's continuous-batching coalescing with the env-configured
// VLMFormula/VLMTable clients.
//
// Constructed from a backend_routing::BackendSpec (kind == Openai). Adding any new
// modality to any OpenAI-compatible model is then a config entry, zero code.
class OpenAIEndpoint final : public formula::IFormulaRecognizer,
                             public table::ITableRecognizer {
public:
  explicit OpenAIEndpoint(backend_routing::BackendSpec spec);

  // --- shared (overrides the identical pure-/virtuals in both bases) ---
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "openai";
  }

  // Async decouple: a remote endpoint is always async-capable once ready. One
  // submit/parse covers both the formula and table interfaces (the futures are
  // raw responses; parse_one applies the configured parser per modality).
  [[nodiscard]] bool supports_async() const noexcept override { return ready_; }
  [[nodiscard]] std::vector<std::future<std::string>>
  submit_async(const GpuImage &page, const std::vector<Box> &boxes,
               cudaStream_t stream) override;
  [[nodiscard]] std::string
  parse_async_result(const std::string &raw) const override {
    return parse_one(raw);
  }
  // Snapshot the configured parser (by value) into a self-contained callable so
  // the deferred finalize never dereferences this object after a pipeline
  // recycle frees it. Defined in the .cpp so this header need not include
  // turbo_ocr/analysis/vlm/openai_policy.h (and its curl / nlohmann dependencies) — the
  // body is a one-line forward to openai_policy::parse_with.
  [[nodiscard]] std::function<std::string(const std::string &)>
  async_result_parser() const override;

  // GET <base_url>/v1/models; resolves the served-model-name when unset.
  // Returns false (=> caller disables the modality cleanly) if unreachable.
  bool health_check();

  // --- IFormulaRecognizer ---
  [[nodiscard]] bool load_model_dir(const std::string &) override {
    return health_check();
  }
  [[nodiscard]] bool load_tokenizer(const std::string &) override { return true; }
  [[nodiscard]] std::vector<formula::FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  // --- ITableRecognizer ---
  [[nodiscard]] bool load() override { return health_check(); }
  [[nodiscard]] std::vector<router::TableResult>
  run(const GpuImage &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, cudaStream_t stream) override;

private:
  // The crop/submit/await machinery is NOT declared here: it is the shared,
  // device-free policy in turbo_ocr/analysis/vlm/openai_policy.h, which this class calls
  // with a D2H closure for the one device-specific step (see the .cpp). Keeping
  // it out of this header also keeps its curl/json dependencies out of the two
  // TUs that include this file just to construct an endpoint.
  std::string parse_one(const std::string &raw) const;

  backend_routing::BackendSpec spec_;
  bool                 ready_ = false;
};

} // namespace turbo_ocr::vlm
