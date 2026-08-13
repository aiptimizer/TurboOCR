#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/base/geometry/box.h"
#include "nvidia/support/gpu_image.h"
// otsl_to_html's ONE declaration. It is device-free and lives in its own header
// so a CPU-only TU can name it without dragging <cuda_runtime.h> in from here.
#include "turbo_ocr/analysis/table/vlm/otsl.h"

namespace turbo_ocr::table {

// VLM-driven table structure backend. Per-page the caller passes a list of
// table region quads; this class crops them out of the GpuImage, PNG-encodes
// them, and POSTs each crop to an OpenAI-compatible chat-completions endpoint
// (vllm-paddlevl, sglang, etc.) with the "Table Recognition:" prompt that
// PaddleOCR-VL-1.5 expects. The model returns an OTSL string which we convert
// to HTML before returning. All in-flight crops share one std::thread group
// so vLLM's continuous batching coalesces them on the server side.
//
// Env vars (mirroring VLMFormula naming so deployments can point both at the
// same vllm-paddlevl endpoint):
//   VLLM_TABLE_BASE_URL    fallback VLLM_BASE_URL (default http://localhost:8003)
//   VLLM_TABLE_MODEL       fallback VLLM_MODEL    (auto-detected via /v1/models)
//   VLLM_TABLE_BATCH       (default 8) — parallel HTTP fanout per page
//   VLLM_TABLE_TIMEOUT_S   (default 60)
//   VLLM_TABLE_MAX_TOKENS  (default 4096)
//   VLLM_TABLE_PROMPT      (default "Table Recognition:")
class VLMTable {
public:
  VLMTable();
  ~VLMTable() noexcept;

  VLMTable(const VLMTable &) = delete;
  VLMTable &operator=(const VLMTable &) = delete;

  // Resolve VLM endpoint + model. Returns false when the endpoint is
  // unreachable so the pipeline can fall back to "tables disabled".
  [[nodiscard]] bool init();

  [[nodiscard]] bool is_ready() const noexcept { return ready_; }

  // For each `region` quad, returns the HTML string (OTSL converted) from
  // the VLM. Empty strings on per-region failure.
  [[nodiscard]] std::vector<std::string>
  run(const GpuImage &page,
      const std::vector<Box> &regions,
      cudaStream_t stream);

  // Async variant: D2H + PNG-encode + submit to VLMCropPool, return futures
  // without blocking. Caller resolves futures and converts OTSL→HTML.
  [[nodiscard]] std::vector<std::future<std::string>>
  submit_async(const GpuImage &page,
               const std::vector<Box> &regions,
               cudaStream_t stream);

private:
  bool single_request(const std::vector<uint8_t> &crop_png,
                      std::string &out_otsl);

  std::string base_url_;
  std::string model_;
  std::string prompt_;
  int         batch_       = 8;
  int         timeout_s_   = 60;
  int         max_tokens_  = 4096;
  bool        ready_       = false;
};

// otsl_to_html is declared in turbo_ocr/analysis/table/vlm/otsl.h (included above) — it
// has no device dependency and must stay reachable from CPU-only TUs.

} // namespace turbo_ocr::table
