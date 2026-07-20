#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/types.h"        // OCRResultItem
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/router/router_types.h" // router::TableResult

namespace turbo_ocr::backend_routing { struct BackendSpec; }
namespace turbo_ocr::recognition { class PaddleRec; }

namespace turbo_ocr::table {

// Backend-agnostic table-structure recognition. Implementations may be a
// split TRT encoder + host GRU decode (SLANeXt) or a remote VLM. Contract
// mirrors IFormulaRecognizer but is
// batch-shaped: load on-disk artefacts (or health-check a remote endpoint),
// then answer run() with one router::TableResult per input region, in input
// order. Local backends loop internally; the VLM backend keeps its own
// continuous-batching crop fanout.
class ITableRecognizer {
public:
  virtual ~ITableRecognizer() noexcept = default;

  // Load every artefact the backend needs. Backends read their own paths
  // from env (TABLE_SLANEXT_*, VLLM_TABLE_*)
  // so this signature stays uniform; remote VLM uses it as a health-check +
  // model resolve. Return false => pipeline disables tables cleanly.
  [[nodiscard]] virtual bool load() = 0;

  // Recognize each table region. `regions` are page-coordinate boxes already
  // adjusted (crop margin / detunion) by the caller. `page_ocr` is the page's
  // text-rec output (the cell-fill source for local structure backends; the
  // VLM backend ignores it). Returns one TableResult per region in input
  // order; an empty html marks a per-region failure (never resized shorter).
  // TableResult.layout_id is left at -1 here and stamped by the caller.
  [[nodiscard]] virtual std::vector<router::TableResult>
  run(const GpuImage &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, cudaStream_t stream) = 0;

  // Opt-in: give a local structure backend the page text recognizer so it can
  // OCR grid cells the page detector left empty (per-cell crop fill). Remote
  // (VLM) backends ignore it. Set by the pipeline before run(); not owned.
  virtual void set_cell_recognizer(recognition::PaddleRec * /*rec*/) noexcept {}

  // --- Async decouple (opt-in) — mirrors IFormulaRecognizer ---------------
  // Remote (VLM) table backends submit crops non-blocking on the GPU worker
  // and await+convert (OTSL->HTML) off it. Local structure backends (SLANeXt)
  // need page_ocr for cell-fill and have no network wait, so they
  // leave these defaulted and keep the synchronous run().
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  // One raw-response (OTSL/HTML) future per region, input order, non-blocking.
  [[nodiscard]] virtual std::vector<std::future<std::string>>
  submit_async(const GpuImage & /*page*/, const std::vector<Box> & /*regions*/,
               cudaStream_t /*stream*/) {
    return {};
  }

  // Convert one raw endpoint response into the cell HTML (OTSL->HTML etc.).
  [[nodiscard]] virtual std::string
  parse_async_result(const std::string &raw) const { return raw; }

  // Self-contained parser snapshot for the deferred (async) path — mirrors
  // IFormulaRecognizer::async_result_parser. Captures only value-state so the
  // deferred finalize can outlive a pipeline recycle that frees the recognizer.
  [[nodiscard]] virtual std::function<std::string(const std::string &)>
  async_result_parser() const {
    return [](const std::string &raw) { return raw; };
  }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;

  [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;
};

// Factory. `backend` from TABLE_BACKEND. Unknown keys return nullptr.
//   "slanext"  — SLANeXt encoder-split (TRT FP16 encoder + host GRU)  [default]
//   "vlm"      — OpenAI-compatible vLLM endpoint (OTSL->HTML), env-configured
std::unique_ptr<ITableRecognizer> make_table_recognizer(std::string_view backend);

// Overload from a resolved backend_routing::BackendSpec: kind==Openai -> the generic
// OpenAIEndpoint; kind==Local -> the engine-keyed factory above.
std::unique_ptr<ITableRecognizer> make_table_recognizer(const backend_routing::BackendSpec &spec);

} // namespace turbo_ocr::table
