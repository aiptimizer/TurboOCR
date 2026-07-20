#pragma once

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::backend_routing { struct BackendSpec; }

namespace turbo_ocr::formula {

struct FormulaEngineResult {
  std::string latex;
  std::size_t token_count = 0;
  bool        hit_eos = false;
  // Status channel: distinguishes "region legitimately had no formula"
  // (ok==true, empty latex) from "the formula stage failed for this region"
  // (ok==false, e.g. sidecar RPC crash/timeout). The pipeline surfaces ok==false
  // as a per-response degraded flag so a degraded stage never looks identical to
  // a clean empty result. Default true: backends opt in by clearing it on error.
  bool        ok = true;
};

// Backend-agnostic formula recognition interface. Implementations may use
// any execution shape internally — an in-process ONNX Runtime graph driven by
// a host AR loop (PP-FormulaNet-S, on the CUDA or CPU execution provider), or
// a remote VLM endpoint.
//
// Contract: load the on-disk artefact bundle from a directory, optionally
// load a tokenizer, then answer run() with a FormulaEngineResult per input
// box in input order.
class IFormulaRecognizer {
public:
  virtual ~IFormulaRecognizer() noexcept = default;

  [[nodiscard]] virtual bool load_model_dir(const std::string &model_dir) = 0;

  // Tokenizer is backend-defined. Some backends embed their tokenizer in
  // the model bundle; pass an empty string in that case and the call is a
  // no-op that returns true.
  [[nodiscard]] virtual bool load_tokenizer(const std::string &path) = 0;

  [[nodiscard]] virtual std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) = 0;

  // --- Async decouple (opt-in) -------------------------------------------
  // Remote (HTTP/VLM) backends can split run() into a fast non-blocking
  // submit (D2H + PNG-encode + global-pool submit, done on the GPU pipeline
  // worker) and a later await+parse (done OFF the GPU worker, on the work
  // pool / HTTP thread). Local TRT/ORT backends leave these defaulted and
  // keep the synchronous run() — supports_async() stays false for them.
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  // Submit each box's crop and return one raw-response future per box, in
  // input order, WITHOUT blocking on the network. Default: not supported.
  [[nodiscard]] virtual std::vector<std::future<std::string>>
  submit_async(const GpuImage & /*page*/, const std::vector<Box> & /*boxes*/,
               cudaStream_t /*stream*/) {
    return {};
  }

  // Turn one raw endpoint response (resolved from a submit_async future) into
  // the final LaTeX. Default identity; remote backends apply their parser.
  [[nodiscard]] virtual std::string
  parse_async_result(const std::string &raw) const { return raw; }

  // Return a SELF-CONTAINED parser for the deferred (async) path. Obtained on
  // the GPU worker while this recognizer is alive; the returned callable must
  // capture only value-state (no pointer back into the recognizer), so the
  // deferred finalize can outlive a pipeline recycle that frees the recognizer.
  // Default: identity (matches parse_async_result's default). Async backends
  // override to snapshot their parser.
  [[nodiscard]] virtual std::function<std::string(const std::string &)>
  async_result_parser() const {
    return [](const std::string &raw) { return raw; };
  }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;

  [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;

  // Optional per-page routing hint set by the pipeline right before run().
  // `page_has_cjk` is true when the page's recognized TEXT contains CJK — a
  // signal that a composite backend should route this page's formulas to a
  // Chinese-capable model even when a given crop is pure math. Default no-op;
  // only the auto composite consumes it.
  virtual void set_context_hint(bool /*page_has_cjk*/) noexcept {}

  // True only for backends that actually consume set_context_hint(). The
  // pipeline skips its O(page-text) CJK scan for every backend that returns
  // false (all single-model engines), so the hint is computed only when it can
  // change routing. Only the auto composite overrides this.
  [[nodiscard]] virtual bool wants_context_hint() const noexcept { return false; }
};

// Factory. `backend` is the key used to select the implementation
// (typically read from FORMULA_BACKEND). Unknown keys return nullptr.
//
// Supported keys:
//   "ppformulanet_s"  — PP-FormulaNet-S, in-process ONNX Runtime (CUDA);
//                       FAST split-graph host loop only (no fused graph)
//   "vlm"             — OpenAI-compatible vLLM endpoint (env-configured)
std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend);

// Overload from a resolved backend_routing::BackendSpec: kind==Openai -> the generic
// OpenAIEndpoint; kind==Local -> the engine-keyed factory above.
std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const backend_routing::BackendSpec &spec);

} // namespace turbo_ocr::formula
