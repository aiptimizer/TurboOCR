#pragma once

// ITableRecognizer — de-CUDA-typed table-structure recognition.
//
// This is the rebuild of src/backends/nvidia/stages/table_recognizer.h. Three
// changes make the interface genuinely backend-agnostic (today it leaks CUDA and
// so only the GPU class can implement it, forcing CpuOcrPipeline to hold a
// concrete CPU twin):
//   1. GpuImage      -> backend::ImageView   (any device-resident page image)
//   2. cudaStream_t  -> backend::DeviceQueue& (opaque ordering primitive)
//   3. set_cell_recognizer(recognition::PaddleRec*) ->
//      set_cell_recognizer(backend::IRecognizer*)  (the page recognizer as an
//      interface, so a SLANeXt/MPSGraph/MIGraphX structure backend fills empty
//      cells with whatever recognizer the active backend provides)
//
// Everything else (load/run batch contract, async decouple for VLM backends,
// self-contained deferred parser) is preserved verbatim so ONE registry + ONE
// dispatch serves every backend.

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/device_queue.h" // backend::DeviceQueue
#include "turbo_ocr/backend/image_view.h"    // backend::ImageView
#include "turbo_ocr/backend/stages.h"        // backend::IRecognizer
#include "turbo_ocr/base/geometry/box.h"   // turbo_ocr::Box
#include "turbo_ocr/core/types.h"          // turbo_ocr::OCRResultItem
#include "turbo_ocr/core/router_types.h"   // router::TableResult

namespace turbo_ocr::backend_routing { struct BackendSpec; }

namespace turbo_ocr::backend {

// Backend-agnostic table-structure recognition. Implementations may be a split
// device encoder + host GRU decode (SLANeXt) or a remote VLM. Batch-shaped: load
// on-disk artefacts (or health-check a remote endpoint), then answer run() with
// one router::TableResult per input region, in input order. Local backends loop
// internally; the VLM backend keeps its own continuous-batching crop fanout.
class ITableRecognizer {
public:
  virtual ~ITableRecognizer() noexcept = default;

  // Load every artefact the backend needs. Backends read their own paths from
  // env (TABLE_SLANEXT_*, VLLM_TABLE_*) so this signature stays uniform; a remote
  // VLM uses it as a health-check + model resolve. Return false => pipeline
  // disables tables cleanly.
  [[nodiscard]] virtual bool load() = 0;

  // Recognize each table region. `page` is a device-resident page image;
  // `regions` are page-coordinate boxes already adjusted (crop margin / detunion)
  // by the caller. `page_ocr` is the page's text-rec output (the cell-fill source
  // for local structure backends; the VLM backend ignores it). `queue` orders the
  // device work. Returns one TableResult per region in input order; an empty html
  // marks a per-region failure (never resized shorter). TableResult.layout_id is
  // left at -1 here and stamped by the caller.
  [[nodiscard]] virtual std::vector<router::TableResult>
  run(const backend::ImageView &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr,
      backend::DeviceQueue &queue) = 0;

  // Opt-in: give a local structure backend the page text recognizer so it can
  // OCR grid cells the page detector left empty (per-cell crop fill). Remote
  // (VLM) backends ignore it. Set by the pipeline before run(); not owned. Now
  // an interface pointer, so any backend's recognizer satisfies it.
  //
  // CONTRACT for overrides — "unwrap-or-null, then forward unconditionally".
  // An implementation typically downcasts `rec` to its own concrete recognizer.
  // It MUST treat a null or foreign-backend `rec` as "clear the slot": unwrap
  // into a local defaulted to nullptr and forward that local every time. Never
  // make the forward conditional on a successful downcast — that silently keeps
  // the previously installed pointer, which the caller has just told us is no
  // longer the active recognizer (and which a pipeline recycle may have freed).
  // This is generic seam policy: it lives here so no vendor arm re-forks it.
  virtual void set_cell_recognizer(backend::IRecognizer * /*rec*/) noexcept {}

  // --- Async decouple (opt-in) — mirrors IFormulaRecognizer ----------------
  // Remote (VLM) table backends submit crops non-blocking on the device worker
  // and await+convert (OTSL->HTML) off it. Local structure backends (SLANeXt)
  // need page_ocr for cell-fill and have no network wait, so they leave these
  // defaulted and keep the synchronous run().
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  // One raw-response (OTSL/HTML) future per region, input order, non-blocking.
  [[nodiscard]] virtual std::vector<std::future<std::string>>
  submit_async(const backend::ImageView & /*page*/,
               const std::vector<Box> & /*regions*/,
               backend::DeviceQueue & /*queue*/) {
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

// NOTE deliberately NO make_table_recognizer(std::string_view) here. Before
// the backend:: rename that declaration silently resolved to the OLD
// CUDA-typed definition (part of the ODR hazard this header's move fixed);
// local engines are minted by the vendor Backend::make_table_recognizer(spec)
// instead.

// Overload from a resolved backend_routing::BackendSpec: kind==Openai -> the
// generic OpenAIEndpoint; kind==Local -> the engine-keyed factory above.
std::unique_ptr<ITableRecognizer>
make_table_recognizer(const backend_routing::BackendSpec &spec);

} // namespace turbo_ocr::backend
