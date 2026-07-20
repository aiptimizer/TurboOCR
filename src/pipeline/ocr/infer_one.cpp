#include "infer_one.h"

#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"

namespace turbo_ocr::pipeline {

namespace {

// Resolve the recognizer for one /infer region, uniformly for either modality:
//   - inline_spec != nullptr -> build a TRANSIENT backend from the spec and load
//     it HERE (on the worker thread, so any TRT build binds to this CUDA
//     context). Ownership stays in `owner`, kept alive for the caller's run();
//     `backend_name` is ignored. A null from the factory (unknown engine) or a
//     failed load surfaces as unavailable below — never dereferenced.
//   - inline_spec == nullptr -> borrow the NAMED registry backend (route default
//     when `backend_name` is empty); not owned.
// Throws BackendUnavailableError (identical message to the per-modality callers)
// when the resolved backend is null or not ready, so a mis-routed request fails
// loud rather than silently producing nothing. Any factory exception (e.g. an
// invalid engine for the modality) propagates unchanged.
template <class Rec, class MakeInline, class LoadInline, class PickNamed>
Rec *resolve_recognizer(const backend_routing::BackendSpec *inline_spec,
                        const std::string &backend_name,
                        const std::string &modality,
                        std::unique_ptr<Rec> &owner, const MakeInline &make,
                        const LoadInline &load, const PickNamed &pick) {
  Rec *rec = nullptr;
  if (inline_spec) {
    owner = make(*inline_spec);
    if (owner && load(*owner)) rec = owner.get();
  } else {
    rec = pick(backend_name);
  }
  if (!rec || !rec->is_ready())
    throw turbo_ocr::BackendUnavailableError(
        modality + " backend '" + (inline_spec ? "<inline>" : backend_name) +
        "' is unavailable (not loaded/ready)");
  return rec;
}

}  // namespace

std::string infer_one_region(
    const GpuImage &gpu_img, int img_w, int img_h, cudaStream_t stream,
    const std::string &modality, const std::string &backend_name,
    const backend_routing::BackendSpec *inline_spec,
    const std::function<table::ITableRecognizer *(const std::string &)>
        &pick_table,
    const std::function<formula::IFormulaRecognizer *(const std::string &)>
        &pick_formula) {
  using turbo_ocr::Box;

  // The whole crop is the single region: corners [tl, tr, br, bl].
  const std::vector<Box> regions{
      Box{{{{0, 0}, {img_w, 0}, {img_w, img_h}, {0, img_h}}}}};

  if (modality == "table") {
    std::unique_ptr<table::ITableRecognizer> transient;
    auto *const rec = resolve_recognizer<table::ITableRecognizer>(
        inline_spec, backend_name, modality, transient,
        [](const backend_routing::BackendSpec &s) {
          return table::make_table_recognizer(s);
        },
        [](table::ITableRecognizer &r) { return r.load(); }, pick_table);
    cudaStreamSynchronize(stream);
    auto r = rec->run(gpu_img, regions, /*page_ocr=*/{}, stream);
    return r.empty() ? std::string() : std::move(r[0].html);
  }
  if (modality == "formula") {
    std::unique_ptr<formula::IFormulaRecognizer> transient;
    auto *const rec = resolve_recognizer<formula::IFormulaRecognizer>(
        inline_spec, backend_name, modality, transient,
        [](const backend_routing::BackendSpec &s) {
          return formula::make_formula_recognizer(s);
        },
        [inline_spec](formula::IFormulaRecognizer &r) {
          return r.load_model_dir(inline_spec->model_path) &&
                 r.load_tokenizer("");
        },
        pick_formula);
    cudaStreamSynchronize(stream);
    auto r = rec->run(gpu_img, regions, stream);
    if (!r.empty() && !r[0].ok)
      // Degraded (transport/timeout), not a legitimately-empty formula. /infer
      // returns bare LaTeX, so at least log it (no-silent-failure) rather than let
      // a VLM timeout look identical to an empty input.
      std::cerr << "[infer_one] formula backend degraded for /infer region "
                   "(transport/parse failure, not empty input)\n";
    return r.empty() ? std::string() : std::move(r[0].latex);
  }
  return "";
}

} // namespace turbo_ocr::pipeline
