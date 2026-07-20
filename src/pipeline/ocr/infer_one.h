#pragma once

#include <functional>
#include <string>

#include <cuda_runtime.h>

#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::backend_routing  { struct BackendSpec; }
namespace turbo_ocr::table    { class ITableRecognizer; }
namespace turbo_ocr::formula  { class IFormulaRecognizer; }

namespace turbo_ocr::pipeline {

// Core of OcrPipeline::infer_one, factored out of the orchestrator: run ONE
// already-uploaded crop (the whole image is the single region) through a
// table/formula backend and return the raw recognized string (HTML for table,
// LaTeX/text for formula).
//
//   - inline_spec == nullptr -> use the recognizer returned by the matching
//     pick callback (the NAMED registry backend, or the route default when
//     `backend_name` is empty).
//   - inline_spec != nullptr -> build a TRANSIENT recognizer from that spec on
//     THIS worker thread (so any TRT build binds to the worker's CUDA context),
//     run it once, and discard it. `backend_name` is ignored.
//
// `gpu_img` describes the crop already resident on the GPU; `img_w`/`img_h` are
// its dimensions (used to span the full-image region box). Returns "" for the
// legitimately-empty result. Throws RoutingConfigError (from the factory) when
// an inline spec names an engine invalid for the modality, and
// BackendUnavailableError when the explicitly requested backend is null/not
// ready — both surfaced unchanged to the caller.
std::string infer_one_region(
    const GpuImage &gpu_img, int img_w, int img_h, cudaStream_t stream,
    const std::string &modality, const std::string &backend_name,
    const backend_routing::BackendSpec *inline_spec,
    const std::function<table::ITableRecognizer *(const std::string &)>
        &pick_table,
    const std::function<formula::IFormulaRecognizer *(const std::string &)>
        &pick_formula);

} // namespace turbo_ocr::pipeline
