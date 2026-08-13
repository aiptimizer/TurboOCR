#include "nvidia/stages/formula_recognizer.h"

#include <iostream>
#include <memory>

#include "nvidia/stages/openai_endpoint.h"
#include "nvidia/stages/auto_cjk_formula.h"
#include "nvidia/stages/ppformulanet_ort.h"
#include "nvidia/stages/vlm_formula.h"
#include "turbo_ocr/backend/routing_config.h"

namespace turbo_ocr::formula {

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(std::string_view backend) {
  if (backend == "ppformulanet_s" || backend == "ppformulanet_plus_s") {
    // Pure in-process ORT decoder (no Python sidecar). GPU FAST host-loop only.
    // The shipped bundle's weights ARE PP-FormulaNet_plus-S (see
    // routing_config.cpp); "ppformulanet_plus_s" is normalized to the
    // historical token at the routing seam, but accept it here too so a
    // direct string_view caller cannot regress the alias.
    return std::make_unique<PPFormulaNetOrt>();
  }
  if (backend == "ppformulanet_plus_m") {
    // PP-FormulaNet_plus-M (B6 encoder + 512-d x 6-layer MBart decoder): the
    // Chinese-formula model. In-process ORT FAST host-loop (encoder.onnx + prep.onnx +
    // static-KV decoder_step.onnx, single greedy token/step). The FAST split graphs are
    // required (no fused fallback).
    return std::make_unique<PPFormulaNetOrt>("ppformulanet_plus_m");
  }
  if (backend == "auto") {
    // Opt-in composite: -S over all crops, plus-M re-run on CJK crops only.
    return std::make_unique<AutoCjkFormula>();
  }
  if (backend == "vlm") {
    return std::make_unique<VLMFormula>();
  }
  std::cerr << "[FormulaRecognizer] unknown FORMULA_BACKEND='" << backend
            << "' (expected 'ppformulanet_s'/'ppformulanet_plus_s', "
               "'ppformulanet_plus_m', 'auto', or 'vlm')\n";
  return nullptr;
}

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind == backend_routing::Kind::Openai)
    return std::make_unique<vlm::OpenAIEndpoint>(spec);
  return make_formula_recognizer(std::string_view{spec.engine});
}

} // namespace turbo_ocr::formula
