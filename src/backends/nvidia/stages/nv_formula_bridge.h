#pragma once

// nv_formula_bridge.h — NEUTRAL boundary between the new and old formula
// interfaces. The namespaces are DISTINCT now (new = backend::IFormulaRecognizer
// / backend::FormulaEngineResult; old = formula::…) — the bridge survives NOT
// for a name clash but because the OLD headers drag <cuda_runtime.h> and
// GpuImage into any TU including them, which the new-world TU must stay free
// of. The bridge speaks a POD result + Box + a POD image + void* stream only.
//
//   nv_formula_recognizer.cpp       (new headers) -> backend::IFormulaRecognizer
//   nv_formula_recognizer_impl.cpp  (old headers) -> NvFormulaImpl over the
//                                    proven formula::PPFormulaNetOrt

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "nvidia/support/nv_image_pod.h"    // GpuImagePod
#include "turbo_ocr/base/geometry/box.h" // turbo_ocr::Box

namespace turbo_ocr::nvidia {

// Neutral mirror of formula::FormulaEngineResult (identical fields).
struct FormulaResultPod {
  std::string latex;
  std::size_t token_count = 0;
  bool hit_eos = false;
  bool ok = true;
};

class NvFormulaImpl {
public:
  virtual ~NvFormulaImpl() = default;
  [[nodiscard]] virtual bool load_model_dir(const std::string &model_dir) = 0;
  [[nodiscard]] virtual bool load_tokenizer(const std::string &path) = 0;
  [[nodiscard]] virtual std::vector<FormulaResultPod>
  run(const GpuImagePod &page, const std::vector<turbo_ocr::Box> &boxes,
      void *stream) = 0;
  [[nodiscard]] virtual bool is_ready() const = 0;
  // Per-page CJK routing hint — must cross the bridge or the auto composite's
  // page-level escalation silently degrades to per-crop-output-only.
  virtual void set_context_hint(bool page_has_cjk) = 0;
  [[nodiscard]] virtual bool wants_context_hint() const = 0;
};

// engine = the routed local engine key ("ppformulanet_s", "ppformulanet_plus_m",
// "auto", …), forwarded to the OLD-side formula::make_formula_recognizer — the
// ONE factory that knows how to build each local recognizer (incl. the auto
// CJK ladder). Returns null for an engine that factory rejects.
std::unique_ptr<NvFormulaImpl> make_nv_formula_impl(const std::string &engine);

} // namespace turbo_ocr::nvidia
