// nv_formula_recognizer_impl.cpp — OLD-headers side of the formula bridge.
// Includes the EXISTING formula/ppformulanet/ppformulanet_ort.h and wraps
// the old-interface recognizers. MUST NOT include the new
// backend/formula_recognizer.h (same-namespace ODR clash) — speaks only the
// neutral bridge vocabulary.

#include "nvidia/stages/nv_formula_bridge.h"

#include <cuda_runtime.h>

#include "nvidia/support/gpu_image.h"
#include "nvidia/stages/formula_recognizer.h" // OLD factory: make_formula_recognizer
#include "nvidia/stages/ppformulanet_ort.h"

namespace turbo_ocr::nvidia {

namespace {
class PPFormulaImpl final : public NvFormulaImpl {
public:
  explicit PPFormulaImpl(std::unique_ptr<formula::IFormulaRecognizer> rec)
      : rec_(std::move(rec)) {}

  bool load_model_dir(const std::string &model_dir) override {
    return rec_->load_model_dir(model_dir);
  }
  bool load_tokenizer(const std::string &path) override {
    return rec_->load_tokenizer(path);
  }

  std::vector<FormulaResultPod> run(const GpuImagePod &page,
                                    const std::vector<turbo_ocr::Box> &boxes,
                                    void *stream) override {
    const decode::GpuImage img{.data = page.data,
                               .step = page.step,
                               .rows = page.rows,
                               .cols = page.cols};
    auto res = rec_->run(img, boxes, static_cast<cudaStream_t>(stream));
    std::vector<FormulaResultPod> out;
    out.reserve(res.size());
    for (auto &r : res)
      out.push_back(FormulaResultPod{std::move(r.latex), r.token_count,
                                     r.hit_eos, r.ok});
    return out;
  }

  bool is_ready() const override { return rec_->is_ready(); }

  void set_context_hint(bool page_has_cjk) override {
    rec_->set_context_hint(page_has_cjk);
  }
  bool wants_context_hint() const override {
    return rec_->wants_context_hint();
  }

private:
  std::unique_ptr<formula::IFormulaRecognizer> rec_;
};
} // namespace

std::unique_ptr<NvFormulaImpl> make_nv_formula_impl(const std::string &engine) {
  // ONE factory builds every local engine (plus-S host loop, plus-M MBart
  // host loop, the auto CJK ladder). The bridge previously hardcoded
  // PPFormulaNetOrt("ppformulanet_s"), which silently served the WRONG graph
  // contract for FORMULA_BACKEND=ppformulanet_plus_m (fast/ subdir layout vs
  // plus-M's in-dir decoder_step.onnx -> fatal at load) and made "auto"
  // unreachable on the unified server.
  const std::string eng = engine.empty() ? "ppformulanet_s" : engine;
  auto rec = formula::make_formula_recognizer(eng);
  if (!rec) return nullptr;
  return std::make_unique<PPFormulaImpl>(std::move(rec));
}

} // namespace turbo_ocr::nvidia
