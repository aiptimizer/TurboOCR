// NvFormulaRecognizer implementation (NEW-headers side). Converts interface
// args to bridge PODs, forwards to NvFormulaImpl, and maps the neutral
// FormulaResultPod back to the new backend::FormulaEngineResult.

#include "nvidia/stages/nv_formula_recognizer.h"

#include "nvidia/support/cuda_common.h"       // cuda_stream
#include "nvidia/support/nv_image_pod.h"      // GpuImagePod
#include "nvidia/stages/nv_formula_bridge.h" // NvFormulaImpl, FormulaResultPod, make_nv_formula_impl

namespace turbo_ocr::nvidia {

NvFormulaRecognizer::NvFormulaRecognizer(std::string engine)
    : engine_(std::move(engine)), impl_(make_nv_formula_impl(engine_)) {}
NvFormulaRecognizer::~NvFormulaRecognizer() noexcept = default;

bool NvFormulaRecognizer::load_model_dir(const std::string &model_dir) {
  return impl_ && impl_->load_model_dir(model_dir);
}

bool NvFormulaRecognizer::load_tokenizer(const std::string &path) {
  return impl_ && impl_->load_tokenizer(path);
}

std::vector<turbo_ocr::backend::FormulaEngineResult>
NvFormulaRecognizer::run(const backend::ImageView &page,
                         const std::vector<turbo_ocr::Box> &boxes,
                         backend::DeviceQueue &queue) {
  std::vector<turbo_ocr::backend::FormulaEngineResult> out;
  if (!impl_)
    return out;
  const GpuImagePod pod{page.data, page.step, page.rows, page.cols};
  auto pods = impl_->run(pod, boxes, static_cast<void *>(cuda_stream(queue)));
  out.reserve(pods.size());
  for (auto &p : pods)
    out.push_back(turbo_ocr::backend::FormulaEngineResult{
        std::move(p.latex), p.token_count, p.hit_eos, p.ok});
  return out;
}

bool NvFormulaRecognizer::is_ready() const noexcept {
  return impl_ && impl_->is_ready();
}

void NvFormulaRecognizer::set_context_hint(bool page_has_cjk) noexcept {
  if (impl_) impl_->set_context_hint(page_has_cjk);
}

bool NvFormulaRecognizer::wants_context_hint() const noexcept {
  return impl_ && impl_->wants_context_hint();
}

} // namespace turbo_ocr::nvidia
