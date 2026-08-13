// CpuFormulaRecognizerAdapter implementation — maps interface args to the host
// cv::Mat OrtFormulaRecognizer and its per-region LaTeX strings back to the
// neutral FormulaEngineResult value type.

#include "cpu/stages/cpu_formula_recognizer.h"

#include <utility>

#include "cpu/support/host_common.h" // to_mat

namespace turbo_ocr::cpu {

bool CpuFormulaRecognizerAdapter::load_model_dir(const std::string &model_dir) {
  // OrtFormulaRecognizer::load() needs the tokenizer too, so defer the real
  // load until load_tokenizer() supplies it; just stash the directory here.
  model_dir_ = model_dir;
  return true;
}

bool CpuFormulaRecognizerAdapter::load_tokenizer(const std::string &path) {
  return impl_.load(model_dir_, path);
}

std::vector<turbo_ocr::backend::FormulaEngineResult>
CpuFormulaRecognizerAdapter::run(const backend::ImageView &page,
                                 const std::vector<turbo_ocr::Box> &boxes,
                                 backend::DeviceQueue & /*queue*/) {
  std::vector<turbo_ocr::backend::FormulaEngineResult> out;
  auto latexes = impl_.recognize_regions(to_mat(page), boxes);
  out.reserve(latexes.size());
  for (auto &tex : latexes) {
    turbo_ocr::backend::FormulaEngineResult r;
    r.latex = std::move(tex);
    r.token_count = 0;   // CPU recognizer does not surface a token count
    r.hit_eos = false;
    r.ok = true;         // per-region empty LaTeX is "no formula", not failure
    out.push_back(std::move(r));
  }
  return out;
}

bool CpuFormulaRecognizerAdapter::is_ready() const noexcept {
  return impl_.ready();
}

} // namespace turbo_ocr::cpu
