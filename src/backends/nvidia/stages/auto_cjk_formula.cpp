#include "nvidia/stages/auto_cjk_formula.h"

#include <filesystem>
#include <iostream>

#include "nvidia/stages/ppformulanet_ort.h"

// text_has_cjk / cjk_stats live in cjk_stats.cpp (turbo_ocr_common) so the
// unit suite can exercise them without linking the CUDA backends.

namespace turbo_ocr::formula {

AutoCjkFormula::AutoCjkFormula()
    : fast_(std::make_unique<PPFormulaNetOrt>("ppformulanet_s")),
      cjk_(std::make_unique<PPFormulaNetOrt>("ppformulanet_plus_m")) {}

AutoCjkFormula::~AutoCjkFormula() noexcept = default;

bool AutoCjkFormula::load_model_dir(const std::string &model_dir) {
  namespace fs = std::filesystem;
  cjk_model_dir_ =
      (fs::path(model_dir).parent_path() / "ppformulanet_plus_m").string();
  if (!fast_->load_model_dir(model_dir)) {
    std::cerr << "[auto_cjk] FATAL: plus-S backend failed to load from "
              << model_dir << '\n';
    return false;
  }
  if (!cjk_->load_model_dir(cjk_model_dir_)) {
    std::cerr << "[auto_cjk] FATAL: plus-M backend failed to load from "
              << cjk_model_dir_
              << " (FORMULA_BACKEND=auto needs both models; use "
                 "FORMULA_BACKEND=ppformulanet_s for plus-S only)\n";
    return false;
  }
  std::cout << "[auto_cjk] loaded plus-S (" << model_dir << ") + plus-M ("
            << cjk_model_dir_ << "); per-crop CJK escalation active\n";
  return true;
}

bool AutoCjkFormula::load_tokenizer(const std::string &path) {
  namespace fs = std::filesystem;
  const std::string cjk_tok =
      (fs::path(cjk_model_dir_) / "tokenizer.json").string();
  return fast_->load_tokenizer(path) && cjk_->load_tokenizer(cjk_tok);
}

std::vector<FormulaEngineResult>
AutoCjkFormula::run(const GpuImage &page, const std::vector<Box> &boxes,
                    cudaStream_t stream) {
  // Fast pass: -S over every crop (EN/Latin — the common case).
  std::vector<FormulaEngineResult> res = fast_->run(page, boxes, stream);

  // Escalate a crop to plus-M when EITHER the page has CJK text (plus-M is the
  // stronger model — wins on hard math on CJK pages too, not just Chinese
  // glyphs) OR -S's own output shows CJK it mangled. On a pure-EN page neither
  // fires, so EN crops keep -S speed and accuracy (no regression).
  std::vector<Box> cjk_boxes;
  std::vector<std::size_t> cjk_idx;
  for (std::size_t i = 0; i < res.size() && i < boxes.size(); ++i) {
    if (res[i].ok && (page_has_cjk_ || text_has_cjk(res[i].latex))) {
      cjk_boxes.push_back(boxes[i]);
      cjk_idx.push_back(i);
    }
  }
  if (!cjk_boxes.empty()) {
    std::vector<FormulaEngineResult> cjk_res =
        cjk_->run(page, cjk_boxes, stream);
    for (std::size_t j = 0; j < cjk_res.size() && j < cjk_idx.size(); ++j)
      res[cjk_idx[j]] = std::move(cjk_res[j]);
  }
  return res;
}

bool AutoCjkFormula::is_ready() const noexcept {
  return fast_ && cjk_ && fast_->is_ready() && cjk_->is_ready();
}

} // namespace turbo_ocr::formula
