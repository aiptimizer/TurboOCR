#include "nvidia/stages/vlm_table_recognizer.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace turbo_ocr::table {

bool VLMTableRecognizer::load() {
  // Build-then-commit: vlm_ stays null on failure so is_ready() reports false
  // and the pipeline disables tables cleanly (no half-initialized backend).
  auto v = std::make_unique<VLMTable>();
  if (!v->init()) {
    std::cerr << "[vlm-table] init/health-check failed — tables disabled\n";
    return false;
  }
  vlm_ = std::move(v);
  std::cout << "[Pipeline] Table backend=vlm (PaddleOCR-VL via vLLM, OTSL->HTML)\n";
  return true;
}

std::vector<router::TableResult>
VLMTableRecognizer::run(const GpuImage &page, const std::vector<Box> &regions,
                        const std::vector<OCRResultItem> & /*page_ocr*/,
                        cudaStream_t stream) {
  // VLMTable reads the table crop directly; page_ocr unused. It returns one
  // HTML per region in input order (fewer/empty only on transport failure).
  std::vector<std::string> htmls;
  if (vlm_) {
    htmls = vlm_->run(page, regions, stream);
  } else {
    // Precondition breach (run() before a successful load()). Surface it and
    // degrade every region rather than dereference a null backend.
    std::cerr << "[vlm-table] run() called with no backend loaded — all "
                 "regions degraded\n";
  }

  if (htmls.size() < regions.size())
    std::cerr << "[vlm-table] VLM returned " << htmls.size() << " of "
              << regions.size()
              << " table HTMLs — missing regions left empty (degraded, not empty input)\n";

  // Always emit exactly one TableResult per region, in input order; an empty
  // html marks a per-region failure (the interface never resizes shorter).
  std::vector<router::TableResult> out;
  out.reserve(regions.size());
  for (std::size_t i = 0; i < regions.size(); ++i) {
    router::TableResult tr;
    tr.layout_id = -1;  // stamped by caller
    tr.html      = (i < htmls.size()) ? std::move(htmls[i]) : std::string{};
    if (tr.html.empty())
      std::cerr << "[vlm-table] region " << i
                << " produced empty HTML (VLM transport/parse failure)\n";
    tr.score = 0.0f;
    tr.box   = regions[i];
    out.push_back(std::move(tr));
  }
  return out;
}

} // namespace turbo_ocr::table
