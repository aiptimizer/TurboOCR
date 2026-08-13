#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "nvidia/stages/table_recognizer.h"
#include "nvidia/stages/vlm_table.h"

namespace turbo_ocr::table {

// Remote VLM table backend (PaddleOCR-VL via vLLM, OTSL->HTML). Reads the
// VLLM_TABLE_* env in load(). Ignores page_ocr (the VLM reads the table image
// directly). Wires the previously-orphaned VLMTable into the pipeline.
class VLMTableRecognizer final : public ITableRecognizer {
public:
  [[nodiscard]] bool load() override;
  [[nodiscard]] std::vector<router::TableResult>
  run(const GpuImage &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, cudaStream_t stream) override;
  [[nodiscard]] bool is_ready() const noexcept override {
    return vlm_ && vlm_->is_ready();
  }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "vlm";
  }

private:
  std::unique_ptr<VLMTable> vlm_;
};

} // namespace turbo_ocr::table
