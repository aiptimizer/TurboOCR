#pragma once

// CpuTableRecognizer — the backend-agnostic backend::ITableRecognizer
// (include/turbo_ocr/backend/table_recognizer.h) for the CpuBackend,
// wrapping the proven table::OrtSlanextTableRecognizer (ORT-CPU encoder + host
// GRU decode). The two de-CUDA changes are absorbed here:
//   ImageView   -> host cv::Mat (zero-copy)
//   DeviceQueue -> dropped (Host queue is a synchronous no-op)
//   set_cell_recognizer(backend::IRecognizer*) -> unwrap to OrtPaddleRec*
//
// No bridge TU is needed (as on NVIDIA), because ort_slanext_table.h does not
// pull in the OLD table/table_recognizer.h, so there is no namespace collision.

#include <memory>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/table_recognizer.h"        // new ITableRecognizer
#include "turbo_ocr/analysis/table/slanext/ort_slanext_table.h" // OrtSlanextTableRecognizer

namespace turbo_ocr::cpu {

class CpuTableRecognizer final : public turbo_ocr::backend::ITableRecognizer {
public:
  CpuTableRecognizer() = default;
  ~CpuTableRecognizer() noexcept override = default;

  [[nodiscard]] bool load() override;
  [[nodiscard]] std::vector<turbo_ocr::router::TableResult>
  run(const backend::ImageView &page, const std::vector<turbo_ocr::Box> &regions,
      const std::vector<turbo_ocr::OCRResultItem> &page_ocr,
      backend::DeviceQueue &queue) override;
  void set_cell_recognizer(backend::IRecognizer *rec) noexcept override;
  [[nodiscard]] bool is_ready() const noexcept override;
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "slanext";
  }

private:
  table::OrtSlanextTableRecognizer impl_;
};

} // namespace turbo_ocr::cpu
