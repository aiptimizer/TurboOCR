#pragma once

// NvTableRecognizer — the NEW backend-agnostic backend::ITableRecognizer
// (include/turbo_ocr/backend/table_recognizer.h) for NVIDIA. It holds
// an opaque NvTableImpl (nv_table_bridge.h) that wraps the proven
// table::SlanextTableRecognizer in a separate TU, so the de-CUDA'd interface is
// satisfied without touching the existing SLANeXt code.
//
// The three de-CUDA changes are absorbed here:
//   ImageView   -> GpuImagePod -> decode::GpuImage (in the impl TU)
//   DeviceQueue -> native_handle() void* stream    -> cudaStream_t
//   set_cell_recognizer(backend::IRecognizer*) -> unwrap to PaddleRec* void*

#include <memory>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/table_recognizer.h" // new backend::ITableRecognizer

namespace turbo_ocr::nvidia {

class NvTableImpl; // nv_table_bridge.h

class NvTableRecognizer final : public turbo_ocr::backend::ITableRecognizer {
public:
  NvTableRecognizer();
  ~NvTableRecognizer() noexcept override;

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
  std::unique_ptr<NvTableImpl> impl_;
};

} // namespace turbo_ocr::nvidia
