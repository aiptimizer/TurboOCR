// nv_table_recognizer_impl.cpp — OLD-headers side of the table bridge. This TU
// includes the EXISTING nvidia/stages/slanext_table_recognizer.h +
// slanext_table_recognizer.h and wraps the proven SlanextTableRecognizer. It
// MUST NOT include the new backend/table_recognizer.h (it must stay CUDA-free;
// this TU exists precisely to confine <cuda_runtime.h>/GpuImage
// clash) — it only speaks the neutral bridge vocabulary from nv_table_bridge.h.

#include "nvidia/stages/nv_table_bridge.h"

#include <cuda_runtime.h>

#include "nvidia/support/gpu_image.h"
#include "nvidia/stages/paddle_rec.h"
#include "nvidia/stages/table_recognizer.h"
#include "nvidia/stages/slanext_table_recognizer.h"

namespace turbo_ocr::nvidia {

namespace {
class SlanextImpl final : public NvTableImpl {
public:
  bool load() override { return rec_.load(); }

  std::vector<router::TableResult>
  run(const GpuImagePod &page, const std::vector<turbo_ocr::Box> &regions,
      const std::vector<turbo_ocr::OCRResultItem> &page_ocr,
      void *stream) override {
    const decode::GpuImage img{.data = page.data,
                               .step = page.step,
                               .rows = page.rows,
                               .cols = page.cols};
    return rec_.run(img, regions, page_ocr,
                    static_cast<cudaStream_t>(stream));
  }

  void set_cell_recognizer(void *paddle_rec) override {
    rec_.set_cell_recognizer(
        static_cast<recognition::PaddleRec *>(paddle_rec));
  }

  bool is_ready() const override { return rec_.is_ready(); }

private:
  table::SlanextTableRecognizer rec_;
};
} // namespace

std::unique_ptr<NvTableImpl> make_nv_table_impl() {
  return std::make_unique<SlanextImpl>();
}

} // namespace turbo_ocr::nvidia
