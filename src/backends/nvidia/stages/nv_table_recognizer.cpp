// NvTableRecognizer implementation (NEW-headers side of the bridge). Converts
// the de-CUDA'd interface args to the neutral bridge POD types and forwards to
// NvTableImpl. Includes nv_stages.h to unwrap a backend::IRecognizer* down to
// the concrete recognition::PaddleRec* the SLANeXt wrapper wants — that is
// namespace-safe here because this TU never includes the OLD
// nvidia/stages/table_recognizer.h.

#include "nvidia/stages/nv_table_recognizer.h"

#include "nvidia/support/cuda_common.h"     // cuda_stream, to_gpu_image
#include "nvidia/stages/nv_stages.h"       // NvRecognizer (to unwrap IRecognizer -> PaddleRec)
#include "nvidia/stages/nv_table_bridge.h" // NvTableImpl, GpuImagePod, make_nv_table_impl

namespace turbo_ocr::nvidia {

NvTableRecognizer::NvTableRecognizer() : impl_(make_nv_table_impl()) {}
NvTableRecognizer::~NvTableRecognizer() noexcept = default;

bool NvTableRecognizer::load() { return impl_ && impl_->load(); }

std::vector<turbo_ocr::router::TableResult>
NvTableRecognizer::run(const backend::ImageView &page,
                       const std::vector<turbo_ocr::Box> &regions,
                       const std::vector<turbo_ocr::OCRResultItem> &page_ocr,
                       backend::DeviceQueue &queue) {
  if (!impl_)
    return {};
  const GpuImagePod pod{page.data, page.step, page.rows, page.cols};
  return impl_->run(pod, regions, page_ocr,
                    static_cast<void *>(cuda_stream(queue)));
}

void NvTableRecognizer::set_cell_recognizer(backend::IRecognizer *rec) noexcept {
  if (!impl_)
    return;
  // The NVIDIA table wrapper fills empty cells via a concrete PaddleRec. Unwrap
  // the interface pointer; a non-NVIDIA recognizer here is a wiring error.
  // Seam contract (backend::ITableRecognizer::set_cell_recognizer): unwrap-or-
  // null, then forward UNCONDITIONALLY so null/foreign clears the slot.
  void *paddle = nullptr;
  if (auto *nv = dynamic_cast<NvRecognizer *>(rec))
    paddle = static_cast<void *>(nv->native());
  impl_->set_cell_recognizer(paddle);
}

bool NvTableRecognizer::is_ready() const noexcept {
  return impl_ && impl_->is_ready();
}

} // namespace turbo_ocr::nvidia
