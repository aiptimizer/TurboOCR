// CpuTableRecognizer implementation — converts the de-CUDA'd interface args to
// the host cv::Mat the SLANeXt CPU backend's ORT encoder reads, and forwards.
//
// This adapter is also what AppleBackend and IntelBackend return from
// make_table_recognizer, so it must stay backend-neutral: the page ImageView and
// the DeviceQueue are passed THROUGH (the wrapped recognizer uses them for
// per-cell crop OCR) alongside the host view of the same pixels.

#include "cpu/stages/cpu_table_recognizer.h"

#include "cpu/support/host_common.h" // to_mat

namespace turbo_ocr::cpu {

bool CpuTableRecognizer::load() { return impl_.load(); }

std::vector<turbo_ocr::router::TableResult>
CpuTableRecognizer::run(const backend::ImageView &page,
                        const std::vector<turbo_ocr::Box> &regions,
                        const std::vector<turbo_ocr::OCRResultItem> &page_ocr,
                        backend::DeviceQueue &queue) {
  // to_mat is the HOST alias of `page` (same bytes) for the ORT-CPU encoder;
  // `page` itself goes with it because the cell recognizer is the active
  // backend's and must see the page in its own address space.
  return impl_.run(to_mat(page), page, regions, page_ocr, queue);
}

void CpuTableRecognizer::set_cell_recognizer(backend::IRecognizer *rec) noexcept {
  // Straight through: the SLANeXt CPU wrapper's cell-fill hook is typed on
  // backend::IRecognizer, so it takes whatever recognizer the active backend
  // installed — an apple::MpsRecognizer or an intel::IntelRecognizer arriving
  // here is a NORMAL configuration, not a wiring error: both backends mint this
  // same adapter for tables. No downcast, so nothing to accidentally make
  // conditional on one succeeding: the seam contract ("forward unconditionally,
  // null clears the slot") holds by construction.
  impl_.set_cell_recognizer(rec);
}

bool CpuTableRecognizer::is_ready() const noexcept { return impl_.ready(); }

} // namespace turbo_ocr::cpu
