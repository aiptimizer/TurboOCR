#pragma once

// host_common.h — small, header-only glue between the device-agnostic backend
// interfaces (include/turbo_ocr/backend/*) and the existing CPU classes
// (include/turbo_ocr/**). It is the CPU analogue of nvidia/support/cuda_common.h, but
// far simpler: the Host address space IS host RAM, so an ImageView and a cv::Mat
// alias the SAME bytes — there is no device transfer and no vendor pointer type.
//
//   backend::ImageView (kind=Host)  <->  cv::Mat (BGR8, non-owning)
//   backend::DeviceQueue            ->   nothing (Host queue is a no-op)
//
// Nothing here re-implements pipeline logic; it only translates vocabulary at
// the seam.

#include <opencv2/core.hpp>

#include "turbo_ocr/backend/image_view.h" // backend::ImageView, DeviceKind

namespace turbo_ocr::cpu {

// Wrap a Host ImageView's pixels as a NON-OWNING cv::Mat (BGR8). Zero-copy: the
// Mat header points straight at ImageView::data with the ImageView's row pitch,
// so the buffer's lifetime stays with whoever owns the ImageView. Callers must
// not let the returned Mat outlive the ImageView's backing store.
[[nodiscard]] inline cv::Mat to_mat(const backend::ImageView &v) noexcept {
  if (v.empty())
    return cv::Mat{};
  return cv::Mat(v.rows, v.cols, CV_8UC3, const_cast<void *>(v.data),
                 v.step ? v.step : static_cast<std::size_t>(v.cols) * 3);
}

// Describe an existing host BGR8 cv::Mat as a Host ImageView (zero-copy). The
// Mat must be continuous-per-row 8UC3; step carries the row pitch in bytes.
[[nodiscard]] inline backend::ImageView to_image_view(const cv::Mat &m) noexcept {
  return backend::ImageView{.data = static_cast<void *>(m.data),
                            .step = static_cast<std::size_t>(m.step),
                            .rows = m.rows,
                            .cols = m.cols,
                            .kind = backend::DeviceKind::Host};
}

} // namespace turbo_ocr::cpu
