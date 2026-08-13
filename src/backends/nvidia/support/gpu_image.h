#pragma once

#include <cstddef>

namespace turbo_ocr::decode {

// Lightweight non-owning GPU image descriptor (replaces cv::cuda::GpuMat).
// Supports designated initializers: GpuImage{.data = ptr, .step = s, .rows = h, .cols = w}
struct GpuImage {
  void *data = nullptr;
  std::size_t step = 0;
  int rows = 0;
  int cols = 0;

  [[nodiscard]] constexpr bool empty() const noexcept { return data == nullptr || rows == 0 || cols == 0; }
};

} // namespace turbo_ocr::decode

// Convenience alias at turbo_ocr level for backward compatibility
namespace turbo_ocr {
using decode::GpuImage;
} // namespace turbo_ocr
