#pragma once

// nv_image_pod.h — a GpuImage-shaped POD used across the table/formula pimpl
// bridges. Neutral (no interface headers, no CUDA): reconstructed as a
// decode::GpuImage on the old-headers side and built from a backend::ImageView
// on the new-headers side.

#include <cstddef>

namespace turbo_ocr::nvidia {

struct GpuImagePod {
  void *data = nullptr;
  std::size_t step = 0;
  int rows = 0;
  int cols = 0;
};

} // namespace turbo_ocr::nvidia
