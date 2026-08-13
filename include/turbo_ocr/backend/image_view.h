#pragma once

// ImageView — device-agnostic, non-owning image handle.
//
// Generalizes turbo_ocr::decode::GpuImage (src/backends/nvidia/support/gpu_image.h)
// by adding a `DeviceKind kind` field so the SAME descriptor can name a pixel
// buffer that lives in CUDA VRAM, a Metal MTLBuffer, a HIP allocation, a Level
// Zero / OpenCL device buffer, or plain host RAM. It carries no device SDK
// types, so it lives in turbo_ocr_common and every layer above the Backend seam
// (pipeline, stages, table/formula, kernels) speaks ImageView instead of a
// vendor pointer.
//
// Contract (identical to GpuImage, plus `kind`):
//   * NON-OWNING. The buffer's lifetime is managed elsewhere (the Backend's
//     device allocator / decoder). Copying an ImageView copies the descriptor,
//     never the pixels.
//   * `data` is a pointer VALID IN THE ADDRESS SPACE named by `kind`. It must
//     never be dereferenced on the host unless kind == Host. Backends validate
//     the pointer against their own allocator/MemoryInfo before binding it.
//   * `step` is the ROW PITCH IN BYTES (>= cols * elem_size). Rows may be padded
//     for alignment; consumers must stride by `step`, not cols*channels.
//   * Layout is interleaved 8-bit BGR (the pipeline's canonical decode format),
//     matching the existing GpuImage/nvJPEG contract, unless a backend documents
//     otherwise for an internal buffer.
//
// The crop math the pipeline already relies on (aabb / clamped_crop_rect in
// common/geometry/box.h) is backend-neutral and operates on the {rows, cols}
// of an ImageView unchanged.

#include <cstddef>

namespace turbo_ocr::backend {

// The memory/execution space a device pointer lives in. This is the single
// device-identity enum shared by ImageView, DeviceQueue, IEngine::caps().io_space,
// and DeviceTensor — a bare void* never encodes its space, so it is always
// carried alongside as a DeviceKind.
//
//   Host  — plain CPU RAM (CpuBackend; also any op running as a host fallback).
//   Cuda  — NVIDIA device memory (cudaMalloc).
//   Metal — Apple MTLBuffer contents (unified memory on Apple silicon).
//   Hip   — AMD ROCm device memory (hipMalloc).
//   L0    — Intel Level Zero / OpenCL / SYCL USM device allocation.
enum class DeviceKind : int {
  Host = 0,
  Cuda = 1,
  Metal = 2,
  Hip = 3,
  L0 = 4,
};

[[nodiscard]] constexpr bool is_host(DeviceKind k) noexcept {
  return k == DeviceKind::Host;
}

// Is a pointer in this device's space directly dereferenceable by the host once
// the queue has drained? True for the host and for unified-memory devices; false
// for discrete VRAM, which needs an explicit D2H copy.
//
// This exists so the SHARED layer never has to write `kind == DeviceKind::Metal`
// (dedup rule 1: orchestration must not know which vendor it is on). It is only
// the DEFAULT answer, derived from the device class — an allocator that knows
// better (an AMD APU, an Intel iGPU, a CUDA managed-memory allocator) overrides
// IDeviceAllocator::host_coherent().
[[nodiscard]] constexpr bool device_is_host_coherent(DeviceKind k) noexcept {
  switch (k) {
  case DeviceKind::Host:  return true;
  case DeviceKind::Metal: return true; // UMA: MTLBuffer.contents is host-addressable
  case DeviceKind::Cuda:
  case DeviceKind::Hip:
  case DeviceKind::L0:
  default:                return false;
  }
}

[[nodiscard]] constexpr const char *device_kind_name(DeviceKind k) noexcept {
  switch (k) {
  case DeviceKind::Host:  return "host";
  case DeviceKind::Cuda:  return "cuda";
  case DeviceKind::Metal: return "metal";
  case DeviceKind::Hip:   return "hip";
  case DeviceKind::L0:    return "l0";
  }
  return "unknown";
}

// Non-owning device-resident image descriptor.
// Supports designated initializers:
//   ImageView{.data = ptr, .step = pitch, .rows = h, .cols = w,
//             .kind = DeviceKind::Metal}
struct ImageView {
  void *data = nullptr;      // pointer valid in `kind`'s address space
  std::size_t step = 0;      // row pitch in BYTES
  int rows = 0;              // image height
  int cols = 0;              // image width
  DeviceKind kind = DeviceKind::Host;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return data == nullptr || rows == 0 || cols == 0;
  }

  [[nodiscard]] constexpr bool is_host() const noexcept {
    return kind == DeviceKind::Host;
  }
};

} // namespace turbo_ocr::backend

// Convenience alias at the turbo_ocr level, mirroring the GpuImage alias in
// nvidia/support/gpu_image.h so call sites can name ImageView without the nested
// namespace.
namespace turbo_ocr {
using backend::DeviceKind;
using backend::ImageView;
} // namespace turbo_ocr
