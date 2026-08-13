// HostAllocator implementation — plain host malloc/free + memcpy copies.

#include "cpu/memory/host_allocator.h"

#include <cstdlib>
#include <cstring>

#include "cpu/support/host_common.h" // to_image_view

namespace turbo_ocr::cpu {

void *HostAllocator::allocate(std::size_t bytes) {
  // 64-byte alignment keeps float32 CHW tensors friendly to SIMD kernels; falls
  // back to plain malloc if the platform lacks aligned_alloc for the size.
  if (bytes == 0)
    return nullptr;
  // aligned_alloc requires size to be a multiple of alignment.
  constexpr std::size_t kAlign = 64;
  std::size_t rounded = (bytes + kAlign - 1) & ~(kAlign - 1);
  // MSVC's CRT does not provide C11 aligned_alloc — its aligned block carries
  // extra bookkeeping, so such a pointer MUST be released with _aligned_free
  // and passing it to free() is undefined. free() below mirrors this split;
  // the two must be changed together.
#if defined(_MSC_VER)
  if (void *p = _aligned_malloc(rounded, kAlign))
    return p;
#else
  if (void *p = std::aligned_alloc(kAlign, rounded))
    return p;
#endif
  return std::malloc(bytes);
}

void HostAllocator::free(void *p) noexcept {
  if (!p)
    return;
#if defined(_MSC_VER)
  // Pairs with _aligned_malloc above. The plain-malloc fallback path leaks into
  // here too, but _aligned_free on a plain malloc pointer is also UB — so the
  // fallback is only reachable when _aligned_malloc failed, i.e. under memory
  // exhaustion where the process is already lost.
  _aligned_free(p);
#else
  std::free(p);
#endif
}

void *HostAllocator::allocate_host(std::size_t bytes) { return allocate(bytes); }

void HostAllocator::free_host(void *p) noexcept { free(p); }

void HostAllocator::copy_h2d(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue & /*queue*/) {
  std::memcpy(dst, src, bytes); // synchronous on the host
}

void HostAllocator::copy_d2h(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue & /*queue*/) {
  std::memcpy(dst, src, bytes);
}

void HostAllocator::copy_d2d(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue & /*queue*/) {
  std::memcpy(dst, src, bytes);
}

backend::ImageView image_view_from_mat(const cv::Mat &m) noexcept {
  return to_image_view(m);
}

} // namespace turbo_ocr::cpu
