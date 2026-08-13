#include "amd/memory/hip_allocator.h"

#include "amd/support/hip_check.h"
#include "amd/queue/hip_queue.h"

#include <hip/hip_runtime.h>

namespace turbo_ocr::amd {

void *HipAllocator::allocate(std::size_t bytes) {
  if (bytes == 0)
    return nullptr;
  HIP_CHECK(hipSetDevice(device_id_));
  void *p = nullptr;
  HIP_CHECK(hipMalloc(&p, bytes));
  return p;
}

void HipAllocator::free(void *p) noexcept {
  if (p)
    hipFree(p); // best-effort
}

void *HipAllocator::allocate_host(std::size_t bytes) {
  if (bytes == 0)
    return nullptr;
  void *p = nullptr;
  HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocDefault));
  return p;
}

void HipAllocator::free_host(void *p) noexcept {
  if (p)
    hipHostFree(p);
}

void HipAllocator::copy_h2d(void *dst, const void *src, std::size_t bytes,
                            DeviceQueue &queue) {
  HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                           hip_stream_of(queue)));
}

void HipAllocator::copy_d2h(void *dst, const void *src, std::size_t bytes,
                            DeviceQueue &queue) {
  HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                           hip_stream_of(queue)));
}

void HipAllocator::copy_d2d(void *dst, const void *src, std::size_t bytes,
                            DeviceQueue &queue) {
  HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice,
                           hip_stream_of(queue)));
}

} // namespace turbo_ocr::amd
