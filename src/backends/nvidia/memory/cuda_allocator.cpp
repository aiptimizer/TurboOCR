// CudaAllocator implementation — cudaMalloc/Free + pinned host + async copies.

#include "nvidia/memory/cuda_allocator.h"

#include <cuda_runtime.h>

#include "nvidia/support/cuda_common.h" // cuda_stream()
#include "nvidia/support/cuda_check.h"

namespace turbo_ocr::nvidia {

void *CudaAllocator::allocate(std::size_t bytes) {
  void *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  return p;
}

void CudaAllocator::free(void *p) noexcept {
  if (p)
    cudaFree(p); // best-effort; matches CudaPtr::~CudaPtr
}

void *CudaAllocator::allocate_host(std::size_t bytes) {
  void *p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, bytes)); // pinned, for fast H2D/D2H staging
  return p;
}

void CudaAllocator::free_host(void *p) noexcept {
  if (p)
    cudaFreeHost(p);
}

void CudaAllocator::copy_h2d(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue &queue) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                             cuda_stream(queue)));
}

void CudaAllocator::copy_d2h(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue &queue) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                             cuda_stream(queue)));
}

void CudaAllocator::copy_d2d(void *dst, const void *src, std::size_t bytes,
                             backend::DeviceQueue &queue) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                             cuda_stream(queue)));
}

} // namespace turbo_ocr::nvidia
