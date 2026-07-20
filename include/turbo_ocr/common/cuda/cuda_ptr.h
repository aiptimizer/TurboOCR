#pragma once

#include "turbo_ocr/common/cuda/cuda_check.h"

#include <cstddef>
#include <utility>

namespace turbo_ocr {

/// RAII wrapper for device memory allocated with cudaMalloc.
/// Eliminates manual cudaFree calls and exception-safety bugs.
template <typename T>
class CudaPtr {
  T *ptr_ = nullptr;

public:
  CudaPtr() = default;

  /// Allocate @p count elements of type T on the device.
  explicit CudaPtr(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~CudaPtr() noexcept {
    if (ptr_)
      cudaFree(ptr_);
  }

  // Move-only
  CudaPtr(CudaPtr &&o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
  CudaPtr &operator=(CudaPtr &&o) noexcept {
    if (this != &o) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = o.ptr_;
      o.ptr_ = nullptr;
    }
    return *this;
  }
  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  T *get() noexcept { return ptr_; }
  const T *get() const noexcept { return ptr_; }

  /// Return address-of the internal pointer (for cudaMalloc-family APIs).
  T **addr() noexcept { return &ptr_; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  /// Release ownership and return the raw pointer.
  T *release() noexcept {
    T *p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  /// Reset with a new allocation (frees old).
  void reset(size_t count) {
    if (ptr_)
      cudaFree(ptr_);
    ptr_ = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }
};

/// RAII wrapper for pinned host memory allocated with cudaMallocHost.
template <typename T>
class CudaHostPtr {
  T *ptr_ = nullptr;

public:
  CudaHostPtr() = default;

  explicit CudaHostPtr(size_t count) {
    CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
  }

  ~CudaHostPtr() noexcept {
    if (ptr_)
      cudaFreeHost(ptr_);
  }

  CudaHostPtr(CudaHostPtr &&o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
  CudaHostPtr &operator=(CudaHostPtr &&o) noexcept {
    if (this != &o) {
      if (ptr_)
        cudaFreeHost(ptr_);
      ptr_ = o.ptr_;
      o.ptr_ = nullptr;
    }
    return *this;
  }
  CudaHostPtr(const CudaHostPtr &) = delete;
  CudaHostPtr &operator=(const CudaHostPtr &) = delete;

  T *get() noexcept { return ptr_; }
  const T *get() const noexcept { return ptr_; }
  T **addr() noexcept { return &ptr_; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  T *release() noexcept {
    T *p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  void reset(size_t count) {
    if (ptr_)
      cudaFreeHost(ptr_);
    ptr_ = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
  }
};

} // namespace turbo_ocr
