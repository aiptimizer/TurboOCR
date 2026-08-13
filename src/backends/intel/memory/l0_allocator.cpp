// L0Allocator — SYCL USM device allocator (oneAPI/DPC++, Level Zero).
//
// TOOLCHAIN: `icpx -fsycl`. NOT compilable on the dev Mac; guarded by
// TURBO_OCR_HAS_SYCL. The guarded-out branch is a plain host-malloc allocator so
// the tree parses AND so a build without an Intel GPU degrades to a coherent
// (host) memory space rather than handing out invalid device pointers —
// has_device() reports which one you got, and OpenVINOEngine::caps().io_space
// follows it.

#include "intel/memory/l0_allocator.h"
#include "intel/queue/l0_device_queue.h"

#if defined(TURBO_OCR_HAS_SYCL)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp> // native L0 interop for OpenVINO
#endif

#include <cstdlib>
#include <cstring>

namespace turbo_ocr::intel {

#if defined(TURBO_OCR_HAS_SYCL)

struct L0Allocator::Impl {
  sycl::context ctx;
  sycl::device dev;
  bool ok = false;

  explicit Impl(int device_id) {
    try {
      if (device_id < 0) {
        dev = sycl::device(sycl::gpu_selector_v);
      } else {
        auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        const auto idx = static_cast<std::size_t>(device_id);
        dev = (idx < gpus.size()) ? gpus[idx] : sycl::device(sycl::gpu_selector_v);
      }
      ctx = sycl::context(dev);
      ok = true;
    } catch (const sycl::exception &) {
      ok = false; // no Intel GPU present; caller degrades to host tensors
    }
  }
};

L0Allocator::L0Allocator(int device_id)
    : impl_(std::make_unique<Impl>(device_id)) {}
L0Allocator::~L0Allocator() = default;

bool L0Allocator::has_device() const noexcept { return impl_->ok; }

void *L0Allocator::allocate(std::size_t bytes) {
  if (bytes == 0)
    return nullptr;
  if (!impl_->ok)
    return std::malloc(bytes);
  return sycl::malloc_device(bytes, impl_->dev, impl_->ctx);
}
void L0Allocator::free(void *p) noexcept {
  if (!p)
    return;
  if (!impl_->ok) {
    std::free(p);
    return;
  }
  sycl::free(p, impl_->ctx);
}

void *L0Allocator::allocate_host(std::size_t bytes) {
  if (bytes == 0)
    return nullptr;
  if (!impl_->ok)
    return std::malloc(bytes);
  return sycl::malloc_host(bytes, impl_->ctx);
}
void L0Allocator::free_host(void *p) noexcept { free(p); }

namespace {
// USM is unified-addressing, so h2d/d2h/d2d are ONE memcpy primitive; the three
// entry points are kept distinct only to preserve the interface's intent (and to
// leave room for direction-specific pinned-staging strategies later).
void enqueue_copy(void *dst, const void *src, std::size_t bytes,
                  backend::DeviceQueue &queue, bool have_device) {
  if (bytes == 0 || dst == nullptr || src == nullptr)
    return;
  auto *native = queue.native_handle();
  if (!have_device || native == nullptr) {
    std::memcpy(dst, src, bytes); // degraded host path
    return;
  }
  auto *q = static_cast<sycl::queue *>(native);
  sycl::event e = q->memcpy(dst, src, bytes);
  // Tell the lane its new tail so DeviceQueue::record() is accurate.
  if (auto *lq = dynamic_cast<L0DeviceQueue *>(&queue))
    lq->note_submission(&e);
}
} // namespace

void L0Allocator::copy_h2d(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &q) {
  enqueue_copy(d, s, n, q, impl_->ok);
}
void L0Allocator::copy_d2h(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &q) {
  enqueue_copy(d, s, n, q, impl_->ok);
}
void L0Allocator::copy_d2d(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &q) {
  enqueue_copy(d, s, n, q, impl_->ok);
}

void *L0Allocator::native_l0_context() const noexcept {
  if (!impl_->ok)
    return nullptr;
  try {
    auto h = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(impl_->ctx);
    return reinterpret_cast<void *>(h);
  } catch (...) {
    return nullptr; // not the Level Zero backend (e.g. OpenCL) — no interop
  }
}
void *L0Allocator::native_l0_device() const noexcept {
  if (!impl_->ok)
    return nullptr;
  try {
    auto h = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(impl_->dev);
    return reinterpret_cast<void *>(h);
  } catch (...) {
    return nullptr; // not the Level Zero backend (e.g. OpenCL) — no interop
  }
}

#else // !TURBO_OCR_HAS_SYCL — coherent host-memory allocator.

struct L0Allocator::Impl {};
L0Allocator::L0Allocator(int) : impl_(std::make_unique<Impl>()) {}
L0Allocator::~L0Allocator() = default;
bool L0Allocator::has_device() const noexcept { return false; }
void *L0Allocator::allocate(std::size_t bytes) {
  return bytes ? std::malloc(bytes) : nullptr;
}
void L0Allocator::free(void *p) noexcept { std::free(p); }
void *L0Allocator::allocate_host(std::size_t bytes) {
  return bytes ? std::malloc(bytes) : nullptr;
}
void L0Allocator::free_host(void *p) noexcept { std::free(p); }
void L0Allocator::copy_h2d(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &) {
  if (d && s && n)
    std::memcpy(d, s, n);
}
void L0Allocator::copy_d2h(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &) {
  if (d && s && n)
    std::memcpy(d, s, n);
}
void L0Allocator::copy_d2d(void *d, const void *s, std::size_t n,
                           backend::DeviceQueue &) {
  if (d && s && n)
    std::memcpy(d, s, n);
}
void *L0Allocator::native_l0_context() const noexcept { return nullptr; }
void *L0Allocator::native_l0_device() const noexcept { return nullptr; }

#endif

} // namespace turbo_ocr::intel
