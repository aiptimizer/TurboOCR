#include "turbo_ocr/decode/host_image_pool.h"

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "turbo_ocr/common/log/logger.h"

namespace turbo_ocr::decode {

namespace {

// Grow in coarse steps so a corpus of slowly increasing sizes does not
// reallocate a slot on every request.
size_t rounded_capacity(size_t bytes, size_t previous, size_t max_block) {
  constexpr size_t kGranule = size_t{4} << 20;
  size_t cap = std::max(bytes, previous * 2);
  cap = (cap + kGranule - 1) / kGranule * kGranule;
  return std::min(cap, std::max(bytes, max_block));
}

} // namespace

namespace {
void *heap_alloc(size_t bytes) noexcept {
  try { return cv::fastMalloc(bytes); } catch (...) { return nullptr; }
}
void heap_free(void *p) noexcept { cv::fastFree(p); }
} // namespace

BlockMemory HostImagePool::heap_memory() noexcept { return {&heap_alloc, &heap_free, "heap"}; }

HostImagePool::HostImagePool(size_t slots, size_t max_block_bytes, BlockMemory memory,
                             size_t threshold_bytes, std::chrono::milliseconds wait)
    : slots_(std::max<size_t>(slots, 1)), max_block_(max_block_bytes),
      memory_(memory.alloc && memory.free ? memory : heap_memory()),
      threshold_(threshold_bytes), wait_(wait), allocator_(*this) {
  blocks_.resize(slots_);
}

HostImagePool::~HostImagePool() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &b : blocks_) release_block_(b);
}

void HostImagePool::release_block_(Block &b) noexcept {
  if (!b.ptr) return;
  if (b.pinned) memory_.free(b.ptr);
  else heap_free(b.ptr);
  b.ptr = nullptr;
  b.cap = 0;
  b.pinned = false;
}

bool HostImagePool::resize_block_(Block &b, size_t bytes) noexcept {
  const size_t cap = rounded_capacity(bytes, b.cap, max_block_);
  release_block_(b);
  if (memory_.alloc != &heap_alloc) {
    if (void *p = memory_.alloc(cap)) {
      b.ptr = p;
      b.cap = cap;
      b.pinned = true;
      return true;
    }
    TOCR_LOG_WARN_RL("host image pool: buffer allocation failed, using the heap for it",
                     "memory", std::string_view(memory_.name), "bytes", cap);
  }
  b.ptr = heap_alloc(cap);
  b.cap = b.ptr ? cap : 0;
  b.pinned = false;
  return b.ptr != nullptr;
}

HostImagePool::Block *HostImagePool::take(size_t bytes) {
  if (bytes > max_block_) return nullptr;
  std::unique_lock<std::mutex> lk(mu_);
  const auto deadline = std::chrono::steady_clock::now() + wait_;
  for (;;) {
    // Best fit among free buffers that are already large enough.
    Block *best = nullptr;
    Block *free_any = nullptr;
    Block *unallocated = nullptr;
    for (auto &b : blocks_) {
      if (b.busy) continue;
      if (!b.ptr) { if (!unallocated) unallocated = &b; continue; }
      if (!free_any) free_any = &b;
      if (b.cap >= bytes && (!best || b.cap < best->cap)) best = &b;
    }
    Block *chosen = best ? best : (unallocated ? unallocated : free_any);
    if (chosen) {
      if (chosen->cap < bytes && !resize_block_(*chosen, bytes)) return nullptr;
      chosen->busy = true;
      return chosen;
    }
    if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) return nullptr;
  }
}

void HostImagePool::give_back(Block *b) noexcept {
  {
    std::lock_guard<std::mutex> lk(mu_);
    b->busy = false;
  }
  cv_.notify_one();
}

HostImagePool::Stats HostImagePool::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  Stats s;
  s.slots = slots_;
  for (const auto &b : blocks_) {
    if (!b.ptr) continue;
    ++s.created;
    s.resident_bytes += b.cap;
    if (b.busy) ++s.in_use;
    if (b.pinned) ++s.pinned;
  }
  s.hits = hits_.load(std::memory_order_relaxed);
  s.fallbacks = fallbacks_.load(std::memory_order_relaxed);
  return s;
}

// ---- cv::MatAllocator -------------------------------------------------------
//
// Mirrors OpenCV's StdMatAllocator for the size/step arithmetic; the only
// difference is where page-sized buffers come from. `handle` on the UMatData
// carries the pool block so deallocate() knows what to return.

cv::UMatData *HostImagePool::Allocator::allocate(int dims, const int *sizes, int type,
                                                 void *data0, size_t *step,
                                                 cv::AccessFlag /*flags*/,
                                                 cv::UMatUsageFlags /*usageFlags*/) const {
  size_t total = CV_ELEM_SIZE(type);
  for (int i = dims - 1; i >= 0; --i) {
    if (step) {
      if (data0 && step[i] != cv::Mat::AUTO_STEP) {
        CV_Assert(total <= step[i]);
        total = step[i];
      } else {
        step[i] = total;
      }
    }
    total *= static_cast<size_t>(sizes[i]);
  }

  auto *u = new cv::UMatData(this);
  u->size = total;
  if (data0) {
    u->data = u->origdata = static_cast<uchar *>(data0);
    u->flags |= cv::UMatData::USER_ALLOCATED;
    return u;
  }
  if (total >= pool_.threshold_) {
    if (Block *b = pool_.take(total)) {
      u->data = u->origdata = static_cast<uchar *>(b->ptr);
      u->handle = b;
      pool_.hits_.fetch_add(1, std::memory_order_relaxed);
      return u;
    }
    pool_.fallbacks_.fetch_add(1, std::memory_order_relaxed);
    TOCR_LOG_WARN_RL("host image pool busy; image allocated on the heap",
                     "bytes", total, "slots", pool_.slots_);
  }
  u->data = u->origdata = static_cast<uchar *>(cv::fastMalloc(total));
  return u;
}

bool HostImagePool::Allocator::allocate(cv::UMatData *u, cv::AccessFlag,
                                        cv::UMatUsageFlags) const {
  return u != nullptr;
}

void HostImagePool::Allocator::deallocate(cv::UMatData *u) const {
  if (!u) return;
  CV_Assert(u->urefcount == 0);
  CV_Assert(u->refcount == 0);
  if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
    if (u->handle) {
      pool_.give_back(static_cast<Block *>(u->handle));
    } else {
      cv::fastFree(u->origdata);
    }
  }
  u->origdata = nullptr;
  u->handle = nullptr;
  delete u;
}

HostImagePool &HostImagePool::install_default(size_t slots, size_t max_block_bytes,
                                              BlockMemory memory) {
  // Leaked on purpose: Mats allocated from it may be destroyed during static
  // teardown, after any scoped object would be gone.
  static HostImagePool *pool = new HostImagePool(slots, max_block_bytes, memory);
  cv::Mat::setDefaultAllocator(&pool->allocator());
  return *pool;
}

} // namespace turbo_ocr::decode
