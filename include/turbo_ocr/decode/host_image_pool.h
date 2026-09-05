#pragma once

// A bounded pool of reusable page-sized host buffers behind a cv::MatAllocator.
//
// Every request that reaches the pipeline as a host image (PNG and other host
// formats on the work threads, PDF pages, the replica's host decodes for
// /ocr/markdown and /infer) allocates tens to hundreds of MB, uses them for
// one request and frees them. Through a general-purpose allocator that is the
// worst possible pattern: the pages park in per-thread arenas, RSS plateaus at
// the sum of every arena's peak and comes back only on the allocator's own
// schedule. This pool takes those allocations out of the allocator's hands:
//
//  - `slots` buffers, each growing to the largest image it has held (capped
//    by `max_block_bytes`, the MAX_IMAGE_PIXELS_MP budget), reused for the
//    life of the process. Host memory for images is a budget chosen at
//    startup (slots × largest image), not a behaviour to observe.
//  - The GPU server hands the pool pinned memory (cudaHostAlloc, injected as
//    a BlockMemory so this module stays CUDA-free): the GPU reads it by DMA,
//    uploads run at full bandwidth, and on unified-memory hosts it never
//    page-faults through pageable heap the allocator may be moving.
//  - Installed as the default cv::MatAllocator, so no call site changes:
//    allocations at or above `threshold_bytes` (page-sized) come from the
//    pool, everything smaller goes to OpenCV's standard allocator untouched.
//  - Never a failure mode: when every slot is busy for longer than a short
//    wait, the allocation falls back to the heap with a rate-limited log.
//    Correctness never depends on the pool; only the memory bound does.
//
// The same shape as the pinned pools in Triton and the buffer reuse in DALI.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <opencv2/core.hpp>

namespace turbo_ocr::decode {

// Where the pool's buffers come from. `alloc` returns nullptr on failure
// (the pool then uses the heap for that buffer and says so once).
struct BlockMemory {
  void *(*alloc)(size_t bytes) noexcept = nullptr;
  void (*free)(void *ptr) noexcept = nullptr;
  const char *name = "heap";
};

class HostImagePool {
 public:
  // Plain heap buffers (cv::fastMalloc); what a CPU build uses.
  [[nodiscard]] static BlockMemory heap_memory() noexcept;

  struct Stats {
    size_t slots = 0;          // maximum number of buffers
    size_t created = 0;        // buffers allocated so far (lazy)
    size_t in_use = 0;         // buffers currently handed out
    size_t resident_bytes = 0; // sum of buffer capacities
    size_t pinned = 0;         // buffers backed by the injected memory (not the heap)
    uint64_t hits = 0;         // allocations served from the pool
    uint64_t fallbacks = 0;    // allocations that fell back to the heap
  };

  // `threshold_bytes`: allocations smaller than this bypass the pool.
  // `max_block_bytes`: largest buffer the pool will grow (requests above it
  // go to the heap; the image caps make that unreachable in practice).
  HostImagePool(size_t slots, size_t max_block_bytes,
                BlockMemory memory = heap_memory(),
                size_t threshold_bytes = size_t{1} << 20,
                std::chrono::milliseconds wait = std::chrono::milliseconds{20});
  ~HostImagePool();
  HostImagePool(const HostImagePool &) = delete;
  HostImagePool &operator=(const HostImagePool &) = delete;

  // The allocator to hand to cv::Mat (a Mat's `allocator` field, or
  // cv::Mat::setDefaultAllocator). Valid for the pool's lifetime.
  [[nodiscard]] cv::MatAllocator &allocator() noexcept { return allocator_; }

  [[nodiscard]] Stats stats() const;
  [[nodiscard]] size_t threshold_bytes() const noexcept { return threshold_; }

  // Install a process-wide pool as OpenCV's default allocator. Idempotent;
  // the pool lives until process exit (Mats may outlive any scope).
  static HostImagePool &install_default(size_t slots, size_t max_block_bytes,
                                        BlockMemory memory);
  [[nodiscard]] const char *memory_name() const noexcept { return memory_.name; }

 private:
  struct Block {
    void *ptr = nullptr;
    size_t cap = 0;
    bool pinned = false;  // allocated through `memory_` (else the heap)
    bool busy = false;
  };

  class Allocator final : public cv::MatAllocator {
   public:
    explicit Allocator(HostImagePool &pool) : pool_(pool) {}
    cv::UMatData *allocate(int dims, const int *sizes, int type, void *data,
                           size_t *step, cv::AccessFlag flags,
                           cv::UMatUsageFlags usageFlags) const override;
    bool allocate(cv::UMatData *data, cv::AccessFlag accessflags,
                  cv::UMatUsageFlags usageFlags) const override;
    void deallocate(cv::UMatData *data) const override;

   private:
    HostImagePool &pool_;
  };

  // Hand out a buffer of at least `bytes` (nullptr: pool busy or over budget).
  [[nodiscard]] Block *take(size_t bytes);
  void give_back(Block *b) noexcept;
  [[nodiscard]] bool resize_block_(Block &b, size_t bytes) noexcept;
  void release_block_(Block &b) noexcept;

  const size_t slots_;
  const size_t max_block_;
  const BlockMemory memory_;
  const size_t threshold_;
  const std::chrono::milliseconds wait_;
  Allocator allocator_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Block> blocks_;  // never reallocated after construction
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> fallbacks_{0};
};

} // namespace turbo_ocr::decode
