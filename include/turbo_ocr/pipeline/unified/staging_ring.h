#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "turbo_ocr/backend/backend.h" // IDeviceAllocator, DeviceBuffer

namespace turbo_ocr::pipeline {

// PAGE STAGING RING — grow-only, owned per replica, never freed per request.
//
// upload_image_ used to allocate_buffer() + free a device buffer PER IMAGE and
// stage through a fresh pageable std::vector. On CUDA that is a cudaMalloc and
// a cudaFree per request — both device-wide synchronizing calls that serialize
// the stream — plus a pageable H2D, which cudaMemcpyAsync cannot actually
// perform asynchronously and which runs at roughly half pinned bandwidth.
// The pre-seam CUDA pipeline used grow-only pitched device buffers plus a
// grow-only PINNED host buffer; this restores that shape generically, so every
// vendor gets it rather than NVIDIA needing a forked pipeline to have it.
//
// THE SLOT IS THE PAIR, AND DEPTH IS THE CALLER'S DECLARATION.
// The pinned host buffer is the SOURCE of an H2D that is genuinely
// asynchronous (cudaMemcpyAsync / hipMemcpyAsync), so it stays live until the
// copy lands — exactly as long as the device buffer does. A ring that
// double-buffered only the device side and shared ONE host buffer let the
// memcpy for page N+1 overwrite the bytes page N's still-pending copy was
// reading. So host and device grow together, per slot.
//
// Depth is not a constant either. `Uploaded` is a non-owning view, so EVERY
// page a caller keeps alive simultaneously needs its own slot:
// run_with_layout holds one, run_pipelined holds two, and
// run_batch_with_layout's batched path holds ALL n (it uploads every page,
// then runs one batched detection and one batched recognition across the
// whole set). A fixed 2-slot ring silently aliased page i onto page i+2 there.
// Callers that hold more than the default declare it with reserve(), and
// trim() hands the surplus back once the views are dead so a 200-page batch
// does not pin 200 page buffers for the life of the replica.
// Pages the single-image and pipelined paths keep simultaneously live, and
// therefore the number of slots a replica retains between requests.
inline constexpr std::size_t kResidentStagingSlots = 2;

struct StagingRing {
  struct Slot {
    void *host = nullptr;              // pinned (allocate_host)
    std::size_t host_cap = 0;
    backend::DeviceBuffer dev;
    std::size_t dev_cap = 0;
  };
  // One page's staging pair. Either pointer being null means the allocation
  // failed and the caller must treat the page as failed rather than copy.
  struct Lease {
    std::uint8_t *host = nullptr;
    void *dev = nullptr;
  };

  backend::IDeviceAllocator *alloc = nullptr;
  std::vector<Slot> slots;
  std::size_t cursor = 0;

  ~StagingRing();
  // Guarantee at least `n` distinct slots before the next n acquire() calls.
  void reserve(std::size_t n);
  // Take the next slot, growing its host+device buffers to `bytes`.
  [[nodiscard]] Lease acquire(std::size_t bytes);
  // Release every slot past `keep`. Only legal once the views into them are
  // dead.
  void trim(std::size_t keep);
};

} // namespace turbo_ocr::pipeline
