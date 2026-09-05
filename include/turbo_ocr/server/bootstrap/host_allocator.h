#pragma once

// Host-memory containment for the long-running server process.
//
// Request handling allocates large, short-lived host buffers (a decoded page
// is tens to hundreds of MB; the base64 routes hold the encoded text as well)
// on whichever work-pool thread took the request. Every general-purpose
// allocator keeps such freed memory in per-thread/per-arena free lists and
// returns it to the OS only on its own schedule:
//
//  - glibc parks freed blocks in per-arena free lists and auto-raises its mmap
//    threshold as large blocks are freed, so later image-sized allocations
//    grow the arena instead of being unmapped; RSS climbs toward a high-water
//    mark and stays there. Fixed thresholds + a periodic malloc_trim(0)
//    return the already-free pages between bursts.
//  - jemalloc (commonly LD_PRELOADed on hosts that share memory with other
//    services) keeps ~4 arenas per CPU, and an arena releases its dirty pages
//    only when ITS decay clock advances, which happens on allocation activity
//    on that arena or through jemalloc's optional background thread. With
//    many work threads spread over many arenas, an arena that goes quiet keeps
//    its peak for as long as the process lives, so RSS plateaus inside a
//    burst, never comes down between bursts, and the floor ratchets up as new
//    peaks land on arenas that had not seen them. Periodically advancing every
//    arena's decay ("arena.<all>.decay") releases what the operator's decay
//    settings say is releasable; it never forces a purge and never touches
//    live memory.
//
// Neither call reclaims live buffers, so the reaper cannot mask a real leak;
// it only removes retention. Cadence is low (seconds) so the cost is noise.
// The allocator is detected at run time, which is what makes an LD_PRELOADed
// jemalloc work without a rebuild.

#include <cstddef>

#include "turbo_ocr/decode/host_image_pool.h"

namespace turbo_ocr::server::bootstrap {

enum class HostAllocator {
  Unknown,   // neither glibc nor jemalloc detected (musl, macOS, ...): nothing to do
  Glibc,     // glibc malloc: mallopt tuning + malloc_trim
  Jemalloc,  // jemalloc (linked or LD_PRELOADed): mallctl arena decay
};

// Detects the allocator serving malloc in this process. jemalloc is
// recognised by its exported mallctl symbol, so a preloaded copy counts.
[[nodiscard]] HostAllocator detect_host_allocator() noexcept;

[[nodiscard]] const char *host_allocator_name(HostAllocator a) noexcept;

// Asks the allocator to hand already-freed memory back to the OS: malloc_trim
// on glibc, decay on every arena on jemalloc. Returns false when the
// allocator offers no such call (Unknown) or the call failed. Safe to call
// from any thread at any time.
bool release_idle_host_memory(HostAllocator a) noexcept;

// Installs the process-wide pinned host image pool (decode/host_image_pool.h)
// as OpenCV's default allocator and logs its budget: `slots` reusable page
// buffers, each growing to at most `max_block_bytes` (the MAX_IMAGE_PIXELS_MP
// budget). TURBO_OCR_DISABLE_HOST_IMAGE_POOL=1 leaves OpenCV's allocator in
// place (troubleshooting only).
void install_host_image_pool(size_t slots, size_t max_block_bytes,
                             decode::BlockMemory memory);

// Process-wide setup, once at startup: applies the glibc thresholds when
// glibc is the allocator, and starts the low-frequency reaper thread that
// calls release_idle_host_memory() every `reaper_period_s` seconds (0
// disables it, as does TURBO_OCR_DISABLE_MALLOC_REAPER=1 in the environment).
// Returns the detected allocator for the startup log.
HostAllocator tune_host_allocator(int reaper_period_s = 5);

} // namespace turbo_ocr::server::bootstrap
