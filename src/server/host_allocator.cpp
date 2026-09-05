#include "turbo_ocr/server/bootstrap/host_allocator.h"

#include <chrono>
#include <cstddef>
#include <thread>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/host_image_pool.h"

namespace turbo_ocr::server::bootstrap {

namespace {

// int mallctl(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen)
using MallctlFn = int (*)(const char *, void *, size_t *, void *, size_t);

// jemalloc's MALLCTL_ARENAS_ALL: the arena index that addresses every arena
// in "arena.<i>.<cmd>" names. Spelled out here so the server does not need
// jemalloc's headers at build time (it is normally only preloaded).
constexpr const char *kDecayAllArenas = "arena.4096.decay";

MallctlFn find_mallctl() noexcept {
#if defined(_WIN32)
  return nullptr;
#else
  // Both spellings: unprefixed is the distro default, je_ the upstream one.
  for (const char *name : {"mallctl", "je_mallctl"}) {
    if (void *sym = dlsym(RTLD_DEFAULT, name))
      return reinterpret_cast<MallctlFn>(sym);
  }
  return nullptr;
#endif
}

MallctlFn mallctl_fn() noexcept {
  static const MallctlFn fn = find_mallctl();
  return fn;
}

} // namespace

HostAllocator detect_host_allocator() noexcept {
  if (mallctl_fn()) return HostAllocator::Jemalloc;
#if defined(__GLIBC__)
  return HostAllocator::Glibc;
#else
  return HostAllocator::Unknown;
#endif
}

const char *host_allocator_name(HostAllocator a) noexcept {
  switch (a) {
    case HostAllocator::Glibc: return "glibc";
    case HostAllocator::Jemalloc: return "jemalloc";
    case HostAllocator::Unknown: break;
  }
  return "unknown";
}

bool release_idle_host_memory(HostAllocator a) noexcept {
  switch (a) {
    case HostAllocator::Jemalloc:
      if (MallctlFn fn = mallctl_fn())
        return fn(kDecayAllArenas, nullptr, nullptr, nullptr, 0) == 0;
      return false;
    case HostAllocator::Glibc:
#if defined(__GLIBC__)
      (void)malloc_trim(0);  // returns whether anything was released; either way the call succeeded
      return true;
#else
      return false;
#endif
    case HostAllocator::Unknown: break;
  }
  return false;
}

void install_host_image_pool(size_t slots, size_t max_block_bytes,
                             decode::BlockMemory memory) {
  if (env::env_present("TURBO_OCR_DISABLE_HOST_IMAGE_POOL")) {
    TOCR_LOG_INFO("Host image pool disabled by TURBO_OCR_DISABLE_HOST_IMAGE_POOL");
    return;
  }
  auto &pool = decode::HostImagePool::install_default(slots, max_block_bytes, memory);
  TOCR_LOG_INFO("Host image pool installed", "slots", slots,
                "memory", std::string_view(pool.memory_name()),
                "max_block_mb", static_cast<int>(max_block_bytes >> 20),
                "budget_mb", static_cast<int>((slots * max_block_bytes) >> 20),
                "threshold_kb", static_cast<int>(pool.threshold_bytes() >> 10));
}

HostAllocator tune_host_allocator(int reaper_period_s) {
  const HostAllocator a = detect_host_allocator();

#if defined(__GLIBC__)
  if (a == HostAllocator::Glibc) {
    // Freeze the mmap threshold so image-sized blocks are unmapped on free
    // (setting it explicitly also disables the upward auto-tuning), bound the
    // main arena's trim threshold, and cap the arena count: with a large
    // work-thread pool the default of 8 per CPU means that many high-water
    // marks. (jemalloc ignores mallopt; nothing to tune there from here.)
    mallopt(M_MMAP_THRESHOLD, 1 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 4 * 1024 * 1024);
    mallopt(M_ARENA_MAX, 8);
  }
#endif

  const bool reaper = reaper_period_s > 0 && a != HostAllocator::Unknown &&
                      !env::env_present("TURBO_OCR_DISABLE_MALLOC_REAPER");
  TOCR_LOG_INFO("Host allocator", "allocator", std::string_view(host_allocator_name(a)),
                "idle_memory_reaper", reaper, "period_s", reaper ? reaper_period_s : 0);
  if (reaper) {
    std::thread([a, reaper_period_s] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(reaper_period_s));
        (void)release_idle_host_memory(a);
      }
    }).detach();
  }
  return a;
}

} // namespace turbo_ocr::server::bootstrap
