#include <catch_amalgamated.hpp>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "turbo_ocr/server/bootstrap/host_allocator.h"

using turbo_ocr::server::bootstrap::detect_host_allocator;
using turbo_ocr::server::bootstrap::HostAllocator;
using turbo_ocr::server::bootstrap::host_allocator_name;
using turbo_ocr::server::bootstrap::release_idle_host_memory;

TEST_CASE("the detected allocator has a name and matches the build", "[host_allocator]") {
  const HostAllocator a = detect_host_allocator();
  CHECK(std::strlen(host_allocator_name(a)) > 0);
  CHECK(std::string(host_allocator_name(HostAllocator::Unknown)) == "unknown");
  CHECK(std::string(host_allocator_name(HostAllocator::Glibc)) == "glibc");
  CHECK(std::string(host_allocator_name(HostAllocator::Jemalloc)) == "jemalloc");
#if defined(__GLIBC__)
  // On glibc systems the answer is glibc unless jemalloc was preloaded or
  // linked into the test binary, which the test runner may legitimately do.
  CHECK((a == HostAllocator::Glibc || a == HostAllocator::Jemalloc));
#else
  CHECK((a == HostAllocator::Unknown || a == HostAllocator::Jemalloc));
#endif
}

TEST_CASE("releasing idle memory is safe on every allocator and never fails for a real one", "[host_allocator]") {
  // Create and free a burst of allocations so there is something to give back.
  {
    std::vector<std::string> junk;
    for (int i = 0; i < 64; ++i) junk.emplace_back(1 << 20, 'x');
  }
  const HostAllocator a = detect_host_allocator();
  const bool released = release_idle_host_memory(a);
  if (a == HostAllocator::Unknown)
    CHECK_FALSE(released);
  else
    CHECK(released);
  // Explicitly unsupported kinds report false instead of pretending.
  CHECK_FALSE(release_idle_host_memory(HostAllocator::Unknown));
  // Calling it repeatedly is idempotent.
  CHECK(release_idle_host_memory(a) == released);
}
