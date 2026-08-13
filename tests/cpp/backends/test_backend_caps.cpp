// CAPABILITY HONESTY — checks over every compiled backend that
// test_seam_conformance.cpp does not already make. Read that file first; this
// one shares its `constructible_backends()` shape deliberately (same
// enumeration, same "skip only when the factory itself declines" rule) but
// asserts a DIFFERENT set of contracts:
//
//   * recommended_pool_size >= 1 — the pool divides replica budgets by it
//     (pool_sizing.h); a backend that ever reports 0 turns that into a
//     division by zero / infinite pool, not a small pool.
//   * a Host-device backend's queue must report is_async()==false — the
//     pipeline uses caps().device to choose the device-staging path and
//     is_async() to decide whether cross-queue event choreography is
//     meaningful; a host backend claiming async would send it down the
//     device staging path over plain host memory for no reason, and worse,
//     would make synchronize()-ordering assumptions the host queue never
//     honours.
//   * make_kernels()->caps().device must agree with the backend's own
//     caps().device — kernels and backend are asked separately by different
//     callers (IKernels::caps() drives per-op fallback placement,
//     Backend::caps() drives pipeline staging), and nothing else checks they
//     describe the same device.
//   * two calls to make_queue() must yield two DISTINCT queues — one per
//     pipeline replica. A backend that handed back a shared singleton would
//     not just fail this assertion: two unique_ptr owners of the same raw
//     pointer double-free it on teardown, so this also guards against a
//     crash-shaped bug, not just a semantic one.

#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/backend/backend_registry.h"
#include "turbo_ocr/backend/kernels.h"

using namespace turbo_ocr;

namespace {

// Same shape as test_seam_conformance.cpp's constructible_backends(): a vendor
// whose factory declines (no device present) is skipped, not failed, because
// that is the registry's documented "compiled in but not usable here".
std::vector<std::pair<std::string, std::unique_ptr<backend::Backend>>>
constructible_backends() {
  std::vector<std::pair<std::string, std::unique_ptr<backend::Backend>>> out;
  for (const auto name : backend::available_backends()) {
    auto bk = backend::make_backend(name);
    if (bk) out.emplace_back(std::string(name), std::move(bk));
  }
  return out;
}


// A REGISTERED backend whose DEVICE is gone is an environment fault, not a code
// fault, and the two must not look alike:
//   * an EMPTY registry means the registration TUs were not force-linked — a
//     broken build, and these suites fail on it (that is why they assert
//     non-empty rather than skipping);
//   * a backend that registers but cannot make a queue means the driver or card
//     is unavailable, which no source change can fix and which every other GPU
//     test in this repo already reports with SKIP.
// Without this split a faulted GPU reads as "the seam is broken", which is
// exactly the wrong thing to tell someone at 3am.
[[nodiscard]] inline bool device_is_usable(backend::Backend &bk) {
  try {
    return bk.make_queue() != nullptr;
  } catch (...) {
    return false;
  }
}

} // namespace

TEST_CASE("recommended_pool_size is never zero", "[backend_caps]") {
  // Non-empty asserted, not skipped: an empty registry means the registrars
  // were not linked into this binary, which is a broken build, not "no
  // backends here" (see backend_registry.h's PULL-IN CONTRACT).
  auto backends = constructible_backends();
  REQUIRE_FALSE(backends.empty());

  for (auto &[name, bk] : backends) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    // The pool divides a VRAM/UMA budget by this (pool_sizing.h); 0 is not a
    // "small pool", it is a crash or an unbounded one.
    CHECK(bk->caps().recommended_pool_size >= 1);
  }
}

TEST_CASE("a Host-device backend's queue is never async", "[backend_caps]") {
  auto backends = constructible_backends();
  REQUIRE_FALSE(backends.empty());

  for (auto &[name, bk] : backends) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    if (bk->caps().device != backend::DeviceKind::Host) continue;
    auto queue = bk->make_queue();
    REQUIRE(queue != nullptr);
    // A host backend claiming async would push callers onto the device
    // staging path (copy-then-sync) over memory that was already directly
    // readable — the wrong branch for every caller that keys off is_async().
    CHECK_FALSE(queue->is_async());
  }
}

TEST_CASE("make_kernels()->caps().device agrees with the backend's own caps()",
          "[backend_caps]") {
  auto backends = constructible_backends();
  REQUIRE_FALSE(backends.empty());

  for (auto &[name, bk] : backends) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    auto kernels = bk->make_kernels();
    REQUIRE(kernels != nullptr);
    // Two independent caps() calls describing the SAME backend must name the
    // same device — the pipeline picks staging from Backend::caps().device and
    // per-op fallback from IKernels::caps().device; a mismatch would place a
    // buffer for one device and run a kernel that expects the other.
    CHECK(kernels->caps().device == bk->caps().device);
  }
}

TEST_CASE("make_queue() yields a fresh queue on every call", "[backend_caps]") {
  auto backends = constructible_backends();
  REQUIRE_FALSE(backends.empty());

  for (auto &[name, bk] : backends) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    auto q1 = bk->make_queue();
    auto q2 = bk->make_queue();
    REQUIRE(q1 != nullptr);
    REQUIRE(q2 != nullptr);
    // Each pipeline replica owns its queue outright (device_queue.h's
    // Ownership note). Two unique_ptrs wrapping the SAME raw queue would
    // double-free on scope exit here, so this line is also a crash guard, not
    // only a semantic one.
    CHECK(q1.get() != q2.get());
    // On a device backend the native handle is the real resource (a
    // cudaStream_t and friends) — two replicas sharing one stream would
    // serialize their device work behind each other's ordering and corrupt
    // concurrent submissions. Host queues are synchronous no-ops with no
    // handle to compare, so this check is meaningful only where a distinct
    // handle is possible.
    if (bk->caps().device != backend::DeviceKind::Host) {
      CHECK(q1->native_handle() != q2->native_handle());
    }
  }
}
