// UnifiedPipelinePool admission tests — the bound on how long a request waits
// for a busy device, and the shed that stops an unbounded queue forming behind
// it.
//
// This is the coverage that replaces the deleted C4 get_with_timeout cases in
// test_gpu_safety.cpp. That primitive was a GPU-only future timeout in
// pipeline_dispatcher.h; the pool that took over its job is device-neutral, so
// its tests belong here, where BOTH configures run them.
//
// The entries hold null pipelines on purpose: every property under test lives in
// the free-list and the waiter accounting, and acquire() never dereferences the
// pipeline. Building real ones would drag a Backend and its models into a unit
// test to exercise none of it.

#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/pipeline/unified/make_infer_func.h"

using turbo_ocr::PoolExhaustedError;
using turbo_ocr::pipeline::UnifiedPipelineEntry;
using turbo_ocr::pipeline::UnifiedPipelinePool;
using namespace std::chrono_literals;

namespace {

std::vector<UnifiedPipelineEntry> empty_entries(std::size_t n) {
  std::vector<UnifiedPipelineEntry> v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    v.push_back(UnifiedPipelineEntry{nullptr, nullptr});
  return v;
}

} // namespace

TEST_CASE("pool hands out every replica before anyone waits", "[pool]") {
  UnifiedPipelinePool pool(empty_entries(3), 50ms, /*max_waiters=*/8);
  REQUIRE(pool.size() == 3);
  REQUIRE(pool.available() == 3);

  auto a = pool.acquire();
  auto b = pool.acquire();
  REQUIRE(pool.available() == 1);
  REQUIRE(pool.waiting() == 0);

  auto c = pool.acquire();
  REQUIRE(pool.available() == 0);
}

TEST_CASE("acquire throws once the deadline elapses, and does not hang", "[pool]") {
  UnifiedPipelinePool pool(empty_entries(1), 30ms, /*max_waiters=*/8);
  auto held = pool.acquire();

  // The deadline is the whole point: a wedged replica must surface as an error
  // the route can map to 503, never as a blocked worker thread.
  const auto t0 = std::chrono::steady_clock::now();
  REQUIRE_THROWS_AS(pool.acquire(), PoolExhaustedError);
  REQUIRE(std::chrono::steady_clock::now() - t0 < 5s);
}

TEST_CASE("a released replica wakes a waiter instead of timing it out", "[pool]") {
  UnifiedPipelinePool pool(empty_entries(1), 5s, /*max_waiters=*/8);
  std::atomic<bool> got{false};

  auto held = std::make_unique<UnifiedPipelinePool::Lease>(pool.acquire());
  std::thread waiter([&] {
    auto lease = pool.acquire(); // blocks until the main thread releases
    got.store(true);
  });

  // Wait for the thread to actually park, so the release below is the thing
  // that unblocks it rather than a race that never blocked at all.
  for (int i = 0; i < 200 && pool.waiting() == 0; ++i)
    std::this_thread::sleep_for(5ms);
  REQUIRE(pool.waiting() == 1);

  held.reset(); // release
  waiter.join();
  REQUIRE(got.load());
  REQUIRE(pool.available() == 1);
}

TEST_CASE("pool sheds immediately once the waiter cap is full", "[pool]") {
  // SHED BEFORE PARKING: admitting past the cap costs a blocked worker thread
  // for the whole deadline and still ends in a failure the client sees as a
  // timeout. The long deadline here is deliberate — it makes a shed impossible
  // to confuse with a deadline expiry.
  //
  // NOTE the sentinel: max_waiters 0 means NO CAP (like acquire_timeout 0
  // meaning no deadline), not "never queue". Passing 0 here expecting an
  // instant shed is how this test was first written, and it waited the full
  // 10s. Hence a cap of 1 plus two contenders.
  UnifiedPipelinePool pool(empty_entries(1), 10s, /*max_waiters=*/1);
  auto held = pool.acquire();

  std::thread parked([&] {
    try { auto lease = pool.acquire(); } catch (const PoolExhaustedError &) {}
  });
  for (int i = 0; i < 200 && pool.waiting() == 0; ++i)
    std::this_thread::sleep_for(5ms);
  REQUIRE(pool.waiting() == 1); // the cap is now full

  const auto t0 = std::chrono::steady_clock::now();
  REQUIRE_THROWS_AS(pool.acquire(), PoolExhaustedError);
  REQUIRE(std::chrono::steady_clock::now() - t0 < 1s);

  { auto drop = std::move(held); } // free the parked thread
  parked.join();
}

TEST_CASE("try_acquire_for reports failure rather than throwing", "[pool]") {
  UnifiedPipelinePool pool(empty_entries(1), 5s, /*max_waiters=*/8);
  auto held = pool.acquire();
  REQUIRE_FALSE(pool.try_acquire_for(20ms).has_value());

  // The readiness probe uses this path, so it must recover once a replica frees.
  auto released = std::move(held);
  { auto drop = std::move(released); }
  REQUIRE(pool.try_acquire_for(20ms).has_value());
}
