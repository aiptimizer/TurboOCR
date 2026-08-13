// WorkPool shutdown semantics — the discard-after-grace backstop.
//
// The scenario these tests pin down: SIGTERM arrives with a deep queue, the
// grace window expires (wait_drain -> false), and the shutdown path calls
// discard_pending(). The contract is drop-what-never-started /
// finish-what-did: queued tasks are dropped and counted, in-flight tasks run
// to completion, later submits are refused, and the destructor no longer
// drains the backlog (which is what used to run up to max_depth tasks of
// teardown after the grace window, until the orchestrator's SIGKILL).

#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <future>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/service/server/work_pool.h"

using turbo_ocr::PoolExhaustedError;
using turbo_ocr::server::WorkPool;
using namespace std::chrono_literals;

TEST_CASE("discard_pending drops queued tasks but in-flight completes",
          "[work_pool]") {
  WorkPool pool(1);

  std::promise<void> started, release;
  std::shared_future<void> gate(release.get_future());
  std::atomic<int> ran{0};

  // Occupy the single worker, then queue five tasks behind it. Wait for the
  // blocker to actually START: until the worker dequeues it, it still sits in
  // the queue and a discard would (correctly) drop it too — which is not the
  // scenario under test.
  pool.submit([&started, gate, &ran] {
    started.set_value();
    gate.wait();
    ran.fetch_add(1);
  });
  started.get_future().wait();
  for (int i = 0; i < 5; ++i)
    pool.submit([&ran] { ran.fetch_add(1); });

  // Grace window "expires": the blocker holds the worker, so nothing drains.
  REQUIRE_FALSE(pool.wait_drain(50ms));
  REQUIRE(pool.queue_depth() == 5);

  const size_t dropped = pool.discard_pending();
  REQUIRE(dropped == 5);
  REQUIRE(pool.discarded_tasks() == 5);
  REQUIRE(pool.queue_depth() == 0);

  // The in-flight task is untouched: release it and the pool quiesces.
  release.set_value();
  REQUIRE(pool.wait_drain(2000ms));
  REQUIRE(ran.load() == 1);  // ONLY the in-flight task ran; none of the queued 5
}

TEST_CASE("submit after discard_pending is refused", "[work_pool]") {
  WorkPool pool(1);
  REQUIRE(pool.discard_pending() == 0);  // empty queue: idempotent, drops none
  REQUIRE_THROWS_AS(pool.submit([] {}), PoolExhaustedError);
  // The refused submit must not count as a shutdown drop — discarded_tasks()
  // answers "how much queued work was shed", not "how many submits bounced".
  REQUIRE(pool.discarded_tasks() == 0);
}

TEST_CASE("destructor after discard does not run the backlog", "[work_pool]") {
  std::atomic<int> ran{0};
  {
    WorkPool pool(1);
    std::promise<void> started, release;
    std::shared_future<void> gate(release.get_future());
    pool.submit([&started, gate] {
      started.set_value();
      gate.wait();
    });
    started.get_future().wait();  // blocker dequeued — see the first test case
    for (int i = 0; i < 8; ++i)
      pool.submit([&ran] { ran.fetch_add(1); });
    REQUIRE(pool.discard_pending() == 8);
    release.set_value();
    // ~WorkPool joins here. Before the discard backstop existed this drained
    // and RAN all 8 queued tasks during teardown.
  }
  REQUIRE(ran.load() == 0);
}

TEST_CASE("normal path is unchanged: tasks run and counters stay zero",
          "[work_pool]") {
  std::atomic<int> ran{0};
  {
    WorkPool pool(2);
    for (int i = 0; i < 16; ++i)
      pool.submit([&ran] { ran.fetch_add(1); });
    REQUIRE(pool.wait_drain(2000ms));
  }
  REQUIRE(ran.load() == 16);
}
