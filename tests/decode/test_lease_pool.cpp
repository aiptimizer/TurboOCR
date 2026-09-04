#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "turbo_ocr/decode/lease_pool.h"

using turbo_ocr::decode::LeasePool;
using namespace std::chrono_literals;

namespace {

struct Widget {
  int id;
};

struct CountingFactory {
  std::shared_ptr<std::atomic<int>> made = std::make_shared<std::atomic<int>>(0);
  std::unique_ptr<Widget> operator()() const {
    return std::make_unique<Widget>(Widget{++*made});
  }
};

} // namespace

TEST_CASE("construction is lazy and never exceeds capacity", "[lease_pool]") {
  CountingFactory f;
  LeasePool<Widget> pool(2, f);
  CHECK(pool.capacity() == 2);
  CHECK(pool.created() == 0);

  auto a = pool.acquire();
  auto b = pool.acquire();
  REQUIRE(a);
  REQUIRE(b);
  CHECK(pool.created() == 2);
  CHECK(*f.made == 2);

  // Exhausted: a bounded wait gives up instead of constructing a third.
  auto c = pool.try_acquire_for(20ms);
  CHECK_FALSE(c.has_value());
  CHECK(pool.created() == 2);

  // Releasing one makes it available again without constructing anew.
  a.release();
  CHECK(pool.idle() == 1);
  auto d = pool.try_acquire_for(20ms);
  REQUIRE(d.has_value());
  REQUIRE(*d);
  CHECK(pool.created() == 2);
  CHECK(*f.made == 2);
}

TEST_CASE("released instances are reused, not rebuilt", "[lease_pool]") {
  CountingFactory f;
  LeasePool<Widget> pool(1, f);
  Widget *first = nullptr;
  {
    auto l = pool.acquire();
    first = l.get();
  }
  CHECK(pool.idle() == 1);
  auto l2 = pool.acquire();
  CHECK(l2.get() == first);
  CHECK(*f.made == 1);
}

TEST_CASE("moved-from leases do not double-release", "[lease_pool]") {
  CountingFactory f;
  LeasePool<Widget> pool(1, f);
  auto a = pool.acquire();
  auto b = std::move(a);
  CHECK_FALSE(a);
  CHECK(b);
  a.release();  // no-op on the moved-from handle
  CHECK(pool.idle() == 0);
  b.release();
  CHECK(pool.idle() == 1);
  b.release();  // idempotent
  CHECK(pool.idle() == 1);
}

TEST_CASE("a throwing factory frees its slot", "[lease_pool]") {
  int calls = 0;
  LeasePool<Widget> pool(1, [&]() -> std::unique_ptr<Widget> {
    if (++calls == 1) throw std::runtime_error("no device");
    return std::make_unique<Widget>(Widget{calls});
  });
  CHECK_THROWS_AS(pool.acquire(), std::runtime_error);
  CHECK(pool.created() == 0);
  auto l = pool.acquire();
  REQUIRE(l);
  CHECK(l->id == 2);
  CHECK(pool.created() == 1);
}

TEST_CASE("a null-returning factory yields empty leases without hanging", "[lease_pool]") {
  LeasePool<Widget> pool(2, []() -> std::unique_ptr<Widget> { return nullptr; });
  auto l = pool.try_acquire_for(50ms);
  REQUIRE(l.has_value());  // answered, not timed out
  CHECK_FALSE(*l);
  CHECK(pool.created() == 0);
  auto m = pool.acquire();  // must not block forever
  CHECK_FALSE(m);
}

TEST_CASE("concurrent leasing never exceeds capacity", "[lease_pool]") {
  constexpr size_t kCap = 3;
  constexpr int kThreads = 16;
  constexpr int kIters = 300;
  CountingFactory f;
  LeasePool<Widget> pool(kCap, f);
  std::atomic<int> in_use{0}, peak{0};

  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < kIters; ++i) {
        auto l = pool.acquire();
        REQUIRE(l);
        int now = ++in_use;
        int seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {
        }
        --in_use;
      }
    });
  }
  for (auto &t : ts) t.join();

  CHECK(peak.load() <= static_cast<int>(kCap));
  CHECK(pool.created() <= kCap);
  CHECK(*f.made <= static_cast<int>(kCap));
  CHECK(pool.idle() == pool.created());  // every lease came back
}

TEST_CASE("a waiter wakes when a lease is returned", "[lease_pool]") {
  CountingFactory f;
  LeasePool<Widget> pool(1, f);
  auto held = pool.acquire();
  std::atomic<bool> got{false};
  std::thread waiter([&] {
    auto l = pool.acquire();
    got = static_cast<bool>(l);
  });
  std::this_thread::sleep_for(30ms);
  CHECK_FALSE(got.load());
  held.release();
  waiter.join();
  CHECK(got.load());
}
