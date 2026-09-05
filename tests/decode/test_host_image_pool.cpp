#include <catch_amalgamated.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/decode/host_image_pool.h"

using turbo_ocr::decode::HostImagePool;

namespace {
constexpr size_t MiB = size_t{1} << 20;

cv::Mat pooled(HostImagePool &pool, int rows, int cols) {
  cv::Mat m;
  m.allocator = &pool.allocator();
  m.create(rows, cols, CV_8UC3);
  return m;
}
} // namespace

TEST_CASE("page-sized images come from the pool, small ones from the heap", "[host_image_pool]") {
  HostImagePool pool(/*slots=*/2, /*max_block=*/64 * MiB);
  {
    cv::Mat big = pooled(pool, 1000, 1000);  // 3 MB
    cv::Mat small = pooled(pool, 16, 16);    // 768 B
    REQUIRE(!big.empty());
    REQUIRE(!small.empty());
    auto s = pool.stats();
    CHECK(s.in_use == 1);
    CHECK(s.created == 1);
    CHECK(s.hits == 1);
    CHECK(s.fallbacks == 0);
    big.at<cv::Vec3b>(999, 999) = cv::Vec3b(1, 2, 3);  // the buffer is writable to its end
  }
  CHECK(pool.stats().in_use == 0);
  CHECK(pool.stats().created == 1);  // kept for reuse
}

TEST_CASE("a released buffer is reused and grows to the largest image", "[host_image_pool]") {
  HostImagePool pool(1, 64 * MiB);
  const uchar *first = nullptr;
  { cv::Mat a = pooled(pool, 1000, 1000); first = a.data; }
  { cv::Mat b = pooled(pool, 1000, 1000); CHECK(b.data == first); }  // same slot, no new allocation
  CHECK(pool.stats().resident_bytes >= 3 * MiB);
  { cv::Mat c = pooled(pool, 3000, 3000); CHECK(!c.empty()); }      // 27 MB: slot grows
  CHECK(pool.stats().created == 1);
  CHECK(pool.stats().resident_bytes >= 27 * MiB);
  { cv::Mat d = pooled(pool, 1000, 1000); CHECK(!d.empty()); }      // fits in the grown slot
  CHECK(pool.stats().resident_bytes >= 27 * MiB);                   // never shrinks
}

TEST_CASE("exhaustion falls back to the heap instead of failing", "[host_image_pool]") {
  HostImagePool pool(1, 64 * MiB, HostImagePool::heap_memory(), 1 * MiB, std::chrono::milliseconds{5});
  cv::Mat a = pooled(pool, 1000, 1000);
  cv::Mat b = pooled(pool, 1000, 1000);  // slot busy -> heap
  REQUIRE(!a.empty());
  REQUIRE(!b.empty());
  auto s = pool.stats();
  CHECK(s.in_use == 1);
  CHECK(s.fallbacks == 1);
  b.release();
  a.release();
  CHECK(pool.stats().in_use == 0);
}

TEST_CASE("images above the budget cap bypass the pool", "[host_image_pool]") {
  HostImagePool pool(2, /*max_block=*/2 * MiB);
  cv::Mat huge = pooled(pool, 2000, 2000);  // 12 MB > cap
  REQUIRE(!huge.empty());
  CHECK(pool.stats().created == 0);
  CHECK(pool.stats().fallbacks == 1);
}

TEST_CASE("copies share the buffer and return it exactly once", "[host_image_pool]") {
  HostImagePool pool(1, 64 * MiB);
  {
    cv::Mat a = pooled(pool, 1000, 1000);
    cv::Mat b = a;              // refcount bump
    cv::Mat c = a(cv::Rect(0, 0, 10, 10));  // view
    a.release();
    CHECK(pool.stats().in_use == 1);
    b.release();
    CHECK(pool.stats().in_use == 1);  // the view still holds it
  }
  CHECK(pool.stats().in_use == 0);
}

TEST_CASE("concurrent users never exceed the slot count and all buffers come back", "[host_image_pool]") {
  constexpr size_t kSlots = 3;
  HostImagePool pool(kSlots, 64 * MiB, HostImagePool::heap_memory(), 1 * MiB, std::chrono::milliseconds{2});
  std::atomic<size_t> peak{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 200; ++i) {
        cv::Mat m = pooled(pool, 800, 800);
        REQUIRE(!m.empty());
        m.at<uchar>(0, 0) = 1;
        const size_t now = pool.stats().in_use;
        size_t seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {
        }
      }
    });
  }
  for (auto &t : ts) t.join();
  CHECK(peak.load() <= kSlots);
  CHECK(pool.stats().in_use == 0);
  CHECK(pool.stats().created <= kSlots);
  CHECK(pool.stats().hits + pool.stats().fallbacks == 8 * 200);
}
