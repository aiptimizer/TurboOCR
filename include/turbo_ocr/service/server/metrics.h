#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/service/server/work_pool.h"

namespace turbo_ocr::server {

/// Prometheus-compatible metrics with zero external dependencies.
/// Thread-safe via atomics only — counters are relaxed RMWs; the histogram
/// uses the release/acquire protocol documented at record_request().
class Metrics {
public:
  // ── Route index (used as label dimension) ──────────────────────────────

  enum Route : int {
    kOcr = 0,
    kOcrRaw,
    kOcrBatch,
    kOcrPixels,
    kOcrPdf,
    kOcrMarkdown,
    kInfer,
    kOcrStream,
    kHealth,
    kOther,
    kRouteCount
  };

  static constexpr const char *route_name(Route r) {
    constexpr const char *names[] = {
        "/ocr", "/ocr/raw", "/ocr/batch", "/ocr/pixels", "/ocr/pdf",
        "/ocr/markdown", "/infer", "/ocr/stream", "/health", "other"};
    static_assert(sizeof(names) / sizeof(names[0]) == kRouteCount,
                  "route label table out of sync with the Route enum");
    return names[r];
  }

  static Route route_from_path(std::string_view path) {
    if (path == "/ocr")          return kOcr;
    if (path == "/ocr/raw")      return kOcrRaw;
    if (path == "/ocr/batch")    return kOcrBatch;
    if (path == "/ocr/pixels")   return kOcrPixels;
    if (path == "/ocr/pdf")      return kOcrPdf;
    if (path == "/ocr/markdown") return kOcrMarkdown;
    if (path == "/infer")        return kInfer;
    if (path == "/ocr/stream")   return kOcrStream;
    if (path == "/health" || path == "/health/live" || path == "/health/ready")
      return kHealth;
    // Any other matched handler (e.g. the CPU build's /profile) is bucketed
    // under "other" so it can't corrupt /health's request/error counters.
    return kOther;
  }

  // ── Recording ──────────────────────────────────────────────────────────

  void record_request(Route route, int http_status, double duration_s) {
    auto &r = routes_[route];
    if (http_status >= 200 && http_status < 300)
      r.ok.fetch_add(1, std::memory_order_relaxed);
    else if (http_status >= 400 && http_status < 500)
      r.client_err.fetch_add(1, std::memory_order_relaxed);
    else if (http_status >= 500)
      r.server_err.fetch_add(1, std::memory_order_relaxed);

    // Histogram: bump _count and _sum BEFORE the per-bucket counter so a
    // concurrent /metrics scrape can never observe a cumulative bucket count
    // exceeding _count (Prometheus requires le-buckets <= +Inf == _count).
    // The bucket increment is RELEASE and serialize() reads it ACQUIRE so the
    // ordering holds on weakly-ordered targets (aarch64), not just x86 TSO —
    // program order alone doesn't order two relaxed RMWs on distinct atomics.
    // Values above all buckets only appear in _count/_sum (+Inf bucket).
    r.hist_count.fetch_add(1, std::memory_order_relaxed);
    r.hist_sum.fetch_add(
        static_cast<uint64_t>(duration_s * 1e6), std::memory_order_relaxed);
    for (size_t i = 0; i < kNumBuckets; ++i) {
      if (duration_s <= kBuckets[i]) {
        r.hist_buckets[i].fetch_add(1, std::memory_order_release);
        break;
      }
    }
  }

  void record_request_size(size_t bytes) {
    request_bytes_total_.fetch_add(bytes, std::memory_order_relaxed);
    request_count_sized_.fetch_add(1, std::memory_order_relaxed);
  }

  void record_pool_exhaustion() {
    pool_exhaustions_.fetch_add(1, std::memory_order_relaxed);
  }

  void set_pool_size(int n) {
    pool_size_.store(n, std::memory_order_relaxed);
  }

  void set_gpu_vram_used_bytes(size_t bytes) {
    gpu_vram_used_.store(bytes, std::memory_order_relaxed);
  }

  void set_gpu_vram_total_bytes(size_t bytes) {
    gpu_vram_total_.store(bytes, std::memory_order_relaxed);
  }

  // ── Saturation gauges ──────────────────────────────────────────────────
  // These are point-in-time depths; the scrape handler is expected to call
  // record_saturation() just before serialize() so the values reflect the
  // moment of the scrape rather than the last request.

  void set_workpool_queue_depth(size_t depth) {
    workpool_queue_depth_.store(depth, std::memory_order_relaxed);
  }

  void set_workpool_inflight(size_t n) {
    workpool_inflight_.store(n, std::memory_order_relaxed);
  }

  void set_workpool_max_depth(size_t depth) {
    workpool_max_depth_.store(depth, std::memory_order_relaxed);
  }

  /// Downstream GPU/pipeline queue depth (the PipelineDispatcher bound that
  /// actually triggers 503s). Setter-only so the pipeline owns the value.
  /// Device-side lease pool: how many requests are blocked waiting for a
  /// replica, how many replicas are free, the age of the longest-held lease,
  /// and how many leases have been observed wedged.
  ///
  /// These close the gap work_pool.h names: /metrics reported only the WorkPool
  /// QUEUE in front of the device, so a server whose replicas were all wedged
  /// looked idle — empty queue, nothing in flight, and no signal at all that
  /// throughput had gone to zero.
  void set_pool_saturation(size_t waiting, size_t available,
                           long long oldest_lease_ms, uint64_t stuck,
                           long long stuck_threshold_ms) {
    pool_waiting_.store(waiting, std::memory_order_relaxed);
    pool_available_.store(available, std::memory_order_relaxed);
    pool_oldest_lease_ms_.store(oldest_lease_ms, std::memory_order_relaxed);
    pool_stuck_leases_.store(stuck, std::memory_order_relaxed);
    pool_stuck_threshold_ms_.store(stuck_threshold_ms, std::memory_order_relaxed);
  }

  void set_dispatcher_queue_depth(size_t depth) {
    dispatcher_queue_depth_.store(depth, std::memory_order_relaxed);
  }

  /// Convenience hook for the /metrics scrape handler: snapshot the WorkPool
  /// saturation in one call. Pass max_depth so the gauge ceiling stays in
  /// sync with the pool's configured bound. escaped/discarded are the pool's
  /// monotonic counters (WorkPool::escaped_tasks / discarded_tasks) — polled
  /// here like the gauges because the dependency points outward (work_pool.h
  /// cannot include this header; it is included BY it).
  void record_saturation(size_t queue_depth, size_t inflight, size_t max_depth,
                         uint64_t escaped = 0, uint64_t discarded = 0) {
    set_workpool_queue_depth(queue_depth);
    set_workpool_inflight(inflight);
    set_workpool_max_depth(max_depth);
    workpool_escaped_tasks_.store(escaped, std::memory_order_relaxed);
    workpool_discarded_tasks_.store(discarded, std::memory_order_relaxed);
  }

  // ── Prometheus text exposition ─────────────────────────────────────────

  [[nodiscard]] std::string serialize() const {
    std::string out;
    out.reserve(4096);

    // requests_total
    out += "# HELP turbo_ocr_requests_total Total HTTP requests by route and status.\n";
    out += "# TYPE turbo_ocr_requests_total counter\n";
    for (int i = 0; i < kRouteCount; ++i) {
      auto &r = routes_[i];
      auto name = route_name(static_cast<Route>(i));
      append_counter(out, "turbo_ocr_requests_total", name, "2xx",
                     r.ok.load(std::memory_order_relaxed));
      append_counter(out, "turbo_ocr_requests_total", name, "4xx",
                     r.client_err.load(std::memory_order_relaxed));
      append_counter(out, "turbo_ocr_requests_total", name, "5xx",
                     r.server_err.load(std::memory_order_relaxed));
    }

    // request_duration_seconds (histogram)
    out += "# HELP turbo_ocr_request_duration_seconds Request latency histogram.\n";
    out += "# TYPE turbo_ocr_request_duration_seconds histogram\n";
    for (int i = 0; i < kRouteCount; ++i) {
      if (i == kHealth) continue;  // skip health from histogram
      auto &r = routes_[i];
      auto name = route_name(static_cast<Route>(i));
      uint64_t cumulative = 0;
      for (size_t b = 0; b < kNumBuckets; ++b) {
        // ACQUIRE pairs with the RELEASE bucket increment so a bucket count
        // is never seen ahead of the _count bump that preceded it.
        cumulative += r.hist_buckets[b].load(std::memory_order_acquire);
        char le_buf[32];
        std::snprintf(le_buf, sizeof(le_buf), "%.3f", kBuckets[b]);
        append_histogram_bucket(out, name, le_buf, cumulative);
      }
      uint64_t count = r.hist_count.load(std::memory_order_relaxed);
      append_histogram_bucket(out, name, "+Inf", count);
      double sum = static_cast<double>(
          r.hist_sum.load(std::memory_order_relaxed)) / 1e6;
      append_histogram_summary(out, name, sum, count);
    }

    // pool_exhaustions_total
    out += "# HELP turbo_ocr_pool_exhaustions_total Times pipeline pool was full (503).\n";
    out += "# TYPE turbo_ocr_pool_exhaustions_total counter\n";
    char buf[128];
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_exhaustions_total %" PRIu64 "\n",
                  pool_exhaustions_.load(std::memory_order_relaxed));
    out += buf;

    // pool_size
    out += "# HELP turbo_ocr_pipeline_pool_size Number of pipeline slots.\n";
    out += "# TYPE turbo_ocr_pipeline_pool_size gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pipeline_pool_size %d\n",
                  pool_size_.load(std::memory_order_relaxed));
    out += buf;

    // Saturation gauges (snapshot at scrape time via record_saturation()).
    out += "# HELP turbo_ocr_workpool_queue_depth Tasks waiting in the WorkPool queue.\n";
    out += "# TYPE turbo_ocr_workpool_queue_depth gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_workpool_queue_depth %zu\n",
                  workpool_queue_depth_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_workpool_inflight Tasks currently executing on WorkPool threads.\n";
    out += "# TYPE turbo_ocr_workpool_inflight gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_workpool_inflight %zu\n",
                  workpool_inflight_.load(std::memory_order_relaxed));
    out += buf;
    size_t wp_max = workpool_max_depth_.load(std::memory_order_relaxed);
    if (wp_max > 0) {
      out += "# HELP turbo_ocr_workpool_max_depth Configured WorkPool queue ceiling (503 above this).\n";
      out += "# TYPE turbo_ocr_workpool_max_depth gauge\n";
      std::snprintf(buf, sizeof(buf), "turbo_ocr_workpool_max_depth %zu\n", wp_max);
      out += buf;
    }
    out += "# HELP turbo_ocr_workpool_escaped_tasks_total Tasks that escaped with an "
           "exception (their HTTP callback never fired; the client hung until its "
           "own timeout). Nonzero and growing = wedged-client signal.\n";
    out += "# TYPE turbo_ocr_workpool_escaped_tasks_total counter\n";
    std::snprintf(buf, sizeof(buf),
                  "turbo_ocr_workpool_escaped_tasks_total %" PRIu64 "\n",
                  workpool_escaped_tasks_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_workpool_discarded_tasks_total Queued tasks dropped at "
           "shutdown after the grace period expired (never started; client saw the "
           "connection close at exit).\n";
    out += "# TYPE turbo_ocr_workpool_discarded_tasks_total counter\n";
    std::snprintf(buf, sizeof(buf),
                  "turbo_ocr_workpool_discarded_tasks_total %" PRIu64 "\n",
                  workpool_discarded_tasks_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_pool_waiting Requests blocked waiting for a pipeline replica.\n";
    out += "# TYPE turbo_ocr_pool_waiting gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_waiting %zu\n",
                  pool_waiting_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_pool_available Free pipeline replicas.\n";
    out += "# TYPE turbo_ocr_pool_available gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_available %zu\n",
                  pool_available_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_pool_oldest_lease_ms Age of the longest-held pipeline lease. A value far above your slowest request means a wedged replica.\n";
    out += "# TYPE turbo_ocr_pool_oldest_lease_ms gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_oldest_lease_ms %lld\n",
                  pool_oldest_lease_ms_.load(std::memory_order_relaxed));
    out += buf;
    // The THRESHOLD beside the counter. Without it a zero counter is
    // ambiguous — no stuck replicas, or the detector switched off? That is
    // exactly what UnifiedPipelinePool::stuck_threshold()'s comment said it was
    // exposed for, and nothing was reading it.
    out += "# HELP turbo_ocr_pool_stuck_lease_threshold_ms TURBO_POOL_STUCK_LEASE_MS. 0 means stuck-lease detection is DISABLED, so the counter below is meaningless rather than reassuring.\n";
    out += "# TYPE turbo_ocr_pool_stuck_lease_threshold_ms gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_stuck_lease_threshold_ms %lld\n",
                  static_cast<long long>(
                      pool_stuck_threshold_ms_.load(std::memory_order_relaxed)));
    out += buf;
    out += "# HELP turbo_ocr_pool_stuck_leases Leases observed held past TURBO_POOL_STUCK_LEASE_MS. Each is a replica effectively lost until restart. Always 0 when the detector is unset.\n";
    out += "# TYPE turbo_ocr_pool_stuck_leases counter\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_pool_stuck_leases %llu\n",
                  static_cast<unsigned long long>(
                      pool_stuck_leases_.load(std::memory_order_relaxed)));
    out += buf;
    out += "# HELP turbo_ocr_dispatcher_queue_depth Requests queued for the GPU pipeline dispatcher.\n";
    out += "# TYPE turbo_ocr_dispatcher_queue_depth gauge\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_dispatcher_queue_depth %zu\n",
                  dispatcher_queue_depth_.load(std::memory_order_relaxed));
    out += buf;

    // GPU VRAM. NOTE: the backend's device_memory() (the scrape handler's source) reports
    // whole-device occupancy, not this process's footprint — on a shared GPU
    // these numbers include every other tenant. See the per-metric HELP below.
    size_t vram_used = gpu_vram_used_.load(std::memory_order_relaxed);
    size_t vram_total = gpu_vram_total_.load(std::memory_order_relaxed);
    if (vram_total > 0) {
      out += "# HELP turbo_ocr_gpu_vram_used_bytes Whole-device GPU memory in use (all processes, not just this server).\n";
      out += "# TYPE turbo_ocr_gpu_vram_used_bytes gauge\n";
      std::snprintf(buf, sizeof(buf), "turbo_ocr_gpu_vram_used_bytes %zu\n", vram_used);
      out += buf;
      out += "# HELP turbo_ocr_gpu_vram_total_bytes Total GPU device memory (shared across all processes).\n";
      out += "# TYPE turbo_ocr_gpu_vram_total_bytes gauge\n";
      std::snprintf(buf, sizeof(buf), "turbo_ocr_gpu_vram_total_bytes %zu\n", vram_total);
      out += buf;
    }

    // Request body sizes
    out += "# HELP turbo_ocr_request_bytes_total Total request body bytes received.\n";
    out += "# TYPE turbo_ocr_request_bytes_total counter\n";
    std::snprintf(buf, sizeof(buf), "turbo_ocr_request_bytes_total %" PRIu64 "\n",
                  request_bytes_total_.load(std::memory_order_relaxed));
    out += buf;
    out += "# HELP turbo_ocr_request_body_avg_bytes Average request body size.\n";
    out += "# TYPE turbo_ocr_request_body_avg_bytes gauge\n";
    uint64_t cnt = request_count_sized_.load(std::memory_order_relaxed);
    uint64_t total_bytes = request_bytes_total_.load(std::memory_order_relaxed);
    double avg = cnt > 0 ? static_cast<double>(total_bytes) / cnt : 0.0;
    std::snprintf(buf, sizeof(buf), "turbo_ocr_request_body_avg_bytes %.0f\n", avg);
    out += buf;

    return out;
  }

  // ── Singleton ──────────────────────────────────────────────────────────

  static Metrics &instance() {
    static Metrics m;
    return m;
  }

private:
  // Histogram bucket boundaries (seconds)
  static constexpr size_t kNumBuckets = 9;
  static constexpr double kBuckets[kNumBuckets] = {
      0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0};

  struct RouteMetrics {
    std::atomic<uint64_t> ok{0};
    std::atomic<uint64_t> client_err{0};
    std::atomic<uint64_t> server_err{0};
    std::array<std::atomic<uint64_t>, kNumBuckets> hist_buckets{};
    std::atomic<uint64_t> hist_sum{0};   // microseconds
    std::atomic<uint64_t> hist_count{0};
  };

  std::array<RouteMetrics, kRouteCount> routes_{};
  std::atomic<uint64_t> pool_exhaustions_{0};
  std::atomic<int> pool_size_{0};
  std::atomic<size_t> workpool_queue_depth_{0};
  std::atomic<size_t> workpool_inflight_{0};
  std::atomic<size_t> workpool_max_depth_{0};
  // Monotonic pool counters mirrored at scrape time (record_saturation).
  std::atomic<uint64_t> workpool_escaped_tasks_{0};
  std::atomic<uint64_t> workpool_discarded_tasks_{0};
  std::atomic<size_t> dispatcher_queue_depth_{0};
  std::atomic<size_t> pool_waiting_{0};
  std::atomic<size_t> pool_available_{0};
  std::atomic<long long> pool_oldest_lease_ms_{0};
  std::atomic<uint64_t> pool_stuck_leases_{0};
  std::atomic<long long> pool_stuck_threshold_ms_{0};
  std::atomic<size_t> gpu_vram_used_{0};
  std::atomic<size_t> gpu_vram_total_{0};
  std::atomic<uint64_t> request_bytes_total_{0};
  std::atomic<uint64_t> request_count_sized_{0};

  // ── Formatting helpers ─────────────────────────────────────────────────

  static void append_counter(std::string &out, const char *metric,
                             const char *route, const char *status,
                             uint64_t val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s{route=\"%s\",status=\"%s\"} %" PRIu64 "\n",
                  metric, route, status, val);
    out += buf;
  }

  static void append_histogram_bucket(std::string &out, const char *route,
                                       const char *le, uint64_t cumulative) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_bucket{route=\"%s\",le=\"%s\"} %" PRIu64 "\n",
        route, le, cumulative);
    out += buf;
  }

  static void append_histogram_summary(std::string &out, const char *route,
                                        double sum, uint64_t count) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_sum{route=\"%s\"} %.6f\n",
        route, sum);
    out += buf;
    std::snprintf(buf, sizeof(buf),
        "turbo_ocr_request_duration_seconds_count{route=\"%s\"} %" PRIu64 "\n",
        route, count);
    out += buf;
  }

};

/// Register the /metrics endpoint and automatic per-request recording.
/// Call this BEFORE drogon::app().run().
///
/// `pool` (optional): snapshot WorkPool saturation gauges at scrape time.
/// `dispatcher_queue_depth` (optional): GPU dispatcher queue depth at scrape.
/// Both default off (CPU build / no dispatcher) so the gauges read 0 only when
/// genuinely unwired, never stale.
/// `pool_saturation` (optional): snapshot the DEVICE-side lease pool at scrape
/// time and run its stuck-lease sweep. Pass it whenever a pipeline pool exists —
/// without it the wedged-replica signal does not exist, and /metrics describes
/// only the queue in front of the device.
/// `device_memory` (optional): (used, total) device bytes at scrape, false when
/// the vendor cannot report them. Injected rather than called directly: this
/// header used to `#include <cuda_runtime_api.h>` and call cudaMemGetInfo under
/// `#ifndef USE_CPU_ONLY`, which put CUDA in a device-neutral service header and
/// would have broken the first non-NVIDIA GPU build (that flag is off there too).
inline void register_metrics_route(
    const WorkPool *pool = nullptr,
    std::function<size_t()> dispatcher_queue_depth = nullptr,
    std::function<void()> pool_saturation = nullptr,
    std::function<bool(size_t &, size_t &)> device_memory = nullptr) {
  // Endpoint — update GPU VRAM + saturation on each scrape (cheap)
  drogon::app().registerHandler(
      "/metrics",
      [pool, dispatcher_queue_depth = std::move(dispatcher_queue_depth),
       pool_saturation = std::move(pool_saturation),
       device_memory = std::move(device_memory)](
          const drogon::HttpRequestPtr &,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        // Device-wide on a shared GPU: these count every tenant, not this
        // server's footprint. The gauges' HELP strings carry the same caveat so
        // leak hunts aren't misled.
        if (device_memory) {
          size_t used = 0, total = 0;
          if (device_memory(used, total)) {
            Metrics::instance().set_gpu_vram_used_bytes(used);
            Metrics::instance().set_gpu_vram_total_bytes(total);
          }
        }
        // Snapshot saturation at the moment of scrape (point-in-time gauges).
        if (pool)
          Metrics::instance().record_saturation(
              pool->queue_depth(), pool->inflight(), pool->max_depth(),
              pool->escaped_tasks(), pool->discarded_tasks());
        if (dispatcher_queue_depth)
          Metrics::instance().set_dispatcher_queue_depth(dispatcher_queue_depth());
        // Also runs the stuck-lease sweep: a scrape is exactly when someone is
        // asking, which is why the pool needs no thread of its own to watch.
        if (pool_saturation) pool_saturation();
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setBody(Metrics::instance().serialize());
        resp->setContentTypeString(
            "text/plain; version=0.0.4; charset=utf-8");
        callback(resp);
      },
      {drogon::Get});
}

} // namespace turbo_ocr::server
