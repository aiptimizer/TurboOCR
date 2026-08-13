#pragma once

// Shared server startup/shutdown boilerplate for the GPU (TRT) and CPU (ORT)
// mains. This is a pure de-duplication of the ~120 lines of near-identical
// signal/shutdown handling, model-path validation, and Drogon HTTP listener
// setup the two binaries used to carry independently. Pipeline/dispatcher/pool
// construction, route registration, and the genuinely backend-specific bits
// stay in each main.
//
// Header-only on purpose: it pulls in gRPC + Drogon, which only the server
// executables link (not turbo_ocr_common), so each main compiles it into its
// own TU rather than forcing those deps onto the shared static lib.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <drogon/HttpAppFramework.h>
#include <grpcpp/server.h>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/server/bootstrap/server_config.h"
#include "turbo_ocr/service/server/work_pool.h"

namespace turbo_ocr::server::bootstrap {

// Host-RSS containment under sustained high-concurrency, large-image load.
// A burst of large pages (OmniDocBench reaches >100 MP; in-spec pages up to
// the MAX_IMAGE_PIXELS_MP cap decode to hundreds of MB of BGR) at concurrency
// ~150 transiently allocates tens of GB of host buffers. glibc keeps that
// memory in its per-arena free lists after the request frees it, and its
// DYNAMIC mmap threshold ratchets UP as large blocks are freed — so later
// image-sized allocations grow the arena instead of being munmap'd, and RSS
// climbs toward a high-water mark without returning toward baseline.
//
//  - Freeze M_MMAP_THRESHOLD so every image-sized allocation is mmap'd and
//    returned to the OS the instant it's freed, not parked in an arena.
//    Setting it explicitly ALSO disables the upward auto-tuning (the ratchet).
//  - Bound M_TRIM_THRESHOLD so the main arena's top is returned promptly.
//  - Cap M_ARENA_MAX: with the large work-thread pool the default (8*ncpu)
//    arenas each retain their own high-water of freed pages.
// NOTE: this bounds the glibc-arena component only. Grow-only per-thread
// (image decode scratch across the work-thread pool) and per-pipeline (pinned
// upload staging, nvJPEG state) buffers still saturate to the largest-image
// footprint; that high-water is bounded but large for very-large-image corpora.
inline void tune_glibc_arenas() {
#if defined(__GLIBC__)
  mallopt(M_MMAP_THRESHOLD, 1 * 1024 * 1024);
  mallopt(M_TRIM_THRESHOLD, 4 * 1024 * 1024);
  mallopt(M_ARENA_MAX, 8);

  // Belt-and-braces: a low-frequency reaper returns each arena's accumulated
  // free pages to the OS (madvise) so idle RSS settles back toward baseline
  // between load bursts instead of pinning the peak. malloc_trim only releases
  // ALREADY-FREE memory, so it never reclaims live buffers; cheap at 5 s cadence.
  if (!env::env_present("TURBO_OCR_DISABLE_MALLOC_REAPER")) {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        malloc_trim(0);
      }
    }).detach();
  }
#endif
}

// --- Graceful-shutdown globals (shared by both mains) -----------------------
//
// These live in an inline-variable home so both binaries see one definition.
// The signal handler may fire on a different thread than the writer in main(),
// hence the atomics; the gRPC server outlives the detached drain thread (it is
// owned by the GrpcHandle on main()'s stack, which is only destroyed after
// run() returns — i.e. after the drain has driven app().quit()).
inline std::atomic<bool> g_shutdown_requested{false};
inline WorkPool *g_work_pool_for_drain = nullptr;
inline grpc::Server *g_grpc_server_for_drain = nullptr;
// Default 30 matches the config default in case the signal fires before main()
// has finished assigning it.
inline std::atomic<int> g_shutdown_grace_seconds{30};

inline int shutdown_grace_seconds() {
  return g_shutdown_grace_seconds.load(std::memory_order_acquire);
}

// Runs from Drogon's main loop (registered via setTermSignalHandler) —
// safe to start a thread, log, and call app().quit(). The detached
// drainer waits for the WorkPool to quiesce before tearing down Drogon
// so inflight requests get to send their response. It ALSO begins the gRPC
// graceful shutdown in parallel (stop admitting new RPCs + drain in-flight up
// to the same deadline) rather than waiting for run() to return — otherwise an
// in-flight gRPC call would be cut mid-response when Shutdown() finally fired.
inline void begin_graceful_shutdown(const char *signal_name) {
  if (g_shutdown_requested.exchange(true)) return;
  TOCR_LOG_INFO("Graceful shutdown requested",
                "signal", std::string_view(signal_name),
                "grace_seconds", shutdown_grace_seconds());
  std::thread([signal_name]() {
    const int grace = shutdown_grace_seconds();
    // ONE wall-clock window shared by both drains. gRPC Shutdown(deadline)
    // BLOCKS until its in-flight RPCs finish or the deadline hits, so running
    // it inline before the HTTP drain would serialize the two (total up to
    // 2*grace, overrunning the k8s termination window). Drain gRPC on its own
    // thread and the WorkPool against the SAME absolute deadline, so both are
    // bounded by one grace window and truly run concurrently.
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::seconds(grace);

    std::thread grpc_thread;
    if (g_grpc_server_for_drain) {
      grpc_thread = std::thread([signal_name, grace]() {
        auto grpc_deadline =
            std::chrono::system_clock::now() + std::chrono::seconds(grace);
        g_grpc_server_for_drain->Shutdown(grpc_deadline);
        TOCR_LOG_INFO("gRPC graceful shutdown complete",
                      "signal", std::string_view(signal_name));
      });
    }
    if (g_work_pool_for_drain) {
      // Remaining budget against the shared deadline (>= 0).
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now());
      if (remaining.count() < 0) remaining = std::chrono::milliseconds(0);
      bool drained = g_work_pool_for_drain->wait_drain(remaining);
      if (drained) {
        TOCR_LOG_INFO("Inflight work drain complete",
                      "drained", drained,
                      "signal", std::string_view(signal_name));
      } else {
        // Grace expired with work still pending. Without this, ~WorkPool
        // would RUN the whole queued backlog during teardown — unbounded time
        // past the grace window, ending in the orchestrator's SIGKILL cutting
        // an in-flight request. Drop what never started (their clients see
        // the connection close at exit — the outcome SIGKILL gave them
        // anyway); in-flight tasks keep running and the pool's destructor
        // still joins them, so started work finishes instead of dying
        // mid-kernel. The count also feeds the /metrics
        // workpool_discarded_tasks_total counter.
        const size_t dropped = g_work_pool_for_drain->discard_pending();
        TOCR_LOG_WARN("Shutdown grace expired — discarding queued work; "
                      "in-flight requests run to completion",
                      "discarded", dropped,
                      "inflight", g_work_pool_for_drain->inflight(),
                      "signal", std::string_view(signal_name));
      }
    }
    if (grpc_thread.joinable()) grpc_thread.join();
    drogon::app().quit();
  }).detach();
}

// Validate model paths up front so a missing models/ tree fails fast with a
// clear error rather than tripping a confusing CUDA/TRT or ORT load failure
// deep in pipeline construction. The "not found" noun and the env-var suffix
// differ between the backends (the GPU main hints "<PURPOSE>_ONNX", the CPU
// main "<PURPOSE>_MODEL"), so both are passed in to keep the message
// byte-for-byte identical to the per-main copies this replaces.
inline void require_model(const std::string &path, const char *purpose,
                          const char *noun, const char *env_suffix) {
  if (!std::filesystem::exists(path)) {
    TOCR_LOG_ERROR("Model file missing",
                   "purpose", std::string_view(purpose),
                   "path", std::string_view(path));
    std::cerr << "[FATAL] " << purpose << " " << noun << " not found at: " << path
              << "\n        Run scripts/models/fetch/download_models.sh or set "
              << purpose << env_suffix << " env var.\n";
    std::exit(1);
  }
}

// Apply the shared Drogon HTTP body-size + listener configuration from the
// ServerConfig, register the graceful-shutdown signal handlers, and run() the
// event loop. Returns once Drogon's loop exits (e.g. via app().quit() from the
// drain thread). The thread count differs per backend (GPU sizes io_threads off
// the pool; CPU uses a fixed 4), so it is passed in. `work_pool` is published to
// the drain globals here, immediately before the signal handlers can fire.
//
// MAX_BODY_MB caps the largest accepted upload; MAX_BODY_MEMORY_MB controls how
// much of each body Drogon buffers in memory before spilling to a temp file. The
// memory cap is clamped to the total cap so the in-RAM default never exceeds the
// accepted body size. Same env vars on both servers, matching the nginx body cap.
inline void run_http_server(const ServerConfig &cfg, int io_threads,
                            WorkPool &work_pool, int pool_size = 0) {
  const int max_body_mb = cfg.max_body_mb;
  int max_body_mem_mb = cfg.max_body_mem_mb;
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;
  size_t max_body_bytes = static_cast<size_t>(max_body_mb) * 1024 * 1024;
  size_t max_mem_bytes  = static_cast<size_t>(max_body_mem_mb) * 1024 * 1024;

  // Graceful shutdown on SIGTERM (Docker / K8s) and SIGINT (Ctrl-C):
  // drain WorkPool inflight up to SHUTDOWN_GRACE_SECONDS before quit().
  g_work_pool_for_drain = &work_pool;
  // The one line that says what the server actually resolved to. Both derived
  // numbers (worker threads, replica count) are the levers throughput turns on,
  // and until this was restored neither was observable in a running deployment.
  TOCR_LOG_INFO("HTTP server starting", "port", cfg.http_port, "io_threads",
                io_threads, "work_threads",
                static_cast<int>(work_pool.num_threads()), "pool_size",
                pool_size, "body_cap_mb", max_body_mb, "body_mem_mb",
                max_body_mem_mb);
  drogon::app()
      .setTermSignalHandler([] { begin_graceful_shutdown("SIGTERM"); })
      .setIntSignalHandler([]  { begin_graceful_shutdown("SIGINT");  })
      .addListener(cfg.host, cfg.http_port)
      .setThreadNum(io_threads)
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(max_body_bytes)
      .setClientMaxMemoryBodySize(max_mem_bytes)
      .run();
}

// Final post-run() gRPC teardown shared by both mains: on a signal-driven exit
// begin_graceful_shutdown() already drained gRPC, so this is an idempotent no-op
// (Shutdown is safe to call twice). On any other exit path it's the one that
// stops the server before the thread join. Either way the CQ must be shut down
// first.
inline void shutdown_grpc_after_run(grpc::Server *server) {
  TOCR_LOG_INFO("HTTP server stopped, shutting down gRPC");
  server->Shutdown();
  TOCR_LOG_INFO("Shutdown complete");
}

// Cache-only readiness view for gRPC Health: reads the verdict the HTTP
// probe refreshes, and flips not-ready the moment a drain begins. Shared by
// both server mains — the CPU and GPU probes differ in HOW they refresh the
// atomic, never in how gRPC consumes it.
[[nodiscard]] inline std::function<bool()>
make_cached_readiness(std::shared_ptr<const std::atomic<bool>> ok) {
  return [ok = std::move(ok)]() -> bool {
    if (g_shutdown_requested.load(std::memory_order_acquire)) return false;
    return ok->load(std::memory_order_acquire);
  };
}

} // namespace turbo_ocr::server::bootstrap
