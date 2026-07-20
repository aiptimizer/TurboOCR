#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "turbo_ocr/common/log/logger.h"

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/grpc/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "readiness_gpu.h"
#include "turbo_ocr/server/bootstrap/server_bootstrap.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/bootstrap/stages_gpu.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/http/common_routes.h"
#include "turbo_ocr/http/image_routes.h"
#include "turbo_ocr/http/pdf_routes.h"

using turbo_ocr::render::PdfRenderer;

namespace bootstrap = turbo_ocr::server::bootstrap;

int main(int argc, char **argv) try {
  const auto cfg = turbo_ocr::server::ServerConfig::load_or_die(argc, argv);
  cfg.log_effective();
  bootstrap::g_shutdown_grace_seconds.store(cfg.shutdown_grace_seconds,
                                            std::memory_order_release);

  // Parallelism lives at the request level (work-pool threads); OpenCV's
  // per-op thread pool on top of that just oversubscribes cores under load.
  cv::setNumThreads(0);

  bootstrap::tune_glibc_arenas();

  if (!cfg.selected_model_name.empty())
    TOCR_LOG_INFO("OCR model selected",
                  "model", std::string_view(cfg.selected_model_name),
                  "det",   std::string_view(cfg.det_onnx),
                  "rec",   std::string_view(cfg.rec_paths.rec),
                  "dict",  std::string_view(cfg.rec_paths.dict));

  turbo_ocr::server::validate_gpu_models(cfg);

  // Build the PdfRenderer here — AFTER the fail-fast model validation above
  // (so a missing model std::exit(1)s without orphaning forked daemons) but
  // BEFORE any CUDA/TRT call below. The renderer fork()s a pool of fastpdf2png
  // daemons, and fork() in a process that has already initialized a CUDA
  // context is undefined behaviour (CUDA's driver threads, pinned allocations
  // and fds don't survive a fork cleanly). sweep_orphan_engine_temps /
  // ensure_trt_engine / cudaMemGetInfo below all touch the CUDA context, so the
  // daemons must be spawned ahead of them. The renderer depends on neither the
  // dispatcher nor any model path, so hoisting it above them is safe.
  const int pdf_daemons = cfg.pdf_daemons;
  const int pdf_workers = cfg.pdf_workers;
  PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);

  // PDF extraction mode default
  const turbo_ocr::pdf::PdfMode default_pdf_mode = cfg.default_pdf_mode;
  if (cfg.default_pdf_mode_was_set) {
    TOCR_LOG_INFO("PDF extraction default mode configured", "mode", turbo_ocr::pdf::mode_name(default_pdf_mode));
  } else {
    TOCR_LOG_INFO("PDF extraction default mode: ocr (override per-request with /ocr/pdf?mode=<geometric|auto|auto_verified>)");
  }
  turbo_ocr::pdf::ensure_pdfium_initialized();

  // Engine builds + pool sizing (stages_gpu.cpp) — first CUDA touch, safely
  // after the renderer daemons forked above.
  const auto stages = turbo_ocr::server::load_gpu_stages(cfg);
  const std::string &layout_model = stages.layout_model;
  const std::string &doc_ori_model = stages.doc_ori_model;

  auto dispatcher = turbo_ocr::pipeline::make_pipeline_dispatcher(
      stages.pool_size, stages.det_model, stages.rec_model, stages.rec_dict,
      stages.cls_model, layout_model, doc_ori_model, cfg.det_cfg);
  // Some pipelines may have failed to init (logged); key downstream sizing
  // off the count actually built.
  int pool_size = static_cast<int>(dispatcher->worker_count());

  // Per-request inference deadline (C4). cfg.request_timeout_ms defaults to 0,
  // which leaves the dispatcher in its legacy unbounded-wait mode (non-breaking
  // for existing deployments). When an operator sets REQUEST_TIMEOUT_MS>0 the
  // deadline is per-image/per-page, so it only trips a genuinely wedged worker,
  // never a normal request. A timed-out future is ABANDONED, so every GPU submit
  // below captures its request inputs BY VALUE (see the infer lambda).
  dispatcher->set_request_timeout_ms(cfg.request_timeout_ms);

  const bool nvjpeg_available = turbo_ocr::server::probe_nvjpeg();
  turbo_ocr::server::ImageDecoder decode =
      turbo_ocr::server::make_gpu_image_decoder(nvjpeg_available);

  // Inference function for shared routes (/ocr base64)
  const bool layout_available = !layout_model.empty();
  turbo_ocr::server::InferFunc infer =
      turbo_ocr::server::make_gpu_infer_func(*dispatcher);

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  if (cfg.http_threads) work_threads = *cfg.http_threads;
  turbo_ocr::server::WorkPool work_pool(work_threads);

  // --- Register all routes ---
  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route(
      &work_pool, [d = dispatcher.get()] { return d->queue_depth(); });
  // Readiness probe (readiness_gpu.cpp): bounded real-inference probe whose
  // verdict feeds both HTTP /health/ready and the gRPC cache-only view.
  auto probe = std::make_shared<turbo_ocr::server::GpuProbeState>();
  auto readiness = turbo_ocr::server::make_gpu_readiness(
      *dispatcher, layout_available, probe);
  turbo_ocr::routes::register_health_route(readiness, &work_pool);
  // Structure-backend availability, read from a warmed pipeline itself (the
  // exact default-entry pointers dispatch_router_ gates `tables=1`/`formulas=1`
  // on) rather than re-deriving from config — single source of truth, can't
  // drift. Computed once, fed to the HTTP routes, the gRPC service, and
  // /capabilities so the fail-loud gate is consistent everywhere.
  // Deadline-free submit().get() (not submit_for_default): this is a trivial
  // pointer read on the warmed pool, not request work, so it must not be bound
  // by REQUEST_TIMEOUT_MS — a stray TimeoutError here would abort startup.
  const auto struct_avail = dispatcher->submit([](auto &e) {
    return std::pair<bool, bool>{e.pipeline->has_default_table_backend(),
                                 e.pipeline->has_default_formula_backend()};
  }).get();
  const bool table_avail   = struct_avail.first;
  const bool formula_avail = struct_avail.second;
  turbo_ocr::routes::register_ocr_base64_route(work_pool, infer, decode,
                                               layout_available, table_avail,
                                               formula_avail);
  turbo_ocr::routes::register_image_routes(work_pool, *dispatcher, decode, nvjpeg_available, layout_available,
                                           table_avail, formula_avail, cfg.max_batch_images);
  turbo_ocr::routes::register_pdf_route(work_pool, *dispatcher, pdf_renderer, default_pdf_mode, layout_available,
                                        table_avail, formula_avail,
                                        /*default_dpi=*/100,
                                        cfg.max_pdf_pages,
                                        /*doc_ori_available=*/!doc_ori_model.empty());
  turbo_ocr::routes::register_ocr_stream_route_gpu(work_pool, *dispatcher, pdf_renderer, decode,
                                                   default_pdf_mode, layout_available,
                                                   table_avail, formula_avail,
                                                   /*default_dpi=*/100,
                                                   cfg.max_pdf_pages,
                                                   /*doc_ori_available=*/!doc_ori_model.empty());

  // GET /capabilities — shared builder (capabilities_route.cpp) so the two
  // builds fill the same document; GPU honors auto_verified and has no
  // /profile endpoint.
  const auto caps = turbo_ocr::routes::make_capabilities_info(
      cfg, /*is_gpu=*/true, layout_available, table_avail, formula_avail,
      /*autorotate_available=*/!doc_ori_model.empty(),
      /*profile_endpoint=*/false, /*honored_auto_verified=*/true,
      /*pdf_default_dpi=*/100);
  const std::string capabilities_json =
      turbo_ocr::routes::build_capabilities_json(caps);
  turbo_ocr::routes::register_capabilities_route(caps);

  // Seed the readiness cache once now (the dispatcher just warmed up), so the
  // gRPC cached-only probe has a real verdict before the first HTTP probe runs.
  (void)readiness();

  // gRPC Health uses a CACHE-ONLY view of the readiness verdict (H2): it must
  // never run the GPU probe inline, because Health() executes on the gRPC CQ
  // poller thread and a fresh GPU pass there would stall every queued RPC on
  // that poller. It reads the last cached value the HTTP /health/ready probe
  // (above) refreshes; the cache TTL keeps it fresh under any real probe
  // traffic and we seeded it just above. This also keeps gRPC liveness/
  // readiness GPU-free (M2): a busy GPU can't make Health() block, so it can't
  // flap the process out of service.
  auto readiness_cached = bootstrap::make_cached_readiness(
      std::shared_ptr<const std::atomic<bool>>(probe, &probe->ok));

  // gRPC — same availability bools as the HTTP routes so the fail-loud gate is
  // consistent across transports (TABLE_BACKEND_DISABLED / FORMULA_BACKEND_DISABLED).
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      *dispatcher, cfg, &pdf_renderer, layout_available, readiness_cached,
      table_avail, formula_avail, capabilities_json);
  // Published for the signal-handler drain thread. Not dangling: grpc_handle
  // owns the server on main()'s stack and is destroyed only after run() returns
  // (i.e. after the drain has driven app().quit()). See server_bootstrap.h.
  // cppcheck-suppress danglingLifetime
  bootstrap::g_grpc_server_for_drain = grpc_handle.server.get();

  // HTTP (Drogon) — behind nginx (port 8000), direct access on 8080
  const int port = cfg.http_port;
  int io_threads = std::max(pool_size, 4);

  // MAX_BODY_MB caps the largest accepted upload (default 100). Same env
  // var the entrypoint uses to render the nginx body cap, so the two
  // layers always agree. MAX_BODY_MEMORY_MB controls how much of each
  // body Drogon buffers in memory before spilling to a temp file —
  // larger values cut /tmp I/O at the cost of letting RSS grow with
  // concurrent connections. Default 1024 MB (1 GiB) effectively keeps
  // every body in RAM until MAX_BODY_MB is raised above 1 GiB, since
  // the memory cap is clamped to the total cap below. Lower it on
  // memory-constrained hosts (e.g. MAX_BODY_MEMORY_MB=50 caps body
  // buffer RSS at ~50 MB × concurrent_requests).
  const int max_body_mb = cfg.max_body_mb;
  int max_body_mem_mb = cfg.max_body_mem_mb;
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;

  TOCR_LOG_INFO("HTTP server starting", "port", port, "io_threads", io_threads,
           "work_threads", work_threads, "pool_size", dispatcher->worker_count(),
           "body_cap_mb_drogon", max_body_mb, "body_cap_mb_nginx", max_body_mb,
           "body_mem_mb", max_body_mem_mb);

  // Background readiness refresher: periodically re-runs the GPU probe (off the
  // gRPC CQ poller) so the cached verdict gRPC Health reads stays fresh even for
  // deployments that probe ONLY gRPC Health and never hit HTTP /health/ready.
  // The probe's own 5s TTL gates the real GPU work, so this just guarantees a
  // wake within that window. Stopped + joined before gRPC/dispatcher teardown.
  std::atomic<bool> refresher_stop{false};
  std::thread readiness_refresher([&readiness, &refresher_stop]() {
    while (!refresher_stop.load(std::memory_order_acquire)) {
      (void)readiness();
      for (int i = 0; i < 30 && !refresher_stop.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // Listener + body-size config + graceful-shutdown signal handlers + run().
  // Blocks until app().quit() (the drain thread) returns.
  bootstrap::run_http_server(cfg, io_threads, work_pool);

  refresher_stop.store(true, std::memory_order_release);
  readiness_refresher.join();
  bootstrap::shutdown_grpc_after_run(grpc_handle.server.get());
  return 0;
} catch (const std::exception &e) {
  // Function-try-block on main(): any exception escaping startup (config,
  // engine build, CUDA/TRT init, listener bind) becomes a clean non-zero exit
  // with a logged reason instead of std::terminate + a useless core dump.
  TOCR_LOG_ERROR("Fatal error during startup", "error", std::string_view(e.what()));
  return 1;
} catch (...) {
  TOCR_LOG_ERROR("Fatal error during startup: unknown exception");
  return 1;
}
