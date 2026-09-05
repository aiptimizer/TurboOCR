#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pool/cpu_pipeline_pool.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/http/common_routes.h"
#include "turbo_ocr/http/cpu_image_routes.h"
#include "turbo_ocr/http/pdf_routes.h"
#include "turbo_ocr/server/bootstrap/stages_cpu.h"
#include "turbo_ocr/grpc/grpc_service.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/bootstrap/host_allocator.h"
#include "turbo_ocr/server/bootstrap/server_bootstrap.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"

namespace bootstrap = turbo_ocr::server::bootstrap;

// CPU-only server entry point: pure orchestration. Model/stage loading lives
// in cpu_stages.cpp, the endpoint handlers in src/routes/, request validation
// in request_gate.h — main() only wires them together in startup order and
// runs the servers.
int main(int argc, char **argv) try {
  using namespace turbo_ocr;
  TOCR_LOG_INFO("PaddleOCR CPU-Only Mode (ONNX Runtime)");

  const auto cfg = server::ServerConfig::load_or_die(argc, argv);
  cfg.log_effective();
  bootstrap::g_shutdown_grace_seconds.store(cfg.shutdown_grace_seconds,
                                            std::memory_order_release);

  const auto &rec_paths = cfg.rec_paths;
  if (!cfg.selected_model_name.empty())
    TOCR_LOG_INFO("OCR model selected",
                  "model", std::string_view(cfg.selected_model_name),
                  "det",   std::string_view(cfg.det_onnx),
                  "rec",   std::string_view(rec_paths.rec),
                  "dict",  std::string_view(rec_paths.dict));

  // Validate model paths up front so a missing models/ tree fails fast with a
  // clear error rather than tripping a confusing ORT load failure deep in
  // pipeline construction. Dict file is also required on CPU.
  bootstrap::require_model(cfg.det_onnx, "DET", "file", "_MODEL");
  bootstrap::require_model(rec_paths.rec, "REC", "file", "_MODEL");
  bootstrap::require_model(rec_paths.dict, "REC_DICT", "file", "_MODEL");
  bootstrap::require_model(cfg.cls_onnx, "CLS", "file", "_MODEL");

  std::string cls_model = cfg.cls_onnx;
  if (cfg.disable_angle_cls) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Build the PdfRenderer FIRST, before the ORT pipeline pool below. The
  // renderer fork()s a pool of fastpdf2png daemons; forking is safest done
  // before the inference backend spins up its own worker threads / runtime
  // state (matching the GPU binary, where forking after CUDA init is UB). The
  // renderer has no dependency on the pool, so hoisting it is safe.
  render::PdfRenderer pdf_renderer(cfg.pdf_daemons, cfg.pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", cfg.pdf_daemons,
                "workers", cfg.pdf_workers);
  pdf::ensure_pdfium_initialized();

  const int pool_size = cfg.pipeline_pool_size.value_or(4);
  TOCR_LOG_INFO("CPU pipeline pool size", "pool_size", pool_size);
  auto pool = pipeline::make_cpu_pipeline_pool(
      pool_size, cfg.det_onnx, rec_paths.rec, rec_paths.dict, cls_model,
      cfg.det_cfg);

  // Stage loading (layout / formula / table / doc-ori) — cpu_stages.cpp.
  // Explicitly configured backends that fail to load throw -> clean exit 1.
  const auto stages = server::load_cpu_stages(*pool, pool_size, cfg);
  const auto infer = server::make_cpu_infer_func(*pool);
  const auto orient_fn = server::make_cpu_orient_func(*pool, stages.doc_ori);
  const server::ImageDecoder decode = server::cpu_decode_image;

  // Work pool for offloading blocking inference from the Drogon event loop.
  server::WorkPool work_pool(std::max(pool_size * 32, 128));
  server::bootstrap::install_host_image_pool(std::max(pool_size * 32, 128) + pool_size,
                                             cfg.max_image_pixels_bytes(),
                                             decode::HostImagePool::heap_memory());
  server::Metrics::instance().set_pool_size(pool_size);
  server::register_observability_middleware();
  server::register_metrics_route(&work_pool);

  // Readiness probe for HTTP /health/ready: bounded try-acquire (the CPU
  // pool's acquire() is unbounded, so a plain acquire could never return
  // NOT-ready and would block the probe forever under saturation). 250ms is
  // long enough to ride out a brief burst but short enough to answer the k8s
  // probe before its timeout. The result is cached so the gRPC Health path
  // below can answer WITHOUT blocking a CQ poller or stealing a pipeline
  // lease — the same split the GPU server uses.
  struct CpuProbeState {
    std::atomic<bool> ok{true};  // seeded Ready: pool exists before traffic
  };
  auto probe = std::make_shared<CpuProbeState>();
  auto readiness = [&pool, probe]() -> bool {
    try {
      const bool ok =
          pool->try_acquire_for(std::chrono::milliseconds(250)).has_value();
      probe->ok.store(ok, std::memory_order_release);
      return ok;
    } catch (...) {
      probe->ok.store(false, std::memory_order_release);
      return false;
    }
  };

  // ---- HTTP routes ----
  routes::register_common_routes(work_pool, infer, decode, stages.layout,
                                 stages.table, stages.formula, readiness);
  routes::register_profile_route();
  routes::register_ocr_pixels_route_cpu(work_pool, infer, stages.layout,
                                        stages.table, stages.formula);
  routes::register_pdf_route(work_pool, infer, pdf_renderer,
                             cfg.default_pdf_mode, stages.layout, stages.table,
                             stages.formula, cfg.max_pdf_pages, orient_fn);
  routes::register_ocr_batch_route_cpu(work_pool, *pool, pool_size, decode,
                                       stages.layout, stages.table,
                                       stages.formula, cfg.max_batch_images);

  // GET /capabilities — advertise this build's honored feature set. The same
  // document rides in gRPC HealthResponse.capabilities_json below. CPU
  // registers /profile and aliases auto_verified->auto (not honored as a
  // distinct path; see pdf_routes.cpp); its /ocr/pdf default DPI is 100.
  const auto caps = routes::make_capabilities_info(
      cfg, /*is_gpu=*/false, stages.layout, stages.table, stages.formula,
      /*autorotate_available=*/stages.doc_ori, /*profile_endpoint=*/true,
      /*honored_auto_verified=*/false, /*pdf_default_dpi=*/100);
  const std::string capabilities_json = routes::build_capabilities_json(caps);
  routes::register_capabilities_route(caps);

  // gRPC. Cache-only readiness: Health() runs inline on a CQ poller thread,
  // where the blocking 250ms lease-stealing probe would compete with real
  // RPCs under saturation. Reads the value the HTTP probe above refreshes.
  auto readiness_cached = bootstrap::make_cached_readiness(
      std::shared_ptr<const std::atomic<bool>>(probe, &probe->ok));
  auto grpc_handle = server::start_grpc_server(
      infer, cfg, &pdf_renderer, stages.layout, readiness_cached,
      stages.table, stages.formula, capabilities_json);
  // Published for the signal-handler drain thread. Not dangling: grpc_handle
  // owns the server on main()'s stack and is destroyed only after run()
  // returns (i.e. after the drain has driven app().quit()).
  // cppcheck-suppress danglingLifetime
  bootstrap::g_grpc_server_for_drain = grpc_handle.server.get();

  TOCR_LOG_INFO("Starting CPU-Only OCR Server", "port", cfg.http_port,
                "grpc_port", cfg.grpc_port, "body_cap_mb", cfg.max_body_mb);

  // Listener + body-size config + graceful-shutdown signal handlers + run().
  // The CPU server uses a fixed 4 IO threads. Blocks until app().quit().
  bootstrap::run_http_server(cfg, /*io_threads=*/4, work_pool);

  bootstrap::shutdown_grpc_after_run(grpc_handle.server.get());
  return 0;
} catch (const std::exception &e) {
  // Function-try-block on main(): any exception escaping startup (config, ORT
  // model load, pool acquire, listener bind) becomes a clean non-zero exit
  // with a logged reason instead of std::terminate + a useless core dump.
  TOCR_LOG_ERROR("Fatal error during startup", "error",
                 std::string_view(e.what()));
  return 1;
} catch (...) {
  TOCR_LOG_ERROR("Fatal error during startup: unknown exception");
  return 1;
}
