// server_main.cpp — the ONE server entry point.
//
// Replaces the forked src/service/server/cuda/gpu_server_main.cpp + cpu_server_main.cpp
// (and, with src/server/stages.cpp, the forked stages_gpu/stages_cpu). The
// two mains were ~90% identical: config load, model-path validation, PDF
// renderer, a pipeline pool, stage loading, an InferFunc, the same route
// registrations, the same gRPC bootstrap, the same graceful shutdown. The only
// real difference was WHICH device built the pool and the InferFunc — which is
// exactly what the Backend seam abstracts.
//
// Startup order (unchanged from both mains):
//   1. ServerConfig::load_or_die + model-path validation (fail fast)
//   2. PdfRenderer FIRST — it fork()s daemons, which must happen before any
//      inference runtime spins up worker threads / device state
//   3. make_backend(name) -> load_stages -> N x UnifiedOcrPipeline -> pool
//   4. the ONE pipeline::make_infer_func(pool) + the backend's decoder/orient
//   5. device-neutral HTTP routes + gRPC + run()
//
// Backend selection: --backend / TURBO_BACKEND / OCR_BACKEND, empty =>
// auto-detect among the vendors compiled into this binary (see the per-vendor
// *_backend_registry TU that this binary links).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/service/grpc/grpc_service.h"
#include "turbo_ocr/service/http/common_routes.h"
#include "turbo_ocr/service/http/image_routes.h" // /ocr/pixels + /profile (InferFunc-based)
#include "turbo_ocr/service/http/pdf_routes.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/service/server/bootstrap/server_bootstrap.h"
#include "turbo_ocr/service/server/bootstrap/server_config.h"
#include "turbo_ocr/service/server/metrics.h"
#include "turbo_ocr/service/server/server_types.h"
#include "turbo_ocr/service/server/work_pool.h"

#include "turbo_ocr/pipeline/unified/make_infer_func.h"  // pipeline::make_infer_func / UnifiedPipelinePool
#include "turbo_ocr/service/server/unified/backend_stages.h"           // build_backend_runtime
#include "turbo_ocr/service/http/unified_routes.h"   // /ocr/batch + /capabilities/backend
#include "turbo_ocr/pipeline/unified/vlm_factory.h"      // register_device_readback / make_allocator_readback

namespace bootstrap = turbo_ocr::server::bootstrap;

namespace {

// --backend <name> (parsed by ServerConfig's CLI) / TURBO_BACKEND /
// OCR_BACKEND. Empty => auto-detect among the compiled-in backends.
std::string resolve_backend_name(const turbo_ocr::server::ServerConfig &cfg) {
  if (!cfg.backend.empty()) return cfg.backend;
  std::string s = turbo_ocr::env::env_or("TURBO_BACKEND", "");
  if (s.empty()) s = turbo_ocr::env::env_or("OCR_BACKEND", "");
  return s;
}

// WHAT THIS PROCESS IS ACTUALLY RUNNING ON. cfg.log_effective() covers the knobs
// ServerConfig owns; this covers every OTHER knob any stage read on the way up
// (engine, kernel and pool tuning that never passed through the config layer).
// Called after warmup so late reads from stage constructors are included, and
// before serving so it is in the log by the first request.
void log_environment_overrides() {
  std::string knobs;
  for (const auto &[k, v] : turbo_ocr::env::observed()) {
    if (!knobs.empty()) knobs += ' ';
    knobs += k + '=' + v;
  }
  if (!knobs.empty())
    TOCR_LOG_INFO("Environment overrides in effect", "knobs", knobs);
}

} // namespace

// Every HTTP route this server serves, in one place.
//
// Lifted out of main() because it is a LIST, not a sequence of steps: main's
// remaining body is genuinely ordered (renderer before device, device before
// pool, pool before serve), while these lines only have to run after the
// closures exist. Mixing the two made a 197-line main in which the ordering
// constraints that matter were indistinguishable from the ones that do not.
//
// Every registrar takes the SAME `rt.available`, so there is no per-endpoint
// argument list to get out of order — which is what once let the gRPC registrar
// take the same flags in a different order, compile cleanly, and silently
// disable a feature.
void register_http_routes(const turbo_ocr::server::ServerConfig &cfg,
                          turbo_ocr::server::BackendRuntime &rt,
                          turbo_ocr::server::WorkPool &work_pool,
                          const turbo_ocr::server::InferFunc &infer,
                          const turbo_ocr::server::EncodedInferFunc &encoded_infer,
                          const turbo_ocr::server::InferOneFunc &infer_one,
                          const turbo_ocr::server::ImageDecoder &decode,
                          const turbo_ocr::server::OrientFunc &orient_fn,
                          turbo_ocr::render::PdfRenderer &pdf_renderer,
                          const std::function<bool()> &readiness,
                          std::string *capabilities_json_out) {
  using namespace turbo_ocr;
  // ---- HTTP routes (all device-neutral: plain closures + bools) ----
  // Every registrar takes the SAME `rt.available` value. There is no longer a
  // per-endpoint argument list to get out of order — which is what allowed the
  // gRPC registrar below to take the same flags in a different order, and a
  // transposition there to compile cleanly and silently disable a feature.
  // encoded_infer is passed so /ocr and /ocr/raw can reach the backend's
  // on-device decoder. It was previously constructed and never consumed —
  // make_encoded_infer_func had a definition, a declaration, and zero call
  // sites, so GPU-direct decode was unreachable on the primary transport.
  routes::register_common_routes(work_pool, infer, decode, rt.available,
                                 readiness, encoded_infer);
  routes::register_profile_route();
  routes::register_ocr_pixels_route(work_pool, infer, rt.available);
  // RESTORED: both were lost when src/cuda/'s duplicate HTTP layer was deleted
  // and are device-agnostic, so they belong on every backend — not behind the
  // `gpu_routes` flag that used to gate them.
  routes::register_ocr_markdown_route(work_pool, infer, decode, rt.available);
  routes::register_infer_route(work_pool, infer_one, decode);
  routes::register_ocr_stream_route(work_pool, infer, pdf_renderer, decode,
                                    cfg.default_pdf_mode, rt.available,
                                    cfg.pdf_default_dpi, cfg.max_pdf_pages,
                                    orient_fn);
  routes::register_pdf_route(work_pool, infer, pdf_renderer,
                             cfg.default_pdf_mode, rt.available,
                             cfg.max_pdf_pages, orient_fn, cfg.pdf_default_dpi);
  // POST /ocr/batch — the ONE batch route (src/service/http/unified_routes.cpp),
  // typed on pipeline::UnifiedPipelinePool instead of a vendor pool, so it
  // serves every backend. Same request/response contract as the CPU route it
  // replaces (see the PARITY NOTE in that file).
  routes::register_ocr_batch_route_unified(work_pool, rt.pool, rt.pool_size,
                                           decode, rt.available,
                                           cfg.max_batch_images);

  // GET /capabilities — advertise this build's honored feature set; the same
  // document rides in the gRPC HealthResponse below.
  auto caps_info = routes::make_capabilities_info(
      cfg, /*is_gpu=*/rt.caps.device != backend::DeviceKind::Host, rt.available,
      /*profile_endpoint=*/true,
      // FALSE, and now true-to-the-code: run_pdf_job resolves auto_verified to
      // auto on every transport, so nothing verifies. Advertising it would
      // promise a cross-check this build does not perform.
      /*honored_auto_verified=*/false, cfg.pdf_default_dpi);
  // The three axes, all reported: what this backend COULD do, what it DID load,
  // and which backend it is. An operator seeing tables supported-but-not-loaded
  // knows to configure a model; seeing it unsupported knows not to try.
  caps_info.implemented = rt.caps.implemented;
  caps_info.backend_name = rt.caps.name;
  caps_info.device_name = std::string(backend::device_kind_name(rt.caps.device));
  caps_info.engine_mode = std::string(backend::engine_mode_name(rt.caps.mode));
  caps_info.has_native_engine = rt.caps.has_native_engine;
  caps_info.has_onnx_engine = rt.caps.has_onnx_engine;
  const std::string capabilities_json = routes::build_capabilities_json(caps_info);
  routes::register_capabilities_route(caps_info, capabilities_json);
  *capabilities_json_out = capabilities_json;
}

// Observability + readiness wiring, extracted verbatim from main().
struct ReadinessFns {
  std::function<bool()> probe_now; // bounded try-acquire; refreshes the cache
  std::function<bool()> cached;    // lock-free read for the gRPC Health path
};

static ReadinessFns
register_observability(turbo_ocr::server::WorkPool &work_pool,
                       const turbo_ocr::server::BackendRuntime &rt) {
  using namespace turbo_ocr;
  server::Metrics::instance().set_pool_size(rt.pool_size);
  server::register_observability_middleware();
  // /metrics now also describes the DEVICE-side lease pool (waiting, available,
  // oldest lease age, stuck-lease count) and runs the stuck-lease sweep on each
  // scrape. Without this the wedged-replica condition is invisible: the WorkPool
  // queue in front of the device drains normally while every replica is lost.
  server::register_metrics_route(
      &work_pool, /*dispatcher_queue_depth=*/nullptr,
      [pool = rt.pool] {
        pool->check_stuck_leases();
        server::Metrics::instance().set_pool_saturation(
            pool->waiting(), pool->available(),
            pool->oldest_lease_age().count(), pool->stuck_leases(),
            pool->stuck_threshold().count());
      },
      // VRAM through the seam, so /metrics names no vendor. The backend outlives
      // every scrape (owned by `rt`).
      [bk = rt.backend.get()](size_t &used, size_t &total) {
        return bk->device_memory(used, total);
      });

  // Readiness: bounded try-acquire on the shared pool (an unbounded acquire
  // could never answer NOT-ready and would block the probe forever under
  // saturation). The verdict is cached so the gRPC Health path can answer
  // WITHOUT blocking a CQ poller or stealing a pipeline lease.
  struct ProbeState {
    std::atomic<bool> ok{true}; // seeded Ready: the pool exists before traffic
  };
  auto probe = std::make_shared<ProbeState>();
  auto pool = rt.pool;
  ReadinessFns fns;
  fns.probe_now = [pool, probe]() -> bool {
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
  fns.cached = bootstrap::make_cached_readiness(
      std::shared_ptr<const std::atomic<bool>>(probe, &probe->ok));
  return fns;
}

int main(int argc, char **argv) try {
  using namespace turbo_ocr;
  TOCR_LOG_INFO("TurboOCR unified server (multi-backend)");

  // BEFORE anything allocates. The glibc arena policy (frozen mmap threshold,
  // bounded trim threshold, capped arena count, plus the malloc_trim reaper)
  // only bounds RSS if it is set before the allocation pattern it is shaping
  // begins — see the rationale on the function itself.
  //
  // This call was LOST in the merge that collapsed the CPU and GPU mains into
  // this one (it lived in gpu_server_main.cpp). The function, its 20 lines of
  // measured rationale, and the TURBO_OCR_DISABLE_MALLOC_REAPER knob that
  // docs/reference/configuration.md still documents all survived — with no
  // caller. The documented high-water-mark mitigation has not been running.
  bootstrap::tune_glibc_arenas();

  const auto cfg = server::ServerConfig::load_or_die(argc, argv);
  cfg.log_effective();
  bootstrap::g_shutdown_grace_seconds.store(cfg.shutdown_grace_seconds,
                                            std::memory_order_release);

  const auto &rec_paths = cfg.rec_paths;
  if (!cfg.selected_model_name.empty())
    TOCR_LOG_INFO("OCR model selected", "model",
                  std::string_view(cfg.selected_model_name), "det",
                  std::string_view(cfg.det_onnx), "rec",
                  std::string_view(rec_paths.rec), "dict",
                  std::string_view(rec_paths.dict));

  // Fail fast on a missing models/ tree rather than deep inside engine load.
  bootstrap::require_model(cfg.det_onnx, "DET", "file", "_MODEL");
  bootstrap::require_model(rec_paths.rec, "REC", "file", "_MODEL");
  bootstrap::require_model(rec_paths.dict, "REC_DICT", "file", "_MODEL");
  bootstrap::require_model(cfg.cls_onnx, "CLS", "file", "_MODEL");
  if (cfg.disable_angle_cls)
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");

  // Build the PdfRenderer FIRST, before any inference runtime. The renderer
  // fork()s a pool of fastpdf2png daemons; forking after a device runtime has
  // initialized is UB (this held for CUDA and holds for Metal/ORT too).
  render::PdfRenderer pdf_renderer(cfg.pdf_daemons, cfg.pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", cfg.pdf_daemons,
                "workers", cfg.pdf_workers);
  pdf::ensure_pdfium_initialized();

  // --- the ONE device bootstrap -------------------------------------------
  const std::string backend_name = resolve_backend_name(cfg);
  auto rt = server::build_backend_runtime(backend_name, cfg);

  // Wire the shared remote-VLM factory's device readback to THIS backend's
  // allocator, so a kind:openai table/formula spec can crop a device-resident
  // page on any vendor. Host/unified-memory backends work without it; discrete
  // VRAM needs it. The backend outlives every request (owned by `rt`).
  //
  // Registered UNDER THE BACKEND'S DeviceKind, not as a process-global: this
  // binary can hold several vendors (see backend/backend_registry.h), and a
  // single global slot would be last-writer-wins across them.
  pipeline::register_device_readback(
      rt.caps.device,
      pipeline::make_allocator_readback(rt.backend->allocator()));

  // The ONE InferFunc. Every route + gRPC consumes this same std::function.
  const server::InferFunc infer = pipeline::make_infer_func(rt.pool);
  // Its encoded-bytes twin. Routes that still hold the ENCODED bytes should
  // prefer this: it defers the decode into the pipeline, where a backend with
  // an on-device decoder (nvJPEG, vImage) avoids a host decode plus a
  // full-frame H2D. Backends without one decode on the host inside the
  // pipeline, so it is safe to pass everywhere and only the backends that
  // benefit change behaviour.
  const server::EncodedInferFunc encoded_infer =
      pipeline::make_encoded_infer_func(rt.pool, *rt.backend);
  // Single-crop, single-backend inference — what POST /infer runs on.
  const server::InferOneFunc infer_one = pipeline::make_infer_one_func(rt.pool);
  const server::ImageDecoder decode = rt.backend->make_image_decoder();
  const server::OrientFunc orient_fn = rt.backend->make_orient_func();

  // NOTE the argument order: WorkPool(num_threads, max_depth) — the first
  // argument is the THREAD count, the second the queue depth. Reading them the
  // other way round and "making the first configurable" would silently cut the
  // worker count to the pool size. WORK_QUEUE_DEPTH=0 keeps WorkPool's own 8192
  // default; that queue is the admission gate in front of the whole server, so
  // it must be reachable without a rebuild.
  //
  // WORKER COUNT. These threads keep `pool_size` replicas fed and absorb
  // blocking host work (decode, JSON, PDF page joins) off the drogon event
  // loop — NOT to add request concurrency, which the replica pool already
  // bounds. A small multiple of the pool leaves headroom without putting dozens
  // of runnable threads on the scheduler ahead of the replicas. Measured on one
  // RTX 5090 (5 replicas, FUNSD-50 at concurrency 16): everything from 20 to 48
  // threads lands at 579-594 img/s, decaying slowly from there (80 -> 576), so
  // the rule targets that flat region rather than a peak. (It replaced a
  // pool_size*32 rule that put 160 threads on a 20-core box.) HTTP_THREADS
  // overrides it — the GPU main had that override and the merged main dropped it.
  int work_threads = std::clamp(rt.pool_size * 4, 16, 64);
  if (cfg.http_threads && *cfg.http_threads > 0) work_threads = *cfg.http_threads;
  server::WorkPool work_pool(work_threads,
                             cfg.work_queue_depth > 0
                                 ? static_cast<size_t>(cfg.work_queue_depth)
                                 : 8192);
  auto probes = register_observability(work_pool, rt);
  const auto &readiness = probes.probe_now;

  // ---- HTTP routes ---- (the list itself lives in register_http_routes)
  std::string capabilities_json;
  register_http_routes(cfg, rt, work_pool, infer, encoded_infer, infer_one,
                       decode, orient_fn, pdf_renderer, readiness,
                       &capabilities_json);
  // GET /capabilities/backend — the backend facts used to live ONLY here,
  // because CapabilitiesInfo sat in a then-frozen header. That freeze ended
  // with the 2026-07-23 merge and `backend`/`device` are now in /capabilities
  // itself (above). This endpoint is kept because it also reports
  // `available_backends` (everything compiled into the binary, not just the
  // active one) and removing a live endpoint would break any client using it.
  routes::register_backend_capabilities_route(rt.caps, rt.pool_size);
  TOCR_LOG_INFO("Capabilities", "backend", rt.caps.name, "device",
                backend::device_kind_name(rt.caps.device));

  auto grpc_handle =
      server::start_grpc_server(infer, cfg, &pdf_renderer, rt.available,
                                probes.cached, capabilities_json,
                                orient_fn, encoded_infer, infer_one);
  // cppcheck-suppress danglingLifetime
  bootstrap::g_grpc_server_for_drain = grpc_handle.server.get();

  TOCR_LOG_INFO("Starting TurboOCR server", "backend", rt.caps.name, "port",
                cfg.http_port, "grpc_port", cfg.grpc_port, "body_cap_mb",
                cfg.max_body_mb);

  // io_threads: the GPU main sized these off its pool, the CPU main pinned 4.
  // Keep the pool-derived sizing for async devices, 4 for the host backend.
  const int io_threads =
      cfg.http_threads.value_or(rt.caps.async ? std::max(4, rt.pool_size) : 4);
  log_environment_overrides();

  bootstrap::run_http_server(cfg, io_threads, work_pool, rt.pool_size);

  bootstrap::shutdown_grpc_after_run(grpc_handle.server.get());
  return 0;
} catch (const std::exception &e) {
  TOCR_LOG_ERROR("Fatal error during startup", "error",
                 std::string_view(e.what()));
  return 1;
} catch (...) {
  TOCR_LOG_ERROR("Fatal error during startup: unknown exception");
  return 1;
}
