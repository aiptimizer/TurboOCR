#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/common/stage_profiler.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/cpu_pipeline_pool.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_bootstrap.h"
#include "turbo_ocr/server/server_config.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/pixel_dims.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;

namespace bootstrap = turbo_ocr::server::bootstrap;

namespace {
// Mirrors reject_unknown_query_params in the shared routes: when
// TURBO_OCR_STRICT_QUERY_PARAMS=1, reject any query param not in `allowed`
// (400 INVALID_PARAMETER). No-op otherwise. Applied to the CPU-inline
// /ocr/pixels and /ocr/batch handlers so strict mode behaves like the GPU build.
[[nodiscard]] bool cpu_reject_unknown_query_params(
    const drogon::HttpRequestPtr &req,
    std::initializer_list<std::string_view> allowed,
    turbo_ocr::server::DrogonCallback &callback) {
  static const bool strict =
      turbo_ocr::server::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  if (!strict) return false;
  for (const auto &kv : req->getParameters()) {
    bool known = false;
    for (auto a : allowed) if (a == kv.first) { known = true; break; }
    if (!known) {
      callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
          "INVALID_PARAMETER",
          std::format("Unknown query parameter '{}' "
                      "(TURBO_OCR_STRICT_QUERY_PARAMS=1)", kv.first)));
      return true;
    }
  }
  return false;
}
}  // namespace

int main(int argc, char **argv) try {
  TOCR_LOG_INFO("PaddleOCR CPU-Only Mode (ONNX Runtime)");

  const auto cfg = turbo_ocr::server::ServerConfig::load_or_die(argc, argv);
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
  auto det_model = cfg.det_onnx;
  auto rec_model = rec_paths.rec;
  auto rec_dict  = rec_paths.dict;
  auto cls_model = cfg.cls_onnx;

  // Validate model paths up front so a missing models/ tree fails fast
  // with a clear error rather than tripping a confusing ORT load failure
  // deep in pipeline construction. Dict file is also required on CPU.
  auto require_model = [](const std::string &path, const char *purpose) {
    bootstrap::require_model(path, purpose, "file", "_MODEL");
  };
  require_model(det_model, "DET");
  require_model(rec_model, "REC");
  require_model(rec_dict, "REC_DICT");
  require_model(cls_model, "CLS");

  if (cfg.disable_angle_cls) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Build the PdfRenderer FIRST, before the ORT pipeline pool below. The
  // renderer fork()s a pool of fastpdf2png daemons; forking is safest done
  // before the inference backend spins up its own worker threads / runtime
  // state (matching the GPU binary, where forking after CUDA init is UB). The
  // renderer has no dependency on the pool, so hoisting it is safe.
  const int pdf_daemons = cfg.pdf_daemons;
  const int pdf_workers = cfg.pdf_workers;
  turbo_ocr::render::PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);
  turbo_ocr::pdf::ensure_pdfium_initialized();

  // Layout model (CPU via ONNX Runtime) — on by default. Optional: a
  // missing layout.onnx soft-disables the stage below rather than aborting.
  std::string layout_model = cfg.layout_onnx;
  const bool layout_disabled = cfg.layout_disabled;
  bool layout_available = false;

  const int pool_size = cfg.pipeline_pool_size.value_or(4);

  TOCR_LOG_INFO("CPU pipeline pool size", "pool_size", pool_size);
  auto pool = turbo_ocr::pipeline::make_cpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model, cfg.det_cfg);

  // Load layout model into each pipeline if enabled
  if (!layout_disabled && !layout_model.empty()) {
    bool all_ok = true;
    for (size_t i = 0; i < static_cast<size_t>(pool_size); ++i) {
      auto handle = pool->acquire();
      if (!handle->load_layout_model(layout_model)) {
        TOCR_LOG_WARN("Layout model not found; layout disabled");
        all_ok = false;
        break;
      }
    }
    if (all_ok) {
      layout_available = true;
      TOCR_LOG_INFO("Layout detection enabled (CPU/ONNX Runtime)");
    }
  } else if (layout_disabled) {
    TOCR_LOG_INFO("Layout detection disabled");
  }

  // CUDA-FREE formula + table stages (optional, env-gated, same knobs as the
  // GPU server). FORMULA_ONNX + FORMULA_TOKENIZER enable PP-FormulaNet-S on
  // ORT-CPU; TABLE_SLANEXT_ENCODER_ONNX enables the SLANeXt ORT-CPU encoder +
  // host GRU decode. A configured-but-unloadable backend is fatal (never serve
  // a silently structure-less pipeline). Loaded into every pipeline in the pool.
  // Lifted to function scope so the route handlers can fail-loud (400
  // TABLE_BACKEND_DISABLED / FORMULA_BACKEND_DISABLED) when a client asks for a
  // stage this server didn't load.
  bool table_available = false, formula_available = false;
  {
    std::string formula_onnx = turbo_ocr::server::env_or("FORMULA_ONNX", "");
    std::string formula_tok = turbo_ocr::server::env_or("FORMULA_TOKENIZER", "");
    // Auto-resolve baked formula weights when only FORMULA_BACKEND is set
    // (parity with the GPU build): default to models/formula/ppformulanet_s
    // (CPU uses the fused inference_trt.onnx inside it via CpuFormulaRecognizer).
    if (turbo_ocr::server::env_or("FORMULA_BACKEND", "") == "ppformulanet_s" &&
        formula_onnx.empty()) {
      formula_onnx = "models/formula/ppformulanet_s";
      if (formula_tok.empty())
        formula_tok = "models/formula/ppformulanet_s/tokenizer.json";
    }
    // Auto-resolve the baked SLANeXt encoder when only TABLE_BACKEND=slanext is
    // set; load_table_backend() reads TABLE_SLANEXT_ENCODER_ONNX from the env.
    if (turbo_ocr::server::env_or("TABLE_BACKEND", "") == "slanext" &&
        turbo_ocr::server::env_or("TABLE_SLANEXT_ENCODER_ONNX", "").empty()) {
      setenv("TABLE_SLANEXT_ENCODER_ONNX",
             "models/table/slanext_encoder/SLANeXt_wired_encoder.onnx", 1);
    }
    const bool want_formula = !formula_onnx.empty() && !formula_tok.empty();
    const bool want_table = !turbo_ocr::server::env_or("TABLE_SLANEXT_ENCODER_ONNX", "").empty();
    if (want_formula || want_table) {
      for (size_t i = 0; i < static_cast<size_t>(pool_size); ++i) {
        auto handle = pool->acquire();
        if (want_formula && !handle->load_formula_model(formula_onnx, formula_tok)) {
          TOCR_LOG_ERROR("CPU formula backend failed to load — refusing to start",
                         "model", std::string_view(formula_onnx));
          return 1;
        }
        if (want_table && !handle->load_table_backend()) {
          TOCR_LOG_ERROR("CPU table backend failed to load — refusing to start");
          return 1;
        }
        // Read availability from what actually loaded into the pipeline (single
        // source of truth for the tables=1/formulas=1 fail-loud gate), not env
        // intent. All pipelines load identically, so the last wins.
        table_available = handle->has_table_backend();
        formula_available = handle->has_formula_backend();
      }
      if (formula_available) TOCR_LOG_INFO("Formula stage enabled (CPU/ONNX Runtime)");
      if (table_available) TOCR_LOG_INFO("Table stage enabled (CPU/ONNX Runtime)");
    }
  }

  // Load the document-orientation model into each pipeline (optional). Powers
  // /ocr/pdf?autorotate=1. Absent model -> autorotate requests are rejected.
  bool doc_ori_available = false;
  if (!cfg.doc_ori_onnx.empty()) {
    bool all_ok = true;
    for (size_t i = 0; i < static_cast<size_t>(pool_size); ++i) {
      auto handle = pool->acquire();
      if (!handle->load_doc_ori_model(cfg.doc_ori_onnx)) { all_ok = false; break; }
    }
    doc_ori_available = all_ok;
    TOCR_LOG_INFO(doc_ori_available
                      ? "Doc-orientation (autorotate) enabled (CPU/ONNX Runtime)"
                      : "Doc-orientation model not found; autorotate disabled");
  }

  turbo_ocr::server::OrientFunc orient_fn;
  if (doc_ori_available)
    orient_fn = [&pool](const cv::Mat &img) -> int {
      auto handle = pool->acquire();
      return handle->detect_orientation(img);
    };

  turbo_ocr::server::InferFunc infer =
      [&pool](const cv::Mat &img,
              const turbo_ocr::server::InferOptions &opts)
          -> turbo_ocr::server::InferResult {
    auto handle = pool->acquire();
    auto out = handle->run_with_layout(img, opts.want_layout,
                                        opts.want_reading_order,
                                        opts.want_tables, opts.want_formulas);
    return turbo_ocr::server::InferResult{
        .results          = std::move(out.results),
        .layout           = std::move(out.layout),
        .reading_order    = std::move(out.reading_order),
        .tables           = std::move(out.tables),
        .formulas         = std::move(out.formulas),
        .formula_degraded = out.formula_degraded,
        .formula_warning  = std::move(out.formula_warning),
        .table_degraded   = out.table_degraded,
        .table_warning    = std::move(out.table_warning),
        .text_degraded    = out.text_degraded,
        .text_warning     = std::move(out.text_warning),
    };
  };

  turbo_ocr::server::ImageDecoder decode = turbo_ocr::server::cpu_decode_image;

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  turbo_ocr::server::WorkPool work_pool(work_threads);

  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route(&work_pool);

  // Readiness probe for HTTP /health/ready: bounded try-acquire (the CPU
  // pool's acquire() is unbounded, so a plain acquire could never return
  // NOT-ready and would block the probe forever under saturation). 250ms is
  // long enough to ride out a brief burst but short enough to answer the k8s
  // probe before its timeout. The result is cached so the gRPC Health path
  // below can answer WITHOUT blocking a CQ poller or stealing a pipeline
  // lease — the same split the GPU server uses (readiness vs readiness_cached).
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
  turbo_ocr::routes::register_common_routes(work_pool, infer, decode,
                                             layout_available, table_available,
                                             formula_available, readiness);

  // GET /capabilities (M6) — advertise this build's honored feature set so a
  // client can discover the known GPU/CPU divergences without trial requests.
  // CPU registers /profile and aliases auto_verified->auto (not honored as a
  // distinct path; see pdf_routes.cpp), and its /ocr/pdf default DPI is 100.
  {
    turbo_ocr::routes::CapabilitiesInfo caps;
    caps.is_gpu              = false;
    caps.layout_available    = layout_available;
    caps.table_available     = table_available;
    caps.formula_available   = formula_available;
    caps.autorotate_available = doc_ori_available;
    caps.profile_endpoint    = true;
    caps.grpc_response_mode  =
        std::string(turbo_ocr::server::detail::grpc_mode_str(cfg.grpc_response_mode));
    caps.honored_auto_verified = false;
    caps.pdf_default_dpi     = 100;
    caps.max_pdf_pages       = cfg.max_pdf_pages;
    caps.max_body_mb         = cfg.max_body_mb;
    caps.max_image_dim       = cfg.max_image_dim;
    caps.max_batch_images    = cfg.max_batch_images;
    turbo_ocr::routes::register_capabilities_route(caps);
  }

  // --- /profile endpoint: read+reset per-stage timing (PROFILE_STAGES=1) ---
  drogon::app().registerHandler(
      "/profile",
      [](const drogon::HttpRequestPtr &,
         std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(turbo_ocr::prof::dump_json_and_reset());
        callback(resp);
      },
      {drogon::Get});

  // --- /ocr/pixels endpoint (raw BGR pixel data, zero decode overhead) ---
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&work_pool, &infer, layout_available, table_available, formula_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        turbo_ocr::server::InferOptions opts;
        if (auto r = turbo_ocr::server::parse_query_options(
                req, layout_available, &opts);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }
        if (cpu_reject_unknown_query_params(
                req, {"layout", "reading_order", "as_blocks", "tables",
                      "formulas", "text", "width", "height", "channels"},
                callback))
          return;
        if (auto r = turbo_ocr::server::check_structure_backends(
                opts, table_available, formula_available);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }

        // Dimensions: query params (preferred) with the legacy X-* headers as a
        // v2.3-compat fallback; conflict -> 400. Shared with the GPU handler.
        const auto dims = turbo_ocr::server::resolve_pixel_dims(req);
        if (!dims.ok()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              dims.error_code.c_str(), dims.error));
          return;
        }
        const int width = dims.width, height = dims.height,
                  channels = dims.channels;
        const bool used_legacy_dim_header = dims.used_legacy_header;

        // Configurable via MAX_IMAGE_DIM (default 16384). Same env var as
        // the GPU path so both servers share one knob.
        const int kMaxPixelDim = turbo_ocr::decode::max_image_dim();
        if (width > kMaxPixelDim || height > kMaxPixelDim) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "DIMENSIONS_TOO_LARGE", std::format("Dimensions {}x{} exceed maximum of {}x{}",
                          width, height, kMaxPixelDim, kMaxPixelDim)));
          return;
        }
        // Pixel-AREA cap, mirroring the GPU /ocr/pixels handler. The body-size
        // check below bounds RAM only as long as MAX_BODY_MB stays small;
        // raising it would let a 268 MP body allocate ~805 MB without this.
        if (turbo_ocr::decode::exceeds_pixel_cap(width, height)) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "PIXELS_TOO_LARGE", std::format("Image area {}x{} exceeds maximum of {} pixels",
                          width, height, turbo_ocr::decode::max_image_pixels())));
          return;
        }

        size_t expected = static_cast<size_t>(width) * height * channels;
        if (req->body().size() != expected) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "BODY_SIZE_MISMATCH", std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                          expected, width, height, channels, req->body().size())));
          return;
        }

        turbo_ocr::server::submit_work(work_pool, std::move(callback),
            [req, &infer, width, height, channels, opts,
             used_legacy_dim_header](turbo_ocr::server::DrogonCallback &cb) {
          turbo_ocr::server::run_with_error_handling(cb, "/ocr/pixels", [&] {
            cv::Mat img(height, width,
                        channels == 3 ? CV_8UC3 : CV_8UC1,
                        const_cast<char *>(req->body().data()));
            // The detector is BGR-only; a 1-channel Mat hits a nullptr
            // memcpy in CpuPaddleDet's cv::split path (SIGSEGV). Expand
            // grayscale up front, matching the GPU /ocr/pixels handler.
            if (channels == 1)
              cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            auto inf = infer(img, opts);
            auto resp = turbo_ocr::server::json_response(
                turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks));
            if (used_legacy_dim_header)
              turbo_ocr::server::stamp_pixel_dim_deprecation(resp);
            cb(resp);
          });
        });
      },
      {drogon::Post});

  // --- /ocr/pdf endpoint (CPU: sequential page OCR) ---
  // pdf_renderer + pdfium are initialized near the top of main(), before the
  // ORT pipeline pool, so the daemon fork()s happen ahead of backend startup.
  const turbo_ocr::pdf::PdfMode default_pdf_mode = cfg.default_pdf_mode;

  turbo_ocr::routes::register_pdf_route(work_pool, infer, pdf_renderer, default_pdf_mode, layout_available,
                                        table_available, formula_available,
                                        cfg.max_pdf_pages, orient_fn);

  // --- /ocr/batch endpoint (CPU version) ---
  const int max_batch_images = cfg.max_batch_images;
  drogon::app().registerHandler(
      "/ocr/batch",
      [&work_pool, &pool, pool_size, &decode, layout_available,
       table_available, formula_available, max_batch_images](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        turbo_ocr::server::InferOptions opts;
        if (auto r = turbo_ocr::server::parse_query_options(
                req, layout_available, &opts);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }
        if (cpu_reject_unknown_query_params(
                req, {"layout", "reading_order", "as_blocks", "tables",
                      "formulas", "text"}, callback))
          return;
        if (auto r = turbo_ocr::server::check_structure_backends(
                opts, table_available, formula_available);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }
        const bool want_layout = opts.want_layout;

        auto json = req->getJsonObject();
        if (!json) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
          return;
        }
        if (!json->isMember("images") || !(*json)["images"].isArray()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
          return;
        }

        auto &images_json = (*json)["images"];
        size_t n = images_json.size();
        if (n == 0) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
          return;
        }
        // Cap before O(n) per-slot allocations (see GPU route): an unbounded
        // images[] is a memory-amplification OOM lever.
        if (n > static_cast<size_t>(max_batch_images)) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "BATCH_TOO_LARGE",
              std::format("images array has {} entries, max is {}", n, max_batch_images)));
          return;
        }

        // Pre-decode base64. asString() throws Json::LogicError on a
        // non-string element ({}/[]), so guard the type — a malformed slot
        // becomes an empty string the per-slot path tags, not a crash.
        auto raw_bytes = std::make_shared<std::vector<std::string>>(n);
        for (size_t i = 0; i < n; ++i) {
          const auto &el = images_json[static_cast<int>(i)];
          if (el.isString())
            (*raw_bytes)[i] = base64_decode(el.asString());
        }

        turbo_ocr::server::submit_work(work_pool, std::move(callback),
            [raw_bytes, n, &pool, pool_size, &decode, opts](turbo_ocr::server::DrogonCallback &cb) {
         turbo_ocr::server::run_with_error_handling(cb, "/ocr/batch", [&] {
          const bool want_layout = opts.want_layout;
          // Same MAX_IMAGE_DIM cap as /ocr, /ocr/raw, /ocr/pixels.
          const int kMaxImageDim = turbo_ocr::decode::max_image_dim();

          // Per-slot batch state. Cardinality MUST equal the input array
          // length so callers can correlate batch_results[i] with their
          // images[i]; failed slots get tagged in `error`, never dropped.
          struct BatchItem {
            // Full per-page result so the serializer emits tables/formulas (+
            // degradation) when the request opted in, identical to /ocr/raw.
            turbo_ocr::pipeline::OcrPipelineResult out;
            std::string error;          // empty when the slot succeeded
          };
          std::vector<BatchItem> batch_items(n);
          std::vector<cv::Mat> imgs(n);

          // Empty inputs short-circuit so we never pay decode cost on
          // slots the caller never filled in.
          for (size_t i = 0; i < n; ++i) {
            if ((*raw_bytes)[i].empty()) batch_items[i].error = "empty";
          }

          // Pre-decode dim sniff: refuses oversized PNG/JPEG/WebP per-slot
          // before the decoder allocates a buffer. Mirrors GPU /ocr/batch:
          // per-side cap, per-image pixel-AREA cap, and an aggregate
          // decoded-pixel budget so the per-image cap is not silently
          // multiplied by the batch size into a host-RAM OOM.
          int64_t cumulative_pixels = 0;
          const int64_t batch_pixel_budget = turbo_ocr::decode::max_batch_pixels();
          for (size_t i = 0; i < n; ++i) {
            if (!batch_items[i].error.empty()) continue;
            const auto &raw = (*raw_bytes)[i];
            if (auto d = turbo_ocr::decode::peek_image_dimensions(
                    reinterpret_cast<const unsigned char *>(raw.data()), raw.size())) {
              if (d->width > kMaxImageDim || d->height > kMaxImageDim) {
                batch_items[i].error = std::format(
                    "dimensions_too_large ({}x{} > {}x{})",
                    d->width, d->height, kMaxImageDim, kMaxImageDim);
              } else if (turbo_ocr::decode::exceeds_pixel_cap(d->width, d->height)) {
                batch_items[i].error = std::format(
                    "pixels_too_large ({}x{} > {} px)",
                    d->width, d->height, turbo_ocr::decode::max_image_pixels());
              } else {
                const int64_t pix = static_cast<int64_t>(d->width) * d->height;
                if (cumulative_pixels + pix > batch_pixel_budget) {
                  batch_items[i].error = std::format(
                      "batch_pixels_exceeded (batch sum > {} px)",
                      batch_pixel_budget);
                } else {
                  cumulative_pixels += pix;
                }
              }
            }
          }

          // Decode every still-valid slot; tag decode failures per-slot.
          for (size_t i = 0; i < n; ++i) {
            if (!batch_items[i].error.empty()) continue;
            const auto &raw = (*raw_bytes)[i];
            imgs[i] = decode(
                reinterpret_cast<const unsigned char *>(raw.data()),
                raw.size());
            if (imgs[i].empty()) {
              batch_items[i].error = "decode_failed";
              continue;
            }
            // Post-decode safety net for residual non-sniffed formats
            // (BMP/PNM): per-side and per-image pixel-AREA cap, matching GPU
            // /ocr/batch. PNG/JPEG/WebP/TIFF/GIF are bounded pre-decode above.
            if (imgs[i].cols > kMaxImageDim || imgs[i].rows > kMaxImageDim) {
              batch_items[i].error = std::format(
                  "dimensions_too_large ({}x{} > {}x{})",
                  imgs[i].cols, imgs[i].rows, kMaxImageDim, kMaxImageDim);
              imgs[i].release();
            } else if (turbo_ocr::decode::exceeds_pixel_cap(imgs[i].cols, imgs[i].rows)) {
              batch_items[i].error = std::format(
                  "pixels_too_large ({}x{} > {} px)",
                  imgs[i].cols, imgs[i].rows, turbo_ocr::decode::max_image_pixels());
              imgs[i].release();
            }
          }

          // Build the work index list (slots that survived all checks).
          std::vector<size_t> valid_indices;
          valid_indices.reserve(n);
          for (size_t i = 0; i < n; ++i) {
            if (batch_items[i].error.empty() && !imgs[i].empty())
              valid_indices.push_back(i);
          }

          std::atomic<size_t> next_valid{0};
          int num_workers = valid_indices.empty()
              ? 0
              : std::min(static_cast<int>(valid_indices.size()), pool_size);
          {
            // Every request keeps ONE guaranteed worker (progress under
            // contention); extra fan-out workers draw from a process-wide
            // permit pool so N concurrent /ocr/batch requests can't create
            // N*pool_size OS threads (most would only block on pool->acquire()
            // anyway). Same bound the gRPC batch path applies.
            static std::counting_semaphore<4096> extra_batch_permits{
                static_cast<std::ptrdiff_t>(turbo_ocr::server::env_int(
                    "BATCH_FANOUT_GLOBAL_WORKERS", 64, 1, 4096))};
            struct Permit {
              bool held = false;
              ~Permit() { if (held) extra_batch_permits.release(); }
            };
            std::vector<Permit> permits(
                static_cast<size_t>(std::max(0, num_workers - 1)));
            std::vector<std::jthread> threads;
            threads.reserve(num_workers);
            const auto worker_body = [&]() {
                // Acquire the pool handle outside the per-image loop —
                // pool exhaustion is a fatal worker-level error (no point
                // trying again), so it tags every remaining slot.
                std::unique_ptr<decltype(pool->acquire())> handle_holder;
                try {
                  handle_holder = std::make_unique<decltype(pool->acquire())>(
                      pool->acquire());
                } catch (const turbo_ocr::PoolExhaustedError &e) {
                  TOCR_LOG_ERROR("Batch worker pool exhausted", "route", "/ocr/batch");
                  // Tag every UNCLAIMED valid slot as failed so callers see it.
                  while (true) {
                    size_t k = next_valid.fetch_add(1);
                    if (k >= valid_indices.size()) break;
                    batch_items[valid_indices[k]].error = "pool_exhausted";
                  }
                  return;
                }
                auto &handle = *handle_holder;
                while (true) {
                  size_t k = next_valid.fetch_add(1);
                  if (k >= valid_indices.size()) break;
                  size_t idx = valid_indices[k];
                  // Per-image try/catch so one image failing does NOT
                  // leave all later slots silently empty with HTTP 200.
                  try {
                    batch_items[idx].out = handle->run_with_layout(
                        imgs[idx], want_layout, opts.want_reading_order,
                        opts.want_tables, opts.want_formulas);
                  } catch (const std::exception &e) {
                    TOCR_LOG_ERROR("Batch image error", "route", "/ocr/batch",
                                   "image_index", idx, "error",
                                   std::string_view(e.what()));
                    batch_items[idx].error = e.what();
                  } catch (...) {
                    TOCR_LOG_ERROR("Batch image error: unknown",
                                   "route", "/ocr/batch", "image_index", idx);
                    batch_items[idx].error = "unknown";
                  }
                }
            };
            // First worker is guaranteed; each subsequent one needs a permit.
            if (num_workers > 0) threads.emplace_back(worker_body);
            for (int w = 1; w < num_workers; ++w) {
              Permit &pm = permits[static_cast<size_t>(w - 1)];
              pm.held = extra_batch_permits.try_acquire();
              if (!pm.held) break;
              threads.emplace_back(worker_body);
            }
          } // jthreads auto-join here (permits release after)

          // Emit per-image results AND a sibling errors array so clients
          // can see which slots failed without having to compare lengths.
          std::string json_str;
          json_str.reserve(batch_items.size() * 1024);
          json_str += "{\"batch_results\":[";
          for (size_t i = 0; i < batch_items.size(); ++i) {
            if (i > 0) json_str += ',';
            json_str += turbo_ocr::emit_pipeline_result_json(batch_items[i].out, opts.want_blocks);
          }
          json_str += "],\"errors\":[";
          for (size_t i = 0; i < batch_items.size(); ++i) {
            if (i > 0) json_str += ',';
            const auto &e = batch_items[i].error;
            if (e.empty()) {
              json_str += "null";
            } else {
              json_str += '"';
              for (char c : e) {
                if (c == '"' || c == '\\') json_str += '\\';
                json_str += c;
              }
              json_str += '"';
            }
          }
          json_str += "]}";
          cb(turbo_ocr::server::json_response(std::move(json_str)));
         });
        });
      },
      {drogon::Post});

  // gRPC server. Cache-only readiness: Health() runs inline on a CQ poller
  // thread, where the blocking 250ms lease-stealing probe would compete with
  // real RPCs under saturation — exactly the flap the GPU path is hardened
  // against. Reads the value the HTTP probe above refreshes.
  auto readiness_cached = [probe]() -> bool {
    if (bootstrap::g_shutdown_requested.load(std::memory_order_acquire))
      return false;
    return probe->ok.load(std::memory_order_acquire);
  };
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      infer, cfg, &pdf_renderer, layout_available, readiness_cached,
      table_available, formula_available);
  // Published for the signal-handler drain thread. Not dangling: grpc_handle
  // owns the server on main()'s stack and is destroyed only after run() returns
  // (i.e. after the drain has driven app().quit()). See server_bootstrap.h.
  // cppcheck-suppress danglingLifetime
  bootstrap::g_grpc_server_for_drain = grpc_handle.server.get();

  // HTTP server (Drogon)
  const int port = cfg.http_port;

  // MAX_BODY_MB caps the largest accepted upload (default 100).
  // MAX_BODY_MEMORY_MB tunes the in-memory buffer threshold (default
  // 1024 MB / 1 GiB); bodies above it spill to /tmp. The cap is
  // clamped to MAX_BODY_MB below, so the default keeps every body in
  // RAM until the operator raises MAX_BODY_MB past 1 GiB. See main.cpp
  // for the full rationale — same knob on both CPU and GPU servers.
  const int max_body_mb = cfg.max_body_mb;
  int max_body_mem_mb = cfg.max_body_mem_mb;
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;

  TOCR_LOG_INFO("Starting CPU-Only OCR Server", "port", port, "grpc_port",
                cfg.grpc_port, "body_cap_mb_drogon", max_body_mb,
                "body_cap_mb_nginx", max_body_mb, "body_mem_mb", max_body_mem_mb);

  // Listener + body-size config + graceful-shutdown signal handlers + run().
  // The CPU server uses a fixed 4 IO threads. Blocks until app().quit().
  bootstrap::run_http_server(cfg, /*io_threads=*/4, work_pool);

  bootstrap::shutdown_grpc_after_run(grpc_handle.server.get());
  return 0;
} catch (const std::exception &e) {
  // Function-try-block on main(): any exception escaping startup (config, ORT
  // model load, pool acquire, listener bind) becomes a clean non-zero exit with
  // a logged reason instead of std::terminate + a useless core dump.
  TOCR_LOG_ERROR("Fatal error during startup", "error", std::string_view(e.what()));
  return 1;
} catch (...) {
  TOCR_LOG_ERROR("Fatal error during startup: unknown exception");
  return 1;
}
