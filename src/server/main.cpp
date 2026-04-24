#include <csignal>
#include <cstdlib>
#include <format>
#include <string>

#include "turbo_ocr/common/logger.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/image_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::decode::FastPngDecoder;
using turbo_ocr::decode::NvJpegDecoder;
using turbo_ocr::render::PdfRenderer;
using turbo_ocr::server::env_or;

namespace {
std::atomic<bool> g_shutdown_requested{false};

void shutdown_handler(int) {
  if (g_shutdown_requested.exchange(true)) return;
  drogon::app().quit();
}
} // namespace

int main() {
  auto rec_paths = turbo_ocr::server::resolve_rec_paths("REC_ONNX");
  if (auto lang = turbo_ocr::server::ocr_lang(); !lang.empty())
    TOCR_LOG_INFO("Language selected via OCR_LANG",
                  "lang",  std::string_view(lang),
                  "rec",   std::string_view(rec_paths.rec),
                  "dict",  std::string_view(rec_paths.dict));
  auto rec_dict = rec_paths.dict;

  // Auto-build TRT engines from ONNX (cached by TRT version + model hash)
  auto det_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("DET_ONNX", "models/det.onnx"), "det");
  auto rec_model = turbo_ocr::engine::ensure_trt_engine(rec_paths.rec, "rec");
  auto cls_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("CLS_ONNX", "models/cls.onnx"), "cls");
  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Optional PP-DocLayoutV3 stage. ON by default — users can disable with
  // DISABLE_LAYOUT=1 to save ~300-500 MB VRAM. Also accepts legacy
  // ENABLE_LAYOUT=0 to disable.
  std::string layout_model;
  bool layout_disabled = turbo_ocr::server::env_enabled("DISABLE_LAYOUT") ||
                          (std::getenv("ENABLE_LAYOUT") && !turbo_ocr::server::env_enabled("ENABLE_LAYOUT"));
  if (!layout_disabled) {
    if (auto *pre = std::getenv("LAYOUT_TRT"); pre && *pre) {
      layout_model = pre;
      TOCR_LOG_INFO("Layout detection enabled", "engine", std::string_view(layout_model));
    } else {
      layout_model = turbo_ocr::engine::ensure_trt_engine(
          env_or("LAYOUT_ONNX", "models/layout/layout.onnx"), "layout");
      if (layout_model.empty()) {
        TOCR_LOG_WARN("Layout model (layout.onnx) not found; layout stage will be disabled");
      } else {
        TOCR_LOG_INFO("Layout detection enabled");
      }
    }
  } else {
    TOCR_LOG_INFO("Layout detection disabled (set DISABLE_LAYOUT=0 to enable)");
  }

  // PDF extraction mode default
  turbo_ocr::pdf::PdfMode default_pdf_mode = turbo_ocr::pdf::PdfMode::Ocr;
  if (auto *m = std::getenv("ENABLE_PDF_MODE"); m && *m) {
    default_pdf_mode = turbo_ocr::pdf::parse_pdf_mode(m);
    TOCR_LOG_INFO("PDF extraction default mode configured", "mode", turbo_ocr::pdf::mode_name(default_pdf_mode));
  } else {
    TOCR_LOG_INFO("PDF extraction default mode: ocr (override per-request with /ocr/pdf?mode=<geometric|auto|auto_verified>)");
  }
  turbo_ocr::pdf::ensure_pdfium_initialized();

  // Pipeline pool
  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE")) {
    pool_size = std::max(1, std::atoi(env));
  } else {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      int vram_gb = static_cast<int>(total_mem >> 30);
      if (vram_gb >= 14) pool_size = 5;
      else if (vram_gb >= 12) pool_size = 3;
      else if (vram_gb >= 8)  pool_size = 2;
      else                     pool_size = 1;
      TOCR_LOG_INFO("Auto-detected pipeline pool size", "pool_size", pool_size, "vram_gb", vram_gb);
    }
  }

  auto dispatcher = turbo_ocr::pipeline::make_pipeline_dispatcher(
      pool_size, det_model, rec_model, rec_dict, cls_model, layout_model);

  // PDF renderer
  int pdf_daemons = 16, pdf_workers = 4;
  if (const char *env = std::getenv("PDF_DAEMONS"))
    pdf_daemons = std::max(1, std::atoi(env));
  if (const char *env = std::getenv("PDF_WORKERS"))
    pdf_workers = std::max(1, std::atoi(env));
  PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);

  // nvJPEG
  TOCR_LOG_INFO("Initializing nvJPEG decoders");
  thread_local NvJpegDecoder tl_nvjpeg;
  bool nvjpeg_available = tl_nvjpeg.available();
  if (nvjpeg_available)
    TOCR_LOG_INFO("nvJPEG GPU-accelerated JPEG decode enabled");
  else
    TOCR_LOG_WARN("nvJPEG not available, using OpenCV JPEG decode");

  // Image decoder: JPEG via nvJPEG, PNG via Wuffs
  turbo_ocr::server::ImageDecoder decode =
      [nvjpeg_available](const unsigned char *data, size_t len) -> cv::Mat {
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
      if (nvjpeg_available) {
        cv::Mat img = tl_nvjpeg.decode(data, len);
        if (!img.empty()) return img;
      }
      if (len > static_cast<size_t>(INT_MAX)) return {};
      return cv::imdecode(
          cv::Mat(1, static_cast<int>(len), CV_8UC1,
                  const_cast<unsigned char *>(data)),
          cv::IMREAD_COLOR);
    }
    if (FastPngDecoder::is_png(data, len))
      return FastPngDecoder::decode(data, len);
    return {};
  };

  // Inference function for shared routes (/ocr base64)
  const bool layout_available = !layout_model.empty();
  turbo_ocr::server::InferFunc infer =
      [&dispatcher](const cv::Mat &img, bool want_layout) -> turbo_ocr::server::InferResult {
    auto out = dispatcher->submit([&img, want_layout](auto &e) {
      return e.pipeline->run_with_layout(img, e.stream, want_layout);
    }).get();
    return turbo_ocr::server::InferResult{
        .results = std::move(out.results),
        .layout  = std::move(out.layout),
    };
  };

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  if (const char *env = std::getenv("HTTP_THREADS"))
    work_threads = std::max(1, std::atoi(env));
  turbo_ocr::server::WorkPool work_pool(work_threads);

  // --- Register all routes ---
  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route();
  turbo_ocr::routes::register_health_route([&dispatcher]() -> bool {
    // Verify GPU is responsive by submitting a no-op to the dispatcher.
    try {
      dispatcher->submit([](auto &) {}).get();
      return true;
    } catch (...) {
      return false;
    }
  });
  turbo_ocr::routes::register_ocr_base64_route(work_pool, infer, decode, layout_available);
  turbo_ocr::routes::register_image_routes(work_pool, *dispatcher, decode, nvjpeg_available, layout_available);
  turbo_ocr::routes::register_pdf_route(work_pool, *dispatcher, pdf_renderer, default_pdf_mode, layout_available);

  // gRPC
  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      *dispatcher, grpc_port, &pdf_renderer, default_pdf_mode, layout_available);

  // HTTP (Drogon) — behind nginx (port 8000), direct access on 8080
  int port = 8080;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));
  int io_threads = std::max(pool_size, 4);

  TOCR_LOG_INFO("HTTP server starting", "port", port, "io_threads", io_threads,
           "work_threads", work_threads, "pool_size", dispatcher->worker_count());

  // Graceful shutdown on SIGTERM (Docker stop) and SIGINT (Ctrl-C)
  std::signal(SIGTERM, shutdown_handler);
  std::signal(SIGINT,  shutdown_handler);

  drogon::app()
      .addListener("0.0.0.0", port)
      .setThreadNum(io_threads)
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(100 * 1024 * 1024)  // 100MB for large PDFs
      .setClientMaxMemoryBodySize(100 * 1024 * 1024)
      .run();

  TOCR_LOG_INFO("HTTP server stopped, shutting down gRPC");
  grpc_handle.server->Shutdown();
  TOCR_LOG_INFO("Shutdown complete");
  return 0;
}
