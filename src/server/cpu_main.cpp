#include <atomic>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/logger.h"

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/cpu_pipeline_pool.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::server::env_or;

int main() {
  LOG_INFO("PaddleOCR CPU-Only Mode (ONNX Runtime)");

  auto det_model = env_or("DET_MODEL", "models/det.onnx");
  auto rec_model = env_or("REC_MODEL", "models/rec.onnx");
  auto rec_dict = env_or("REC_DICT", "models/keys.txt");
  auto cls_model = env_or("CLS_MODEL", "models/cls.onnx");
  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Layout model (CPU via ONNX Runtime) — on by default
  std::string layout_model = env_or("LAYOUT_ONNX", "models/layout/layout.onnx");
  bool layout_disabled = turbo_ocr::server::env_enabled("DISABLE_LAYOUT");
  bool layout_available = false;

  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE"))
    pool_size = std::max(1, std::atoi(env));

  LOG_INFO("CPU pipeline pool size", "pool_size", pool_size);
  auto pool = turbo_ocr::pipeline::make_cpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model);

  // Load layout model into each pipeline if enabled
  if (!layout_disabled && !layout_model.empty()) {
    bool all_ok = true;
    for (size_t i = 0; i < static_cast<size_t>(pool_size); ++i) {
      auto handle = pool->acquire();
      if (!handle->load_layout_model(layout_model)) {
        LOG_WARN("Layout model not found; layout disabled");
        all_ok = false;
        break;
      }
    }
    if (all_ok) {
      layout_available = true;
      LOG_INFO("Layout detection enabled (CPU/ONNX Runtime)");
    }
  } else if (layout_disabled) {
    LOG_INFO("Layout detection disabled");
  }

  turbo_ocr::server::InferFunc infer =
      [&pool](const cv::Mat &img, bool want_layout) -> turbo_ocr::server::InferResult {
    auto handle = pool->acquire();
    auto out = handle->run_with_layout(img, want_layout);
    return turbo_ocr::server::InferResult{
        .results = std::move(out.results),
        .layout  = std::move(out.layout),
    };
  };

  turbo_ocr::server::ImageDecoder decode = turbo_ocr::server::cpu_decode_image;

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  turbo_ocr::server::WorkPool work_pool(work_threads);

  turbo_ocr::routes::register_common_routes(work_pool, infer, decode, layout_available);

  // --- /ocr/pixels endpoint (raw BGR pixel data, zero decode overhead) ---
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&work_pool, &infer](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        bool want_layout = false;
        if (auto err = turbo_ocr::server::parse_layout_query(
                req, /*layout_available=*/false, &want_layout);
            !err.empty()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
          return;
        }

        auto w_str = req->getHeader("X-Width");
        auto h_str = req->getHeader("X-Height");
        auto c_str = req->getHeader("X-Channels");

        if (w_str.empty() || h_str.empty()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
                                    "MISSING_HEADER", "Missing X-Width or X-Height headers"));
          return;
        }

        int width, height, channels;
        try {
          width = std::stoi(w_str);
          height = std::stoi(h_str);
          channels = c_str.empty() ? 3 : std::stoi(c_str);
        } catch (const std::exception &) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "INVALID_HEADER", "Invalid X-Width, X-Height, or X-Channels header value"));
          return;
        }

        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "INVALID_DIMENSIONS", "Invalid dimensions or channels"));
          return;
        }

        constexpr int kMaxPixelDim = 16384;
        if (width > kMaxPixelDim || height > kMaxPixelDim) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "DIMENSIONS_TOO_LARGE", std::format("Dimensions {}x{} exceed maximum of {}x{}",
                          width, height, kMaxPixelDim, kMaxPixelDim)));
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
            [req, &infer, width, height, channels, want_layout](turbo_ocr::server::DrogonCallback &cb) {
          turbo_ocr::server::run_with_error_handling(cb, "/ocr/pixels", [&] {
            cv::Mat img(height, width,
                        channels == 3 ? CV_8UC3 : CV_8UC1,
                        const_cast<char *>(req->body().data()));
            auto inf = infer(img, want_layout);
            cb(turbo_ocr::server::json_response(
                turbo_ocr::results_to_json(inf.results, inf.layout)));
          });
        });
      },
      {drogon::Post});

  // --- /ocr/pdf endpoint (CPU: sequential page OCR) ---
  int pdf_daemons = 4, pdf_workers = 2;
  if (const char *env = std::getenv("PDF_DAEMONS"))
    pdf_daemons = std::max(1, std::atoi(env));
  if (const char *env = std::getenv("PDF_WORKERS"))
    pdf_workers = std::max(1, std::atoi(env));
  turbo_ocr::render::PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);
  turbo_ocr::pdf::ensure_pdfium_initialized();

  turbo_ocr::pdf::PdfMode default_pdf_mode = turbo_ocr::pdf::PdfMode::Ocr;
  if (auto *m = std::getenv("ENABLE_PDF_MODE"); m && *m)
    default_pdf_mode = turbo_ocr::pdf::parse_pdf_mode(m);

  turbo_ocr::routes::register_pdf_route(work_pool, infer, pdf_renderer, default_pdf_mode, layout_available);

  // --- /ocr/batch endpoint (CPU version) ---
  drogon::app().registerHandler(
      "/ocr/batch",
      [&work_pool, &pool, pool_size, &decode](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

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

        // Pre-decode base64
        auto raw_bytes = std::make_shared<std::vector<std::string>>(n);
        for (size_t i = 0; i < n; ++i)
          (*raw_bytes)[i] = base64_decode(images_json[static_cast<int>(i)].asString());

        turbo_ocr::server::submit_work(work_pool, std::move(callback),
            [raw_bytes, n, &pool, pool_size, &decode](turbo_ocr::server::DrogonCallback &cb) {
          std::vector<cv::Mat> imgs;
          imgs.reserve(n);
          for (size_t i = 0; i < n; ++i) {
            auto &raw = (*raw_bytes)[i];
            if (raw.empty()) continue;
            cv::Mat img = decode(
                reinterpret_cast<const unsigned char *>(raw.data()),
                raw.size());
            if (!img.empty())
              imgs.push_back(img);
          }

          if (imgs.empty()) {
            cb(turbo_ocr::server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "No valid images"));
            return;
          }

          std::vector<std::vector<OCRResultItem>> batch_results(imgs.size());
          std::atomic<size_t> next_idx{0};

          int num_workers = std::min(static_cast<int>(imgs.size()), pool_size);
          {
            std::vector<std::jthread> threads;
            threads.reserve(num_workers);
            for (int w = 0; w < num_workers; ++w) {
              threads.emplace_back([&]() {
                try {
                  auto handle = pool->acquire();
                  while (true) {
                    size_t idx = next_idx.fetch_add(1);
                    if (idx >= imgs.size()) break;
                    batch_results[idx] = handle->run(imgs[idx]);
                  }
                } catch (const turbo_ocr::PoolExhaustedError &) {
                  LOG_ERROR("Batch worker error: pool exhausted", "route", "/ocr/batch");
                } catch (const std::exception &e) {
                  LOG_ERROR("Batch worker error", "route", "/ocr/batch", "error", std::string_view(e.what()));
                } catch (...) {
                  LOG_ERROR("Batch worker error: unknown exception", "route", "/ocr/batch");
                }
              });
            }
          } // jthreads auto-join here

          std::string json_str;
          json_str.reserve(batch_results.size() * 1024);
          json_str += "{\"batch_results\":[";
          for (size_t i = 0; i < batch_results.size(); ++i) {
            if (i > 0) json_str += ',';
            json_str += results_to_json(batch_results[i]);
          }
          json_str += "]}";
          cb(turbo_ocr::server::json_response(std::move(json_str)));
        });
      },
      {drogon::Post});

  // gRPC server
  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      infer, grpc_port, &pdf_renderer, default_pdf_mode, layout_available);

  // HTTP server (Drogon)
  int port = 8080;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));

  LOG_INFO("Starting CPU-Only OCR Server", "port", port, "grpc_port", grpc_port);

  drogon::app()
      .addListener("0.0.0.0", port)
      .setThreadNum(4)
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(100 * 1024 * 1024)
      .setClientMaxMemoryBodySize(100 * 1024 * 1024)
      .run();

  grpc_handle.server->Shutdown();
  return 0;
}
