#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/pipeline/gpu_pipeline_pool.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/http_routes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::decode::FastPngDecoder;
using turbo_ocr::decode::NvJpegDecoder;
using turbo_ocr::render::PdfRenderer;
using turbo_ocr::server::env_or;

int main() {
  auto rec_dict = env_or("REC_DICT", "models/keys.txt");

  // Auto-build TRT engines from ONNX (cached by TRT version + model hash)
  auto det_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("DET_ONNX", "models/det.onnx"), "det");
  auto rec_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("REC_ONNX", "models/rec.onnx"), "rec");
  auto cls_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("CLS_ONNX", "models/cls.onnx"), "cls");
  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    std::cout << "Angle classification disabled via DISABLE_ANGLE_CLS=1" << '\n';
  }

  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE")) {
    pool_size = std::max(1, std::atoi(env));
  } else {
    // Auto-detect pool size from GPU VRAM (~1.4GB per pipeline + 2GB overhead)
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      int vram_gb = static_cast<int>(total_mem >> 30);
      if (vram_gb >= 14) pool_size = 5;
      else if (vram_gb >= 12) pool_size = 3;
      else if (vram_gb >= 8)  pool_size = 2;
      else                     pool_size = 1;
      std::cout << std::format("Auto-detected pool_size={} for {}GB VRAM\n", pool_size, vram_gb);
    } else {
      pool_size = 4; // fallback
    }
  }

  auto pool = turbo_ocr::pipeline::make_gpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model);

  // PDF renderer: persistent fastpdf2png v2.0 daemons (raw PPM, /dev/shm)
  int pdf_daemons = 16;
  int pdf_workers = 4;
  if (const char *env = std::getenv("PDF_DAEMONS"))
    pdf_daemons = std::max(1, std::atoi(env));
  if (const char *env = std::getenv("PDF_WORKERS"))
    pdf_workers = std::max(1, std::atoi(env));
  PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  std::cout << std::format("PDF renderer: {} daemons x {} workers (raw PPM, /dev/shm)\n", pdf_daemons, pdf_workers);

  // Thread-local nvJPEG decoders for concurrent JPEG decode
  std::cout << "Initializing nvJPEG decoders..." << '\n';
  thread_local NvJpegDecoder tl_nvjpeg;
  bool nvjpeg_available = tl_nvjpeg.available();
  if (nvjpeg_available)
    std::cout << "nvJPEG GPU-accelerated JPEG decode enabled" << '\n';
  else
    std::cout << "nvJPEG not available, using OpenCV JPEG decode" << '\n';

  // GPU image decoder: JPEG via nvJPEG (GPU), PNG via Wuffs, else unsupported
  turbo_ocr::server::ImageDecoder decode =
      [nvjpeg_available](const unsigned char *data, size_t len) -> cv::Mat {
    // JPEG: magic bytes 0xFF 0xD8
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
      if (nvjpeg_available) {
        cv::Mat img = tl_nvjpeg.decode(data, len);
        if (!img.empty()) return img;
      }
      // Fallback to OpenCV for JPEG
      if (len > static_cast<size_t>(INT_MAX)) return {};
      return cv::imdecode(
          cv::Mat(1, static_cast<int>(len), CV_8UC1,
                  const_cast<unsigned char *>(data)),
          cv::IMREAD_COLOR);
    }
    // PNG: Wuffs fast decoder
    if (FastPngDecoder::is_png(data, len)) {
      return FastPngDecoder::decode(data, len);
    }
    // Unsupported format
    return {};
  };

  // Inference function for shared routes — acquires GPU pipeline + stream
  turbo_ocr::server::InferFunc infer = [&pool](const cv::Mat &img) {
    auto handle = pool->acquire();
    return handle->pipeline->run(img, handle->stream);
  };

  crow::SimpleApp app;

  // Register shared routes: /health, /ocr (but NOT /ocr/raw — we override it below)
  turbo_ocr::server::register_health_route(app);
  turbo_ocr::server::register_ocr_base64_route(app, infer, decode);

  // --- /ocr/raw endpoint: GPU-direct JPEG decode, Wuffs PNG ---
  // For JPEG: acquire pipeline first, decode directly to GPU buffer, skip CPU->GPU upload.
  // For PNG/other: decode to CPU cv::Mat, then run through normal pipeline path.
  CROW_ROUTE(app, "/ocr/raw")
      .methods(crow::HTTPMethod::Post)(
          [&pool, &decode, nvjpeg_available](const crow::request &req) {

    if (req.body.empty())
      return crow::response(400, "Empty body");

    const auto *data = reinterpret_cast<const unsigned char *>(req.body.data());
    size_t len = req.body.size();

    try {
      // JPEG with nvJPEG available: GPU-direct decode path
      if (nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
        // Get image dimensions without decoding
        auto [w, h] = tl_nvjpeg.get_dimensions(data, len);
        if (w > 0 && h > 0) {
          // Acquire pipeline FIRST to get the stream and GPU buffer
          auto handle = pool->acquire();
          auto [d_buf, pitch] = handle->pipeline->ensure_gpu_buf(h, w);

          // Decode JPEG directly into the pipeline's GPU buffer
          if (tl_nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, handle->stream)) {
            turbo_ocr::GpuImage gpu_img{.data = d_buf, .step = pitch, .rows = h, .cols = w};
            auto results = handle->pipeline->run(gpu_img, handle->stream);

            auto json_str = turbo_ocr::results_to_json(results);
            auto resp = crow::response(200, std::move(json_str));
            resp.set_header("Content-Type", "application/json");
            return resp;
          }
          // decode_to_gpu failed — fall through to CPU decode with this handle
          cv::Mat img = tl_nvjpeg.decode(data, len);
          if (img.empty()) {
            if (len > static_cast<size_t>(INT_MAX)) return crow::response(400, "Failed to decode image");
            img = cv::imdecode(
                cv::Mat(1, static_cast<int>(len), CV_8UC1,
                        const_cast<unsigned char *>(data)),
                cv::IMREAD_COLOR);
          }
          if (img.empty()) return crow::response(400, "Failed to decode image");
          auto results = handle->pipeline->run(img, handle->stream);
          auto json_str = turbo_ocr::results_to_json(results);
          auto resp = crow::response(200, std::move(json_str));
          resp.set_header("Content-Type", "application/json");
          return resp;
        }
      }

      // Non-JPEG (PNG, etc.) or nvJPEG not available: standard CPU decode path
      cv::Mat img = decode(data, len);
      if (img.empty())
        return crow::response(400, "Failed to decode image");

      auto handle = pool->acquire();
      auto results = handle->pipeline->run(img, handle->stream);
      auto json_str = turbo_ocr::results_to_json(results);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/raw] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/raw] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });

  // --- /ocr/batch endpoint (GPU: nvJPEG batch decode + parallel pipeline) ---
  CROW_ROUTE(app, "/ocr/batch")
      .methods(crow::HTTPMethod::Post)(
          [&pool, &decode,
           nvjpeg_available](const crow::request &req) {
    auto x = crow::json::load(req.body);
    if (!x)
      return crow::response(400, "Invalid JSON");

    if (!x.has("images"))
      return crow::response(400, "Missing images array");

    auto &images_json = x["images"];
    size_t n = images_json.size();
    if (n == 0)
      return crow::response(400, "Empty images array");

    // Step 1: Base64-decode all images upfront (keep raw bytes alive)
    std::vector<std::string> raw_bytes(n);
    for (size_t i = 0; i < n; ++i) {
      raw_bytes[i] = base64_decode(images_json[i].s());
    }

    // Step 2: Separate JPEGs for batch decode vs non-JPEGs for individual decode
    std::vector<cv::Mat> imgs(n);
    std::vector<size_t> jpeg_indices;
    std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

    if (nvjpeg_available) {
      for (size_t i = 0; i < n; ++i) {
        auto &raw = raw_bytes[i];
        if (raw.size() >= 2 &&
            static_cast<unsigned char>(raw[0]) == 0xFF &&
            static_cast<unsigned char>(raw[1]) == 0xD8) {
          jpeg_indices.push_back(i);
          jpeg_buffers.emplace_back(
              reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
        }
      }
    }

    // Step 3: Batch-decode JPEGs if we have 2+ (otherwise single decode is fine)
    if (jpeg_buffers.size() >= 2) {
      auto batch_mats = tl_nvjpeg.batch_decode(jpeg_buffers);
      for (size_t j = 0; j < jpeg_indices.size(); ++j) {
        imgs[jpeg_indices[j]] = std::move(batch_mats[j]);
      }
    }

    // Step 4: Decode remaining images (non-JPEGs + single JPEGs + failed batch entries)
    for (size_t i = 0; i < n; ++i) {
      if (!imgs[i].empty())
        continue;  // Already batch-decoded
      auto &raw = raw_bytes[i];
      if (raw.empty())
        continue;
      imgs[i] = decode(
          reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    }

    // Separate valid images from failed decodes, preserving original indices
    std::vector<cv::Mat> valid_imgs;
    std::vector<size_t> valid_indices;  // maps valid_imgs[j] -> original index i
    valid_imgs.reserve(n);
    valid_indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (!imgs[i].empty()) {
        valid_imgs.push_back(std::move(imgs[i]));
        valid_indices.push_back(i);
      }
    }

    if (valid_imgs.empty())
      return crow::response(400, "No valid images");

    // Cross-image batched OCR: acquire ONE pipeline, call run_batch() which
    // merges recognition boxes from all images into larger TRT batches.
    // kMaxBatchImages=8, so chunk if needed.
    constexpr int kMaxBatch = 8;
    std::vector<std::vector<OCRResultItem>> all_results(n);  // sized to original n

    try {
      auto handle = pool->acquire();
      for (size_t offset = 0; offset < valid_imgs.size(); offset += kMaxBatch) {
        size_t end = std::min(offset + kMaxBatch, valid_imgs.size());
        std::vector<cv::Mat> chunk(
            std::make_move_iterator(valid_imgs.begin() + offset),
            std::make_move_iterator(valid_imgs.begin() + end));

        auto chunk_results = handle->pipeline->run_batch(chunk, handle->stream);

        for (size_t j = 0; j < chunk_results.size(); ++j) {
          all_results[valid_indices[offset + j]] = std::move(chunk_results[j]);
        }
      }
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[Batch] run_batch error: {}", e.what()) << '\n';
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[Batch] run_batch error: unknown exception" << '\n';
      return crow::response(500, "Inference error");
    }

    std::string json_str;
    json_str.reserve(n * 1024);
    json_str += "{\"batch_results\":[";
    for (size_t i = 0; i < n; ++i) {
      if (i > 0) json_str += ',';
      json_str += results_to_json(all_results[i]);
    }
    json_str += "]}";
    auto resp = crow::response(200, std::move(json_str));
    resp.set_header("Content-Type", "application/json");
    return resp;
  });

  // --- /ocr/pixels endpoint: raw BGR pixel data, zero decode overhead ---
  CROW_ROUTE(app, "/ocr/pixels")
      .methods(crow::HTTPMethod::Post)(
          [&pool](const crow::request &req) {
    auto w_str = req.get_header_value("X-Width");
    auto h_str = req.get_header_value("X-Height");
    auto c_str = req.get_header_value("X-Channels");

    if (w_str.empty() || h_str.empty())
      return crow::response(400, "Missing X-Width or X-Height headers");

    int width, height, channels;
    try {
      width = std::stoi(w_str);
      height = std::stoi(h_str);
      channels = c_str.empty() ? 3 : std::stoi(c_str);
    } catch (const std::exception &) {
      return crow::response(400, "Invalid X-Width, X-Height, or X-Channels header value");
    }

    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3))
      return crow::response(400, "Invalid dimensions or channels");

    // Prevent excessive memory allocation from crafted dimensions
    constexpr int kMaxPixelDim = 16384;
    if (width > kMaxPixelDim || height > kMaxPixelDim)
      return crow::response(400, std::format("Dimensions {}x{} exceed maximum of {}x{}", width, height, kMaxPixelDim, kMaxPixelDim));

    size_t expected = static_cast<size_t>(width) * height * channels;
    if (req.body.size() != expected)
      return crow::response(400,
          std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                      expected, width, height, channels, req.body.size()));

    // Zero-copy wrap: req.body lives for entire handler, and pipeline uploads to GPU
    // synchronously in run(), so no clone needed.
    cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                const_cast<char *>(req.body.data()));

    try {
      auto handle = pool->acquire();
      auto results = handle->pipeline->run(img, handle->stream);

      auto json_str = results_to_json(results);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/pixels] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/pixels] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });

  // --- /ocr/pdf endpoint: streamed render + parallel OCR ---
  // Uses inotify to overlap PDF rendering with OCR: as each page finishes
  // rendering to PPM, OCR starts immediately on that page while later pages
  // are still being rendered. This eliminates the sequential bottleneck.
  CROW_ROUTE(app, "/ocr/pdf")
      .methods(crow::HTTPMethod::Post)(
          [&pool, pool_size, &pdf_renderer](const crow::request &req) {
    if (req.body.empty())
      return crow::response(400, "Empty body");

    // Optional DPI from query param: /ocr/pdf?dpi=100
    int dpi = 100;
    if (req.url_params.get("dpi"))
      dpi = std::atoi(req.url_params.get("dpi"));
    if (dpi < 50 || dpi > 600)
      return crow::response(400, "DPI must be between 50 and 600");

    const auto *pdf_data = reinterpret_cast<const uint8_t *>(req.body.data());
    const auto pdf_len = req.body.size();

    // Shared state for streamed render + OCR pipeline
    std::mutex results_mutex;
    std::vector<std::vector<OCRResultItem>> page_results;
    std::atomic<bool> pool_exhausted{false};

    // OCR work queue: pages ready for inference
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::vector<std::pair<int, cv::Mat>> ready_pages; // (page_idx, image)
    std::atomic<bool> render_complete{false};
    std::atomic<int> pages_queued{0};

    // Launch OCR worker threads that consume pages as they become available
    int num_workers = std::min(pool_size, 16); // cap at pool size
    std::vector<std::jthread> ocr_threads;
    ocr_threads.reserve(num_workers);
    for (int w = 0; w < num_workers; ++w) {
      ocr_threads.emplace_back([&]() {
        try {
          auto handle = pool->acquire();
          while (true) {
            // Wait for a page to be available
            std::pair<int, cv::Mat> work;
            {
              std::unique_lock<std::mutex> lock(queue_mutex);
              queue_cv.wait(lock, [&]() {
                return !ready_pages.empty() || render_complete.load(std::memory_order_acquire);
              });
              if (ready_pages.empty()) {
                if (render_complete.load(std::memory_order_acquire)) return;
                continue;
              }
              work = std::move(ready_pages.back());
              ready_pages.pop_back();
            }

            if (work.second.empty()) continue;
            auto result = handle->pipeline->run(work.second, handle->stream);

            std::lock_guard<std::mutex> rlock(results_mutex);
            if (work.first < static_cast<int>(page_results.size()))
              page_results[work.first] = std::move(result);
          }
        } catch (const turbo_ocr::PoolExhaustedError &) {
          pool_exhausted.store(true, std::memory_order_relaxed);
          std::cerr << "[PDF] OCR worker: pool exhausted\n";
        } catch (const std::exception &e) {
          std::cerr << std::format("[PDF] OCR worker error: {}", e.what()) << '\n';
        } catch (...) {
          std::cerr << "[PDF] OCR worker error: unknown exception" << '\n';
        }
      });
    }

    // Streamed render: callback fires for each page as soon as its PPM is ready
    int num_pages = 0;
    try {
      num_pages = pdf_renderer.render_streamed(pdf_data, pdf_len, dpi,
          [&](int page_idx, cv::Mat img) {
            // Ensure results vector is large enough (first callback tells us pages exist)
            {
              std::lock_guard<std::mutex> rlock(results_mutex);
              if (page_idx >= static_cast<int>(page_results.size()))
                page_results.resize(page_idx + 1);
            }
            // Enqueue for OCR
            {
              std::lock_guard<std::mutex> lock(queue_mutex);
              ready_pages.emplace_back(page_idx, std::move(img));
              pages_queued.fetch_add(1, std::memory_order_relaxed);
            }
            queue_cv.notify_one();
          });
    } catch (const std::exception &e) {
      // Signal workers to stop, then clean up
      render_complete.store(true, std::memory_order_release);
      queue_cv.notify_all();
      ocr_threads.clear();
      std::cerr << std::format("[/ocr/pdf] PDF render failed: {}\n", e.what());
      return crow::response(400, "PDF render failed");
    }

    // Ensure results vector is properly sized
    {
      std::lock_guard<std::mutex> rlock(results_mutex);
      page_results.resize(num_pages);
    }

    // Signal render complete and wake all workers
    render_complete.store(true, std::memory_order_release);
    queue_cv.notify_all();

    // Wait for all OCR workers to finish
    ocr_threads.clear();

    if (num_pages == 0)
      return crow::response(400, "PDF contains no pages");

    if (pool_exhausted.load(std::memory_order_relaxed))
      return crow::response(503, "Service overloaded, try again later");

    // Build JSON: {"pages": [{"page": 1, "results": [...]}, ...]}
    size_t n_pages = static_cast<size_t>(num_pages);
    std::string json_str;
    json_str.reserve(n_pages * 1024);
    json_str += "{\"pages\":[";
    for (size_t i = 0; i < n_pages; ++i) {
      if (i > 0) json_str += ',';
      json_str += "{\"page\":";
      json_str += std::to_string(i + 1);
      json_str += ",";
      auto page_json = results_to_json(page_results[i]);
      // Append the JSON without leading '{' and trailing '}' -- avoids substr allocation
      json_str.append(page_json.data() + 1, page_json.size() - 2);
      json_str += '}';
    }
    json_str += "]}";

    auto resp = crow::response(200, std::move(json_str));
    resp.set_header("Content-Type", "application/json");
    return resp;
  });

  // Start gRPC server on background thread (shares the same pipeline pool)
  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));

  auto grpc_handle = turbo_ocr::server::start_grpc_server(*pool, grpc_port);

  int http_threads = std::max(pool_size * 4, 8);
  if (const char *env = std::getenv("HTTP_THREADS"))
    http_threads = std::max(1, std::atoi(env));

  int port = 8000;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));

  std::cout << std::format("HTTP server on port {} ({} threads, pool={})\n", port, http_threads, pool_size);
  app.port(port).timeout(120).concurrency(http_threads).run();

  // Shutdown gRPC when HTTP server exits
  grpc_handle.server->Shutdown();

  return 0;
}
