#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <string_view>

#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/decode/fast_png_decoder.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/pipeline/gpu_pipeline_pool.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/server/env_utils.h"
#include "ocr.grpc.pb.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::decode::FastPngDecoder;
using turbo_ocr::decode::NvJpegDecoder;
using turbo_ocr::server::env_or;

// nvJPEG GPU-accelerated JPEG decoder (one per thread)
static thread_local NvJpegDecoder tl_nvjpeg;
static bool g_nvjpeg_available = false;

// Decode image from raw bytes: JPEG (nvJPEG/OpenCV) and PNG (Wuffs) only.
static cv::Mat decode_image(std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  auto len = image_data.size();
  // JPEG: GPU-accelerated decode via nvJPEG, fallback to OpenCV
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
    if (g_nvjpeg_available) {
      cv::Mat img = tl_nvjpeg.decode(data, len);
      if (!img.empty()) return img;
    }
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                                const_cast<unsigned char *>(data)),
                        cv::IMREAD_COLOR);
  }
  // PNG: Wuffs fast decoder
  if (FastPngDecoder::is_png(data, len))
    return FastPngDecoder::decode(data, len);
  // Unsupported format
  return {};
}

// Response mode: json_bytes sends pre-serialized JSON in a single bytes field
// (matches HTTP path speed), structured sends traditional protobuf fields.
enum class ResponseMode { json_bytes, structured };
static ResponseMode g_response_mode = ResponseMode::json_bytes;

// Fill gRPC response -- fast path: single JSON bytes field (zero per-field allocs)
static void fill_response_json(ocr::OCRResponse *response,
                                const std::vector<OCRResultItem> &results) {
  response->set_num_detections(static_cast<int>(results.size()));
  response->set_json_response(results_to_json(results));
}

// Fill gRPC response -- structured protobuf with Reserve (reduces alloc overhead)
static void fill_response_structured(ocr::OCRResponse *response,
                                      const std::vector<OCRResultItem> &results) {
  response->set_num_detections(static_cast<int>(results.size()));
  response->mutable_results()->Reserve(static_cast<int>(results.size()));
  for (const auto &item : results) {
    auto *result = response->add_results();
    result->set_text(item.text);
    result->set_confidence(item.confidence);
    result->mutable_bounding_box()->Reserve(4);
    for (int k = 0; k < 4; ++k) {
      auto *bbox = result->add_bounding_box();
      bbox->mutable_x()->Reserve(1);
      bbox->mutable_y()->Reserve(1);
      bbox->add_x(static_cast<float>(item.box[k][0]));
      bbox->add_y(static_cast<float>(item.box[k][1]));
    }
  }
}

// Dispatch to active response mode
static void fill_response(ocr::OCRResponse *response,
                           const std::vector<OCRResultItem> &results) {
  if (g_response_mode == ResponseMode::json_bytes)
    fill_response_json(response, results);
  else
    fill_response_structured(response, results);
}

class OCRServiceImpl final : public ocr::OCRService::Service {
public:
  explicit OCRServiceImpl(turbo_ocr::pipeline::GpuPipelinePool &pool)
      : pool_(&pool) {}

  grpc::Status Recognize(grpc::ServerContext * /*context*/,
                         const ocr::OCRRequest *request,
                         ocr::OCRResponse *response) override {
    if (request->image().empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Empty image");

    cv::Mat img = decode_image(request->image());
    if (img.empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Decode failed");

    try {
      auto handle = pool_->acquire();
      auto results = handle->pipeline->run(img, handle->stream);
      fill_response(response, results);
      return grpc::Status::OK;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Pipeline pool exhausted");
    } catch (const std::exception &e) {
      std::cerr << std::format("[gRPC] Inference error: {}\n", e.what());
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    } catch (...) {
      std::cerr << "[gRPC] Inference error: unknown exception\n";
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    }
  }

  grpc::Status RecognizeBatch(grpc::ServerContext *, const ocr::OCRBatchRequest *,
                              ocr::OCRBatchResponse *) override {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                        "Use concurrent Recognize calls instead");
  }

private:
  turbo_ocr::pipeline::GpuPipelinePool *pool_ = nullptr;
};

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
    std::cout << "Angle classification disabled via DISABLE_ANGLE_CLS=1\n";
  }

  // Response serialization mode
  if (const char *mode_env = std::getenv("GRPC_RESPONSE_MODE")) {
    if (std::strcmp(mode_env, "structured") == 0) {
      g_response_mode = ResponseMode::structured;
      std::cout << "gRPC response mode: STRUCTURED (protobuf fields with Reserve)\n";
    } else {
      std::cout << "gRPC response mode: JSON_BYTES (pre-serialized, matches HTTP speed)\n";
    }
  } else {
    std::cout << "gRPC response mode: JSON_BYTES (default, set GRPC_RESPONSE_MODE=structured for protobuf)\n";
  }

  constexpr int kMaxGrpcMessageSize = 100 * 1024 * 1024;

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
      std::cout << std::format("Auto-detected pool_size={} for {}GB VRAM\n", pool_size, vram_gb);
    }
  }

  auto pool = turbo_ocr::pipeline::make_gpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model);

  // Probe nvJPEG availability on the main thread
  g_nvjpeg_available = tl_nvjpeg.available();
  std::cout << std::format("nvJPEG GPU JPEG decode: {}\n",
                           g_nvjpeg_available ? "available" : "unavailable (OpenCV fallback)");

  auto service = std::make_unique<OCRServiceImpl>(*pool);
  std::cout << std::format("Mode: POOL (pool_size={})\n", pool_size);

  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));
  auto server_address = std::format("0.0.0.0:{}", grpc_port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxReceiveMessageSize(kMaxGrpcMessageSize);

  int cqs = 10;
  if (const char *env = std::getenv("GRPC_CQS"))
    cqs = std::max(1, std::atoi(env));
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, cqs * 2);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.AddChannelArgument(GRPC_ARG_MINIMAL_STACK, 1);

  auto server = builder.BuildAndStart();
  std::cout << std::format("gRPC OCR server listening on {}\n", server_address);
  server->Wait();
  return 0;
}
