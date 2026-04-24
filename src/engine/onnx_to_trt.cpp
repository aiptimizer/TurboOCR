#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/common/cuda_check.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace turbo_ocr::engine {

static class BuildLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cerr << "[TRT Build] " << msg << '\n';
  }
} s_logger;

std::string get_engine_cache_dir() {
  // User override
  if (auto *env = std::getenv("TRT_ENGINE_CACHE"))
    return env;

  // Try ~/.cache/turbo-ocr/
  if (auto *home = std::getenv("HOME")) {
    auto dir = std::string(home) + "/.cache/turbo-ocr";
    fs::create_directories(dir);
    return dir;
  }

  // Fallback to /tmp
  auto dir = std::string("/tmp/turbo-ocr-engines");
  fs::create_directories(dir);
  return dir;
}

std::string get_cached_engine_path(const std::string &onnx_path,
                                   const std::string &type) {
  auto cache_dir = get_engine_cache_dir();

  // Build a cache key from: onnx file size + mtime + TRT version
  auto onnx_size = fs::file_size(onnx_path);
  auto onnx_mtime = fs::last_write_time(onnx_path).time_since_epoch().count();

  int trt_major = 0, trt_minor = 0, trt_patch = 0;
#ifdef NV_TENSORRT_MAJOR
  trt_major = NV_TENSORRT_MAJOR;
  trt_minor = NV_TENSORRT_MINOR;
  trt_patch = NV_TENSORRT_PATCH;
#endif

  // GPU compute capability (engines are GPU-architecture specific)
  int gpu_major = 0, gpu_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_minor, cudaDevAttrComputeCapabilityMinor, 0));

  // Cache key includes: onnx identity, TRT version, GPU arch, and profile version.
  // Bump kProfileVersion when optimization profiles change for det/rec/cls.
  // Adding a NEW model type (e.g. "layout") does NOT require a bump because
  // the cache key includes `type` — new types live in their own hash space.
  static constexpr int kProfileVersion = 20260424;

  auto key = "v" + std::to_string(kProfileVersion) + ":" + type + ":" +
      onnx_path + ":" + std::to_string(onnx_size) + ":" +
      std::to_string(onnx_mtime) + ":" + std::to_string(trt_major) + "." +
      std::to_string(trt_minor) + "." + std::to_string(trt_patch) + ":sm" +
      std::to_string(gpu_major) + "." + std::to_string(gpu_minor);
  auto hash = std::hash<std::string>{}(key);

  return cache_dir + "/" + type + "_" + std::to_string(hash) + ".trt";
}

static bool build_engine(const std::string &onnx_path,
                          const std::string &trt_path,
                          const std::string &type) {
  std::cout << "Building TRT engine: " << onnx_path << " -> " << trt_path << '\n';

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(s_logger));
  if (!builder) return false;

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) return false;

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, s_logger));
  if (!parser || !parser->parseFromFile(onnx_path.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    return false;

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  // Layout needs more workspace than det/rec/cls because PP-DocLayoutV3 is a
  // DETR-family model with large attention matmuls; 1 GB is fine for the
  // others.
  size_t workspace_bytes = (type == "layout") ? (4ULL << 30) : (1ULL << 30);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  config->setBuilderOptimizationLevel(5);

  auto profile = builder->createOptimizationProfile();
  auto input = network->getInput(0);

  if (type == "det") {
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 32, 32});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 640, 640});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{1, 3, 960, 960});
  } else if (type == "rec") {
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 48, 48});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 48, 320});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{32, 3, 48, 4000});
  } else if (type == "layout") {
    // PP-DocLayoutV3 has 3 inputs: image [B,3,800,800], im_shape [B,2],
    // scale_factor [B,2]. paddle2onnx does not guarantee input ordering, so
    // dispatch by name. Batch profile: min=1, opt=4, max=8 (measured sweet
    // spot on RTX 5090: 1.18 ms / image at B=4).
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *in = network->getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{4, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{8, 3, 800, 800});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{4, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{8, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for layout model: " << name << '\n';
        return false;
      }
    }
  } else {
    // cls: PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25), input
    // 80x160. Must match kClsImageH/kClsImageW in classification/paddle_cls.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{128, 3, 80, 160});
  }
  config->addOptimizationProfile(profile);

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) return false;

  fs::create_directories(fs::path(trt_path).parent_path());

  // Write to a temp file first, then atomic rename (prevents corruption from
  // concurrent builds in multi-replica Docker deployments)
  auto tmp_path = trt_path + ".tmp." + std::to_string(getpid());
  std::ofstream file(tmp_path, std::ios::binary);
  if (!file) return false;
  file.write(static_cast<const char *>(plan->data()), plan->size());
  file.close();
  fs::rename(tmp_path, trt_path);

  std::cout << "Built: " << trt_path << " ("
            << static_cast<double>(plan->size()) / (1024 * 1024) << " MB)\n";
  return true;
}

std::string ensure_trt_engine(const std::string &onnx_path,
                               const std::string &type) {
  if (!fs::exists(onnx_path)) {
    std::cerr << "[TRT] ONNX not found: " << onnx_path << '\n';
    return {};
  }

  auto trt_path = get_cached_engine_path(onnx_path, type);

  if (fs::exists(trt_path)) {
    std::cout << "Using cached engine: " << trt_path << '\n';
    return trt_path;
  }

  if (!build_engine(onnx_path, trt_path, type)) {
    std::cerr << "[TRT] Failed to build engine from: " << onnx_path << '\n';
    return {};
  }

  return trt_path;
}

} // namespace turbo_ocr::engine
