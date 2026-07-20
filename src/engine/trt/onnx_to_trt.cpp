// ONNX -> TensorRT engine build + cached-engine entry point. Cache-key and
// per-type optimization profiles live in trt_engine_cache.cpp /
// trt_profiles.cpp.

#include "turbo_ocr/engine/trt/onnx_to_trt.h"
#include "engine_internal.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace turbo_ocr::engine {

static class BuildLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // Concurrent TRT builds (e.g. det/rec/cls/layout in parallel during
      // warmup) can interleave on std::cerr. Build phase only — no impact on
      // request hot path.
      static std::mutex log_mu;
      std::lock_guard<std::mutex> lock(log_mu);
      std::cerr << "[TRT Build] " << msg << '\n';
    }
  }
} s_logger;

// A QDQ-bearing ONNX (modelopt INT8/FP8 output) carries explicit Quantize/
// Dequantize nodes whose zero-point dtypes pin the compute precision. TRT 10
// honors light QDQ under a weakly-typed FP16 network, but heavier modelopt
// exports (INT32 zero-points on weight reshapes, mixed FP8/INT8) hit
// "DequantizeLayer can only run in INT8/FP8/FP4/INT4" under weak typing — the
// modelopt-recommended fix is a STRONGLY-TYPED network, where every tensor's
// type comes from the graph and the autotuner only picks tactics. Detected by
// the quantized-variant filename convention used across the repo's scripts.
[[nodiscard]] static bool is_quantized_onnx(const std::string &p) {
  return p.find(".int8.") != std::string::npos ||
         p.find("_int8") != std::string::npos ||
         p.find("_fp8") != std::string::npos ||
         p.find(".fp8.") != std::string::npos;
}

static bool build_engine(const std::string &onnx_path,
                          const std::string &trt_path,
                          const std::string &type) {
  const bool quantized = is_quantized_onnx(onnx_path);
  // Quantized (QDQ-embedded) ONNX. det/rec exports carry LIGHT per-tensor/
  // per-channel Q/DQ on convs and build correctly under a WEAKLY-typed FP16
  // network: TRT honors the embedded Q/DQ (running those ops in INT8) and keeps
  // everything else in FP16. Heavier exports (mixed FP8/INT8, INT32 zero-points)
  // need a STRONGLY-typed network where every tensor's type comes from the graph
  // — reserved for non-det/rec models. Engaged only when the operator explicitly
  // points DET_ONNX/REC_ONNX at a quantized-named file, so it's opt-in and the
  // default FP16 models are unaffected.
  const bool strongly_typed = quantized && type != "det" && type != "rec";
  std::cout << "Building TRT engine: " << onnx_path << " -> " << trt_path;
  if (type == "det")
    std::cout << " (det_max_side=" << detail::read_det_effective_max_side() << ")";
  if (quantized)
    std::cout << (strongly_typed ? " [strongly-typed QDQ]" : " [weakly-typed FP16+QDQ]");
  std::cout << '\n';

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(s_logger));
  if (!builder) return false;

  // Weakly-typed FP16 by default (bit 0 clear). Quantized QDQ ONNX needs a
  // strongly-typed network instead (bit 0 = kSTRONGLY_TYPED): TRT then reads
  // precision from the graph and kFP16 must NOT be set (it's rejected).
  const uint32_t net_flags =
      strongly_typed
          ? (1U << static_cast<uint32_t>(
                 nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
          : 0U;
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(net_flags));
  if (!network) return false;

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, s_logger));
  if (!parser) return false;
  if (!parser->parseFromFile(onnx_path.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    // v6 introduces RepLKFPN/LightSVTR ops; the parser builds clean today, but
    // surface every error so a future op regression is diagnosable rather than
    // a bare "Failed to build engine".
    for (int i = 0; i < parser->getNbErrors(); ++i)
      std::cerr << "[TRT] ONNX parse error in " << onnx_path << ": "
                << parser->getError(i)->desc() << '\n';
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  // Layout needs more workspace than det/rec/cls because PP-DocLayoutV3 is a
  // DETR-family model with large attention matmuls; 1 GB is fine for the
  // others. Det workspace scales with DET_MAX_SIDE: at 4096 the activation
  // tensors are ~17× larger than at 960, so 1 GB can be tight. Capped at
  // 4 GB (same ceiling as layout) so even 4096-side builds fit on a 16 GB
  // card alongside rec/cls/layout.
  size_t workspace_bytes;
  if (type == "layout") {
    workspace_bytes = 4ULL << 30;          // 4 GiB (PP-DocLayoutV3 DETR attention)
  } else if (type == "slanext_encoder") {
    workspace_bytes = 2ULL << 30;          // 2 GiB (PP-LCNet CNN encoder, 488²)
  } else if (type == "det") {
    // (det_max/960)² × 1 GiB, capped at 4 GiB. 960 is the historical baseline
    // for the scale ratio: at 2048 → ~4 GiB, at 4096 → 4 GiB (cap). det_max is
    // the effective profile MAX, so workspace scales with the same value the
    // profile MAX uses below.
    int det_max = detail::read_det_effective_max_side();
    double scale = (static_cast<double>(det_max) / 960.0) *
                   (static_cast<double>(det_max) / 960.0);
    size_t scaled = static_cast<size_t>((1ULL << 30) * scale);
    // TRT_DET_WORKSPACE_GB raises the 4 GiB ceiling on cards with headroom:
    // the medium det at DET_MAX_SIDE_LIMIT=2560 has tactics needing ~4.06 GiB,
    // so the default cap fails its build by ~60 MB even on a 32 GB card.
    size_t cap = 4ULL << 30;
    if (const char *env = std::getenv("TRT_DET_WORKSPACE_GB"); env && *env) {
      int gb = std::atoi(env);
      if (gb >= 1 && gb <= 24) {
        cap = static_cast<size_t>(gb) << 30;
      } else {
        std::cerr << "[TRT] TRT_DET_WORKSPACE_GB=\"" << env
                  << "\" out of range [1,24] — keeping the 4 GiB default\n";
      }
    }
    workspace_bytes = std::min(scaled, cap);
    if (workspace_bytes < (1ULL << 30)) workspace_bytes = 1ULL << 30;
  } else if (type == "slanext_wired" || type == "slanext_wireless" ||
             type == "formula") {
    workspace_bytes = 2ULL << 30;          // 2 GiB (plan 07 §5)
  } else if (type == "rec") {
    // v6 rec (EncoderWithLightSVTR) fuses the whole graph into one large
    // attention ForeignNode; 1 GiB starves the tactic search on the medium
    // tier ("could not find any implementation"). 4 GiB matches layout.
    workspace_bytes = 4ULL << 30;          // 4 GiB
  } else {
    workspace_bytes = 1ULL << 30;          // cls, table_cls, table_cell_*
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
  // Strongly-typed networks reject kFP16 — precision is fixed by the graph.
  if (!strongly_typed)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // TRT_OPT_LEVEL: 0..5, default 5. Lower trades runtime perf for build time.
  const int opt_level = detail::read_trt_opt_level();
  config->setBuilderOptimizationLevel(opt_level);
  std::cout << "[TRT] builder optimization level: " << opt_level << '\n';

  if (!detail::add_optimization_profiles(*builder, *network, *config, type))
    return false;

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) return false;

  std::error_code ec;
  fs::create_directories(fs::path(trt_path).parent_path(), ec);
  if (ec) {
    std::cerr << "[TRT] Failed to create cache dir for " << trt_path << ": "
              << ec.message() << '\n';
    return false;
  }

  // Write to a temp file first, then atomic rename (prevents corruption from
  // concurrent builds in multi-replica Docker deployments).
  //
  // Both the open and the close-after-write check log on failure with the
  // strerror text. Earlier the open path was a silent `return false`, and
  // a uid-mismatch on a bind-mounted engine cache (host uid 1000, container
  // ocr uid 1001) hit it as EACCES — which surfaced only as "Failed to
  // build engine from <onnx>" from the caller, despite TRT having built
  // the engine successfully. Hours of bisection later, ergo this errno log.
  const auto tmp_path = trt_path + ".tmp." + std::to_string(getpid());
  std::ofstream file(tmp_path, std::ios::binary);
  if (!file) {
    std::cerr << "[TRT] Failed to open temp file " << tmp_path << ": "
              << std::strerror(errno) << '\n';
    return false;
  }
  file.write(static_cast<const char *>(plan->data()), plan->size());
  file.close();
  if (!file) {
    std::cerr << "[TRT] Failed to write temp file " << tmp_path
              << " (disk full?): " << std::strerror(errno) << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }
  fs::rename(tmp_path, trt_path, ec);
  if (ec) {
    std::cerr << "[TRT] Failed to rename " << tmp_path << " -> " << trt_path
              << ": " << ec.message() << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }

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
    std::cout << "Using cached engine: " << trt_path;
    if (type == "det")
      std::cout << " (det_max_side=" << detail::read_det_effective_max_side() << ")";
    std::cout << '\n';
    return trt_path;
  }

  if (!build_engine(onnx_path, trt_path, type)) {
    std::cerr << "[TRT] Failed to build engine from: " << onnx_path << '\n';
    return {};
  }

  return trt_path;
}

} // namespace turbo_ocr::engine
