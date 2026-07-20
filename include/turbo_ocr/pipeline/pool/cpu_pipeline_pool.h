#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "turbo_ocr/pipeline/ocr/cpu_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pool/pipeline_pool.h"
#include "turbo_ocr/common/errors.h"

namespace turbo_ocr::pipeline {

/// Convenience alias — a PipelinePool of CpuOcrPipeline.
/// Uses infinite timeout (CPU pipelines are cheap; blocking is safe).
using CpuPipelinePool = PipelinePool<CpuOcrPipeline>;

/// Factory: create, init, warmup CPU pipelines and return a pool.
[[nodiscard]] inline std::unique_ptr<CpuPipelinePool> make_cpu_pipeline_pool(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const DetInferConfig &det_cfg = {turbo_ocr::detection::kDetResizeDefault,
                                     turbo_ocr::detection::kDbDefaults}) {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Pool] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<CpuOcrPipeline>> pipelines;
  for (int i = 0; i < pool_size; ++i) {
    auto pipeline = std::make_unique<CpuOcrPipeline>();
    if (pipeline->init(det_model, rec_model, rec_dict, cls_model, det_cfg)) {
      pipelines.push_back(std::move(pipeline));
    } else {
      std::cerr << std::format("[Pool] Failed to init CPU pipeline {} of {}", i, pool_size)
                << '\n';
    }
  }

  if (pipelines.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(
        std::format("[Pool] All {} CPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Warming up {} CPU pipelines...", pipelines.size())
            << '\n';
  for (auto &p : pipelines) {
    p->warmup();
  }
  std::cout << "CPU pipeline warmup complete." << '\n';

  // Infinite timeout for CPU pipelines (blocks until one is available)
  return std::make_unique<CpuPipelinePool>(std::move(pipelines),
                                           std::chrono::seconds::max());
}

} // namespace turbo_ocr::pipeline
