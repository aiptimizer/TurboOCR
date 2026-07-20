#pragma once

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/server/model_catalog.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace turbo_ocr::pipeline {

using turbo_ocr::server::DetInferConfig;

/// Abstract interface for OCR pipelines (GPU and CPU).
/// GPU-specific overloads (with cudaStream_t, GpuImage, batch) remain
/// as non-virtual extensions on OcrPipeline.
class IOcrPipeline {
public:
  virtual ~IOcrPipeline() = default;

  // `det_cfg` is the selected model's official detection inference config
  // (resize policy + DB params); threaded into the detector at load. Env
  // per-field overrides are applied inside the detector so env always wins.
  [[nodiscard]] virtual bool init(const std::string &det_model,
                                  const std::string &rec_model,
                                  const std::string &rec_dict,
                                  const std::string &cls_model = "",
                                  const DetInferConfig &det_cfg =
                                      {turbo_ocr::detection::kDetResizeDefault,
                                       turbo_ocr::detection::kDbDefaults}) = 0;

  virtual void warmup() = 0;

  [[nodiscard]] virtual std::vector<OCRResultItem> run(const cv::Mat &img) = 0;
};

} // namespace turbo_ocr::pipeline
