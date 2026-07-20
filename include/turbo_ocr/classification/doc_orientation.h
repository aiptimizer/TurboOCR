#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/engine/trt/trt_engine.h"

namespace turbo_ocr::classification {

/// GPU document-orientation classifier (PP-LCNet_x1_0_doc_ori, TensorRT).
/// Detects a page's clockwise rotation ∈ {0,90,180,270}. Preprocessing runs
/// on the CPU (opt-in, once per page); only the 224x224 inference is on GPU.
class DocOrientation {
public:
  DocOrientation() = default;
  ~DocOrientation() noexcept = default;

  [[nodiscard]] bool load_model(const std::string &model_path);
  void allocate_buffers();

  /// Returns the page's detected clockwise rotation in degrees (0/90/180/270),
  /// or 0 if not loaded or low-confidence. `bgr` is the full rendered page.
  [[nodiscard]] int detect(const cv::Mat &bgr, cudaStream_t stream = 0);

private:
  std::unique_ptr<engine::TrtEngine> engine_;
  CudaPtr<float> d_input_;
  CudaPtr<float> d_output_;
  CudaHostPtr<float> h_input_;
  CudaHostPtr<float> h_output_;
  bool buffers_allocated_ = false;
};

} // namespace turbo_ocr::classification
