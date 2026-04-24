#pragma once

#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/cuda_ptr.h"

namespace turbo_ocr::classification {

/// GPU angle classifier using TensorRT (flips 180-degree text crops).
class PaddleCls {
public:
  PaddleCls() = default;
  ~PaddleCls() noexcept = default; // RAII handles all GPU cleanup

  /// Load a TensorRT classification engine and allocate GPU buffers.
  [[nodiscard]] bool load_model(const std::string &model_path);

  // Classify crops and flip 180 degree boxes in-place.
  // boxes: [tl, tr, br, bl] -- rotated boxes get corners swapped.
  void run(const GpuImage &img, std::vector<Box> &boxes,
           cudaStream_t stream = 0);

  // Eagerly allocate all GPU/pinned buffers (called during warmup)
  void allocate_buffers();

private:
  static constexpr int kClsBatchNum = 128;
  // PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25) expects
  // 80x160 input. The v4 shape (48x192) worked on GPU only because the
  // TRT engine is built with a dynamic shape profile; CPU ONNX Runtime
  // rejects it outright.
  static constexpr int kClsImageH = 80;
  static constexpr int kClsImageW = 160;
  static constexpr float kClsThresh = 0.9f;

  std::unique_ptr<engine::TrtEngine> engine_;

  // Pre-allocated GPU buffers (RAII -- no manual cudaFree needed)
  CudaPtr<float> d_batch_input_;
  CudaPtr<float> d_output_;
  CudaPtr<float> d_M_invs_;
  CudaPtr<int> d_crop_widths_;

  // Pinned CPU for async download (RAII -- no manual cudaFreeHost needed)
  CudaHostPtr<float> h_output_;
  CudaHostPtr<float> h_M_invs_;
  CudaHostPtr<int> h_crop_widths_;

  bool buffers_allocated_ = false;
};

} // namespace turbo_ocr::classification
