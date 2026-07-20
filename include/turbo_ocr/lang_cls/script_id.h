#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "turbo_ocr/lang_cls/script_id_types.h"

namespace turbo_ocr::lang_cls {

using turbo_ocr::script_id::ScriptId;

struct ScriptIdResult {
  ScriptId script = ScriptId::Latin;
  float    confidence = 0.0f;
};

// GPU per-text-line script classifier. Input contract (locked by
// turbostruct-rs/models/lang_cls/meta.json + script_id training recipe):
//   shape  [N, 3, 32, 128] float32 RGB CHW
//   norm   (pixel/255 - 0.5) / 0.5
//   output [N, 7] logits, argmax → ScriptId (Latin=0..Thai=6)
//
// The TRT engine is built by onnx_to_trt.cpp under the "script_id" type
// with workspace 256 MiB, FP16, and shape profile (1/32/512, 3, 32, 128).
class ScriptIdEngine {
 public:
  ScriptIdEngine() = default;
  ~ScriptIdEngine() noexcept = default;

  [[nodiscard]] bool load_model(const std::string& model_path);

  // Classify text-line crops defined by quad boxes in `page`. Returns one
  // ScriptIdResult per input box in the same order. Boxes are warped from
  // the page GpuImage exactly like PaddleCls (perspective M_inv per quad
  // → cuda_batch_roi_warp). Caller-owned `stream`; the call syncs the
  // stream once before returning.
  [[nodiscard]] std::vector<ScriptIdResult>
  classify_batch(const GpuImage& page,
                 const std::vector<Box>& boxes,
                 cudaStream_t stream = 0);

  void allocate_buffers();

 private:
  static constexpr int kMaxBatch    = 512;
  static constexpr int kImageH      = 32;
  static constexpr int kImageW      = 128;
  static constexpr int kNumClasses  = 7;
  // Below this confidence the dispatcher should fall back to its default
  // language; the engine still reports the argmax for visibility.
  static constexpr float kMinConf   = 0.6f;

  std::unique_ptr<engine::TrtEngine> engine_;

  CudaPtr<float>  d_batch_input_;
  CudaPtr<float>  d_output_;
  CudaPtr<float>  d_M_invs_;
  CudaPtr<int>    d_crop_widths_;

  CudaHostPtr<float> h_output_;
  CudaHostPtr<float> h_M_invs_;
  CudaHostPtr<int>   h_crop_widths_;

  bool buffers_allocated_ = false;

  // Serialises classify_batch() across workers when shared via shared_ptr.
  mutable std::mutex inference_mutex_;
};

}  // namespace turbo_ocr::lang_cls
