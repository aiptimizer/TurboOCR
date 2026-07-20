#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

// GPU layout detector using PP-DocLayoutV3 via TensorRT (fp16, dynamic batch).
//
// Usage is split into two phases so the CPU doesn't have to block in the
// middle of OcrPipeline::run():
//
//   1. enqueue(gpu_img, H, W, stream)  — async: preprocess kernel + H2D
//      of im_shape/scale_factor + TRT execute, all on `stream`. Returns
//      immediately without any stream sync (no readback here).
//
//   2. collect(threshold) — issues both D2H copies (count tensor out1 +
//      the fixed-size detection buffer out0) back-to-back on the enqueue
//      stream and drains them with a single cudaStreamSynchronize (stream
//      ordering places the D2H copies strictly after the TRT execute), then
//      CPU-decodes h_out0_ into LayoutBox vectors (the model already ran NMS
//      on-GPU; the host step is decode + postfilter, not NMS).
//
// Splitting like this means the pipeline overlap between det_N and rec_{N-1}
// is preserved — layout adds zero wall-clock in the common case.
class PaddleLayout {
public:
  PaddleLayout() = default;
  // All device/host/engine resources are RAII members (CudaPtr / CudaHostPtr /
  // unique_ptr), so no hand-written teardown is needed.
  ~PaddleLayout() noexcept = default;

  // Load the TensorRT engine and allocate GPU + pinned-host buffers for
  // batch=1 inference. The engine itself supports batches up to 8.
  [[nodiscard]] bool load_model(const std::string &trt_path);

  // Enqueue one image's inference on `stream`. gpu_img is BGR uint8 on
  // device (typically the same GpuImage already passed to detection).
  // (orig_h, orig_w) are forwarded as `im_shape` so the model projects its
  // 800x800 output boxes back into original coordinates.
  //
  // Does NOT touch h_out0_ and does NOT block the CPU. Safe to call
  // concurrently with subsequent work on the same stream.
  [[nodiscard]] bool
  enqueue(const GpuImage &gpu_img, int orig_h, int orig_w,
          cudaStream_t stream);

  // Drain the enqueue stream, D2H the detection + count tensors, then decode
  // into LayoutBox vectors. Must be called exactly once after enqueue().
  [[nodiscard]] std::vector<LayoutBox>
  collect(float score_threshold = 0.3f);

  static constexpr int kInputSize = 800;
  // PP-DocLayoutV3's decoder emits up to 300 queries per image. Most FUNSD
  // pages produce 10-30 detections after score filtering.
  static constexpr int kMaxDetections = 300;
  // Batched profile MAX from onnx_to_trt.cpp: {1-4-8, 3, 800, 800}.
  static constexpr int kMaxBatch = 8;

private:
  std::unique_ptr<engine::TrtEngine> engine_;

  // Per-enqueue state that collect() needs. pending_stream_ doubles as the
  // "enqueue succeeded" flag: collect() bails when it is null so a failed
  // enqueue can never decode a previous request's stale buffers.
  int pending_orig_h_ = 0;
  int pending_orig_w_ = 0;
  cudaStream_t pending_stream_ = nullptr;

  // Device buffers sized for batch=1 inference.
  CudaPtr<float>   d_image_;          // [1, 3, 800, 800]
  CudaPtr<float>   d_im_shape_;       // [1, 2]
  CudaPtr<float>   d_scale_factor_;   // [1, 2]
  CudaPtr<float>   d_out0_;           // [kMaxDetections, 7] raw detections
  CudaPtr<int32_t> d_out1_;           // [1] num valid detections
  // fetch_name_2 is a (N, 200, 200) int32 per-proposal mask tensor we don't
  // need. We still have to provide an address (TRT requires addresses for
  // every output) so we allocate a dummy buffer and ignore its contents.
  CudaPtr<int32_t> d_out2_;

  // Pinned host staging so the D2H readback and stream.sync don't stall
  // on pageable memory.
  CudaHostPtr<float>   h_out0_;       // [kMaxDetections, 7]
  CudaHostPtr<int32_t> h_out1_;       // [1]
  CudaHostPtr<float>   h_im_shape_;   // [1, 2]
  CudaHostPtr<float>   h_scale_factor_; // [1, 2]

  // Cached tensor names (paddle2onnx may emit them in any order).
  std::string name_image_;
  std::string name_im_shape_;
  std::string name_scale_factor_;
  std::string name_out0_;   // the (N, 7) detection tensor
  std::string name_out1_;   // the (B,) count tensor
  std::string name_out2_;   // the mask tensor we ignore

  [[nodiscard]] bool init_buffers();
  [[nodiscard]] bool discover_tensor_names();
};

} // namespace turbo_ocr::layout
