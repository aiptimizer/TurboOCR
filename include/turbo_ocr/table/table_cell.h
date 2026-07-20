#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt/trt_engine.h"

namespace turbo_ocr::table {

// One cell-det detection in PAGE (original-image) pixels.
struct CellDetBox {
  int cls;
  float score;
  std::array<int, 4> aabb; // [x0, y0, x1, y1] in page coords
};

// RT-DETR-L wired/wireless table-cell detector. Mirrors PaddleLayout's
// enqueue/collect/event pattern but inputs a SUB-RECT of the page GpuImage
// (no extra crop). One instance per variant (wired, wireless).
//
//   1. enqueue(page, rect, stream)  — async preprocess + im_shape/scale_factor
//      H2D + TRT execute. Records d2h_event_.
//   2. collect(score_threshold)     — block on d2h_event_, D2H, filter, NMS,
//      clamp, project to PAGE coords (offset by rect_x/rect_y).
class TableCellDet {
public:
  static constexpr int kInputSize     = 640;
  static constexpr int kMaxDetections = 300;

  TableCellDet() = default;
  ~TableCellDet() noexcept;

  TableCellDet(const TableCellDet&) = delete;
  TableCellDet& operator=(const TableCellDet&) = delete;

  [[nodiscard]] bool load_model(const std::string& trt_path);

  // Enqueue inference on the given sub-rect of `page`. Coordinates are in
  // page pixels. The model receives a 640×640 resize of the sub-rect.
  [[nodiscard]] bool enqueue(const GpuImage& page,
                             int rect_x, int rect_y,
                             int rect_w, int rect_h,
                             cudaStream_t stream);

  // Collect after enqueue(). Returns detections in PAGE coordinates.
  [[nodiscard]] std::vector<CellDetBox>
  collect(float score_threshold = 0.4f, float nms_iou = 0.5f);

private:
  [[nodiscard]] bool discover_tensor_names();
  [[nodiscard]] bool init_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;

  cudaEvent_t d2h_event_ = nullptr;

  // Per-enqueue state for collect().
  int pending_rect_x_ = 0;
  int pending_rect_y_ = 0;
  int pending_rect_w_ = 0;
  int pending_rect_h_ = 0;
  cudaStream_t pending_stream_ = nullptr;

  // Device buffers — batch=1.
  CudaPtr<float> d_image_;        // [1, 3, 640, 640]
  CudaPtr<float> d_im_shape_;     // [1, 2]
  CudaPtr<float> d_scale_factor_; // [1, 2]
  CudaPtr<float> d_out0_;         // [kMaxDetections, 6]

  // Optional secondary output buffers (some PaddleX exports emit a (B,)
  // count tensor alongside the (N,6) detection tensor). We bind addresses
  // even when we don't read them — TRT requires every output to have one.
  CudaPtr<int32_t> d_out_aux_;    // shape varies, sized for safety

  CudaHostPtr<float> h_out0_;
  CudaHostPtr<float> h_im_shape_;
  CudaHostPtr<float> h_scale_factor_;

  // Tensor names (paddle2onnx may emit them in any order).
  std::string name_image_;
  std::string name_im_shape_;
  std::string name_scale_factor_;
  std::string name_out0_;                          // (N, 6) detections
  std::vector<std::string> aux_output_names_;      // ignored outputs (count etc.)
};

} // namespace turbo_ocr::table
