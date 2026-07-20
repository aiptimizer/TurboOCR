#include "turbo_ocr/table/table_cell.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::table {

using turbo_ocr::engine::TrtEngine;

TableCellDet::~TableCellDet() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool TableCellDet::discover_tensor_names() {
  // Inputs: image (4-D), im_shape (2-D), scale_factor (2-D). paddle2onnx does
  // not guarantee order — dispatch by name (matches paddle_layout).
  for (const auto& n : engine_->input_names()) {
    if (n == "im_shape") {
      name_im_shape_ = n;
    } else if (n == "scale_factor") {
      name_scale_factor_ = n;
    } else if (n == "image" || n.find("image") != std::string::npos) {
      if (name_image_.empty()) name_image_ = n;
    }
  }
  if (name_image_.empty() || name_im_shape_.empty() || name_scale_factor_.empty()) {
    std::cerr << "[table_cell] expected inputs image/im_shape/scale_factor, got: ";
    for (const auto& n : engine_->input_names()) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }

  // Outputs: the (N, 6) detection tensor + optionally a (B,) count tensor.
  // Identify the (N, 6) tensor by rank == 2 with last extent == 6.
  const auto& outs = engine_->output_names();
  if (outs.empty()) {
    std::cerr << "[table_cell] engine has no outputs\n";
    return false;
  }
  for (const auto& name : outs) {
    auto dims = engine_->tensor_shape(name);
    const bool is_det =
        (dims.nbDims == 2 && dims.d[1] == 6) ||
        (dims.nbDims == 3 && dims.d[2] == 6);
    if (is_det && name_out0_.empty()) {
      name_out0_ = name;
    } else {
      aux_output_names_.push_back(name);
    }
  }
  if (name_out0_.empty()) {
    std::cerr << "[table_cell] could not find (N,6) detection output among: ";
    for (const auto& n : outs) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }
  return true;
}

bool TableCellDet::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_im_shape_.reset(2);
  d_scale_factor_.reset(2);
  d_out0_.reset(static_cast<std::size_t>(kMaxDetections) * 6);

  // Generous scratch for aux outputs (e.g. (B,) int32 count, etc.).
  if (!aux_output_names_.empty()) {
    d_out_aux_.reset(static_cast<std::size_t>(kMaxDetections) * 4);
  }

  h_out0_.reset(static_cast<std::size_t>(kMaxDetections) * 6);
  h_im_shape_.reset(2);
  h_scale_factor_.reset(2);

  engine_->set_tensor_address(name_image_,        d_image_.get());
  engine_->set_tensor_address(name_im_shape_,     d_im_shape_.get());
  engine_->set_tensor_address(name_scale_factor_, d_scale_factor_.get());
  engine_->set_tensor_address(name_out0_,         d_out0_.get());
  for (const auto& n : aux_output_names_) {
    engine_->set_tensor_address(n, d_out_aux_.get());
  }

  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool TableCellDet::load_model(const std::string& trt_path) {
  engine_ = std::make_unique<TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[table_cell] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

bool TableCellDet::enqueue(const GpuImage& page,
                           int rect_x, int rect_y,
                           int rect_w, int rect_h,
                           cudaStream_t stream) {
  // Cleared up front; only re-armed on full success (final step below).
  // collect() treats a null pending_stream_ as "the last enqueue failed" and
  // bails, rather than draining the stream and decoding a stale buffer left by
  // a previous successful enqueue. Mirrors PaddleLayout's contract.
  pending_stream_ = nullptr;

  if (!engine_) return false;
  if (rect_w <= 0 || rect_h <= 0) return false;

  // Clamp to page bounds.
  rect_x = std::clamp(rect_x, 0, page.cols - 1);
  rect_y = std::clamp(rect_y, 0, page.rows - 1);
  rect_w = std::clamp(rect_w, 1, page.cols - rect_x);
  rect_h = std::clamp(rect_h, 1, page.rows - rect_y);

  pending_rect_x_ = rect_x;
  pending_rect_y_ = rect_y;
  pending_rect_w_ = rect_w;
  pending_rect_h_ = rect_h;

  // 1. Fused preprocess: sub-rect → 640×640 /255 BGR CHW. Reuses the layout
  //    kernel with sub-rect overload (added in src/table/kernels/).
  kernels::cuda_fused_resize_normalize_layout(
      page, rect_x, rect_y, rect_w, rect_h,
      d_image_.get(), kInputSize, kInputSize, stream);

  // 2. im_shape / scale_factor — PaddleX convention (matches paddle_layout):
  //    im_shape    = resized dims (always 640×640)
  //    scale_factor= resized / original = 640/rect_h, 640/rect_w
  h_im_shape_.get()[0] = static_cast<float>(kInputSize);
  h_im_shape_.get()[1] = static_cast<float>(kInputSize);
  h_scale_factor_.get()[0] =
      static_cast<float>(kInputSize) / static_cast<float>(rect_h);
  h_scale_factor_.get()[1] =
      static_cast<float>(kInputSize) / static_cast<float>(rect_w);
  CUDA_CHECK(cudaMemcpyAsync(d_im_shape_.get(), h_im_shape_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_scale_factor_.get(), h_scale_factor_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));

  nvinfer1::Dims4 img_dims{1, 3, kInputSize, kInputSize};
  nvinfer1::Dims2 vec_dims{1, 2};
  if (!engine_->set_input_shape(name_image_, img_dims))        return false;
  if (!engine_->set_input_shape(name_im_shape_, vec_dims))     return false;
  if (!engine_->set_input_shape(name_scale_factor_, vec_dims)) return false;

  if (!engine_->execute(stream)) {
    std::cerr << "[table_cell] TRT execute failed\n";
    return false;
  }

  pending_stream_ = stream;
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  return true;
}

std::vector<CellDetBox>
TableCellDet::collect(float score_threshold, float nms_iou) {
  std::vector<CellDetBox> out;
  if (!engine_ || !d2h_event_) return out;

  // A null pending_stream_ means the matching enqueue() bailed (bad rect,
  // execute() failure, ...). d2h_event_ would still hold a prior successful
  // recording, so synchronizing on it and decoding h_out0_/d_out0_ here would
  // silently serve the previous request's detections. Bail instead.
  const cudaStream_t stream = pending_stream_;
  if (!stream) return out;

  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  auto dims = engine_->tensor_shape(name_out0_);
  int n_rows = (dims.nbDims >= 2 ? dims.d[0] : 0);
  if (n_rows <= 0) return out;
  n_rows = std::min(n_rows, kMaxDetections);

  CUDA_CHECK(cudaMemcpyAsync(h_out0_.get(), d_out0_.get(),
                              sizeof(float) * static_cast<std::size_t>(n_rows) * 6,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const int rx = pending_rect_x_;
  const int ry = pending_rect_y_;
  const int rw = pending_rect_w_;
  const int rh = pending_rect_h_;

  // 1. Filter by score, drop degenerate, project to PAGE coords.
  std::vector<CellDetBox> kept;
  kept.reserve(n_rows);
  const float* rows = h_out0_.get();
  for (int i = 0; i < n_rows; ++i) {
    const float* row = rows + i * 6;
    const float score = row[1];
    if (score < score_threshold) continue;

    // Sub-rect-local coords from the model (rt-detr postproc projects with
    // scale_factor → coords are in pixels of the input image, which is the
    // sub-rect at its native size).
    // std::lround, not truncation-toward-zero: static_cast<int> bakes a ~1px
    // downward bias into every cell box that can flip borderline cell<->OCR IoU.
    int x0 = static_cast<int>(std::lround(row[2]));
    int y0 = static_cast<int>(std::lround(row[3]));
    int x1 = static_cast<int>(std::lround(row[4]));
    int y1 = static_cast<int>(std::lround(row[5]));
    if (x1 <= x0 || y1 <= y0) continue;

    // Clamp to sub-rect bounds before projecting.
    x0 = std::clamp(x0, 0, rw - 1);
    y0 = std::clamp(y0, 0, rh - 1);
    x1 = std::clamp(x1, 0, rw - 1);
    y1 = std::clamp(y1, 0, rh - 1);
    if (x1 <= x0 || y1 <= y0) continue;

    CellDetBox cb;
    cb.cls = static_cast<int>(row[0]);
    cb.score = score;
    cb.aabb = {rx + x0, ry + y0, rx + x1, ry + y1};
    kept.push_back(cb);
  }

  if (kept.empty()) return out;

  // 2. Same-class NMS, IoU > nms_iou. Score-desc. Width/height use
  //    `(x2 - x1 + 1)` to stay byte-equal to PaddleX.
  std::sort(kept.begin(), kept.end(),
            [](const CellDetBox& a, const CellDetBox& b) {
              return a.score > b.score;
            });

  // Precompute each box's area once. The PaddleX (x2-x1+1)*(y2-y1+1) area is a
  // property of a single box, so recomputing it per candidate pair inside the
  // O(n^2) loop is pure redundancy; hoisting it is byte-identical (same
  // operands, same float rounding) and cuts area work from O(n^2) to O(n).
  std::vector<float> area(kept.size());
  for (std::size_t i = 0; i < kept.size(); ++i) {
    const auto& b = kept[i].aabb;
    area[i] = static_cast<float>(b[2] - b[0] + 1) *
              static_cast<float>(b[3] - b[1] + 1);
  }

  const auto box_iou = [](const CellDetBox& a, const CellDetBox& b,
                          float area_a, float area_b) -> float {
    const int ix1 = std::max(a.aabb[0], b.aabb[0]);
    const int iy1 = std::max(a.aabb[1], b.aabb[1]);
    const int ix2 = std::min(a.aabb[2], b.aabb[2]);
    const int iy2 = std::min(a.aabb[3], b.aabb[3]);
    const int iw = std::max(0, ix2 - ix1 + 1);
    const int ih = std::max(0, iy2 - iy1 + 1);
    const float inter = static_cast<float>(iw) * static_cast<float>(ih);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
  };

  // char, not vector<bool>: avoids the bit-proxy indexing overhead; identical
  // semantics.
  std::vector<char> suppressed(kept.size(), 0);
  out.reserve(kept.size());
  for (std::size_t i = 0; i < kept.size(); ++i) {
    if (suppressed[i]) continue;
    out.push_back(kept[i]);
    for (std::size_t j = i + 1; j < kept.size(); ++j) {
      if (suppressed[j]) continue;
      if (kept[i].cls != kept[j].cls) continue;
      if (box_iou(kept[i], kept[j], area[i], area[j]) > nms_iou) {
        suppressed[j] = 1;
      }
    }
  }

  return out;
}

} // namespace turbo_ocr::table
