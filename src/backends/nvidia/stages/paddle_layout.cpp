#include "nvidia/stages/paddle_layout.h"

#include "turbo_ocr/analysis/layout/picodet_decode.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "nvidia/support/cuda_check.h"
#include "nvidia/kernels_cuda/kernels_cuda.h"
#include "turbo_ocr/analysis/layout/layout_postfilter.h"
#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::layout {

bool PaddleLayout::discover_tensor_names() {
  // PP-DocLayoutV3 has:
  //   inputs : image (4-D), im_shape (2-D), scale_factor (2-D)
  //   outputs: fetch_name_0 (N, 7), fetch_name_1 (B,), fetch_name_2 (N, 200, 200)
  // paddle2onnx does not guarantee input order, so dispatch by name and rank.
  for (const auto &n : engine_->input_names()) {
    if (n == "image" || n.find("image") != std::string::npos) {
      if (name_image_.empty()) name_image_ = n;
    } else if (n == "im_shape") {
      name_im_shape_ = n;
    } else if (n == "scale_factor") {
      name_scale_factor_ = n;
    }
  }
  if (name_image_.empty() || name_im_shape_.empty() || name_scale_factor_.empty()) {
    std::cerr << "[layout] expected inputs image/im_shape/scale_factor, got: ";
    for (const auto &n : engine_->input_names()) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }

  const auto &outs = engine_->output_names();
  if (outs.size() != 3) {
    std::cerr << "[layout] expected 3 outputs, got " << outs.size() << '\n';
    return false;
  }
  // Identify by rank: the (N, 7) detection tensor has nbDims == 2 and last
  // extent 7; the count tensor has nbDims == 1; the mask tensor has nbDims
  // == 3. We query shapes after setting a concrete input shape, so do this
  // once from the engine binding metadata (TRT stores the profile's max
  // dims with the -1 dynamic placeholders stripped to the kMAX profile).
  for (const auto &name : outs) {
    auto dims = engine_->tensor_shape(name);
    if (dims.nbDims == 3 && dims.d[1] == 200 && dims.d[2] == 200) {
      name_out2_ = name;
    } else if (dims.nbDims == 1) {
      name_out1_ = name;
    } else {
      // (-1, 7) — the detection tensor
      name_out0_ = name;
    }
  }
  if (name_out0_.empty() || name_out1_.empty() || name_out2_.empty()) {
    std::cerr << "[layout] could not classify outputs by shape; outs=";
    for (const auto &n : outs) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }
  return true;
}

bool PaddleLayout::init_buffers() {
  // Image: [1, 3, 800, 800] float
  d_image_.reset(static_cast<size_t>(1) * 3 * kInputSize * kInputSize);
  d_im_shape_.reset(2);
  d_scale_factor_.reset(2);

  // Detection output (N, 7). We reserve kMaxDetections rows.
  d_out0_.reset(static_cast<size_t>(kMaxDetections) * 7);
  d_out1_.reset(1);
  // Mask tensor we don't read: (kMaxDetections, 200, 200) int32 ≈ 48 MB. We
  // still have to own the buffer so TRT has a valid address to write to.
  d_out2_.reset(static_cast<size_t>(kMaxDetections) * 200 * 200);

  h_out0_.reset(static_cast<size_t>(kMaxDetections) * 7);
  h_out1_.reset(1);
  h_im_shape_.reset(2);
  h_scale_factor_.reset(2);

  // Bind tensor addresses once. We re-bind after select_profile, but
  // PaddleLayout only uses profile 0 so once is enough.
  engine_->set_tensor_address(name_image_,        d_image_.get());
  engine_->set_tensor_address(name_im_shape_,     d_im_shape_.get());
  engine_->set_tensor_address(name_scale_factor_, d_scale_factor_.get());
  engine_->set_tensor_address(name_out0_,         d_out0_.get());
  engine_->set_tensor_address(name_out1_,         d_out1_.get());
  engine_->set_tensor_address(name_out2_,         d_out2_.get());
  return true;
}

bool PaddleLayout::load_model(const std::string &trt_path) {
  engine_ = std::make_unique<engine::TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[layout] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

bool PaddleLayout::enqueue(const GpuImage &gpu_img, int orig_h, int orig_w,
                           cudaStream_t stream) {
  // Cleared up front; only re-armed on full success (step 5 below). collect()
  // treats a null pending_stream_ as "the last enqueue failed" and bails,
  // rather than draining a stream and decoding a stale buffer.
  pending_stream_ = nullptr;
  pending_orig_h_ = orig_h;
  pending_orig_w_ = orig_w;
  if (!engine_) return false;

  // 1. Preprocess: fused resize-to-800x800 + pixel/255 on GPU. Writes
  //    directly into d_image_ (CHW float, batch=1).
  kernels::cuda_fused_resize_normalize_layout(
      gpu_img, d_image_.get(), kInputSize, kInputSize, stream);

  // 2. Fill im_shape and scale_factor, then async H2D.
  //    PaddleX convention: im_shape = [resized_h, resized_w] (always 800x800),
  //    scale_factor = [h_scale, w_scale] = [800/orig_h, 800/orig_w].
  h_im_shape_.get()[0] = static_cast<float>(kInputSize);
  h_im_shape_.get()[1] = static_cast<float>(kInputSize);
  h_scale_factor_.get()[0] =
      static_cast<float>(kInputSize) / static_cast<float>(orig_h);
  h_scale_factor_.get()[1] =
      static_cast<float>(kInputSize) / static_cast<float>(orig_w);
  CUDA_CHECK(cudaMemcpyAsync(d_im_shape_.get(), h_im_shape_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_scale_factor_.get(), h_scale_factor_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));

  // 3. Set per-input shapes (batch=1). Idempotent in TRT, so cheap.
  nvinfer1::Dims4 img_dims{1, 3, kInputSize, kInputSize};
  nvinfer1::Dims2 vec_dims{1, 2};
  if (!engine_->set_input_shape(name_image_, img_dims))        return false;
  if (!engine_->set_input_shape(name_im_shape_, vec_dims))     return false;
  if (!engine_->set_input_shape(name_scale_factor_, vec_dims)) return false;

  // 4. Execute (async). The stream continues without blocking.
  if (!engine_->execute(stream)) {
    std::cerr << "[layout] TRT execute failed\n";
    return false;
  }

  // 5. Publish the stream. The D2H readback is deferred to collect() to keep
  //    enqueue() fully async — and to avoid the implicit GPU sync that
  //    getTensorShape() triggers on DETR models with data-dependent output
  //    shapes. The copies collect() issues on this same stream are ordered
  //    after the execute above, so a single stream sync there suffices.
  pending_stream_ = stream;
  return true;
}

std::vector<LayoutBox> PaddleLayout::collect(float score_threshold) {
  std::vector<LayoutBox> out;
  if (!engine_) return out;

  // A null pending_stream_ means the matching enqueue() bailed (e.g. execute()
  // returned false). Draining a stream and decoding h_out0_ here would silently
  // serve the previous request's layout. Bail instead.
  cudaStream_t stream = pending_stream_;
  if (!stream) return out;

  // Single host<->device sync for the whole readback. Both D2H copies are
  // issued on the stream the TRT execute ran on, so stream ordering places
  // them strictly after it — one cudaStreamSynchronize drains the execute AND
  // both copies. We copy the full fixed-size detection buffer (kMaxDetections
  // rows, ~8 KB) unconditionally rather than reading the count first and then
  // sizing a second copy; that removes the mid-readback sync the old two-phase
  // path needed, so a burst-idle GPU sees one round-trip instead of three.
  CUDA_CHECK(cudaMemcpyAsync(h_out1_.get(), d_out1_.get(), sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      h_out0_.get(), d_out0_.get(),
      sizeof(float) * static_cast<size_t>(kMaxDetections) * 7,
      cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // The detection count comes from the model's own count tensor (out1), NOT
  // from getTensorShape(out0). out0's (N,7) shape has a data-dependent first
  // dim; querying it via getTensorShape() without an IOutputAllocator is
  // unreliable across repeated executions — it returns the correct N on the
  // first request and a stale/zero N on later ones, which silently dropped
  // layout out of every consecutive response. out1[0] is written by the
  // model's NMS on every run and is authoritative.
  int n_rows = h_out1_.get()[0];
  if (n_rows <= 0) return out;
  n_rows = std::min(n_rows, kMaxDetections);

  // Opt-in diagnostic: surfaces the divergence between the authoritative
  // count and the previously-trusted getTensorShape() value.
  static const bool kLayoutDebug = env::env_present("TURBO_LAYOUT_DEBUG");
  if (kLayoutDebug) {
    const auto sd = engine_->tensor_shape(name_out0_);
    std::cerr << "[layout-dbg] out1_count=" << h_out1_.get()[0]
              << " getTensorShape(out0).d[0]="
              << (sd.nbDims >= 2 ? sd.d[0] : -1) << '\n';
  }

  const int orig_h = pending_orig_h_;
  const int orig_w = pending_orig_w_;

  // Decode rows: [class_id, score, xmin, ymin, xmax, ymax, read_order].
  // With correct im_shape/scale_factor (PaddleX convention), the model's
  // postprocessor outputs coordinates directly in the original image space.
  // A few hundred boxes is decoded/NMS'd on the host by design: a GPU NMS
  // kernel over so few boxes would cost more in launch + result-D2H overhead
  // than it saves.
  const float *const rows = h_out0_.get();
  // THE shared PicoDet row decoder (layout/picodet_decode.h) — same policy and
  // the same fail-loud non-finite guard as every other backend. (A few hundred
  // boxes decode on the host by design: a GPU NMS kernel over so few boxes
  // costs more in launch + D2H than it saves.)
  out = decode_picodet_rows(rows, n_rows, /*stride=*/7, /*count=*/nullptr,
                            score_threshold, orig_h, orig_w);

  return postfilter_layout_boxes(std::move(out), orig_h, orig_w);
}

} // namespace turbo_ocr::layout
