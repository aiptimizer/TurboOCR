#include "turbo_ocr/lang_cls/script_id.h"

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/geometry/perspective.h"
#include "turbo_ocr/kernels/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace turbo_ocr::lang_cls {

using turbo_ocr::Box;
using turbo_ocr::GpuImage;
using turbo_ocr::engine::TrtEngine;

bool ScriptIdEngine::load_model(const std::string& model_path) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  return engine_->load();
}

void ScriptIdEngine::allocate_buffers() {
  if (buffers_allocated_) return;

  const int bs = kMaxBatch;

  d_batch_input_ = CudaPtr<float>(static_cast<size_t>(bs) * 3 * kImageH * kImageW);
  d_output_      = CudaPtr<float>(static_cast<size_t>(bs) * kNumClasses);
  d_M_invs_      = CudaPtr<float>(static_cast<size_t>(bs) * 9);
  d_crop_widths_ = CudaPtr<int>(static_cast<size_t>(bs));

  h_output_      = CudaHostPtr<float>(static_cast<size_t>(bs) * kNumClasses);
  h_M_invs_      = CudaHostPtr<float>(static_cast<size_t>(bs) * 9);
  h_crop_widths_ = CudaHostPtr<int>(static_cast<size_t>(bs));

  engine_->bind_io(d_batch_input_.get(), d_output_.get());

  buffers_allocated_ = true;
}

std::vector<ScriptIdResult>
ScriptIdEngine::classify_batch(const GpuImage& page,
                               const std::vector<Box>& boxes,
                               cudaStream_t stream) {
  std::vector<ScriptIdResult> results;
  if (boxes.empty()) [[unlikely]] return results;

  std::lock_guard<std::mutex> lock(inference_mutex_);

  allocate_buffers();

  const int total = static_cast<int>(boxes.size());
  results.resize(total);

  for (int beg = 0; beg < total; beg += kMaxBatch) {
    const int cur = std::min(kMaxBatch, total - beg);

    float* h_M = h_M_invs_.get();
    int*   h_w = h_crop_widths_.get();
    for (int j = 0; j < cur; ++j) {
      auto ct = turbo_ocr::compute_crop_transform(boxes[beg + j], kImageH, kImageW);
      h_w[j] = ct.crop_width;
      std::memcpy(h_M + j * 9, ct.M_inv, 9 * sizeof(float));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_M_invs_.get(), h_M,
                                static_cast<size_t>(cur) * 9 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_crop_widths_.get(), h_w,
                                static_cast<size_t>(cur) * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    // Bit-identical preprocessing to PaddleCls / PaddleRec: perspective warp
    // quad → RGB CHW with (pixel/255 - 0.5)/0.5 normalization. Matches the
    // script_id training recipe (mean/std = 0.5/0.5, RGB).
    turbo_ocr::kernels::cuda_batch_roi_warp(
        page, d_M_invs_.get(), d_crop_widths_.get(),
        d_batch_input_.get(), cur, kImageH, kImageW, stream);

    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0] = cur;
    dims.d[1] = 3;
    dims.d[2] = kImageH;
    dims.d[3] = kImageW;
    if (!engine_->infer_dynamic(dims, stream)) {
      throw turbo_ocr::InferenceError("ScriptId TRT inference failed");
    }

    CUDA_CHECK(cudaMemcpyAsync(h_output_.get(), d_output_.get(),
                                static_cast<size_t>(cur) * kNumClasses * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float* h_out = h_output_.get();
    for (int j = 0; j < cur; ++j) {
      const float* row = h_out + j * kNumClasses;
      float mx = row[0];
      for (int c = 1; c < kNumClasses; ++c) mx = std::max(mx, row[c]);
      float sum = 0.0f;
      int   best_id = 0;
      float best_exp = 0.0f;
      for (int c = 0; c < kNumClasses; ++c) {
        float e = std::exp(row[c] - mx);
        sum += e;
        if (e > best_exp) { best_exp = e; best_id = c; }
      }
      const float score = (sum > 0.0f) ? (best_exp / sum) : 0.0f;
      results[beg + j].script     = static_cast<ScriptId>(best_id);
      results[beg + j].confidence = score;
    }
  }

  return results;
}

}  // namespace turbo_ocr::lang_cls
