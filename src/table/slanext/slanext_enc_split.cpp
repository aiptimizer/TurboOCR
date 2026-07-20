#include "turbo_ocr/table/slanext/slanext_enc_split.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/table/slanext/slanext_host_decode.h"
#include "turbo_ocr/table/slanext/slanext_postprocess.h"

namespace turbo_ocr::table {

using engine::TrtEngine;

// The host GRU+attention AR decode and the decoder-weight blob loader are
// shared with the CPU backend via slanext_host_decode.{h,cpp}: ONE arithmetic
// source compiled with the same -ffast-math + -fopenmp-simd source properties
// in both targets, so GPU and CPU table-structure output stay bit-identical by
// construction (previously two verbatim copies kept in sync by review).
static_assert(SlanextEncSplit::kTenc      == SlanextDecoderWeights::kTenc);
static_assert(SlanextEncSplit::kCtx       == SlanextDecoderWeights::kCtx);
static_assert(SlanextEncSplit::kHidden    == SlanextDecoderWeights::kHidden);
static_assert(SlanextEncSplit::kVocab     == SlanextDecoderWeights::kVocab);
static_assert(SlanextEncSplit::kLoc       == SlanextDecoderWeights::kLoc);
static_assert(SlanextEncSplit::kMaxTokens == SlanextDecoderWeights::kMaxTokens);
static_assert(SlanextEncSplit::kInputSize == SlanextDecoderWeights::kInputSize);

SlanextEncSplit::~SlanextEncSplit() noexcept {
  if (d2h_event_) cudaEventDestroy(d2h_event_);
}

bool SlanextEncSplit::discover_tensor_names() {
  const auto& ins = engine_->input_names();
  const auto& outs = engine_->output_names();
  if (ins.empty() || outs.empty()) {
    std::cerr << "[slanext-split] missing input/output tensors\n";
    return false;
  }
  input_name_ = ins.front();
  output_name_ = outs.front();
  return true;
}

bool SlanextEncSplit::init_buffers() {
  d_image_.reset(static_cast<std::size_t>(1) * 3 * kInputSize * kInputSize);
  d_feat_.reset(static_cast<std::size_t>(1) * kTenc * kCtx);
  h_feat_.reset(static_cast<std::size_t>(1) * kTenc * kCtx);
  engine_->set_tensor_address(input_name_, d_image_.get());
  engine_->set_tensor_address(output_name_, d_feat_.get());
  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool SlanextEncSplit::load_model(const std::string& encoder_trt_path,
                                 const std::string& decoder_bin_path,
                                 const std::string& dict_path) {
  engine_ = std::make_unique<TrtEngine>(encoder_trt_path);
  if (!engine_->load()) {
    std::cerr << "[slanext-split] failed to load encoder engine: "
              << encoder_trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  if (!weights_.load(decoder_bin_path)) return false;
  try {
    dict_ = dict_path.empty() ? default_dict() : CharDict::from_path(dict_path);
  } catch (const std::exception& e) {
    std::cerr << "[slanext-split] dict load failed: " << e.what() << '\n';
    return false;
  }
  return true;
}

StructureResult SlanextEncSplit::infer(const GpuImage& page, const Box& region,
                                       cudaStream_t stream) {
  StructureResult empty{};
  if (!engine_) return empty;

  auto bb = aabb(region);
  const int rx = std::clamp(bb[0], 0, page.cols - 1);
  const int ry = std::clamp(bb[1], 0, page.rows - 1);
  const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
  const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
  if (rw <= 0 || rh <= 0) return empty;

  // Encoder: ResizeByLong-488 + ImageNet norm + pad (cuda_fused_slanext_pre),
  // then one TRT execute -> feature [1, 256, 96].
  kernels::cuda_fused_slanext_pre_rgb(page, rx, ry, rw, rh, d_image_.get(), stream);
  nvinfer1::Dims4 in_dims{1, 3, kInputSize, kInputSize};
  if (!engine_->set_input_shape(input_name_, in_dims)) {
    std::cerr << "[slanext-split] encoder set_input_shape failed for region ["
              << rx << ',' << ry << ',' << rw << 'x' << rh
              << "] — table region DROPPED\n";
    return empty;
  }
  if (!engine_->execute(stream)) {
    std::cerr << "[slanext-split] encoder execute failed for region ["
              << rx << ',' << ry << ',' << rw << 'x' << rh
              << "] — table region DROPPED\n";
    return empty;
  }
  CUDA_CHECK(cudaMemcpyAsync(h_feat_.get(), d_feat_.get(),
                             sizeof(float) * kTenc * kCtx,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  // Host AR decode (GRU + additive attention + structure/loc heads) — the
  // shared implementation; quads project back to the region pixels (rw, rh).
  return slanext_host_decode(h_feat_.get(), weights_, dict_, rw, rh);
}

}  // namespace turbo_ocr::table
