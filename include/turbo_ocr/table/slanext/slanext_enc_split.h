#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "turbo_ocr/table/slanext/slanext_dict.h"
#include "turbo_ocr/table/slanext/slanext_host_decode.h"
#include "turbo_ocr/table/table_types.h"

namespace turbo_ocr::table {

// SLANeXt encoder-split table-structure backend.
//
// The fused SLANeXt ONNX (CNN encoder + autoregressive GRU-attention decoder)
// can't build under TensorRT (the decoder Loop trips makeScopeNodesContiguous).
// So we split it: the CNN encoder runs on a TRT FP16 engine producing the
// feature [1, T_enc=256, C=96]; the small GRU+attention+2-head decoder runs
// on the host (CPU) per token. Per-step compute is tiny (256-hidden GRU,
// 96-dim attention over 256 positions, 50-token classifier, 8-coord bbox), so
// the host loop is fast and avoids the unbuildable Loop entirely.
// The host decode is numpy-parity-verified at 100% token match vs the full ONNX.
class SlanextEncSplit {
 public:
  static constexpr int kInputSize = 488;   // ResizeByLong-488 letterbox
  static constexpr int kTenc      = 256;   // encoder feature sequence length
  static constexpr int kCtx       = 96;    // encoder feature channel/context dim
  static constexpr int kHidden    = 256;
  static constexpr int kVocab     = 50;
  static constexpr int kLoc       = 8;
  static constexpr int kMaxTokens = 501;

  SlanextEncSplit() = default;
  ~SlanextEncSplit() noexcept;

  SlanextEncSplit(const SlanextEncSplit&) = delete;
  SlanextEncSplit& operator=(const SlanextEncSplit&) = delete;

  // Build/load the TRT encoder engine, the decoder weight blob (16 float32
  // tensors in fixed order), and the structure dict.
  [[nodiscard]] bool load_model(const std::string& encoder_trt_path,
                                const std::string& decoder_bin_path,
                                const std::string& dict_path);

  // Encode one table region on the GPU, decode on the host, return the
  // structure token sequence + per-cell quads (original-region pixels).
  [[nodiscard]] StructureResult infer(const GpuImage& page,
                                      const Box& region,
                                      cudaStream_t stream);

 private:
  [[nodiscard]] bool discover_tensor_names();
  [[nodiscard]] bool init_buffers();

  std::unique_ptr<engine::TrtEngine> engine_;
  cudaEvent_t d2h_event_ = nullptr;

  CudaPtr<float>     d_image_;   // [1,3,488,488]
  CudaPtr<float>     d_feat_;    // [1,256,96]
  CudaHostPtr<float> h_feat_;

  std::string input_name_;
  std::string output_name_;

  // Decoder weights + host AR decode shared with the CPU backend
  // (slanext_host_decode.{h,cpp} — one arithmetic source, one blob loader).
  SlanextDecoderWeights weights_;

  CharDict dict_;
};

}  // namespace turbo_ocr::table
