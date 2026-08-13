#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/table_types.h"

namespace turbo_ocr::table {

// SLANeXt host decoder weights (16 float32 tensors in the fixed blob order
// emitted by the Python extractor) + the host GRU+attention+2-head AR decode.
// CUDA-free: the ONE decode implementation shared by the CPU encoder-split
// backend and the GPU SlanextEncSplit (whose TRT encoder produces the same
// feature map). Compiled into both targets with identical -ffast-math +
// -fopenmp-simd source properties so the token/bbox streams stay bit-identical.
struct SlanextDecoderWeights {
  static constexpr int kTenc   = 256;  // encoder feature sequence length
  static constexpr int kCtx    = 96;   // encoder feature channel/context dim
  static constexpr int kHidden = 256;
  static constexpr int kVocab  = 50;
  static constexpr int kLoc    = 8;
  static constexpr int kMaxTokens = 501;
  static constexpr int kInputSize = 488;

  std::vector<float> lin0_, lin1w_, lin1b_, lin2_, lin3w_, lin3b_,
      lin4w_, lin4b_, lin5w_, lin5b_, lin6w_, lin6b_,
      gw0_, gw1_, gb0_, gb1_;

  // Load the decoder weight blob (16 tensors, fixed order). Returns false on a
  // truncated or oversized (wrong) file.
  [[nodiscard]] bool load(const std::string &bin_path);
};

// Run the host AR decode on one encoder feature [kTenc, kCtx] row-major and
// project quads back to the original region (ori_w/ori_h pixels). Identical
// maths to SlanextEncSplit::infer's host loop.
StructureResult slanext_host_decode(const float *feat,
                                    const SlanextDecoderWeights &w,
                                    const CharDict &dict, int ori_w, int ori_h);

}  // namespace turbo_ocr::table
