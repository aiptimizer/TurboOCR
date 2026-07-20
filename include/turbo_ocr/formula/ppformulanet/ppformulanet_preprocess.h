#pragma once

#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace turbo_ocr::formula {

// PP-FormulaNet-S preprocess constants shared by the GPU (.cu) and CPU paths.
// S is the 384x384 encoder input; MEAN/STD are the grayscale normalization;
// PAD_VAL is the normalized-zero pad value used to letterbox to S x S.
inline constexpr int kFormulaInputSize = 384;
inline constexpr float kFormulaMean = 0.7931f;
inline constexpr float kFormulaStd = 0.1738f;
inline constexpr float kFormulaPadVal = (0.0f - kFormulaMean) / kFormulaStd;

// CUDA-FREE preprocess for ONE BGR8 formula crop. Mirrors the Python sidecar:
// margin-crop -> AR-preserving PIL-bicubic resize (longer side 384) -> /255,
// (x-mean)/std on BGR floats -> BGR2GRAY -> center-pad to 384x384 with PAD_VAL.
// `out` must point to kFormulaInputSize*kFormulaInputSize floats. Pure C++/OpenCV,
// no CUDA. Identical bytes to the GPU path's host preprocess.
void formula_preprocess_one(const uint8_t *bgr, int w, int h, float *out);

// Mode-collapse detector (CUDA-free). PP-FormulaNet-S sometimes collapses on
// hard/non-formula crops into a long repetitive run; flagged formulas are
// dropped to empty when the opt-in PPFNS_DROP_COLLAPSE is set. The 3-gram
// uniqueness test catches repetition without dropping legitimate long matrices.
bool formula_is_mode_collapsed(const std::vector<int64_t> &toks,
                               const std::string &latex);

}  // namespace turbo_ocr::formula
