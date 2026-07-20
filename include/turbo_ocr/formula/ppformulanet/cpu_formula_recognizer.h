#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/formula/ppformulanet/formula_tokenizer.h"
#include "turbo_ocr/formula/ppformulanet/ort_session.h"

namespace turbo_ocr::formula {

// Pure-CPU PP-FormulaNet-S recognizer for the CUDA-FREE build. Runs the fused
// PP-FormulaNet-S graph (encoder + autoregressive Loop) on ORT's
// CPUExecutionProvider and decodes with the HF byte-level BPE tokenizer. No
// CUDA, no TensorRT, no Python sidecar.
//
// This is the CPU sibling of PPFormulaNetOrt's `cpu_` branch, refactored to
// take host cv::Mat crops directly (the CPU OCR pipeline has no GpuImage). It
// shares the bit-exact preprocess (formula_preprocess_one) with the GPU path.
class CpuFormulaRecognizer {
 public:
  CpuFormulaRecognizer() = default;

  // Load the fused inference_trt.onnx (path or its parent dir) + tokenizer.json.
  [[nodiscard]] bool load(const std::string &model_path,
                          const std::string &tokenizer_json);

  [[nodiscard]] bool ready() const noexcept { return ready_; }

  // Recognize one formula crop -> LaTeX. Empty string on an empty/failed crop.
  [[nodiscard]] std::string recognize(const cv::Mat &bgr_crop);

  // Recognize each region of `page` (BGR8) given page-coordinate boxes; returns
  // one LaTeX string per box in input order (empty on failure / no formula).
  [[nodiscard]] std::vector<std::string>
  recognize_regions(const cv::Mat &page, const std::vector<Box> &boxes);

 private:
  OrtSession fused_;
  std::optional<FormulaTokenizer> tok_;
  bool ready_ = false;
};

}  // namespace turbo_ocr::formula
