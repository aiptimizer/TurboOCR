#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/types.h"  // OCRResultItem
#include "turbo_ocr/router/router_types.h"
#include "turbo_ocr/table/slanext/slanext_dict.h"
#include "turbo_ocr/table/slanext/slanext_host_decode.h"

namespace turbo_ocr::formula { }  // fwd-keep ordering

namespace turbo_ocr::recognition { class CpuPaddleRec; }

namespace turbo_ocr::table {

// CUDA-FREE SLANeXt encoder-split table backend. The CPU sibling of
// SlanextEncSplit: a CPU ResizeByLong-488 + ImageNet-norm + pad preprocess
// (replacing cuda_fused_slanext_pre_rgb) feeds an ORT CPUExecutionProvider
// encoder session; the existing host GRU+attention decoder (slanext_host_decode)
// produces the structure tokens + per-cell quads. No CUDA, no TensorRT.
class CpuSlanextEncoder {
 public:
  CpuSlanextEncoder() = default;
  ~CpuSlanextEncoder();

  CpuSlanextEncoder(const CpuSlanextEncoder &) = delete;
  CpuSlanextEncoder &operator=(const CpuSlanextEncoder &) = delete;

  // Load the encoder ONNX (ORT-CPU), the decoder weight blob, and the dict.
  [[nodiscard]] bool load(const std::string &encoder_onnx,
                          const std::string &decoder_bin,
                          const std::string &dict_path);

  // Encode + host-decode one table region (page-coordinate box) of a BGR8 page.
  [[nodiscard]] StructureResult infer(const cv::Mat &page, const Box &region);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  SlanextDecoderWeights weights_;
  CharDict dict_;
};

// Full CPU table recognizer: per region, CpuSlanextEncoder -> structure +
// quads, then geometry-match the page text-OCR into cells + reconstruct HTML.
// Mirrors SlanextTableRecognizer but on host cv::Mat inputs.
class CpuSlanextTableRecognizer {
 public:
  // Reads the same env knobs as the GPU path: TABLE_SLANEXT_ENCODER_ONNX
  // (+ derived decoder/dict), TABLE_SLANEXT_DICT. Returns false (tables
  // disabled) when unconfigured/unloadable.
  [[nodiscard]] bool load();
  [[nodiscard]] bool ready() const noexcept { return wired_ != nullptr; }

  // Optional per-cell crop OCR for grid cells the page detector left empty.
  void set_cell_recognizer(recognition::CpuPaddleRec *r) noexcept { cell_rec_ = r; }

  // One TableResult per region (layout_id left -1; stamped by the caller).
  [[nodiscard]] std::vector<router::TableResult>
  run(const cv::Mat &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr);

 private:
  std::unique_ptr<CpuSlanextEncoder> wired_;
  recognition::CpuPaddleRec *cell_rec_ = nullptr;  // not owned
};

}  // namespace turbo_ocr::table
