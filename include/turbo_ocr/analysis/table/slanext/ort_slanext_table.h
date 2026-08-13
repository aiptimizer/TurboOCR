#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/types.h"  // OCRResultItem
#include "turbo_ocr/core/router_types.h"
#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/slanext/slanext_host_decode.h"

namespace turbo_ocr::formula { }  // fwd-keep ordering

namespace turbo_ocr::backend {
class DeviceQueue;
class IRecognizer;
struct ImageView;
}  // namespace turbo_ocr::backend

namespace turbo_ocr::table {

// CUDA-FREE SLANeXt encoder-split table backend. The CPU sibling of
// SlanextEncSplit: a CPU ResizeByLong-488 + ImageNet-norm + pad preprocess
// (replacing cuda_fused_slanext_pre_rgb) feeds an ORT CPUExecutionProvider
// encoder session; the existing host GRU+attention decoder (slanext_host_decode)
// produces the structure tokens + per-cell quads. No CUDA, no TensorRT.
class OrtSlanextEncoder {
 public:
  OrtSlanextEncoder() = default;
  ~OrtSlanextEncoder();

  OrtSlanextEncoder(const OrtSlanextEncoder &) = delete;
  OrtSlanextEncoder &operator=(const OrtSlanextEncoder &) = delete;

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

// Full CPU table recognizer: per region, OrtSlanextEncoder -> structure +
// quads, then geometry-match the page text-OCR into cells + reconstruct HTML.
// Mirrors SlanextTableRecognizer but on host cv::Mat inputs.
//
// Not only CpuBackend's: AppleBackend and IntelBackend also answer a local
// slanext spec with a cpu::CpuTableRecognizer around this class (a
// device-resident SLANeXt is a TODO on both), so nothing here may be typed on a
// CPU-only stage class — the rest of the pipeline around it is Metal/OpenVINO.
class OrtSlanextTableRecognizer {
 public:
  // Reads the same env knobs as the GPU path: TABLE_SLANEXT_ENCODER_ONNX
  // (+ derived decoder/dict), TABLE_SLANEXT_DICT. Returns false (tables
  // disabled) when unconfigured/unloadable.
  [[nodiscard]] bool load();
  [[nodiscard]] bool ready() const noexcept { return wired_ != nullptr; }

  // Optional per-cell crop OCR for grid cells the page detector left empty.
  // Not owned; nullptr disables cell fill.
  //
  // Typed on the SEAM interface, never on a concrete recognizer: when this took
  // a recognition::OrtPaddleRec*, the CpuTableRecognizer adapter could only pass
  // one on CpuBackend, so on Apple/Intel the hook stayed null and every cell the
  // page text detector under-segmented came back empty with no signal — the very
  // loss the crop fill exists to recover.
  void set_cell_recognizer(backend::IRecognizer *r) noexcept { cell_rec_ = r; }

  // One TableResult per region (layout_id left -1; stamped by the caller).
  //
  // `page` is the host BGR8 page the ORT-CPU encoder reads. `page_view` is that
  // SAME page in the ACTIVE backend's address space (on CpuBackend the two alias
  // the same bytes; on Apple it is the Shared MTLBuffer, host-readable under UMA
  // but also the key the Metal kernels map back to their sampler texture). The
  // cell recognizer belongs to the active backend, so it must be handed
  // `page_view` — a host-only alias would not resolve on its device. `queue`
  // orders that recognizer's work; the encoder path here is synchronous host ORT
  // and does not touch it.
  [[nodiscard]] std::vector<router::TableResult>
  run(const cv::Mat &page, const backend::ImageView &page_view,
      const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, backend::DeviceQueue &queue);

 private:
  std::unique_ptr<OrtSlanextEncoder> wired_;
  backend::IRecognizer *cell_rec_ = nullptr;  // not owned
};

}  // namespace turbo_ocr::table
