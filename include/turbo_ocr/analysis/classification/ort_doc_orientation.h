#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "turbo_ocr/onnx/ort_engine.h"

namespace turbo_ocr::classification {

/// CPU document-orientation classifier (PP-LCNet_x1_0_doc_ori, ONNX Runtime).
/// Mirror of DocOrientation for the CPU-only build. Detects a page's
/// clockwise rotation ∈ {0,90,180,270}.
class OrtDocOrientation {
public:
  /// Pin the ONNX Runtime execution provider this stage's engine runs on — the
  /// FAST path of backend/engine_mode.h (the .onnx as-is on a vendor EP, no
  /// graph build). MUST be called before load_model(); with no call the engine
  /// keeps reading ORT_EP from the environment exactly as before.
  void set_ep_config(const backend::EpConfig &ep) {
    ep_ = ep;
    ep_set_ = true;
  }

  OrtDocOrientation() = default;
  ~OrtDocOrientation() noexcept = default;

  [[nodiscard]] bool load_model(const std::string &model_path);

  /// Page's detected clockwise rotation (0/90/180/270), or 0 if unavailable.
  [[nodiscard]] int detect(const cv::Mat &bgr);

private:
  // Explicit execution provider (set_ep_config); unset => env-driven engine.
  backend::EpConfig ep_{};
  bool ep_set_ = false;
  std::unique_ptr<engine::OrtEngine> engine_;
};

} // namespace turbo_ocr::classification
