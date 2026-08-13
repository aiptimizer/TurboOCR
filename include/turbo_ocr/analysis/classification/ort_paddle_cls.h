#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/onnx/ort_engine.h"
#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::classification {

/// CPU angle classifier using ONNX Runtime (flips 180-degree text crops).
class OrtPaddleCls {
public:
  /// Pin the ONNX Runtime execution provider this stage's engine runs on — the
  /// FAST path of backend/engine_mode.h (the .onnx as-is on a vendor EP, no
  /// graph build). MUST be called before load_model(); with no call the engine
  /// keeps reading ORT_EP from the environment exactly as before.
  void set_ep_config(const backend::EpConfig &ep) {
    ep_ = ep;
    ep_set_ = true;
  }

  OrtPaddleCls() = default;
  ~OrtPaddleCls() noexcept = default;

  /// Load an ONNX classification model.
  [[nodiscard]] bool load_model(const std::string &model_path);

  // Classify crops and flip 180-degree boxes in-place.
  void run(const cv::Mat &img, std::vector<Box> &boxes);

private:
  // Explicit execution provider (set_ep_config); unset => env-driven engine.
  backend::EpConfig ep_{};
  bool ep_set_ = false;
  // PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25) expects
  // 80x160 input. The v4 shape (48x192) trips an ONNX Runtime shape check
  // on the crops produced by tall/narrow text lines.
  static constexpr int kClsImageH = 80;
  static constexpr int kClsImageW = 160;
  static constexpr float kClsThresh = 0.9f;

  std::unique_ptr<engine::OrtEngine> engine_;
};

} // namespace turbo_ocr::classification
