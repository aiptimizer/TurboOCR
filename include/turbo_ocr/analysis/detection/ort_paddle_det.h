#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/onnx/ort_engine.h"
#include "turbo_ocr/analysis/detection/det_config.h"
#include "turbo_ocr/core/db_post_config.h" // shared kMinBoxSide/kMinUnclippedSide
#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::detection {

/// CPU text detector using ONNX Runtime (DB post-processing).
class OrtPaddleDet {
public:
  /// Pin the ONNX Runtime execution provider this stage's engine runs on — the
  /// FAST path of backend/engine_mode.h (the .onnx as-is on a vendor EP, no
  /// graph build). MUST be called before load_model(); with no call the engine
  /// keeps reading ORT_EP from the environment exactly as before.
  void set_ep_config(const backend::EpConfig &ep) {
    ep_ = ep;
    ep_set_ = true;
  }

  OrtPaddleDet() = default;
  ~OrtPaddleDet() noexcept = default;

  /// Load an ONNX detection model. resize/db are this model's official
  /// PaddleOCR detection config (server::DetInferConfig fields); both default
  /// to the bootstrap-installed per-model base (det_config.h
  /// set_det_config_base — the tier's registry row), so a no-arg load runs
  /// the tier's real thresholds. Env vars layered on top
  /// (read_det_resize/read_db_params) always win.
  [[nodiscard]] bool load_model(const std::string &model_path,
                                const DetResizeParams &resize = det_resize_base(),
                                const DbParams &db = det_db_base());

  // Run detection on a CPU cv::Mat image (BGR, uint8)
  [[nodiscard]] std::vector<Box> run(const cv::Mat &img);

private:
  // Explicit execution provider (set_ep_config); unset => env-driven engine.
  backend::EpConfig ep_{};
  bool ep_set_ = false;
  // Per-model resize policy (this model's official config + env overrides).
  // Set from read_det_resize(cfg) in load_model(); drives compute_det_resize()
  // in run(). Kept in lockstep with the GPU detector so the two paths agree.
  DetResizeParams resize_ = kDetResizeDefault;

  // DB post-processing parameters (PP-OCRv6 defaults). Set from
  // detection/det_config.h read_db_params() in load_model(); env-overridable
  // via DET_DB_THRESH/DET_BOX_THRESH/DET_UNCLIP.
  float db_thresh_ = kDbDefaults.thresh;
  float box_thresh_ = kDbDefaults.box_thresh;
  float unclip_ratio_ = kDbDefaults.unclip_ratio;
  // Side limits come from the SHARED pair in core/db_post_config.h (same
  // namespace, so unqualified uses resolve there). This class used to carry
  // its own literal copies — the worst possible place for a silent fork,
  // because this detector is the CPU REFERENCE every backend is golden-diffed
  // against: a change to the shared limits would have left the reference on
  // the stale values and made every backend "diverge" from a wrong baseline.

  std::unique_ptr<engine::OrtEngine> engine_;

  // Reusable buffers (avoid per-call heap allocation)
  std::vector<cv::Point> shifted_buf_;
  cv::Mat mask_buf_;
  std::vector<std::vector<cv::Point>> contours_buf_;
  std::vector<cv::Vec4i> hierarchy_buf_;
  std::vector<float> input_data_buf_;
  std::vector<int64_t> input_shape_buf_{1, 3, 0, 0};

  // Preprocess scratch (reused across calls). resized_/float_img_ are reused by
  // the default normalize path; bgr_ planes by the fused path; bitmap_ by post.
  cv::Mat resized_;
  cv::Mat float_img_;
  cv::Mat bgr_[3];
  cv::Mat bitmap_;

};

} // namespace turbo_ocr::detection
