#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/common/geometry/box.h"

namespace turbo_ocr::detection {

/// CPU text detector using ONNX Runtime (DB post-processing).
class CpuPaddleDet {
public:
  CpuPaddleDet() = default;
  ~CpuPaddleDet() noexcept = default;

  /// Load an ONNX detection model. resize/db are this model's official
  /// PaddleOCR detection config (server::DetInferConfig fields); both default to
  /// the kDetResizeDefault/kDbDefaults base so explicit-DET-override callers and
  /// tests keep working. Env vars layered on top (read_det_resize/read_db_params)
  /// always win.
  [[nodiscard]] bool load_model(const std::string &model_path,
                                const DetResizeParams &resize = kDetResizeDefault,
                                const DbParams &db = kDbDefaults);

  // Run detection on a CPU cv::Mat image (BGR, uint8)
  [[nodiscard]] std::vector<Box> run(const cv::Mat &img);

private:
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
  static constexpr float kMinBoxSide = 3.0f;
  static constexpr float kMinUnclippedSide = 5.0f;

  std::unique_ptr<engine::CpuEngine> engine_;

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
