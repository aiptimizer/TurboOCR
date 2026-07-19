#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/detection/det_postprocess.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/stage_profiler.h"

#include <opencv2/imgproc.hpp>

using namespace turbo_ocr;
using namespace turbo_ocr::detection;
using turbo_ocr::engine::CpuEngine;

namespace {
// Fuse resize-normalize-pack into one split + 3 scaled convertTo passes
// (writes straight into the NCHW buffer). Changes float rounding vs the
// per-channel MatExpr path, so it sits behind an env flag for A/B until the
// lead confirms box count + text_sig are unchanged.
bool det_fused_pre() {
  static const bool e = turbo_ocr::env::env_enabled("TURBO_DET_FUSED_PRE");
  return e;
}
} // namespace

bool CpuPaddleDet::load_model(const std::string &model_path,
                             const DetResizeParams &resize, const DbParams &db) {
  // Same per-model resize + DB-param source as the GPU detector + the TRT
  // builder; env overrides (read_det_resize/read_db_params) win over the base.
  resize_ = read_det_resize(resize);
  const DbParams eff_db = read_db_params(db);
  db_thresh_ = eff_db.thresh;
  box_thresh_ = eff_db.box_thresh;
  unclip_ratio_ = eff_db.unclip_ratio;
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

std::vector<Box> CpuPaddleDet::run(const cv::Mat &img) {
  // Fail LOUD on an unexpected channel count. The split-into-planes preprocess
  // below assumes BGR CV_8UC3 (callers expand grayscale upstream); a non-3-channel
  // input would otherwise leave the r/g planes with stale bytes and normalize a
  // silently-wrong tensor. The old memcpy path crashed here — keep the fail-loud
  // contract rather than silently corrupt OCR.
  CV_Assert(img.type() == CV_8UC3);
  const int h = img.rows;
  const int w = img.cols;
  const auto [resize_h, resize_w] = compute_det_resize(h, w, resize_);

  const int plane_size = resize_h * resize_w;
  input_data_buf_.resize(3 * plane_size);

  {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_PRE);
    cv::resize(img, resized_, cv::Size(resize_w, resize_h));

    // Channel order is BGR: the det models are exported with DecodeImage
    // img_mode: BGR, and mean/std [0.485,0.456,0.406]/[0.229,0.224,0.225]
    // apply POSITIONALLY to that BGR tensor (channel 0 = B gets 0.485).
    // This matches cuda_fused_resize_normalize_det exactly — feeding RGB
    // here measurably loses marginal detections vs the GPU path.
    constexpr float kMean0 = 0.485f, kMean1 = 0.456f, kMean2 = 0.406f;
    constexpr float kStd0 = 0.229f, kStd1 = 0.224f, kStd2 = 0.225f;
    if (det_fused_pre()) {
      // Single pass: BGR uint8 -> planar BGR float with per-channel
      // (1/(255*std)) scale and (-mean/std) shift, written into the NCHW buffer.
      cv::split(resized_, bgr_);
      cv::Mat p0(resize_h, resize_w, CV_32F, input_data_buf_.data());
      cv::Mat p1(resize_h, resize_w, CV_32F,
                 input_data_buf_.data() + plane_size);
      cv::Mat p2(resize_h, resize_w, CV_32F,
                 input_data_buf_.data() + 2 * plane_size);
      bgr_[0].convertTo(p0, CV_32F, 1.0 / (255.0 * kStd0), -kMean0 / kStd0);
      bgr_[1].convertTo(p1, CV_32F, 1.0 / (255.0 * kStd1), -kMean1 / kStd1);
      bgr_[2].convertTo(p2, CV_32F, 1.0 / (255.0 * kStd2), -kMean2 / kStd2);
    } else {
      // Default path: same two-step normalize as the original (u8/255, then the
      // folded (x-mean)/std convert), but cv::split writes straight into the
      // NCHW buffer planes and the per-channel normalize runs in place -- no
      // split temporaries, no MatExpr temporaries, no final memcpy.
      resized_.convertTo(float_img_, CV_32F, 1.0 / 255.0);
      cv::Mat p0(resize_h, resize_w, CV_32F, input_data_buf_.data());
      cv::Mat p1(resize_h, resize_w, CV_32F,
                 input_data_buf_.data() + plane_size);
      cv::Mat p2(resize_h, resize_w, CV_32F,
                 input_data_buf_.data() + 2 * plane_size);

      // cv::Mat BGR split lands positionally: ch0(B)->plane0, ch1(G)->plane1,
      // ch2(R)->plane2.
      cv::Mat planes[3] = {p0, p1, p2};
      cv::split(float_img_, planes);

      p0.convertTo(p0, CV_32F, 1.0 / kStd0, -1.0 * kMean0 / kStd0);
      p1.convertTo(p1, CV_32F, 1.0 / kStd1, -1.0 * kMean1 / kStd1);
      p2.convertTo(p2, CV_32F, 1.0 / kStd2, -1.0 * kMean2 / kStd2);
    }
    input_shape_buf_ = {1, 3, static_cast<int64_t>(resize_h),
                        static_cast<int64_t>(resize_w)};
  }

  CpuEngine::InferResult result;
  {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_INFER);
    result = engine_->infer(input_data_buf_.data(), input_shape_buf_);
  }

  if (result.data.empty())
    return {};

  turbo_ocr::prof::Scope _s(turbo_ocr::prof::DET_POST);

  // Output shape: [1, 1, resize_h, resize_w]
  cv::Mat pred_map(resize_h, resize_w, CV_32F, result.data.data());

  // Threshold to bitmap. compare(CMP_GT) yields a CV_8U 0/255 mask in one SIMD
  // pass -- identical predicate to THRESH_BINARY, no float temp / extra convert.
  cv::compare(pred_map, db_thresh_, bitmap_, cv::CMP_GT);

  // Find contours and extract boxes
  return extract_boxes_from_bitmap(
      pred_map, bitmap_, h, w, resize_h, resize_w,
      box_thresh_, unclip_ratio_, kMinBoxSide, kMinUnclippedSide,
      shifted_buf_, mask_buf_, contours_buf_, hierarchy_buf_);
}
