#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/onnx/ort_engine.h"
#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/analysis/recognition/rec_geometry.h"

namespace turbo_ocr::recognition {

/// CPU text recognizer using ONNX Runtime (CRNN + CTC decoding).
class OrtPaddleRec {
public:
  /// Pin the ONNX Runtime execution provider this stage's engine runs on — the
  /// FAST path of backend/engine_mode.h (the .onnx as-is on a vendor EP, no
  /// graph build). MUST be called before load_model(); with no call the engine
  /// keeps reading ORT_EP from the environment exactly as before.
  void set_ep_config(const backend::EpConfig &ep) {
    ep_ = ep;
    ep_set_ = true;
  }

  OrtPaddleRec();
  ~OrtPaddleRec() noexcept = default;

  /// Load an ONNX recognition model and probe output dimensions.
  [[nodiscard]] bool load_model(const std::string &model_path);
  /// Load the character dictionary for CTC decoding.
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  // Run recognition on image crops defined by boxes.
  // img is the original full image (BGR, uint8).
  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run(const cv::Mat &img, const std::vector<Box> &boxes);

  /// Crops this recognizer FAILED on in the last run — engine produced no
  /// usable logits, so the entry stayed {"",0}. results is pre-sized, so the
  /// returned length always equals boxes.size() and the pipeline's under-return
  /// check cannot see the loss; this is how it surfaces. Mirrors
  /// PaddleRec/RocmRecognizer/IntelRecognizer/MpsRecognizer.
  [[nodiscard]] int last_dropped_crops() const noexcept { return dropped_crops_; }

private:
  int dropped_crops_ = 0;
  // Explicit execution provider (set_ep_config); unset => env-driven engine.
  backend::EpConfig ep_{};
  bool ep_set_ = false;
  std::vector<std::string> label_list_;
  int rec_image_h_ = 48;
  int rec_image_w_ = 320;

  // Batched recognition (env REC_BATCH_N; 1 = scalar path). Each crop's width is
  // snapped up to a fine bucket (ceil(target_w / rec_bucket_step_) * step), and
  // a batch only holds crops from the SAME bucket, so padding per crop is always
  // <= step-1 regardless of the width spread. Smaller REC_BUCKET_STEP -> less
  // padding (better accuracy) but more distinct ORT shapes / smaller batches.
  int rec_batch_num_ = 1;
  int rec_bucket_step_ = 16;
  // REC_FIXED_BUCKETS=1: snap crop widths to the fixed kRecWidthBuckets table
  // ({320,480,...,4000}, 9 buckets) instead of the fine step-16 grid, so a whole
  // image's crops collapse into <=9 STATIC shapes. Fewer distinct shapes => real
  // multi-row batches on MLAS and, crucially, static shapes an accelerator EP
  // (CoreML/ANE) can compile once and dispatch few times. Trades extra padding
  // for far fewer, shape-stable Runs. Off by default (step-16 path unchanged).
  bool rec_fixed_buckets_ = false;
  // Diagnostics (A/B isolation). REC_ZEROCOPY=0 uses the copy infer_batch()
  // instead of the zero-copy view; REC_SELFTEST=1 runs a batch-of-2-identical
  // crops check once to verify multi-row batches decode consistently.
  bool rec_zerocopy_ = true;
  bool rec_selftest_ = false;

  std::unique_ptr<engine::OrtEngine> engine_;

  // kMaxRecWidth lives in rec_geometry.h (shared with the TRT recognizer).

  // Probed after load via probe_output_dims. The initializers are placeholders
  // only: actual_num_classes_ must stay >= the widest tier (medium/small CTC
  // width 18,710) until the probe overwrites it with the true width.
  int actual_seq_len_ = 600;
  int actual_num_classes_ = 20000;

  // Recognizer input width for a box: natural content width (after the
  // vertical-text swap) floored at kMinRecWidth, capped at kMaxRecWidth.
  [[nodiscard]] int rec_target_width(const Box &box) const;

  // Warp one detection quad from the FULL image straight into a NCHW float
  // buffer of width target_w: content columns rendered by a single
  // perspective warp, remaining columns zero-padded (mid-gray). One resample,
  // mirroring the GPU batch_roi_warp kernel — the old crop-then-resize path
  // resampled twice and stretched sub-kMinRecWidth crops, destroying small
  // glyphs.
  void preprocess_box(const cv::Mat &img, const Box &box, int target_w,
                      std::vector<float> &buffer);

  // Batched path: bucket crops by rounded width, one ORT Run per bucket batch.
  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run_batched(const cv::Mat &img, const std::vector<Box> &boxes);

  // Reused across batches to avoid per-call heap churn.
  std::vector<float> batch_buf_;   // {B,3,48,pad_w}
  std::vector<float> scratch_chw_; // {3,48,target_w} for one crop
  cv::Mat warped_;                 // preprocess_box warp output (u8 BGR)
  cv::Mat float_crop_;             // preprocess_box normalized float crop
};

} // namespace turbo_ocr::recognition
