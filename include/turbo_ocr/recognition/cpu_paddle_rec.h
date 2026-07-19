#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/recognition/rec_geometry.h"

namespace turbo_ocr::recognition {

/// CPU text recognizer using ONNX Runtime (CRNN + CTC decoding).
class CpuPaddleRec {
public:
  CpuPaddleRec();
  ~CpuPaddleRec() noexcept = default;

  /// Load an ONNX recognition model and probe output dimensions.
  [[nodiscard]] bool load_model(const std::string &model_path);
  /// Load the character dictionary for CTC decoding.
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  // Run recognition on image crops defined by boxes.
  // img is the original full image (BGR, uint8).
  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run(const cv::Mat &img, const std::vector<Box> &boxes);

private:
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
  // Diagnostics (A/B isolation). REC_ZEROCOPY=0 uses the copy infer_batch()
  // instead of the zero-copy view; REC_SELFTEST=1 runs a batch-of-2-identical
  // crops check once to verify multi-row batches decode consistently.
  bool rec_zerocopy_ = true;
  bool rec_selftest_ = false;

  std::unique_ptr<engine::CpuEngine> engine_;

  // kMaxRecWidth lives in rec_geometry.h (shared with the TRT recognizer).

  // Probed after load via probe_output_dims. The initializers are placeholders
  // only: actual_num_classes_ must stay >= the widest tier (medium/small CTC
  // width 18,710) until the probe overwrites it with the true width.
  int actual_seq_len_ = 600;
  int actual_num_classes_ = 20000;

  // Preprocess a crop into NCHW float buffer
  void preprocess_crop(const cv::Mat &crop, int target_w,
                       std::vector<float> &buffer);

  // Batched path: bucket crops by rounded width, one ORT Run per bucket batch.
  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run_batched(const cv::Mat &img, const std::vector<Box> &boxes);

  // Reused across batches to avoid per-call heap churn.
  std::vector<float> batch_buf_;   // {B,3,48,pad_w}
  std::vector<float> scratch_chw_; // {3,48,target_w} for one crop
};

} // namespace turbo_ocr::recognition
