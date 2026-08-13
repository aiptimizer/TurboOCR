#include "turbo_ocr/analysis/recognition/ort_paddle_rec.h"
#include "turbo_ocr/analysis/recognition/ctc_decode.h"

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/base/geometry/perspective.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <mutex>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/base/log/stage_profiler.h"
#include "turbo_ocr/backend/engine_mode.h"

using namespace turbo_ocr::recognition;
using turbo_ocr::engine::OrtEngine;
using turbo_ocr::Box;

OrtPaddleRec::OrtPaddleRec() {
  label_list_.push_back("blank");
  rec_batch_num_ = turbo_ocr::env::env_int("REC_BATCH_N", rec_batch_num_, 1, 256);
  rec_bucket_step_ =
      turbo_ocr::env::env_int("REC_BUCKET_STEP", rec_bucket_step_, 1, kMaxRecWidth);
  rec_fixed_buckets_ = turbo_ocr::env::env_enabled("REC_FIXED_BUCKETS");
  rec_zerocopy_ = turbo_ocr::env::env_or("REC_ZEROCOPY", "1") != "0";
  rec_selftest_ = turbo_ocr::env::env_enabled("REC_SELFTEST");
}

bool OrtPaddleRec::load_model(const std::string &model_path) {
  engine_ = ep_set_ ? std::make_unique<OrtEngine>(model_path, ep_)
                    : std::make_unique<OrtEngine>(model_path);
  if (!engine_->load())
    return false;

  // Probe output dims at max width to get the true seq_len
  std::vector<int64_t> probe_shape = {1, 3, static_cast<int64_t>(rec_image_h_),
                                       static_cast<int64_t>(kMaxRecWidth)};
  engine_->probe_output_dims(probe_shape, actual_seq_len_, actual_num_classes_);
  TOCR_LOG_INFO("OrtPaddleRec output dims probed", "seq_len", actual_seq_len_,
                "num_classes", actual_num_classes_);

  // Device-class EPs (CUDA / DirectML / MIGraphX — the same set whose fp16
  // lives in the model file) pay a fixed per-Run launch+copy+sync cost that
  // dwarfs a tiny rec forward. The compiled-in defaults (scalar, fine width
  // steps) are CPU-tuned, and nothing in the tree ever set REC_BATCH_N — so
  // on those EPs recognition ran ONE ORT Run PER CROP. Measured on an
  // RTX 5090 (FUNSD page, 38 crops): 412 ms/page scalar vs 128 ms with the
  // 9-bucket ladder + batch 32. Bump the DEFAULTS on device EPs; an explicit
  // env override still wins, and the CPU-class defaults are untouched so the
  // FUNSD-gated CPU path is byte-for-byte the configuration that was gated.
  if (backend::fp16_support_for(engine_->provider()) ==
      backend::Fp16Support::Model) {
    if (turbo_ocr::env::env_or("REC_BATCH_N", "").empty())
      rec_batch_num_ = 32;
    if (turbo_ocr::env::env_or("REC_FIXED_BUCKETS", "").empty())
      rec_fixed_buckets_ = true;
  }
  return true;
}

bool OrtPaddleRec::load_dict(const std::string &dict_path) {
  if (!load_label_dict(dict_path, label_list_))
    return false;
  // load_model probed the engine before load_dict (cpu_ocr_pipeline.cpp), so
  // actual_num_classes_ holds the real output width here, not the placeholder.
  // A dict whose [blank]+chars+space count differs from it silently maps every
  // class to the wrong glyph — fail loud at boot instead.
  const int probed_width = actual_num_classes_;
  if (static_cast<int>(label_list_.size()) != probed_width)
    throw turbo_ocr::ModelLoadError(std::format(
        "Recognition dict/model mismatch: {} produced {} classes but model "
        "output width is {} (expected blank+chars+space == width)",
        dict_path, label_list_.size(), probed_width));
  return true;
}

int OrtPaddleRec::rec_target_width(const Box &box) const {
  return rec_input_width(box, rec_image_h_);
}

void OrtPaddleRec::preprocess_box(const cv::Mat &img, const Box &box,
                                  int target_w, std::vector<float> &buffer) {
  // target_w >= the natural content width by construction (rec_target_width),
  // so ct.crop_width here is that same content width.
  const auto ct =
      turbo_ocr::compute_crop_transform(box, rec_image_h_, target_w);
  const int content_w = ct.crop_width;

  // Single warp from the ORIGINAL image to the recognizer input size — the
  // same dst->src mapping the GPU batch_roi_warp kernel evaluates.
  // WARP_INVERSE_MAP: ct.M_inv maps destination pixels back to source
  // coordinates; BORDER_REPLICATE == the kernel's source clamp.
  const cv::Matx33f m_inv(ct.M_inv[0], ct.M_inv[1], ct.M_inv[2],
                          ct.M_inv[3], ct.M_inv[4], ct.M_inv[5],
                          ct.M_inv[6], ct.M_inv[7], ct.M_inv[8]);
  cv::warpPerspective(img, warped_, m_inv, cv::Size(content_w, rec_image_h_),
                      cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                      cv::BORDER_REPLICATE);

  // Normalize to [-1, 1]: pixel/127.5 - 1.0
  warped_.convertTo(float_crop_, CV_32F, 1.0 / 127.5, -1.0);

  // NCHW RGB planes of width target_w; columns [content_w, target_w) stay 0
  // (mid-gray in normalized space), matching the GPU kernel's padding.
  const size_t plane = static_cast<size_t>(rec_image_h_) * target_w;
  buffer.assign(3 * plane, 0.0f);
  cv::Mat channels[3];
  cv::split(float_crop_, channels);
  // RGB order (R=channels[2], G=channels[1], B=channels[0])
  for (int c = 0; c < 3; ++c) {
    const cv::Mat &src = channels[2 - c];
    for (int r = 0; r < rec_image_h_; ++r)
      std::memcpy(buffer.data() + c * plane +
                      static_cast<size_t>(r) * target_w,
                  src.ptr<float>(r),
                  static_cast<size_t>(content_w) * sizeof(float));
  }
}

std::vector<std::pair<std::string, float>>
OrtPaddleRec::run(const cv::Mat &img, const std::vector<Box> &boxes) {
  if (rec_batch_num_ > 1)
    return run_batched(img, boxes);

  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty())
    return results;

  dropped_crops_ = 0;
  results.resize(boxes.size());

  // Process each box one at a time
  std::vector<float> input_buf;
  std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(rec_image_h_), 0};

  namespace prof = turbo_ocr::prof;
  for (size_t i = 0; i < boxes.size(); i++) {
    int target_w;
    {
      prof::Scope _s(prof::REC_PRE);
      target_w = rec_target_width(boxes[i]);
      preprocess_box(img, boxes[i], target_w, input_buf);
    }

    input_shape[3] = static_cast<int64_t>(target_w);
    OrtEngine::InferResult result;
    {
      prof::Scope _s(prof::REC_INFER);
      result = engine_->infer(input_buf.data(), input_shape);
    }

    if (result.shape.size() >= 3) {
      prof::Scope _s(prof::REC_DECODE);
      int seq_len = static_cast<int>(result.shape[1]);
      int num_classes = static_cast<int>(result.shape[2]);
      results[i] = ctc_greedy_decode_raw(result.data.data(), seq_len, num_classes, label_list_);
    } else {
      // Same no-silent-failure rule as the batched path below.
      ++dropped_crops_;
      TOCR_LOG_WARN_RL("cpu rec: engine returned a rank<3 output for a crop; "
                       "dropping it (it is not blank)");
    }
  }

  return results;
}

std::vector<std::pair<std::string, float>>
OrtPaddleRec::run_batched(const cv::Mat &img, const std::vector<Box> &boxes) {
  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty())
    return results;
  dropped_crops_ = 0;
  results.resize(boxes.size());

  namespace prof = turbo_ocr::prof;
  const int total = static_cast<int>(boxes.size());

  // Diagnostic self-test (REC_SELFTEST=1): runs ONE crop duplicated into a
  // 2-row batch through both the zero-copy view and the copy path. If a row pair
  // MISMATCHes, multi-row batch output is being mis-handled (view bug if only
  // "view" mismatches; model/decode bug if both do).
  if (rec_selftest_) {
    static std::once_flag selftest_flag;
    std::call_once(selftest_flag, [&] {
      int tw = rec_target_width(boxes[0]);
      std::vector<float> one;
      preprocess_box(img, boxes[0], tw, one);
      const size_t re = static_cast<size_t>(3) * rec_image_h_ * tw;
      std::vector<float> two(2 * re);
      std::memcpy(two.data(), one.data(), re * sizeof(float));
      std::memcpy(two.data() + re, one.data(), re * sizeof(float));
      std::vector<int64_t> sh = {2, 3, static_cast<int64_t>(rec_image_h_),
                                 static_cast<int64_t>(tw)};
      auto log_pair = [&](const float *base, int sl, int nc, const char *tag) {
        const size_t ro = static_cast<size_t>(sl) * nc;
        auto r0 = ctc_greedy_decode_raw(base, sl, nc, label_list_);
        auto r1 = ctc_greedy_decode_raw(base + ro, sl, nc, label_list_);
        TOCR_LOG_INFO("rec selftest", "path", tag, "row0", r0.first, "row1",
                      r1.first, "verdict",
                      r0.first == r1.first ? "MATCH" : "MISMATCH");
      };
      auto v = engine_->infer_batch_view(two.data(), sh);
      if (v.shape.size() >= 3)
        log_pair(v.data, static_cast<int>(v.shape[1]),
                 static_cast<int>(v.shape[2]), "view");
      auto rc = engine_->infer_batch(two.data(), sh);
      if (rc.shape.size() >= 3)
        log_pair(rc.data.data(), static_cast<int>(rc.shape[1]),
                 static_cast<int>(rc.shape[2]), "copy");
    });
  }

  // Phase 1: compute each box's natural target width (identical math to the
  // scalar path) and snap it UP to a fine width bucket
  // (ceil(target_w / step) * step). Crops are grouped strictly by bucket so a
  // batch only ever holds crops within one step of each other -> padding per
  // crop is at most step-1, regardless of the global width distribution. (A
  // plain consecutive-N grouping after sorting over-pads crops at a bucket
  // boundary, e.g. several 320px crops batched with a 900px one.) orig_idx
  // restores box order.
  struct BatchCrop {
    int orig_idx = 0;
    int target_w = 0;
    int bucket_w = 0;
  };
  std::vector<BatchCrop> crops(total);
  {
    prof::Scope _s(prof::REC_PRE);
    for (int i = 0; i < total; i++) {
      int target_w = rec_target_width(boxes[i]);
      int bucket_w = rec_fixed_buckets_ ? snap_width_bucket(target_w)
                                        : snap_width_step(target_w, rec_bucket_step_);
      crops[i] = {i, target_w, bucket_w};
    }
  }

  std::ranges::sort(crops, {}, &BatchCrop::bucket_w);

  // Phase 2: one ORT Run per (bucket, batch) group of <= rec_batch_num_ crops,
  // all sharing the same bucket width. Each crop keeps its own content width
  // (bit-identical preprocessing to the scalar path) and is right-padded with
  // zeros up to the shared bucket width (<= step-1 of padding). Zero in
  // normalized space (== mid-gray 127.5 in pixel space) matches the padding the
  // rec model was exported/trained with, so padded columns decode to blank.
  const size_t plane = static_cast<size_t>(rec_image_h_);
  int beg = 0;
  while (beg < total) {
    const int pad_w = crops[beg].bucket_w;
    int end = beg;
    while (end < total && end - beg < rec_batch_num_ &&
           crops[end].bucket_w == pad_w)
      end++;
    const int cur = end - beg;

    const size_t row_elems = static_cast<size_t>(3) * rec_image_h_ * pad_w;
    {
      prof::Scope _s(prof::REC_PRE);
      batch_buf_.assign(static_cast<size_t>(cur) * row_elems, 0.0f);
      for (int j = 0; j < cur; j++) {
        const auto &bc = crops[beg + j];
        preprocess_box(img, boxes[bc.orig_idx], bc.target_w, scratch_chw_);
        float *dst = batch_buf_.data() + static_cast<size_t>(j) * row_elems;
        const float *src = scratch_chw_.data();
        const size_t copy_bytes = static_cast<size_t>(bc.target_w) * sizeof(float);
        for (int c = 0; c < 3; c++) {
          for (int r = 0; r < rec_image_h_; r++) {
            std::memcpy(dst + (static_cast<size_t>(c) * plane + r) * pad_w,
                        src + (static_cast<size_t>(c) * plane + r) * bc.target_w,
                        copy_bytes);
          }
        }
      }
    }

    std::vector<int64_t> shape = {cur, 3, static_cast<int64_t>(rec_image_h_),
                                  static_cast<int64_t>(pad_w)};
    // Default zero-copy: the view points into ORT-owned memory, valid until the
    // next infer call (we fully decode this batch first). REC_ZEROCOPY=0 falls
    // back to the proven copy path for A/B isolation. result/view are declared
    // here so the backing buffer stays alive through the decode loop.
    const float *logits_base = nullptr;
    int seq_len = 0;
    int num_classes = 0;
    OrtEngine::InferResult result;
    OrtEngine::InferView view;
    {
      prof::Scope _s(prof::REC_INFER);
      if (rec_zerocopy_) {
        view = engine_->infer_batch_view(batch_buf_.data(), shape);
        if (view.shape.size() >= 3) {
          logits_base = view.data;
          seq_len = static_cast<int>(view.shape[1]);
          num_classes = static_cast<int>(view.shape[2]);
        }
      } else {
        result = engine_->infer_batch(batch_buf_.data(), shape);
        if (result.shape.size() >= 3) {
          logits_base = result.data.data();
          seq_len = static_cast<int>(result.shape[1]);
          num_classes = static_cast<int>(result.shape[2]);
        }
      }
    }

    if (logits_base) {
      prof::Scope _s(prof::REC_DECODE);
      const size_t row_out = static_cast<size_t>(seq_len) * num_classes;
      // Decode the rows the engine RETURNED, not the rows we requested. An
      // execution provider may resolve a dynamic batch dim to a compiled
      // shape and hand back fewer rows (the detector documents the same EP
      // liberty for its canvas, ort_paddle_det.cpp) — iterating to `cur`
      // would read past the logits buffer and emit plausible garbage text
      // for real crops. Shorted slots keep their pre-sized {"",0} and are
      // counted through the one silent-loss channel.
      const int64_t rows_ret = rec_zerocopy_
                                   ? (view.shape.empty() ? 0 : view.shape[0])
                                   : (result.shape.empty() ? 0 : result.shape[0]);
      const int n_dec = static_cast<int>(
          std::min<int64_t>(cur, std::max<int64_t>(rows_ret, 0)));
      for (int j = 0; j < n_dec; j++) {
        const float *logits = logits_base + static_cast<size_t>(j) * row_out;
        results[crops[beg + j].orig_idx] =
            ctc_greedy_decode_raw(logits, seq_len, num_classes, label_list_);
      }
      if (n_dec < cur) {
        dropped_crops_ += cur - n_dec;
        TOCR_LOG_WARN_RL("cpu rec: engine returned fewer batch rows than "
                         "requested; dropping the shorted crops",
                         "requested", cur, "returned", n_dec);
      }
    } else {
      // NO-SILENT-FAILURE. results was pre-sized, so this chunk stays {"",0} and
      // the returned LENGTH still equals boxes.size() — the pipeline's
      // under-return check structurally cannot see it. Without this counter a
      // failed chunk is indistinguishable from a genuinely blank region.
      //
      // Every per-vendor recognizer already does this (paddle_rec.cpp,
      // rocm_stages.cpp, intel_stages.cpp, mps_stages.mm). THIS is the one
      // shared by all five vendors on --engine-mode onnx, and it was the only
      // one missing it — the project rule inverted: policy implemented four
      // times per-backend and absent from the single shared impl.
      dropped_crops_ += cur;
      TOCR_LOG_WARN_RL("cpu rec: engine produced no usable logits; dropping "
                       "crops (they are not blank)",
                       "dropped", static_cast<long long>(cur));
    }

    beg = end;
  }

  return results;
}
