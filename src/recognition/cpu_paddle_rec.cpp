#include "turbo_ocr/recognition/cpu_paddle_rec.h"
#include "turbo_ocr/recognition/crop_utils.h"
#include "turbo_ocr/recognition/ctc_decode.h"

#include "turbo_ocr/common/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/stage_profiler.h"

using namespace turbo_ocr::recognition;
using turbo_ocr::engine::CpuEngine;
using turbo_ocr::Box;

CpuPaddleRec::CpuPaddleRec() {
  label_list_.push_back("blank");
  if (const char *env = std::getenv("REC_BATCH_N")) {
    int n = std::atoi(env);
    if (n > 1)
      rec_batch_num_ = n;
  }
  if (const char *env = std::getenv("REC_BUCKET_STEP")) {
    int s = std::atoi(env);
    if (s >= 1)
      rec_bucket_step_ = s;
  }
  if (const char *env = std::getenv("REC_ZEROCOPY"))
    rec_zerocopy_ = (std::strcmp(env, "0") != 0);
  if (const char *env = std::getenv("REC_SELFTEST"))
    rec_selftest_ = (std::strcmp(env, "1") == 0);
}

bool CpuPaddleRec::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  if (!engine_->load())
    return false;

  // Probe output dims at max width to get the true seq_len
  std::vector<int64_t> probe_shape = {1, 3, static_cast<int64_t>(rec_image_h_),
                                       static_cast<int64_t>(kMaxRecWidth)};
  engine_->probe_output_dims(probe_shape, actual_seq_len_, actual_num_classes_);
  std::cout << std::format("[CpuPaddleRec] Output dims: seq_len={} num_classes={}",
                          actual_seq_len_, actual_num_classes_) << '\n';
  return true;
}

bool CpuPaddleRec::load_dict(const std::string &dict_path) {
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

void CpuPaddleRec::preprocess_crop(const cv::Mat &crop, int target_w,
                                    std::vector<float> &buffer) {
  // Resize to (rec_image_h_, target_w)
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(target_w, rec_image_h_));

  // Convert to float, normalize to [-1, 1]: pixel/127.5 - 1.0
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 127.5, -1.0);

  // Convert BGR to RGB and to NCHW layout
  int plane_size = rec_image_h_ * target_w;
  buffer.resize(3 * plane_size);

  cv::Mat channels[3];
  cv::split(float_img, channels);

  // RGB order (R=channels[2], G=channels[1], B=channels[0])
  std::memcpy(buffer.data(), channels[2].data, plane_size * sizeof(float));
  std::memcpy(buffer.data() + plane_size, channels[1].data,
              plane_size * sizeof(float));
  std::memcpy(buffer.data() + 2 * plane_size, channels[0].data,
              plane_size * sizeof(float));
}

std::vector<std::pair<std::string, float>>
CpuPaddleRec::run(const cv::Mat &img, const std::vector<Box> &boxes) {
  if (rec_batch_num_ > 1)
    return run_batched(img, boxes);

  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty())
    return results;

  results.resize(boxes.size());

  // Process each box one at a time
  std::vector<float> input_buf;
  std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(rec_image_h_), 0};

  namespace prof = turbo_ocr::prof;
  for (size_t i = 0; i < boxes.size(); i++) {
    int target_w;
    {
      prof::Scope _s(prof::REC_PRE);
      cv::Mat cropped = get_rotate_crop_image(img, boxes[i]);

      float ar = (cropped.rows > 0)
                     ? static_cast<float>(cropped.cols) / cropped.rows
                     : 0;
      target_w = natural_rec_width(ar, rec_image_h_, rec_image_w_);

      preprocess_crop(cropped, target_w, input_buf);
    }

    input_shape[3] = static_cast<int64_t>(target_w);
    CpuEngine::InferResult result;
    {
      prof::Scope _s(prof::REC_INFER);
      result = engine_->infer(input_buf.data(), input_shape);
    }

    if (result.shape.size() >= 3) {
      prof::Scope _s(prof::REC_DECODE);
      int seq_len = static_cast<int>(result.shape[1]);
      int num_classes = static_cast<int>(result.shape[2]);
      results[i] = ctc_greedy_decode_raw(result.data.data(), seq_len, num_classes, label_list_);
    }
  }

  return results;
}

std::vector<std::pair<std::string, float>>
CpuPaddleRec::run_batched(const cv::Mat &img, const std::vector<Box> &boxes) {
  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty())
    return results;
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
      cv::Mat c0 = get_rotate_crop_image(img, boxes[0]);
      float ar = (c0.rows > 0) ? static_cast<float>(c0.cols) / c0.rows : 0;
      int tw = natural_rec_width(ar, rec_image_h_, rec_image_w_);
      std::vector<float> one;
      preprocess_crop(c0, tw, one);
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
        std::cout << std::format("[SELFTEST] {} row0='{}' row1='{}' {}", tag,
                                 r0.first, r1.first,
                                 r0.first == r1.first ? "MATCH" : "MISMATCH")
                  << '\n';
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

  // Phase 1: crop each box once, compute its natural target width (identical
  // math to the scalar path) and snap it UP to a fine width bucket
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
    cv::Mat crop;
  };
  std::vector<BatchCrop> crops(total);
  {
    prof::Scope _s(prof::REC_PRE);
    for (int i = 0; i < total; i++) {
      cv::Mat cropped = get_rotate_crop_image(img, boxes[i]);
      float ar = (cropped.rows > 0)
                     ? static_cast<float>(cropped.cols) / cropped.rows
                     : 0;
      int target_w = natural_rec_width(ar, rec_image_h_, rec_image_w_);
      int bucket_w = snap_width_step(target_w, rec_bucket_step_);
      crops[i] = {i, target_w, bucket_w, std::move(cropped)};
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
        preprocess_crop(bc.crop, bc.target_w, scratch_chw_);
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
    CpuEngine::InferResult result;
    CpuEngine::InferView view;
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
      for (int j = 0; j < cur; j++) {
        const float *logits = logits_base + static_cast<size_t>(j) * row_out;
        results[crops[beg + j].orig_idx] =
            ctc_greedy_decode_raw(logits, seq_len, num_classes, label_list_);
      }
    }

    beg = end;
  }

  return results;
}
