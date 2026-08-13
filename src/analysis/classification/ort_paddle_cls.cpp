#include "turbo_ocr/analysis/classification/ort_paddle_cls.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/geometry/perspective.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::classification;
using turbo_ocr::engine::OrtEngine;
using turbo_ocr::Box;

namespace {
// Warp one detection quad from the FULL image into the NCHW RGB float plane
// at `dst` (3*H*W floats): content rendered at its natural width by a single
// perspective warp, remaining columns zero (mid-gray) — the same geometry
// cuda_batch_roi_warp gives the GPU classifier. A crop-then-resize here would
// both resample twice and stretch narrow crops across the full canvas,
// diverging from the GPU's padded layout.
inline void pack_cls_box(const cv::Mat &img, const Box &box, int H, int W,
                         float *dst) {
  const auto ct = turbo_ocr::compute_crop_transform(box, H, W);
  const int content_w = ct.crop_width;
  const cv::Matx33f m_inv(ct.M_inv[0], ct.M_inv[1], ct.M_inv[2],
                          ct.M_inv[3], ct.M_inv[4], ct.M_inv[5],
                          ct.M_inv[6], ct.M_inv[7], ct.M_inv[8]);
  cv::Mat warped;
  cv::warpPerspective(img, warped, m_inv, cv::Size(content_w, H),
                      cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                      cv::BORDER_REPLICATE);
  cv::Mat float_img;
  warped.convertTo(float_img, CV_32F, 1.0 / 127.5, -1.0);
  const size_t plane = static_cast<size_t>(H) * W;
  std::fill(dst, dst + 3 * plane, 0.0f);
  cv::Mat channels[3];
  cv::split(float_img, channels);
  // RGB order (R=channels[2], G=channels[1], B=channels[0])
  for (int c = 0; c < 3; ++c) {
    const cv::Mat &src = channels[2 - c];
    for (int r = 0; r < H; ++r)
      std::memcpy(dst + c * plane + static_cast<size_t>(r) * W,
                  src.ptr<float>(r),
                  static_cast<size_t>(content_w) * sizeof(float));
  }
}

inline bool cls_batch_enabled() {
  static const bool e = turbo_ocr::env::env_enabled("CLS_BATCH");
  return e;
}
} // namespace

bool OrtPaddleCls::load_model(const std::string &model_path) {
  engine_ = ep_set_ ? std::make_unique<OrtEngine>(model_path, ep_)
                    : std::make_unique<OrtEngine>(model_path);
  return engine_->load();
}

void OrtPaddleCls::run(const cv::Mat &img, std::vector<Box> &boxes) {
  if (boxes.empty())
    return;

  const int plane = kClsImageH * kClsImageW;

  // Batched path: one ORT Run for every box (cls.onnx has a fixed 80x160
  // input and a dynamic batch dim, so all crops share one tensor). Output
  // is {N,2}; row i drives the same swap decision as the scalar path.
  if (cls_batch_enabled()) {
    const int n = static_cast<int>(boxes.size());
    std::vector<float> input_buf(static_cast<size_t>(n) * 3 * plane);
    for (int i = 0; i < n; ++i) {
      pack_cls_box(img, boxes[i], kClsImageH, kClsImageW,
                   input_buf.data() + static_cast<size_t>(i) * 3 * plane);
    }
    const std::vector<int64_t> shape = {n, 3, kClsImageH, kClsImageW};
    auto result = engine_->infer_batch(input_buf.data(), shape);
    if (static_cast<int>(result.data.size()) >= 2 * n) {
      for (int i = 0; i < n; ++i) {
        float s0 = result.data[2 * i];
        float s180 = result.data[2 * i + 1];
        if (s180 > s0 && s180 > kClsThresh) {
          std::swap(boxes[i][0], boxes[i][2]);
          std::swap(boxes[i][1], boxes[i][3]);
        }
      }
    }
    return;
  }

  std::vector<float> input_buf(3 * plane);
  const std::vector<int64_t> input_shape = {1, 3, kClsImageH, kClsImageW};
  for (size_t i = 0; i < boxes.size(); ++i) {
    pack_cls_box(img, boxes[i], kClsImageH, kClsImageW, input_buf.data());
    auto result = engine_->infer(input_buf.data(), input_shape);
    if (result.data.size() >= 2) {
      float score_0 = result.data[0];
      float score_180 = result.data[1];
      if (score_180 > score_0 && score_180 > kClsThresh) {
        std::swap(boxes[i][0], boxes[i][2]);
        std::swap(boxes[i][1], boxes[i][3]);
      }
    }
  }
}
