#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Shared, backend-agnostic logic for the document-orientation classifier
// (PP-LCNet_x1_0_doc_ori). The GPU (TensorRT) and CPU (ONNX Runtime) variants
// both preprocess identically on the CPU — this is opt-in (autorotate=1) and
// runs once per page, so a fused GPU preprocess kernel would be over-engineering.
namespace turbo_ocr::classification {

inline constexpr int kDocOriSize = 224;          // model input H==W
inline constexpr int kDocOriResizeShort = 256;   // resize_short before crop
// Minimum probability MARGIN (top1 - top2) to ACT on a non-zero prediction.
// PP-LCNet_x1_0_doc_ori's graph ends in a softmax, so its 4 outputs are
// probabilities summing to 1 — a clear rotation scores ~0.93 for its class
// (margin ~0.9), while a blank/ambiguous page spreads near 0.25 each (margin
// ~0). 0.25 cleanly separates confident detections from noise without
// rejecting any real rotation.
inline constexpr float kDocOriMargin = 0.25f;

// ImageNet normalization (RGB order), matching the model's inference.yml.
inline constexpr std::array<float, 3> kDocOriMean{0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kDocOriStd{0.229f, 0.224f, 0.225f};

// Resize-short(256) -> center-crop(224) -> BGR->RGB -> /255 -> ImageNet
// normalize -> CHW float. Returns a flat [3*224*224] buffer ready for the
// model input. `bgr` is an 8-bit 3-channel image.
[[nodiscard]] inline std::vector<float>
doc_ori_preprocess(const cv::Mat &bgr) {
  const cv::Mat &img = bgr;
  const int h = img.rows, w = img.cols;
  const double scale =
      static_cast<double>(kDocOriResizeShort) / std::min(h, w);
  cv::Mat resized;
  cv::resize(img, resized,
             cv::Size(static_cast<int>(std::lround(w * scale)),
                      static_cast<int>(std::lround(h * scale))),
             0, 0, cv::INTER_LINEAR);
  // Center crop 224x224.
  const int x = (resized.cols - kDocOriSize) / 2;
  const int y = (resized.rows - kDocOriSize) / 2;
  cv::Mat crop = resized(cv::Rect(x, y, kDocOriSize, kDocOriSize)).clone();

  std::vector<float> out(static_cast<size_t>(3) * kDocOriSize * kDocOriSize);
  const int plane = kDocOriSize * kDocOriSize;
  for (int r = 0; r < kDocOriSize; ++r) {
    const auto *row = crop.ptr<cv::Vec3b>(r);
    for (int c = 0; c < kDocOriSize; ++c) {
      const cv::Vec3b &px = row[c];  // BGR
      // CHW with RGB channel order.
      for (int ch = 0; ch < 3; ++ch) {
        float v = static_cast<float>(px[2 - ch]) / 255.0f;  // BGR[2-ch] = RGB[ch]
        out[static_cast<size_t>(ch) * plane + r * kDocOriSize + c] =
            (v - kDocOriMean[ch]) / kDocOriStd[ch];
      }
    }
  }
  return out;
}

// Argmax over the 4 class probabilities (the model output is already softmax)
// with a top1-vs-top2 margin gate. Returns the page's detected clockwise
// rotation ∈ {0, 90, 180, 270}; returns 0 (treat as upright) when the page is
// upright or the top-two classes are within kDocOriMargin (ambiguous).
[[nodiscard]] inline int doc_ori_label(const float *probs4) {
  int best = 0;
  for (int i = 1; i < 4; ++i) if (probs4[i] > probs4[best]) best = i;
  if (best == 0) return 0;  // upright — nothing to correct
  float second = 0.0f;
  for (int i = 0; i < 4; ++i) if (i != best) second = std::max(second, probs4[i]);
  if (probs4[best] - second < kDocOriMargin) return 0;  // ambiguous
  static constexpr int kDeg[4] = {0, 90, 180, 270};
  return kDeg[best];
}

// Rotate an image to upright given the detected clockwise-rotation label.
// label is the page's CW rotation, so we undo it (CCW by the same amount).
inline void rotate_upright(cv::Mat &img, int label_deg) {
  switch (label_deg) {
    case 90:  cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE); break;
    case 180: cv::rotate(img, img, cv::ROTATE_180); break;
    case 270: cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE); break;
    default: break;  // 0 — already upright
  }
}

} // namespace turbo_ocr::classification
