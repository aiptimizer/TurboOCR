#pragma once

// NvDocOrientation — thin wrapper over classification::DocOrientation
// (PP-LCNet page-orientation classifier). This stage's public input is a
// host cv::Mat (the rendered page), so it is NOT a device stage interface; it
// feeds server::OrientFunc (make_orient_func) directly. Preprocessing runs on
// the CPU once per page and only the 224x224 inference is on GPU, exactly as
// today — wrapping keeps that behaviour byte-identical.

#include <memory>
#include <string>

#include "nvidia/stages/doc_orientation.h"

namespace cv {
class Mat;
}

namespace turbo_ocr::nvidia {

class NvDocOrientation {
public:
  [[nodiscard]] bool load(const std::string &model_path) {
    ready_ = ori_.load_model(model_path);
    if (ready_)
      ori_.allocate_buffers();
    return ready_;
  }
  [[nodiscard]] bool is_ready() const noexcept { return ready_; }

  // Clockwise rotation in {0,90,180,270}; 0 when not loaded / low-confidence.
  // Runs on the default stream (page-level, once per request — no per-page
  // stream choreography needed).
  [[nodiscard]] int detect(const cv::Mat &bgr) { return ori_.detect(bgr); }

private:
  classification::DocOrientation ori_;
  bool ready_ = false;
};

} // namespace turbo_ocr::nvidia
