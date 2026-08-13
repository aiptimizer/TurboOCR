#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"

#include "turbo_ocr/analysis/classification/doc_orientation_common.h"

using namespace turbo_ocr::classification;
using turbo_ocr::engine::OrtEngine;

bool OrtDocOrientation::load_model(const std::string &model_path) {
  engine_ = ep_set_ ? std::make_unique<OrtEngine>(model_path, ep_)
                    : std::make_unique<OrtEngine>(model_path);
  return engine_->load();
}

int OrtDocOrientation::detect(const cv::Mat &bgr) {
  if (!engine_ || bgr.empty() || bgr.type() != CV_8UC3) return 0;
  std::vector<float> input = doc_ori_preprocess(bgr);
  auto res = engine_->infer(
      input.data(), {1, 3, kDocOriSize, kDocOriSize});
  if (res.data.size() < 4) return 0;
  return doc_ori_label(res.data.data());
}
