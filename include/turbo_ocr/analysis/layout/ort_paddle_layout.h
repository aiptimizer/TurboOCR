#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/layout_types.h"

namespace turbo_ocr::layout {

/// CPU layout detection using ONNX Runtime (PP-DocLayoutV3).
/// Drop-in replacement for PaddleLayout when no GPU is available.
class OrtPaddleLayout {
public:
  OrtPaddleLayout();
  ~OrtPaddleLayout() noexcept;

  // NOTE ON THREADS — there is deliberately no setter here any more.
  //
  // This stage's intra-op cap (2) now goes through
  // turbo_ocr::host_ort_intra_op_threads(2), the ONE policy every host ORT
  // stage consults, so the three caps (engine::OrtEngine 4, formula OrtSession
  // 4, layout 2) cannot drift when someone tunes one of them. A per-stage
  // setter here was the first thing I wrote and it was the wrong shape: it
  // made layout's cap tunable and left the other two stuck.
  //
  // The cap dominates this model. Measured M-series, 800x800, median of 10:
  // 2 threads 1010 ms, ORT-default 341 ms, 8 threads 312 ms. See
  // common/host_ort_threads.h for the precedence rules and the measurements
  // on the other two stages.

  // CoreML execution-provider opt-in (Apple only). Call BEFORE load_model();
  // default off, so no existing caller changes behaviour.
  //
  // READ THIS BEFORE ENABLING IT ANYWHERE NEW. On ORT 1.24.4 the CoreML EP
  // compiled this graph into 144 partitions and returned NaN for every score
  // and every box. A NaN score compares false against the threshold, so the
  // decoder dropped every row and produced an EMPTY layout — fast, HTTP 200,
  // and indistinguishable from a blank page. It measured as a 2.6x win. That
  // is fixed as of the ORT 1.27.1 bump, which is why this is back.
  //
  // The finite-check in run() is what makes this safe to switch on at all:
  // keep it, on every ORT version, forever. It is the only thing standing
  // between a provider regression and a silently empty layout.
  void set_use_coreml(bool on) noexcept { use_coreml_ = on; }

  [[nodiscard]] bool load_model(const std::string &onnx_path);

  /// Run layout detection on a CPU image. Returns detected layout boxes.
  [[nodiscard]] std::vector<LayoutBox> run(const cv::Mat &img,
                                           float score_threshold = 0.3f);

  static constexpr int kInputSize = 800;
  static constexpr int kMaxDetections = 300;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool use_coreml_ = false; // opt-in; see set_use_coreml
};

} // namespace turbo_ocr::layout
