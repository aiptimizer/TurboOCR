#pragma once

// FFDetr — the learned half of "Prepare Form".
//
// The four geometry detectors in field_detector.h read a raster the way a
// person does: a rule, a box, a label with space after it. That finds blanks
// that are DRAWN. It cannot find a blank that is only implied by the design of
// the page (a bare cell in a shaded panel, a name line with no rule under it),
// and it cannot tell a signature line from any other write-on line, because in
// morphology it is the same object.
//
// FFDetr is an RF-DETR detector trained on CommonForms — ~450k pages filtered
// out of Common Crawl for having real fillable widgets, so its targets are the
// rectangles the documents' own authors chose, not a heuristic's idea of them.
// It predicts the three widget classes a PDF form dictionary actually
// distinguishes on a raster: Text, Checkbox, Signature.
//
//   arXiv 2509.16506 · https://huggingface.co/jbarrow/FFDetr · Apache-2.0
//
// It is NOT a replacement for the geometry detectors and is not wired as one.
// The model finds fields the morphology cannot see; the morphology gives
// pixel-exact edges where a box really is drawn, which a detector regressing
// normalised coordinates does not. They are merged, and a rectangle both
// argued for comes out as "ffdetr+box" with the confidence of the agreement —
// which is the only signal available for ranking proposals a human will review.
//
// OPTIONAL BY CONSTRUCTION: no model file, no FieldModel, and field detection
// falls back to exactly the geometry-only behaviour it had before.

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/forms/form_field.h"

namespace turbo_ocr::forms {

struct FieldModelOptions {
  // 1024 is not a tunable. commonforms' own FFDetrDetector::extract_widgets
  // overwrites its image_size argument with 1024 before every call, so 1024 is
  // the only resolution this checkpoint has ever been evaluated at, and the
  // ONNX graph is exported with that shape baked in.
  int image_size = 1024;

  // The reference defaults, kept verbatim so a rectangle this server proposes
  // is the same rectangle the published model proposes.
  float confidence = 0.40f;
  float nms_iou = 0.10f; // class-agnostic, as FFDetrDetector does it

  // Intra-op cap for the ORT session, fed through the shared host policy in
  // common/host_ort_threads.h (ORT_NUM_THREADS still overrides it).
  //
  // 8, not the 4 the other host stages use, and the difference is deliberate.
  // The 4 elsewhere exists because those stages share the CPU with det/rec on
  // a pool of workers, where a per-session thread-per-core oversubscribes.
  // This stage is the opposite case on every count: it runs ONLY on ?fields=1,
  // exactly one instance at a time (the callers serialise it), and it is by
  // far the heaviest host graph in the tree — ~0.5 s against layout's ~17 ms.
  // Measured, one page, CPU provider, median of 15 interleaved runs:
  //
  //      2 threads  1884 ms
  //      4 threads   990 ms      <- the inherited default
  //      6 threads   670 ms
  //      8 threads   520 ms      <- 1.9x, for threads that are otherwise idle
  //     10 threads   482 ms      <- +8% more, on a 10 P-core machine
  //
  // 8 takes almost all of the available scaling while still leaving headroom
  // for the OCR workers that may be running other pages of the same PDF.
  int stage_default_threads = 8;
};

// One FieldModel drives one ONNX session and owns per-instance staging
// buffers, so an instance is single-threaded by construction — the pipeline
// pool gives each worker its own, exactly as it does for the layout stage.
class FieldModel {
public:
  // Returns nullptr when the file is missing or will not load. That is a
  // supported outcome, not an error: the caller degrades to geometry-only.
  [[nodiscard]] static std::unique_ptr<FieldModel>
  load(const std::string &onnx_path, const FieldModelOptions &opt = {});

  ~FieldModel();
  FieldModel(const FieldModel &) = delete;
  FieldModel &operator=(const FieldModel &) = delete;

  // Proposals in PAGE PIXELS of `page`, unlabelled — labelling is the caller's
  // job because it owns the OCR. `source` is "ffdetr" on every one.
  [[nodiscard]] std::vector<FormField> run(const cv::Mat &page);

  [[nodiscard]] const FieldModelOptions &options() const noexcept;

private:
  FieldModel();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::forms
