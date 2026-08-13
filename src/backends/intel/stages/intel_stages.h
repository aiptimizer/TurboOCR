#pragma once

// Intel device stages — IntelDetector / IntelRecognizer / IntelClassifier /
// IntelLayout, implementing backend/stages.h by wiring SyclKernels (device
// pre/post) to an OpenVINOEngine (forward pass), keeping data in USM device
// memory and crossing back to the host only at the small result. This is the
// Intel analogue of nv_stages.cpp, and it is deliberately THIN.
//
// ================= WHAT IS *NOT* HERE (and must never be) ===================
// Per the rebuild's dedup rules, generic policy is shared. These stages CALL
// shared helpers instead of restating them:
//
//   crop width / geometry   turbo_ocr::compute_crop_transform (perspective.h)
//                           recognition::rec_input_width, kMaxRecWidth
//   width bucketing +       recognition::plan_rec_batches, group_by_width_bucket,
//   batch ladder            batch_ladder_for_width, rec_shape_matrix
//                           (include/turbo_ocr/analysis/recognition/rec_batching.h)
//   det resize policy       detection::read_det_resize / compute_det_resize
//   DB thresholds           detection::read_db_params
//   DB box extraction       detection::extract_boxes_from_bitmap (via IKernels)
//   CTC decode + dict       recognition::ctc_greedy_decode / load_label_dict
//   box reading order       turbo_ocr::sorted_boxes
//
// The cautionary precedent from the plan is the Apple rec-ladder bug: a backend
// that hardcoded its own width ladder silently squashed long lines and lost
// accuracy that the shared path never lost. An earlier draft of this file had
// the identical defect (a private kMaxW = 320 and an inline CTC loop); it is
// gone. If you find yourself typing a bucket table, a batch size table, a
// normalization constant that another backend also needs, or a decode loop —
// it belongs above the seam.
//
// What IS legitimately here: buffer sizing/reuse in USM, the (width,batch)
// prebuild call, tensor binding, and the order of device calls. That is device
// mechanics, not policy.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel/engine/openvino_engine.h"
#include "intel/kernels_sycl/sycl_kernels.h"
#include "intel/memory/l0_allocator.h"
#include "turbo_ocr/backend/stages.h"

namespace turbo_ocr::intel {

// Per-stage device context: the shared allocator plus this stage's own kernels
// and engine (an ov::InferRequest is not thread-safe, so a stage owns its
// engine). Built by IntelBackend::load_stages and moved into each stage.
struct StageDeps {
  std::shared_ptr<L0Allocator> alloc;
  // The SEAM interface, not SyclKernels. The stages only ever call IKernels ops
  // and consult caps(), so pinning the concrete type here bought nothing and
  // cost correctness: it forced the no-DPC++ build to use SyclKernels' no-op
  // stubs, which silently fed detection an untouched buffer (zero boxes at full
  // inference cost). Typed as the interface, a build without SYCL simply
  // supplies the SHARED HostKernels instead.
  std::unique_ptr<backend::IKernels> kernels;
  std::unique_ptr<OpenVINOEngine> engine;
};

class IntelDetector final : public backend::IDetector {
public:
  explicit IntelDetector(StageDeps deps);
  ~IntelDetector() override;
  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::Box>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

class IntelRecognizer final : public backend::IRecognizer {
public:
  explicit IntelRecognizer(StageDeps deps);
  ~IntelRecognizer() override;
  [[nodiscard]] bool load(const std::string &model_path) override;
  // Character dictionary via the SHARED recognition::load_label_dict (which
  // prepends "blank" and appends " "), so index semantics match every other
  // backend. Called by IntelBackend right after load().
  [[nodiscard]] bool load_dict(const std::string &dict_path);
  [[nodiscard]] std::vector<backend::RecResult>
  run(const backend::ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;

  // Crops this recognizer FAILED on in the last run(), reported through the
  // SHARED seam so the pipeline can mark the page text_degraded. Necessary
  // because this backend PRE-SIZES its result vector and leaves failed chunks
  // empty: the returned length always equals boxes.size(), so the pipeline's
  // under-return check structurally cannot see the loss.
  [[nodiscard]] int last_dropped_crops() const noexcept override {
    return dropped_crops_;
  }


  struct Impl;

private:
  // Reset at the top of every run(); see last_dropped_crops().
  int dropped_crops_ = 0;

  std::unique_ptr<Impl> impl_;
};

class IntelClassifier final : public backend::IClassifier {
public:
  explicit IntelClassifier(StageDeps deps);
  ~IntelClassifier() override;
  [[nodiscard]] bool load(const std::string &model_path) override;
  void run(const backend::ImageView &img, std::vector<turbo_ocr::Box> &boxes,
          backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

class IntelLayout final : public backend::ILayout {
public:
  explicit IntelLayout(StageDeps deps);
  ~IntelLayout() override;
  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      float score_threshold, backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::intel
