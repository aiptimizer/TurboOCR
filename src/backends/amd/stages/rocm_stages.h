#pragma once

// ROCm stage classes — the AMD implementations of the four backend stage
// interfaces. Each mirrors its NVIDIA twin (PaddleDet/PaddleRec/PaddleCls/
// PaddleLayout) but expressed through the device-agnostic seam:
//   GpuImage        -> backend::ImageView (a HIP device buffer)
//   cudaStream_t    -> backend::DeviceQueue& (a HipStreamQueue)
//   engine::TrtEngine -> amd::MIGraphXEngine (IEngine)
//   src/backends/nvidia/kernels_cuda/*.cu  -> amd::HipKernels (IKernels)
//
// DEVICE-RESIDENCY: the image, the normalized input tensor, the model logits /
// pred-map, the threshold bitmap, the CCL scratch, the rec/cls warp batch, and
// the argmax outputs all stay in Hip memory. Each run() crosses back to the host
// only at the small result boundary (DB boxes, argmax indices for CTC collapse,
// the cls flip decision) — returning the pure HOST types the interfaces mandate.
//
// Each stage owns ONE MIGraphXEngine (not thread-safe → one per pipeline entry)
// and BORROWS the pipeline entry's shared IKernels + IDeviceAllocator.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/backend.h"  // IDeviceAllocator
#include "turbo_ocr/backend/engine.h"   // IEngine
#include "turbo_ocr/backend/kernels.h"  // IKernels
#include "turbo_ocr/backend/stages.h"   // IDetector/IRecognizer/IClassifier/ILayout

namespace turbo_ocr::amd {

using backend::DeviceQueue;
using backend::IDeviceAllocator;
using backend::IKernels;
using backend::ImageView;

// Common wiring every ROCm stage shares. `engine` is created per stage in
// load().
//
// THREADING (load-bearing): `kernels` is a shared_ptr, and each StageSet gets
// its OWN HipKernels instance — it is deliberately NOT a per-backend singleton.
// HipKernels owns mutable device scratch (the decode pool, the CCL label map,
// the JFA seed buffers, the pinned box staging), all sized to the last image it
// saw. Two pipeline entries running concurrently on two queues would trample
// each other's CCL scratch and produce silently wrong boxes. The allocator, by
// contrast, IS safely shared: hipMalloc/hipFree are thread-safe and it holds no
// per-call state.
struct StageDeps {
  std::shared_ptr<IKernels> kernels; // per-StageSet; NOT shared across entries
  IDeviceAllocator *alloc = nullptr; // shared singleton (stateless, thread-safe)
  int device_id = 0;
};

// ---- Detection -------------------------------------------------------------
class RocmDetector final : public backend::IDetector {
public:
  RocmDetector(StageDeps deps, std::string rec_dict_unused = {});
  ~RocmDetector() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::Box>
  run(const ImageView &img, int orig_h, int orig_w, DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  // NOTE (dedup): there are deliberately NO set_max_side()/set_db() knobs. The
  // det resize limits and the DB thresholds are SHARED policy read from
  // detection::read_det_resize()/read_db_params() (det_config.h), including
  // their DET_* env overrides — the same source the CUDA and CPU detectors use.
  // A backend-local override here is how one device quietly ends up with a
  // different recall from the rest.

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  bool ready_ = false;
};

// ---- Recognition -----------------------------------------------------------
class RocmRecognizer final : public backend::IRecognizer {
public:
  explicit RocmRecognizer(StageDeps deps);
  ~RocmRecognizer() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  // Loads the character dictionary (one token per line). Required before run().
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  [[nodiscard]] std::vector<backend::RecResult>
  run(const ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }


  // Crops this recognizer FAILED on in the last run(), reported through the
  // SHARED seam so the pipeline can mark the page text_degraded. Necessary
  // because this backend PRE-SIZES its result vector and leaves failed chunks
  // empty: the returned length always equals boxes.size(), so the pipeline's
  // under-return check structurally cannot see the loss.
  [[nodiscard]] int last_dropped_crops() const noexcept override {
    return dropped_crops_;
  }

private:
  // Reset at the top of every run(); see last_dropped_crops().
  int dropped_crops_ = 0;

  struct Impl;
  std::unique_ptr<Impl> p_;
  bool ready_ = false;
};

// ---- Classification (text-line angle 0/180) --------------------------------
class RocmClassifier final : public backend::IClassifier {
public:
  explicit RocmClassifier(StageDeps deps);
  ~RocmClassifier() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  void run(const ImageView &img, std::vector<turbo_ocr::Box> &boxes,
          DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  bool ready_ = false;
};

// ---- Layout (PP-DocLayoutV3, multi-IO) -------------------------------------
class RocmLayout final : public backend::ILayout {
public:
  explicit RocmLayout(StageDeps deps);
  ~RocmLayout() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const ImageView &img, int orig_h, int orig_w, float score_threshold,
      DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  bool ready_ = false;
};

} // namespace turbo_ocr::amd
