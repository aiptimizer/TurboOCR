#pragma once

// NvDetector / NvRecognizer / NvClassifier / NvLayout — the NVIDIA stage
// interfaces (backend/stages.h) over the existing, proven GPU stage classes:
//
//   backend::IDetector    -> detection::PaddleDet
//   backend::IRecognizer  -> recognition::PaddleRec
//   backend::IClassifier  -> classification::PaddleCls
//   backend::ILayout      -> layout::PaddleLayout   (two-phase enqueue/collect)
//
// The backend stage namespace (turbo_ocr::backend) does NOT collide with the
// concrete stage namespaces (detection/recognition/classification/layout), so a
// single TU includes both and forwards, converting ImageView<->GpuImage and
// DeviceQueue->cudaStream_t. Device buffers, baked CUDA graphs, the
// deferred-sync multi-slot rec queue and DB CCL/JFA post-proc stay inside the
// wrapped classes: wrapping, not re-deriving, is the point.
//
// But wrapping preserves the CODE, not the CALLS. Anything the pre-seam
// pipeline drove from its own init has no caller until a wrapper makes one —
// that is how bake_graphs() ended up defined and never invoked, costing ~14%
// throughput with nothing failing. Every hook the old pipeline called from
// outside the stage classes belongs in this file.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/stages.h"

#include "nvidia/stages/paddle_cls.h"
#include "nvidia/stages/paddle_det.h"
#include "nvidia/stages/paddle_layout.h"
#include "nvidia/stages/paddle_rec.h"

namespace turbo_ocr::nvidia {

// ---- Detection -------------------------------------------------------------
class NvDetector final : public backend::IDetector {
public:
  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::Box>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] std::vector<std::vector<turbo_ocr::Box>>
  run_batch(const std::vector<backend::ImageView> &imgs,
            const std::vector<std::pair<int, int>> &orig_dims,
            backend::DeviceQueue &queue) override;
  // The real ceiling, forwarded rather than restated: PaddleDet sizes its batch
  // canvas, per-slice metadata and pinned staging for exactly kMaxBatchSize.
  // Capability only — coalescing needs the POLICY knob
  // (BackendCaps::preferred_batch_size / TURBO_DET_BATCH), which CudaBackend
  // leaves at 1 until someone measures the right N per device.
  [[nodiscard]] int max_batch_size() const noexcept override {
    return detection::PaddleDet::kMaxBatchSize;
  }
  // TWO-PHASE DETECTION (stages.h). The pipeline uses it to keep two pages in
  // flight: collect N's boxes, enqueue N+1's detection, then run cls+rec for N
  // while N+1 detects. Until this existed, supports_async() was false on the one
  // backend that ships, so UnifiedOcrPipeline::run_pipelined()'s whole ring was
  // dead code on CUDA and every multi-page request ran strictly serially.
  [[nodiscard]] bool supports_async() const noexcept override { return true; }
  // The seam's single-outstanding-future rule is PaddleDet's constraint here,
  // not a policy choice: its device scratch is a single set per instance, so a
  // second submission would overwrite the map the first has not read yet.
  [[nodiscard]] backend::BoxesFuture
  enqueue(const backend::ImageView &img, int orig_h, int orig_w,
          backend::DeviceQueue &queue) override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] detection::PaddleDet *native() noexcept { return &det_; }

  // The lane and event are this object's, so it destroys them.
  ~NvDetector() override;

private:
  // THE PRIVATE LANE, and the event that orders it. Both are required by the
  // seam contract and neither is optional on discrete VRAM:
  //
  //  * Submitting the forward pass on the caller's `queue` would buy nothing —
  //    the pipeline runs page N's cls+rec on that same lane straight after, so
  //    the device would simply execute det(N+1) after rec(N) and the only thing
  //    gained is an earlier host return.
  //  * A private lane is UNORDERED against `queue`, where the caller staged the
  //    page H2D. Without waiting on an event recorded there, the forward pass
  //    can read the page buffer before the upload lands.
  cudaStream_t det_stream_ = nullptr;
  cudaEvent_t upload_done_ = nullptr;

  detection::PaddleDet det_;
  bool ready_ = false;
};

// ---- Recognition -----------------------------------------------------------
class NvRecognizer final : public backend::IRecognizer {
public:
  [[nodiscard]] bool load(const std::string &model_path) override;
  // Separate dict load (PaddleRec needs it before run()); the merged
  // BackendConfig carries rec_dict, so CudaBackend calls this after load().
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  [[nodiscard]] std::vector<backend::RecResult>
  run(const backend::ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] std::vector<std::vector<backend::RecResult>>
  run_multi(const std::vector<backend::ImageCrops> &items,
            backend::DeviceQueue &queue) override;
  // Bakes PaddleRec's per-(batch,width)-profile CUDA graphs onto this replica's
  // stream — the call the seam had no hook for (see the file header).
  // Idempotent and self-gating: with TURBO_OCR_CUDA_GRAPHS=0, or an engine
  // built without the static profiles, it does nothing.
  void warmup(backend::DeviceQueue &queue) override;
  // Crops whose engine output overflowed the decode buffers. Forwarding it
  // restores the text_degraded accounting the unified pipeline had lost — a
  // partial recognition failure was returning a thin page with the flag unset.
  [[nodiscard]] int last_dropped_crops() const noexcept override {
    return rec_.last_dropped_crops();
  }
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  // Exposed so the table recognizer's cell-fill can reach the concrete
  // PaddleRec the SLANeXt wrapper still expects (bridged as void*).
  [[nodiscard]] recognition::PaddleRec *native() noexcept { return &rec_; }

private:
  recognition::PaddleRec rec_;
  bool ready_ = false;
};

// ---- Classification --------------------------------------------------------
class NvClassifier final : public backend::IClassifier {
public:
  [[nodiscard]] bool load(const std::string &model_path) override;
  void run(const backend::ImageView &img, std::vector<turbo_ocr::Box> &boxes,
           backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] classification::PaddleCls *native() noexcept { return &cls_; }

private:
  classification::PaddleCls cls_;
  bool ready_ = false;
};

// ---- Layout ----------------------------------------------------------------
class NvLayout final : public backend::ILayout {
public:
  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      float score_threshold, backend::DeviceQueue &queue) override;

  // NVIDIA preserves the two-phase overlap (enqueue, do other work, collect) —
  // its whole reason to exist.
  [[nodiscard]] bool supports_async() const noexcept override { return true; }
  [[nodiscard]] backend::LayoutFuture
  enqueue(const backend::ImageView &img, int orig_h, int orig_w,
          float score_threshold, backend::DeviceQueue &queue) override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] layout::PaddleLayout *native() noexcept { return &layout_; }

private:
  layout::PaddleLayout layout_;
  bool ready_ = false;
};

} // namespace turbo_ocr::nvidia
