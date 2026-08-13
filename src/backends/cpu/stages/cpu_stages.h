#pragma once

// CpuDetector / CpuRecognizer / CpuClassifier / CpuLayout — the CpuBackend stage
// interfaces (backend/stages.h) over the existing, proven CPU stage classes:
//
//   backend::IDetector    -> detection::OrtPaddleDet
//   backend::IRecognizer  -> recognition::OrtPaddleRec
//   backend::IClassifier  -> classification::OrtPaddleCls
//   backend::ILayout      -> layout::OrtPaddleLayout   (synchronous only)
//
// The backend stage namespace (turbo_ocr::backend) does not collide with the
// concrete stage namespaces, so a single TU includes both and simply forwards,
// converting a Host ImageView back to the non-owning cv::Mat the CPU classes
// take and discarding the (no-op) Host DeviceQueue. This is the CPU peer of
// nvidia/stages/nv_stages.h — thin wrapping, no pipeline logic re-implemented.

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/stages.h"

#include "turbo_ocr/analysis/classification/ort_paddle_cls.h"
#include "turbo_ocr/backend/backend.h"   // StageSet, BackendConfig, EpConfig
#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"
#include "turbo_ocr/analysis/detection/ort_paddle_det.h"
#include "turbo_ocr/analysis/layout/ort_paddle_layout.h"
#include "turbo_ocr/analysis/recognition/ort_paddle_rec.h"

namespace turbo_ocr::cpu {

// ---- Detection -------------------------------------------------------------
class CpuDetector final : public backend::IDetector {
public:
  // Pin the execution provider for this stage (backend/engine_mode.h FAST
  // path). Call BEFORE load(); unset => the engine reads ORT_EP from env.
  void set_ep_config(const backend::EpConfig &ep) { det_.set_ep_config(ep); }

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::Box>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] detection::OrtPaddleDet *native() noexcept { return &det_; }

private:
  detection::OrtPaddleDet det_;
  bool ready_ = false;
};

// ---- Recognition -----------------------------------------------------------
class CpuRecognizer final : public backend::IRecognizer {
public:
  // Pin the execution provider for this stage (backend/engine_mode.h FAST
  // path). Call BEFORE load(); unset => the engine reads ORT_EP from env.
  void set_ep_config(const backend::EpConfig &ep) { rec_.set_ep_config(ep); }

  [[nodiscard]] bool load(const std::string &model_path) override;
  // OrtPaddleRec needs the dict loaded before run(); BackendConfig carries
  // rec_dict, so CpuBackend calls this after load() (mirrors NvRecognizer).
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  [[nodiscard]] std::vector<backend::RecResult>
  run(const backend::ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  // Exposed so the table recognizer's cell-fill can reach the concrete
  // OrtPaddleRec the SLANeXt wrapper expects.
  [[nodiscard]] recognition::OrtPaddleRec *native() noexcept { return &rec_; }

  // Forward the SHARED onnx recognizer's drop count through the seam. Without
  // this override the base returns 0 (stages.h) and a partial failure on
  // --engine-mode onnx is invisible on EVERY vendor — the one arm that had no
  // accounting while all four native ones did.
  [[nodiscard]] int last_dropped_crops() const noexcept override {
    return rec_.last_dropped_crops();
  }

private:
  recognition::OrtPaddleRec rec_;
  bool ready_ = false;
};

// ---- Classification --------------------------------------------------------
class CpuClassifier final : public backend::IClassifier {
public:
  // Pin the execution provider for this stage (backend/engine_mode.h FAST
  // path). Call BEFORE load(); unset => the engine reads ORT_EP from env.
  void set_ep_config(const backend::EpConfig &ep) { cls_.set_ep_config(ep); }

  [[nodiscard]] bool load(const std::string &model_path) override;
  void run(const backend::ImageView &img, std::vector<turbo_ocr::Box> &boxes,
          backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] classification::OrtPaddleCls *native() noexcept { return &cls_; }

private:
  classification::OrtPaddleCls cls_;
  bool ready_ = false;
};

// ---- Layout ----------------------------------------------------------------
class CpuLayout final : public backend::ILayout {
public:
  // NO-OP by design: layout::OrtPaddleLayout builds its own Ort::Session rather
  // than going through engine::OrtEngine, so it has no provider switch to set.
  // Layout therefore always runs on the default CPU provider, even when the
  // rest of the stage set is on a vendor EP. Kept as a member so every stage
  // presents the same interface to make_onnx_stages().
  void set_ep_config(const backend::EpConfig &) {}

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      float score_threshold, backend::DeviceQueue &queue) override;
  // No device-side overlap on the host: the default enqueue() returns an
  // already-computed future (supports_async() stays false).
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  [[nodiscard]] layout::OrtPaddleLayout *native() noexcept { return &layout_; }

private:
  layout::OrtPaddleLayout layout_;
  bool ready_ = false;
};


// ---- THE shared ONNX ("fast") stage set ------------------------------------
//
// det/cls/rec/layout + doc-orientation built on engine::OrtEngine over a given
// execution provider. This is the FAST path for EVERY vendor
// (backend/engine_mode.h): NVIDIA on the CUDA EP, Intel on OpenVINO, Apple on
// CoreML, AMD on MIGraphX all run THIS code with a different EpConfig — the
// per-vendor part is one provider string, not a stage set each.
//
// Returns the doc-orientation model alongside the StageSet because the seam
// keeps it on the Backend (it becomes an OrientFunc), not in StageSet.
struct OnnxStageSet {
  backend::StageSet stages;
  std::unique_ptr<classification::OrtDocOrientation> doc_ori; // null unless loaded
};

[[nodiscard]] OnnxStageSet make_onnx_stages(const backend::BackendConfig &cfg,
                                            const backend::EpConfig &ep);

// ---- Vendor-neutral mode policy (used by EVERY backend) --------------------
//
// The Auto-fallback rule, the "explicit native with no artefact is an error"
// rule, and the provider lookup are written ONCE here. A vendor supplies only
// its own `native_available` probe (does MY graph engine have something to
// load?) and gets identical semantics — otherwise each backend re-invents the
// policy and they drift on exactly the corner that matters: whether a missing
// artefact is a fallback or a failure.
[[nodiscard]] backend::EngineMode
resolve_engine_mode(std::string_view vendor, const backend::BackendConfig &cfg,
                    bool native_available);

// The vendor's FAST stage set: make_onnx_stages() with the provider from
// backend::onnx_provider_for(vendor), unless the caller pinned one in cfg.ep.
[[nodiscard]] OnnxStageSet
make_vendor_onnx_stages(std::string_view vendor, const backend::BackendConfig &cfg);

} // namespace turbo_ocr::cpu
