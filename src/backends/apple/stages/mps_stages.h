#pragma once

// MpsDetector / MpsRecognizer / MpsClassifier / MpsLayout — the Apple stage
// interfaces (backend/stages.h). Each combines an MpsEngine (MPSGraph forward
// pass) with MetalKernels (Metal pre/post) and returns HOST types
// (vector<Box>, vector<pair<string,float>>) — the device/host boundary.
//
// MpsRecognizer is the proven, measured path (tools/probes/apple/mps_ocr.mm:140-161): warp
// every crop on the GPU, run rec + a GPU argmax head in the SAME command buffer
// as the warp (BatchScope => one submission), then only the tiny [B,T] token
// indices + scores cross to the host for ctc_greedy_decode. Data stays resident
// on the GPU across warp -> rec -> argmax; nothing but ~14 KB of indices leaves.
//
// MpsDetector runs the DB forward pass resident, then does DB post-process on the
// HOST (extract_boxes_from_bitmap) — the caps().db_postprocess == false fallback,
// reading the prob map through unified memory (coherent, no PCIe D2H). This is
// the mps_ocr.mm detPost path (mps_ocr.mm:99-107), bit-accurate to the pipeline.
//
// MpsClassifier is structural (angle cls not in the measured POC; validate its
// golden output before production). MpsLayout is a STUB: PP-DocLayoutV3 is
// multi-IO (image + im_shape + scale_factor) and the single-input MPSGraph
// builder does not yet handle it — load() returns false, so the backend reports
// layout unavailable (see README TODOs).

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/stages.h"
#include "turbo_ocr/analysis/classification/cls_config.h" // cls canvas + threshold (SHARED)
#include "turbo_ocr/analysis/detection/det_config.h" // DbParams (SHARED det policy)

#include "apple/engine/ane_rec_engine.h"
#include "apple/engine/mps_engine.h"
#include "apple/kernels_metal/metal_kernels.h"
#include "apple/memory/metal_allocator.h"

namespace turbo_ocr::apple {

// ---- Detection -------------------------------------------------------------
class MpsDetector final : public backend::IDetector {
public:
  explicit MpsDetector(std::shared_ptr<MetalAllocator> alloc);
  ~MpsDetector() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::Box>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

  // --- Two-phase async detection (stages.h) ---------------------------------
  // Enabled unless TURBO_APPLE_DET_ASYNC=0.
  //
  // WHY THIS NEEDS A PRIVATE COMMAND QUEUE (genuine device mechanics): an
  // MTLCommandQueue executes its command buffers IN SUBMISSION ORDER. If image
  // N+1's detection were submitted on the pipeline's shared queue, image N's
  // recognition warp — submitted after it — could not complete until that
  // detection had, so the host would reach the ANE predict only after detection
  // was already done and nothing would overlap. On its own queue the detection
  // runs on the GPU while the host drives the Neural Engine for image N. The
  // POLICY (which images are in flight, and when) stays in the shared pipeline;
  // only this ordering mechanism is Apple-local.
  //
  // SINGLE SLOT (the seam's rule, and the reason for it here): `out_buf_` is one
  // buffer, so the outstanding future must be collected — which fully consumes
  // the prob map on the host — before the next enqueue().
  [[nodiscard]] bool supports_async() const noexcept override;
  [[nodiscard]] backend::BoxesFuture enqueue(const backend::ImageView &img,
                                             int orig_h, int orig_w,
                                             backend::DeviceQueue &queue) override;

private:
  // preprocess + forward on `queue`, no host wait. Returns false if it could not
  // be submitted (in which case out_buf_ still holds the PREVIOUS page).
  [[nodiscard]] bool submit_forward_(const backend::ImageView &img,
                                     backend::DeviceQueue &queue);
  // host DB post-process of out_buf_ -> boxes in ORIGINAL coordinates.
  [[nodiscard]] std::vector<turbo_ocr::Box> db_postprocess_(int orig_h, int orig_w);
  // Run the SHARED fixed-canvas decision (detection::pick_det_canvas) for this
  // page and warn once if the single compiled canvas is far off the policy's
  // aspect. See the definition for why a fixed canvas is a hard MPSGraph
  // constraint and how to widen it without a code change.
  void check_canvas_policy_(int orig_h, int orig_w) const;

  std::shared_ptr<MetalAllocator> alloc_;
  MetalKernels kernels_;
  // MULTI-CANVAS: MPSGraph is single-shape, so one compiled engine serves one
  // static canvas. load() discovers every det_c<H>x<W>/ export subdir (or one
  // flat graph.json — the pre-multi-canvas bundle layout, still honored) and
  // holds one engine + resident buffer pair per canvas; select_canvas_() runs
  // the SHARED aspect picker (detection::pick_det_canvas — the same policy the
  // Intel backend's per-canvas static compile uses) before every submit. The
  // single-slot async contract makes `active_` stable between enqueue() and
  // collect(), so the futures read through it safely.
  struct DetCanvas {
    MpsEngine engine;
    int c = 3, h = 0, w = 0;
    backend::DeviceBuffer in_buf;  // [1,3,h,w] normalized input
    backend::DeviceBuffer out_buf; // [1,1,h,w] prob map
  };
  std::vector<std::unique_ptr<DetCanvas>> canvases_;
  DetCanvas *active_ = nullptr; // canvas of the LAST select_canvas_()
  // Pick (and make active) the loaded canvas nearest the shared policy's
  // aspect for this page. No-op with a single canvas.
  void select_canvas_(int orig_h, int orig_w);
  // SHARED DB thresholds, resolved ONCE at load() from the per-tier base plus
  // the DET_DB_THRESH / DET_BOX_THRESH / DET_UNCLIP env overrides
  // (detection::read_db_params). These were hardcoded in db_postprocess_(),
  // which pinned box_thresh at the tiny tier's 0.40 for every tier and made the
  // env knobs dead on this backend alone.
  turbo_ocr::detection::DbParams db_ = turbo_ocr::detection::kDbDefaults;
  // Private lane for enqueue(); created lazily on first async use.
  std::unique_ptr<backend::DeviceQueue> async_q_;
  // Optional CoreML det forward (TURBO_APPLE_DET_COREML=<path.mlpackage>):
  // same normalized in_buf canvas in (zero-copy — Metal SHARED storage), same
  // out_buf prob map out, so db_postprocess_ and every caller are untouched.
  // The package encodes ONE input shape, so it binds to the one loaded canvas
  // whose dims it matches (coreml_canvas_) and other canvases use MPSGraph.
  struct CoremlDet;
  std::unique_ptr<CoremlDet> coreml_;
  const DetCanvas *coreml_canvas_ = nullptr;
  bool ready_ = false;
};

// ---- Recognition (the proven fused path, WIDTH-BUCKETED) --------------------
// A single static-width recognizer tops out ~81.5% F1 on FUNSD: narrow widths
// truncate wide lines, the widest over-pads short lines and the CTC rec is not
// perfectly padding-invariant. The proven standalone harness
// (tools/probes/apple/mps_ocr_funsd_bucket.mm, 84.6% F1) warps each line at natural aspect
// into the SMALLEST width bucket that fits (minimal padding => rec stays
// in-distribution). MpsRecognizer reproduces that inside the IRecognizer seam:
// load(base_dir) discovers rec_b{320,480,800,1200,1600} export subdirs and holds
// one MpsEngine per width; run() routes each box to its bucket, warps+recs each
// group in one command buffer, then reassembles results in original box order.
// load() also accepts a single export dir (contains graph.json) => one bucket.
class MpsRecognizer final : public backend::IRecognizer {
public:
  MpsRecognizer(std::shared_ptr<MetalAllocator> alloc, std::string dict_path);
  ~MpsRecognizer() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<backend::RecResult>
  run(const backend::ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      backend::DeviceQueue &queue) override;
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


  // One width bucket: its own compiled engine + per-batch device scratch.
  // The scratch is sized ONCE at load() for the bucket's top ladder rung (from
  // the shared recognition::batch_ladder_for_width), so nothing allocates in
  // the hot path either.
  struct Bucket {
    int width = 0;                    // rec input W (== export bucket width)
    // Exactly one executor is non-null. `ane` is chosen for narrow buckets when
    // a CoreML package exists (see MpsRecognizer::load); it is a HARDWARE
    // choice — the Apple Neural Engine is a third compute engine with no
    // analogue on CUDA — never a routing/policy choice. Which lines land in
    // this bucket and at which batch size is decided by the SHARED planner.
    std::unique_ptr<MpsEngine> engine;
    std::unique_ptr<AneRecEngine> ane;
    backend::DeviceBuffer h_buf;      // [maxb*9] inverse homographies
    backend::DeviceBuffer cw_buf;     // [maxb] content widths
    backend::DeviceBuffer crops_buf;  // [maxb,3,rech,width] warped crops
    backend::DeviceBuffer idx_buf;    // [maxb,seq] argmax indices (I32)
    backend::DeviceBuffer max_buf;    // [maxb,seq] argmax scores  (F32)
    std::vector<int> rungs;           // shared batch ladder, filtered for width
    int max_batch = 0;                // rungs.back()
    int T = 0;                        // rec time steps (== width/8 for rec_tiny)
  };

  std::shared_ptr<MetalAllocator> alloc_;
  MetalKernels kernels_;
  std::vector<Bucket> buckets_; // ascending by width
  std::vector<int> bucket_widths_;              // ascending, shared planner input
  std::vector<std::vector<int>> bucket_rungs_;  // per-bucket SUPPORTED batches
  std::vector<std::string> labels_;
  std::string dict_path_;
  int rech_ = 48;               // rec input H (shared across buckets)
  bool ready_ = false;
  // Reusable per-page scratch (no allocation in run()).
  std::vector<std::vector<int>> lists_;
};

// ---- Classification (text-line 0/180 angle) --------------------------------
// GOLDEN-DIFFED against OrtPaddleCls by tests/cpp/backends/turbo_golden.cpp:
//   DISABLE_COREML=1 turbo_golden --backend apple --ref cpu --stage cls \
//       --images <funsd_cache> --count 10
// which feeds this stage the CPU reference's boxes and compares flip decisions:
// measured 98.81% agreement on FUNSD pages 0-9 (ctest tripwire 0.98, registered
// as golden_apple_cls). The residual is Metal-bilinear vs cv::warpPerspective
// resampling in warp_crops, not the model — the now-deleted Apple-only harness
// cls_golden_apple.mm isolated that by feeding host crops to the graph directly
// and measured the forward pass bit-accurate to ORT (max prob delta 1.1e-5 over
// 2712 crops). load() takes the MPSGraph export of
// models/cls.onnx at its real [B,3,80,160] input:
//   python tools/modelgen/mps_export_rec.py models/cls.onnx <dir>/cls_b160 80 160
class MpsClassifier final : public backend::IClassifier {
public:
  explicit MpsClassifier(std::shared_ptr<MetalAllocator> alloc);
  ~MpsClassifier() override;

  [[nodiscard]] bool load(const std::string &model_path) override;
  void run(const backend::ImageView &img, std::vector<turbo_ocr::Box> &boxes,
          backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }

private:
  std::shared_ptr<MetalAllocator> alloc_;
  MetalKernels kernels_;
  MpsEngine engine_;
  // PP-OCRv5 text-line orientation classifier (PP-LCNet_x0_25) input; overridden
  // from the export's graph.json input_shape in load(). Defaults come from the
  // SHARED classification header — this was one of FIVE copies of 80/160/0.9,
  // and the threshold constant is now gone entirely (the decision lives in
  // classification::should_flip_180*).
  int clsh_ = turbo_ocr::classification::kClsImageH;
  int clsw_ = turbo_ocr::classification::kClsImageW;
  backend::DeviceBuffer h_buf_, cw_buf_, crops_buf_, idx_buf_, max_buf_;
  std::vector<int> rungs_;   // SHARED batch ladder filtered for the cls canvas
  int max_batch_ = 0;        // rungs_.back()
  bool ready_ = false;
};

// ---- Layout (STUB: multi-IO not yet supported) -----------------------------
class MpsLayout final : public backend::ILayout {
public:
  explicit MpsLayout(std::shared_ptr<MetalAllocator> alloc) : alloc_(std::move(alloc)) {}
  ~MpsLayout() override = default;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const backend::ImageView &img, int orig_h, int orig_w, float score_threshold,
      backend::DeviceQueue &queue) override;
  [[nodiscard]] bool is_ready() const noexcept override { return false; }

private:
  std::shared_ptr<MetalAllocator> alloc_;
};

} // namespace turbo_ocr::apple
