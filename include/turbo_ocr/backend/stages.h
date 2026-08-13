#pragma once

// Stage interfaces — IDetector / IRecognizer / IClassifier / ILayout.
//
// These are what collapse the two forked pipelines (OcrPipeline vs
// CpuOcrPipeline, PaddleDet vs OrtPaddleDet, …) into ONE orchestration. Each
// takes a device-resident ImageView plus a DeviceQueue and returns HOST types
// (vector<Box>, vector<pair<string,float>>, vector<LayoutBox>) — the boundary
// between device work and the host assembly logic (ctc_decode, det_postprocess,
// reading_order, docassembly) that is written exactly once above the seam.
//
// A TRT/CUDA impl, an MPSGraph/Metal impl, a MIGraphX impl, an OpenVINO impl,
// and an ORT-CPU impl all satisfy the same interface; the pipeline holds
// unique_ptr<IDetector> etc. and never names a vendor. Inputs are device
// buffers, outputs are host POD — so a backend keeps the image, the boxes, the
// crops, and the logits resident on its device end-to-end and only crosses back
// to the host at the (small) result.
//
// The existing NVIDIA signatures this generalizes:
//   PaddleDet::run(const GpuImage&, int orig_h, int orig_w, cudaStream_t)
//   PaddleRec::run(const GpuImage&, const vector<Box>&, cudaStream_t) + run_multi
//   PaddleCls::run(const GpuImage&, vector<Box>&, cudaStream_t)  (flips in place)
//   PaddleLayout::enqueue(GpuImage,…,stream) + collect(score)    (two-phase)
// each becomes GpuImage->ImageView, cudaStream_t->DeviceQueue&.

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/device_queue.h"     // DeviceQueue
#include "turbo_ocr/backend/image_view.h"        // ImageView
#include "turbo_ocr/base/geometry/box.h"       // turbo_ocr::Box
#include "turbo_ocr/core/types.h"              // (kept for OCRResultItem users)
#include "turbo_ocr/core/layout_types.h"       // turbo_ocr::layout::LayoutBox

namespace turbo_ocr::backend {

// One (text, confidence) pair per recognized crop — the exact host type the
// existing recognizers already return (matches PaddleRec/OrtPaddleRec).
using RecResult = std::pair<std::string, float>;

// ---- Two-phase stage handoff ----------------------------------------------
//
// StageFuture<T> is the ONE shape every asynchronous stage uses. A stage's
// enqueue() submits device work and returns immediately; collect() blocks until
// that work has completed and hands back the HOST result. Between the two calls
// the caller is free to submit OTHER work — which is the whole point: it is what
// lets image N+1's detection overlap image N's recognition.
//
// This generalizes the two-phase shape the main-tree NVIDIA PaddleLayout already
// proved (src/backends/nvidia/stages/paddle_layout.h:52,57 — enqueue(stream) then
// collect(score)), so CUDA can adopt it by renaming: enqueue() records its
// existing cudaEvent_t, collect() does the cudaEventSynchronize + the host
// post-process it already performs.
//
// Design notes:
//  * ONE virtual call per IMAGE, never per crop — the completion is a
//    std::function stored once per enqueue, not a hot-loop dispatch.
//  * The future OWNS whatever host state the completion needs (the backend
//    captures it), so no `Pending` member state is inherited by every impl.
//    ILayout used to carry exactly that in a base-class `pending_` — a bare,
//    non-owning ImageView stashed across the gap plus a `have_pending_` flag in
//    NvLayout that no destructor drained; it is a StageFuture now, so an
//    abandoned layout submission is waited for like every other stage's.
//  * ONE future outstanding per stage instance. Every stage that submits owns a
//    single set of device scratch (MpsDetector's out_buf_, PaddleDet's
//    d_output_, PaddleLayout's pinned staging), so a second enqueue() before
//    collect() overwrites what the first has not read. This is the contract for
//    every stage rather than a per-stage number: nothing in the pipeline keeps
//    two of one stage in flight, so a knob saying otherwise advertised a schedule
//    no caller could ask for.
//  * ready() is advisory. collect() is the only thing that guarantees ordering.
//  * A DEFAULT-CONSTRUCTED future is empty; collect() on it returns T{}.
template <class T>
class StageFuture {
public:
  StageFuture() = default;
  explicit StageFuture(std::function<T()> complete)
      : complete_(std::move(complete)) {}

  // A future over an ALREADY-computed value — how a synchronous backend
  // satisfies the async interface at zero cost.
  static StageFuture ready(T v) {
    return StageFuture([v = std::move(v)]() mutable { return std::move(v); });
  }

  [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(complete_); }

  // Block until the work completes and return the host result. Single-shot:
  // after collect() the future is empty.
  [[nodiscard]] T collect() {
    if (!complete_) return T{};
    auto f = std::move(complete_);
    complete_ = nullptr;
    return f();
  }

  // DRAINING DESTRUCTOR — the whole reason this is not `= default`.
  //
  // An abandoned future is not merely a lost result: the completion is the ONLY
  // thing that waits on the submission. Destroying it without collecting leaves
  // device work in flight while the buffers it reads are released, and on a
  // single-slot stage (MpsDetector: out_buf_ is allocated once at load() and
  // reused for EVERY page, single-slot) the next enqueue() then writes the same
  // buffer the abandoned pass is still writing.
  //
  // That is reachable today: run_pipelined() has an in-flight detection when it
  // calls layout/rec/router, none of which is wrapped, and /ocr/batch RETRIES
  // per-image on the same lease after a throw. So an exception mid-loop unwound
  // straight into a device-buffer race.
  //
  // noexcept + swallow: a destructor must not throw, and by this point the
  // result is already unwanted — we are draining for ORDERING, not for the
  // value.
  ~StageFuture() {
    if (!complete_) return;
    try {
      auto f = std::move(complete_);
      complete_ = nullptr;
      (void)f();
    } catch (...) { // NOLINT(bugprone-empty-catch) — reason above and below
      // Deliberately swallowed: see above.
    }
  }

  StageFuture(const StageFuture &) = delete;
  StageFuture &operator=(const StageFuture &) = delete;
  StageFuture(StageFuture &&) noexcept = default;
  StageFuture &operator=(StageFuture &&) noexcept = default;

private:
  std::function<T()> complete_;
};

using BoxesFuture = StageFuture<std::vector<turbo_ocr::Box>>;
using LayoutFuture = StageFuture<std::vector<turbo_ocr::layout::LayoutBox>>;

// The one default layout score threshold, so it is not re-spelled at every call
// site (it was duplicated in the pipeline AND defaulted in ILayout::collect).
inline constexpr float kDefaultLayoutScoreThreshold = 0.3f;

// A page image + the detection boxes to recognize within it, for batched
// multi-image recognition (generalizes recognition::ImageCrops{GpuImage; boxes}).
struct ImageCrops {
  ImageView img;
  std::vector<turbo_ocr::Box> boxes;
};

// ---- Detection -------------------------------------------------------------
// DB text detector. Runs resize+normalize, the model forward pass, threshold,
// and DB post-process (CCL + unclip) on the device, returns page-coordinate
// quad boxes to the host.
class IDetector {
public:
  virtual ~IDetector() = default;
  [[nodiscard]] virtual bool load(const std::string &model_path) = 0;

  // `orig_h`/`orig_w` are the ORIGINAL image dims (boxes are returned in
  // original coordinates even though the model runs on a resized canvas).
  [[nodiscard]] virtual std::vector<turbo_ocr::Box>
  run(const ImageView &img, int orig_h, int orig_w, DeviceQueue &queue) = 0;

  // --- Two-phase async detection (opt-in) -----------------------------------
  //
  // enqueue() submits the forward pass and returns without blocking; the
  // future's collect() waits for it and post-processes, returning boxes in
  // ORIGINAL coordinates exactly as run() does. run_pipelined() uses this to
  // keep two images in flight: collect N's boxes, enqueue N+1's detection, then
  // run cls+rec for N while N+1 detects.
  //
  // CONTRACT — the overlap is a property of the LANE, not of the call:
  //  * Submit on a lane of your OWN, not on `queue`. The pipeline runs N's
  //    cls+rec on `queue` right after this call, so submitting there just
  //    orders det(N+1) behind rec(N) and buys nothing but an early host return.
  //    `queue` is still passed because it names the device and because the
  //    synchronous default forwards to run(). MpsDetector is the reference.
  //  * Order your lane AFTER `queue`: the page was staged H2D there, and a
  //    private lane is unordered against it. Record a DeviceEvent on `queue`
  //    and wait it (device_queue.h). Free on unified memory — which is the only
  //    reason Apple gets away without it — mandatory on discrete VRAM.
  //  * ONE future outstanding per stage instance — collect() before the next
  //    enqueue(). See StageFuture above.
  //  * `img` must stay alive and unmodified until collect() returns (ImageView
  //    is non-owning).
  //
  // Default is fully synchronous, so a backend that ignores this compiles and
  // behaves as before and run_pipelined() degrades to the serial control flow.
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  [[nodiscard]] virtual BoxesFuture enqueue(const ImageView &img, int orig_h,
                                            int orig_w, DeviceQueue &queue) {
    return BoxesFuture::ready(run(img, orig_h, orig_w, queue));
  }

  // The largest `imgs.size()` run_batch() can process in ONE device submission.
  //
  // 1 (the default) means run_batch() has no native batched path — its default
  // implementation below just loops run(), which is correct but buys nothing. A
  // backend whose detector genuinely binds a [N,c,h,w] input returns N here.
  //
  // This is the CAPABILITY; BackendCaps::preferred_batch_size is the POLICY, and
  // only the POLICY switches coalescing on: the shared cross-request batcher
  // (include/turbo_ocr/pipeline/unified/stage_batcher.h) builds a scheduler only
  // when preferred_batch_size > 1, and then clamps it to this ceiling. State the
  // truth here freely — reporting a real ceiling changes no schedule, so a stage
  // never has to under-report its ability to avoid a side effect.
  [[nodiscard]] virtual int max_batch_size() const noexcept { return 1; }

  // Optional batched detection over N images (default: loop run()). `orig_dims`
  // is per-image {h, w}; returns one box list per input image, in input order.
  //
  // CONTRACT (relied on by the cross-request batcher): the returned vector has
  // EXACTLY imgs.size() entries in input order, and entry i depends only on
  // imgs[i]. Callers may pass images belonging to unrelated requests in one
  // call, so an implementation must not let one image's content influence
  // another's boxes (no shared thresholding across the batch, no per-batch
  // canvas chosen from image 0).
  [[nodiscard]] virtual std::vector<std::vector<turbo_ocr::Box>>
  run_batch(const std::vector<ImageView> &imgs,
            const std::vector<std::pair<int, int>> &orig_dims,
            DeviceQueue &queue) {
    std::vector<std::vector<turbo_ocr::Box>> out;
    out.reserve(imgs.size());
    for (std::size_t i = 0; i < imgs.size(); ++i)
      out.push_back(run(imgs[i], orig_dims[i].first, orig_dims[i].second, queue));
    return out;
  }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
};

// ---- Recognition -----------------------------------------------------------
// CRNN/CTC text recognizer. Warps+normalizes each box's crop, runs the batched
// forward pass, and does argmax + CTC decode; returns (text, score) per box.
class IRecognizer {
public:
  virtual ~IRecognizer() = default;
  [[nodiscard]] virtual bool load(const std::string &model_path) = 0;

  [[nodiscard]] virtual std::vector<RecResult>
  run(const ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
      DeviceQueue &queue) = 0;

  // One-time device-side preparation for THIS replica's queue, called from the
  // pipeline's warmup() before the first request. Recognition is the launch-
  // bound stage (a page's crops cost a few hundred small kernels), so a backend
  // that can pre-record the submission does it here rather than re-issue it per
  // request — NVIDIA bakes one CUDA graph per batch/width profile, measured at
  // +14% throughput. Best-effort: a backend that cannot pre-build returns
  // having done nothing and run() still works.
  virtual void warmup(DeviceQueue &queue) { (void)queue; }

  // Optional multi-image recognition (generalizes PaddleRec::run_multi). Returns
  // one result list per ImageCrops entry, in input order. Default: loop run().
  [[nodiscard]] virtual std::vector<std::vector<RecResult>>
  run_multi(const std::vector<ImageCrops> &items, DeviceQueue &queue) {
    std::vector<std::vector<RecResult>> out;
    out.reserve(items.size());
    for (const auto &it : items)
      out.push_back(run(it.img, it.boxes, queue));
    return out;
  }

  // Crops whose recognition FAILED during the last run()/run_multi() on this
  // object — an OOM'd chunk, a not-ready engine, a forward pass that threw.
  //
  // WHY IT IS HERE AND NOT IN EACH BACKEND. Every implementation pre-sizes its
  // result vector and writes an EMPTY entry for a failed crop, so the vector
  // still has one entry per box and the pipeline's under-return check cannot
  // see the loss: a partial recognition failure came back as a thin page with
  // no warning at all. The count is the only thing that distinguishes "these
  // crops were genuinely blank" from "these crops were never recognized", and it
  // must therefore live in the SHARED seam — a per-backend fix would leave the
  // other three silently wrong, which is exactly the defect class this project's
  // dedup rule exists to prevent.
  //
  // Contract: reset at the START of each run()/run_multi(), so it always
  // describes the call that just returned. Default 0 = "this backend does not
  // report", which is today's behaviour for anything that has not opted in.
  [[nodiscard]] virtual int last_dropped_crops() const noexcept { return 0; }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
};

// ---- Classification (text-line angle) --------------------------------------
// 0/180 orientation classifier. FLIPS boxes IN PLACE (rotates the quad corner
// order) for crops the model marks upside-down — matching PaddleCls's contract
// — so the boxes feed recognition directly.
//
// The flipped BOXES are the output; there is deliberately no flipped COUNT. It
// used to return one, but no caller ever read it and the two backends whose
// underlying class does not expose it (PaddleCls) hard-coded `return 0` — an
// interface that promises a number two implementations fabricate.
class IClassifier {
public:
  virtual ~IClassifier() = default;
  [[nodiscard]] virtual bool load(const std::string &model_path) = 0;

  virtual void run(const ImageView &img, std::vector<turbo_ocr::Box> &boxes,
                   DeviceQueue &queue) = 0;

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
};

// ---- Layout ----------------------------------------------------------------
// PP-DocLayoutV3 region detector (multi-IO: image + im_shape + scale_factor).
// The primary API is a single synchronous run() returning host LayoutBoxes in
// ORIGINAL coordinates. enqueue() is the same two-phase handoff det/rec use, so
// a backend that can overlap (NVIDIA on its own stream) submits and the pipeline
// collects after recognition; the default returns an already-computed future so
// a backend need only implement the synchronous path.
class ILayout {
public:
  virtual ~ILayout() = default;
  [[nodiscard]] virtual bool load(const std::string &model_path) = 0;

  [[nodiscard]] virtual std::vector<turbo_ocr::layout::LayoutBox>
  run(const ImageView &img, int orig_h, int orig_w, float score_threshold,
      DeviceQueue &queue) = 0;

  // --- Optional async overlap (opt-in) --------------------------------------
  //
  // Same contract as IDetector::enqueue: submit on a lane of your OWN ordered
  // after `queue`, ONE future outstanding, and `img` must stay alive and
  // unmodified until collect() returns. The pipeline gives an async layout its
  // own DeviceQueue and the cross-lane barrier (enqueue_layout_).
  //
  // `score_threshold` is taken HERE rather than at collect() because it is a
  // parameter of the request, not of the wait — the split version needed a
  // default argument on a virtual, which resolves from the STATIC type and so
  // was a second copy of the constant in every override.
  [[nodiscard]] virtual bool supports_async() const noexcept { return false; }

  [[nodiscard]] virtual LayoutFuture enqueue(const ImageView &img, int orig_h,
                                             int orig_w, float score_threshold,
                                             DeviceQueue &queue) {
    return LayoutFuture::ready(
        run(img, orig_h, orig_w, score_threshold, queue));
  }

  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
};

} // namespace turbo_ocr::backend
