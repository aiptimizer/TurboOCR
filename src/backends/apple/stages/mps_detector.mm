// MpsDetector implementation (see mps_stages.h) — split from mps_stages.mm,
// which holds the recognizer/classifier/layout stages. Mirrors
// tools/probes/apple/mps_ocr.mm's proven det path.

#import "apple/stages/mps_stages.h"
#import "apple/queue/metal_device_queue.h"
#import "apple/support/apple_profile.h"
#import "apple/support/apple_contention.h"
#import "apple/support/coreml_compile.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "apple/stages/stage_tier.h"
#include "turbo_ocr/base/env_utils.h"        // env::* — every read is recorded
#include "turbo_ocr/core/norm_params.h"      // SHARED norm factories
#include "turbo_ocr/core/db_post_config.h"   // shared DB geometry limits
#include "turbo_ocr/analysis/detection/det_config.h"      // read_db_params + env overrides
#include "turbo_ocr/analysis/detection/det_postprocess.h" // extract_boxes_from_bitmap
#include "turbo_ocr/core/model_catalog.h"    // kV6DetConfig[Tiny] per-tier DB base

namespace turbo_ocr::apple {

// ===========================================================================
// MpsDetector
// ===========================================================================

// Optional CoreML det forward (see the header). GPU compute units — this is a
// KERNEL-QUALITY experiment (CoreML's conv kernels vs MPSGraph's on the same
// GPU), not an engine move; MLComputeUnitsCPUAndGPU keeps it off the ANE,
// whose capacity the rec hybrid already uses.
struct MpsDetector::CoremlDet {
  MLModel *model = nil;
  NSString *in_key = nil;
  NSString *out_key = nil;
  // Serial lane for the async det path: enqueue() must not block on the
  // predict, and the stage is single-slot anyway (out_buf_).
  dispatch_queue_t lane = nullptr;
  dispatch_semaphore_t done = nullptr;
  bool pending_ok = false;

  bool load(const std::string &pkg, int c, int h, int w) {
    @autoreleasepool {
      NSString *p = [NSString stringWithUTF8String:pkg.c_str()];
      if (![[NSFileManager defaultManager] fileExistsAtPath:p]) {
        NSLog(@"[apple] det CoreML package not found: %@ — using MPSGraph", p);
        return false;
      }
      NSURL *cu = coreml_compiled_url(p);
      if (!cu) return false;
      MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
      cfg.computeUnits = MLComputeUnitsCPUAndGPU;
      NSError *err = nil;
      model = [MLModel modelWithContentsOfURL:cu configuration:cfg error:&err];
      if (!model) {
        NSLog(@"[apple] det CoreML load failed: %@ — using MPSGraph",
              err.localizedDescription);
        return false;
      }
      NSDictionary *ins = model.modelDescription.inputDescriptionsByName;
      NSDictionary *outs = model.modelDescription.outputDescriptionsByName;
      if (ins.count != 1 || outs.count < 1) {
        NSLog(@"[apple] det CoreML package has %lu inputs / %lu outputs — "
              @"expected 1/1+; using MPSGraph", (unsigned long)ins.count,
              (unsigned long)outs.count);
        model = nil;
        return false;
      }
      in_key = ins.allKeys[0];
      out_key = outs.allKeys[0];
      // The canvas is compiled into the MPSGraph engine AND baked into the
      // package; a mismatch would predict on a differently-scaled page and
      // return boxes for an image nobody sent.
      MLFeatureDescription *ind = ins[in_key];
      NSArray<NSNumber *> *shape = ind.multiArrayConstraint.shape;
      if (shape.count == 4 &&
          (shape[1].intValue != c || shape[2].intValue != h ||
           shape[3].intValue != w)) {
        NSLog(@"[apple] det CoreML input [%d,%d,%d] does not match the det "
              @"canvas [%d,%d,%d] — using MPSGraph", shape[1].intValue,
              shape[2].intValue, shape[3].intValue, c, h, w);
        model = nil;
        return false;
      }
      lane = dispatch_queue_create("turbo.det.coreml", DISPATCH_QUEUE_SERIAL);
      done = dispatch_semaphore_create(0);
      return true;
    }
  }

  // in/out are the HOST views of the Metal SHARED in_buf_/out_buf_ — the input
  // MLMultiArray wraps in-place (zero-copy); the prob map is copied out once.
  bool predict(float *in, float *out, int c, int h, int w) {
    @autoreleasepool {
      NSError *err = nil;
      MLMultiArray *x = [[MLMultiArray alloc]
          initWithDataPointer:in
                        shape:@[ @1, @(c), @(h), @(w) ]
                     dataType:MLMultiArrayDataTypeFloat32
                      strides:@[ @((long)c * h * w), @((long)h * w), @(w), @1 ]
                  deallocator:nil
                        error:&err];
      if (!x) return false;
      MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
          initWithDictionary:@{in_key : [MLFeatureValue featureValueWithMultiArray:x]}
                       error:&err];
      if (!fp) return false;
      id<MLFeatureProvider> o = [model predictionFromFeatures:fp error:&err];
      if (!o) {
        NSLog(@"[apple] det CoreML predict failed: %@", err.localizedDescription);
        return false;
      }
      MLMultiArray *pm = [o featureValueForName:out_key].multiArrayValue;
      if (!pm) return false;
      const size_t n = (size_t)h * w;
      // Convert by the DECLARED dtype (the rec score head taught us CoreML
      // freely returns fp16 where the source model said fp32).
      const MLMultiArrayDataType dt = pm.dataType;
      __block bool copied = false;
      [pm getBytesWithHandler:^(const void *bytes, NSInteger size) {
        if (dt == MLMultiArrayDataTypeFloat16 && (size_t)size >= n * 2) {
          const __fp16 *sp = (const __fp16 *)bytes;
          for (size_t i = 0; i < n; ++i) out[i] = (float)sp[i];
          copied = true;
        } else if ((size_t)size >= n * 4) {
          std::memcpy(out, bytes, n * sizeof(float));
          copied = true;
        }
      }];
      return copied;
    }
  }
};

MpsDetector::MpsDetector(std::shared_ptr<MetalAllocator> alloc)
    : alloc_(std::move(alloc)) {}
MpsDetector::~MpsDetector() = default;

bool MpsDetector::load(const std::string &model_path) {
  // SHARED DB policy, resolved ONCE: the bootstrap-installed per-model base
  // (det_config.h set_det_config_base — tiny's box_thresh is 0.40, medium/
  // small 0.45) + DET_* env overrides. This used to sniff the tier out of the
  // model PATH to pick the base — a private workaround for the plumbing the
  // unified merge dropped, and wrong for any custom-named model file; the
  // installed base carries the registry's pairing for every backend.
  db_ = turbo_ocr::detection::read_db_params();

  // CANVAS DISCOVERY, mirroring the recognizer's bucket discovery one class
  // below: a flat export (graph.json directly in model_path — the
  // pre-multi-canvas bundle layout) is one canvas; otherwise every
  // det_c<H>x<W>/ subdir carrying a graph.json becomes one. Each canvas gets
  // its own engine AND its own resident buffers — sharing one buffer pair
  // across shapes would make the single-slot async contract a lie the moment
  // two consecutive pages pick different canvases.
  std::vector<std::string> dirs;
  {
    std::error_code ec;
    if (std::filesystem::exists(std::filesystem::path(model_path) / "graph.json", ec)) {
      dirs.push_back(model_path);
    } else if (std::filesystem::is_directory(model_path, ec)) {
      for (const auto &e : std::filesystem::directory_iterator(model_path, ec)) {
        if (!e.is_directory()) continue;
        const std::string name = e.path().filename().string();
        int ph = 0, pw = 0;
        if (std::sscanf(name.c_str(), "det_c%dx%d", &ph, &pw) == 2 &&
            std::filesystem::exists(e.path() / "graph.json", ec)) {
          dirs.push_back(e.path().string());
        }
      }
      std::sort(dirs.begin(), dirs.end()); // deterministic load order
    }
  }
  if (dirs.empty()) {
    NSLog(@"[apple] det: no export (graph.json or det_c<H>x<W>/) under %s",
          model_path.c_str());
    return false;
  }
  for (const auto &d : dirs) {
    auto cv = std::make_unique<DetCanvas>();
    if (!cv->engine.load(d)) {
      NSLog(@"[apple] det canvas failed to load, skipping: %s", d.c_str());
      continue;
    }
    auto chw = cv->engine.input_chw();
    if (chw.size() == 3) { cv->c = chw[0]; cv->h = chw[1]; cv->w = chw[2]; }
    // Resident input canvas + prob-map output (DBNet output is [1,1,h,w]).
    cv->in_buf = alloc_->allocate_buffer((size_t)cv->c * cv->h * cv->w * sizeof(float));
    cv->out_buf = alloc_->allocate_buffer((size_t)cv->h * cv->w * sizeof(float));
    if (!cv->in_buf || !cv->out_buf) continue;
    canvases_.push_back(std::move(cv));
  }
  if (canvases_.empty()) return false;
  active_ = canvases_.front().get();
  if (canvases_.size() > 1) {
    std::string set;
    for (const auto &cv : canvases_)
      set += std::to_string(cv->h) + "x" + std::to_string(cv->w) + " ";
    NSLog(@"[apple] det canvases: %s(picked per page by the shared aspect policy)",
          set.c_str());
  }

  // Optional CoreML forward — an A/B knob, off unless the env names a package.
  // The package encodes ONE input shape; bind it to the loaded canvas whose
  // dims it accepts, and every other canvas keeps the MPSGraph engine. Every
  // failure path falls back loudly (never silently: a het det would return
  // DIFFERENT boxes, not slower ones).
  if (const std::string cm = env::env_or("TURBO_APPLE_DET_COREML", "");
      !cm.empty()) {
    for (const auto &cv : canvases_) {
      auto det = std::make_unique<CoremlDet>();
      if (det->load(cm, cv->c, cv->h, cv->w)) {
        coreml_ = std::move(det);
        coreml_canvas_ = cv.get();
        NSLog(@"[apple] det forward on CoreML(GPU) for canvas %dx%d: %s",
              cv->h, cv->w, cm.c_str());
        break;
      }
    }
  }
  ready_ = true;
  return ready_;
}

void MpsDetector::select_canvas_(int orig_h, int orig_w) {
  if (canvases_.size() <= 1) return;
  // Warmup/thumbnail inputs have no meaningful aspect — keep whatever canvas
  // is active rather than flapping on a 32x32 synthetic image.
  if (orig_h < 128 || orig_w < 128) return;
  std::vector<std::pair<int, int>> avail;
  avail.reserve(canvases_.size());
  for (const auto &cv : canvases_) avail.emplace_back(cv->h, cv->w);
  const auto [h, w] = turbo_ocr::detection::pick_det_canvas(orig_h, orig_w, avail);
  for (const auto &cv : canvases_)
    if (cv->h == h && cv->w == w) { active_ = cv.get(); return; }
}

bool MpsDetector::submit_forward_(const backend::ImageView &img,
                                  backend::DeviceQueue &queue) {
  // Resident preprocess: resize the page texture to the det canvas, normalize
  // with ImageNet stats on BGR planes (PaddleOCR DBNet) — the resident analogue
  // of mps_ocr.mm:74-79's host cv::resize+split. TODO(apple-det-preproc-golden): golden-diff vs cv::resize
  // (Metal bilinear vs INTER_LINEAR) before treating det as bit-locked.
  // SHARED det normalization (ImageNet over pixel/255, BGR-positional). Never
  // retyped here — see turbo_ocr/core/norm_params.h.
  const backend::NormParams det = backend::norm::imagenet_bgr();

  DetCanvas &cv = *active_;
  bool fwd_ok = false;
  {
    backend::BatchScope batch(queue); // fuse resize + forward in one command buffer
    kernels_.resize_normalize(img, static_cast<float *>(cv.in_buf.data()), cv.w,
                              cv.h, det, queue);
    backend::DeviceTensor in{
        .name = cv.engine.input_names().empty() ? std::string{} : cv.engine.input_names()[0],
        .data = cv.in_buf.data(), .space = backend::DeviceKind::Metal,
        .dtype = backend::DType::F32, .shape = {1, cv.c, cv.h, cv.w}};
    backend::DeviceTensor out{
        .name = {}, .data = cv.out_buf.data(), .space = backend::DeviceKind::Metal,
        .dtype = backend::DType::F32, .shape = {1, 1, cv.h, cv.w}};
    std::vector<backend::OutputLease> leases;
    fwd_ok = cv.engine.run({in}, {out}, leases, queue);
  } // BatchScope closes => the command buffer is COMMITTED here. Never
    // synchronize() with it still open (device_queue.h contract).
  return fwd_ok;
}

std::vector<turbo_ocr::Box> MpsDetector::db_postprocess_(int orig_h, int orig_w) {
  TURBO_APPLE_PROF("det.dbpost(host)");
  // HOST DB post-process — bit-identical to mps_ocr.mm:99-107, reading the prob
  // map through unified memory. Boxes come back in ORIGINAL coordinates.
  const DetCanvas &cvs = *active_;
  const float *map = static_cast<const float *>(cvs.out_buf.data());
  cv::Mat pred(cvs.h, cvs.w, CV_32F, const_cast<float *>(map));
  cv::Mat bitmap;
  // SHARED DB parameters (detection::read_db_params), resolved once at load().
  //
  // These three thresholds used to be hardcoded here (0.2 / 0.40 / 1.4), which
  // made DET_DB_THRESH / DET_BOX_THRESH / DET_UNCLIP dead on Apple and pinned
  // box_thresh at the TINY tier's value — 0.40 is correct for kV6DetConfigTiny
  // but small/medium want 0.45, so a tier switch silently kept tiny's recall
  // curve. Routing through the shared reader keeps the per-tier default AND
  // makes the env overrides work, without changing the tiny numbers (kDbDefaults
  // is 0.2/0.45/1.4; db_.box_thresh is set to the tier value at load()).
  // The geometry limits likewise come from detection::kMinBoxSide /
  // kMinUnclippedSide, which is where metal_kernels.mm's stray 2.0 diverged.
  cv::threshold(pred, bitmap, db_.thresh, 255, cv::THRESH_BINARY);
  bitmap.convertTo(bitmap, CV_8U);

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  std::vector<std::vector<cv::Point>> contours_buf;
  std::vector<cv::Vec4i> hier_buf;
  return turbo_ocr::detection::extract_boxes_from_bitmap(
      pred, bitmap, orig_h, orig_w, cvs.h, cvs.w, db_.box_thresh, db_.unclip_ratio,
      turbo_ocr::detection::kMinBoxSide,
      turbo_ocr::detection::kMinUnclippedSide,
      shifted_buf, mask_buf, contours_buf, hier_buf);
}

// SHARED canvas decision for a FIXED-CANVAS detector.
//
// MPSGraph compiles one graph per static input shape, so this backend holds a
// nearest loaded det canvas (each from its export's graph.json) and a page far
// from every canvas's aspect is
// resized onto it. That used to be an Apple-LOCAL stretch: the shared
// aspect-preserving resize policy (short >= 64, long <= 1280, /32-rounded) was
// never consulted, which made DET_LIMIT_TYPE / DET_LIMIT_SIDE_LEN / DET_MAX_SIDE
// dead on this backend and diverged the effective det input from every other
// backend for varied-aspect documents. (MEASURED: on FUNSD the canvas choice
// moves F1 by < 0.2pt because every page has a similar aspect — the divergence
// is real but invisible in that gate.)
//
// The decision now goes through detection::pick_det_canvas(), the SHARED
// expression of "run the normal policy, then map it onto a canvas the backend
// actually has". With one export the answer is always that export, so nothing
// changes numerically — but the constraint is explicit, shared-owned, and it
// WARNS when the page's policy aspect is far from the canvas's, which is the
// signal that a second export is needed. Exporting more canvases (and passing
// them all here) is then a data change, not a code change.
void MpsDetector::check_canvas_policy_(int orig_h, int orig_w) const {
  // Ignore warmup/thumbnail inputs: a synthetic 32x32 warmup image has no
  // meaningful aspect and would fire the warning on every process start.
  if (orig_h < 128 || orig_w < 128) return;
  std::vector<std::pair<int, int>> have;
  have.reserve(canvases_.size());
  for (const auto &cv : canvases_) have.emplace_back(cv->h, cv->w);
  const auto [ch, cw] = turbo_ocr::detection::pick_det_canvas(
      orig_h, orig_w, have);
  const auto [want_h, want_w] = turbo_ocr::detection::compute_det_resize(
      orig_h, orig_w, turbo_ocr::detection::read_det_resize());
  const double want_ar = (double)want_w / std::max(1, want_h);
  const double have_ar = (double)cw / std::max(1, ch);
  const double err = std::abs(std::log(have_ar / want_ar));
  if (err > 0.15) { // ~16% aspect deviation even from the BEST loaded canvas
    static bool warned = false;
    if (!warned) {
      warned = true;
      NSLog(@"[apple] nearest det canvas %dx%d is %.0f%% off the shared resize "
            @"policy's %dx%d for this page shape (loaded canvases: %zu). "
            @"MPSGraph is single-shape per canvas, so the page is stretched. "
            @"Export a closer canvas (tools/modelgen/apple/) if this corpus "
            @"has varied aspects.",
            ch, cw, err * 100.0, want_h, want_w, canvases_.size());
    }
  }
}

std::vector<turbo_ocr::Box> MpsDetector::run(const backend::ImageView &img,
                                             int orig_h, int orig_w,
                                             backend::DeviceQueue &queue) {
  if (!ready_ || img.empty()) return {};
  select_canvas_(orig_h, orig_w);
  check_canvas_policy_(orig_h, orig_w);
  // The CoreML package is bound to ONE canvas; other canvases use MPSGraph.
  if (coreml_ && active_ == coreml_canvas_) {
    // Resize+normalize stays a Metal kernel (its own command buffer, CHECKED
    // sync — CoreML reads in_buf_ on the host side right after); only the
    // forward pass moves to CoreML. Same in_buf_ in, same out_buf_ out, so
    // db_postprocess_ is byte-for-byte the MPSGraph path's.
    bool ok;
    {
      TURBO_APPLE_PROF("det.coreml(resize+sync)");
      auto &mq = as_metal(queue);
      const unsigned long long mark = mq.sync_mark();
      {
        backend::BatchScope batch(queue);
        kernels_.resize_normalize(img, static_cast<float *>(active_->in_buf.data()),
                                  active_->w, active_->h,
                                  backend::norm::imagenet_bgr(), queue);
      }
      ok = mq.sync_ok(mark);
    }
    if (ok) {
      TURBO_APPLE_PROF("det.coreml(predict)");
      TURBO_APPLE_STAT(det_coreml);
      ok = coreml_->predict(static_cast<float *>(active_->in_buf.data()),
                            static_cast<float *>(active_->out_buf.data()),
                            active_->c, active_->h, active_->w);
    }
    if (!ok) {
      NSLog(@"[apple] det CoreML forward FAILED — returning no boxes for this "
            @"page (out_buf_ holds the previous page's prob map)");
      return {};
    }
    TURBO_APPLE_STAT(det_dbpost);
    return db_postprocess_(orig_h, orig_w);
  }
  bool fwd_ok;
  {
    TURBO_APPLE_PROF("det.gpu(resize+fwd+sync)");
    TURBO_APPLE_STAT(det_gpu);
    auto &mq = as_metal(queue);
    const unsigned long long mark = mq.sync_mark();
    { TURBO_APPLE_STAT(det_encode); fwd_ok = submit_forward_(img, queue); }
    // sync_ok(), not synchronize(): ENCODE success says nothing about whether
    // the GPU ran the work. A command buffer that fails at EXECUTION leaves
    // out_buf_ holding the previous page's bytes while every encode-level check
    // still reports success — that is the whole-page-mix-up signature.
    { TURBO_APPLE_STAT(det_sync); fwd_ok = mq.sync_ok(mark) && fwd_ok; }
  }
  // out_buf_ is allocated once at load() and reused for EVERY page. If the
  // forward pass did not run, it still holds the previous page's probability
  // map, and the DB post-process would happily return THAT page's boxes.
  if (!fwd_ok) {
    NSLog(@"[apple] det forward FAILED — returning no boxes for this page "
          @"(out_buf_ holds the previous page's prob map)");
    return {};
  }
  TURBO_APPLE_STAT(det_dbpost);
  return db_postprocess_(orig_h, orig_w);
}

bool MpsDetector::supports_async() const noexcept {
  static const bool on = env::env_or("TURBO_APPLE_DET_ASYNC", "1") != "0";
  return ready_ && on;
}

backend::BoxesFuture MpsDetector::enqueue(const backend::ImageView &img,
                                          int orig_h, int orig_w,
                                          backend::DeviceQueue &queue) {
  if (!supports_async() || img.empty())
    return backend::BoxesFuture::ready(run(img, orig_h, orig_w, queue));

  // Private lane — see the header for why the shared queue cannot be used.
  if (!async_q_) async_q_ = std::make_unique<MetalDeviceQueue>();

  select_canvas_(orig_h, orig_w);
  if (coreml_ && active_ == coreml_canvas_) {
    // Async shape of the CoreML path: the resize rides the private lane, the
    // predict runs on the CoreML serial lane so enqueue() returns without
    // blocking, and collect() waits the semaphore. Single-slot (out_buf_ and
    // pending_ok), exactly like the MPSGraph future below.
    check_canvas_policy_(orig_h, orig_w);
    auto &amq = as_metal(*async_q_);
    const unsigned long long mark = amq.sync_mark();
    {
      backend::BatchScope batch(*async_q_);
      kernels_.resize_normalize(img, static_cast<float *>(active_->in_buf.data()),
                                active_->w, active_->h,
                                backend::norm::imagenet_bgr(), *async_q_);
    }
    CoremlDet *cd = coreml_.get();
    dispatch_async(cd->lane, ^{
      // The GPU wait inside sync_ok is bounded (TURBO_APPLE_GPU_TIMEOUT_MS)
      // and CoreML's predict returns an error rather than hanging, so the
      // semaphore below is always signalled.
      bool ok = as_metal(*async_q_).sync_ok(mark);
      if (ok)
        ok = cd->predict(static_cast<float *>(active_->in_buf.data()),
                         static_cast<float *>(active_->out_buf.data()),
                         active_->c, active_->h, active_->w);
      cd->pending_ok = ok;
      dispatch_semaphore_signal(cd->done);
    });
    return backend::BoxesFuture([this, cd, orig_h, orig_w] {
      {
        TURBO_APPLE_PROF("det.coreml(wait)");
        dispatch_semaphore_wait(cd->done, DISPATCH_TIME_FOREVER);
      }
      if (!cd->pending_ok) {
        NSLog(@"[apple] det CoreML forward FAILED (async) — returning no boxes");
        return std::vector<turbo_ocr::Box>{};
      }
      return db_postprocess_(orig_h, orig_w);
    });
  }

  auto &amq = as_metal(*async_q_);
  const unsigned long long mark = amq.sync_mark();
  const bool fwd_ok = submit_forward_(img, *async_q_);
  // The completion captures ONLY host scalars; the caller guarantees `img`
  // outlives collect() (IDetector::enqueue contract) and out_buf_ belongs to
  // this stage, whose single slot may not be re-enqueued before collect().
  return backend::BoxesFuture([this, orig_h, orig_w, fwd_ok, mark] {
    bool ok = fwd_ok;
    {
      TURBO_APPLE_PROF("det.async(wait)");
      // Same reasoning as the synchronous path: encode success does not imply
      // execution success, and this lane is the one that runs under the
      // heaviest cross-client contention.
      ok = as_metal(*async_q_).sync_ok(mark) && ok;
    }
    if (!ok) {
      NSLog(@"[apple] det forward FAILED (async) — returning no boxes");
      return std::vector<turbo_ocr::Box>{};
    }
    return db_postprocess_(orig_h, orig_w);
  });
}


} // namespace turbo_ocr::apple
