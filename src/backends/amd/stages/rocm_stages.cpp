// rocm_stages.cpp — device-resident ROCm implementations of IDetector,
// IRecognizer, IClassifier, ILayout. Structure mirrors the NVIDIA reference
// (paddle_det / paddle_rec / paddle_cls / paddle_layout): MIGraphX forward pass
// with hipMalloc I/O, HipKernels pre/post, data resident until the small host
// result. Host decode seams (CTC collapse, DB rescale, layout parse) are the
// ONLY host crossings and reuse the shared CUDA-free geometry headers.

#include "amd/stages/rocm_stages.h"

#include "amd/support/hip_check.h"
#include "amd/engine/migraphx_engine.h"
#include "amd/memory/hip_allocator.h"
#include "amd/queue/hip_queue.h"

// SHARED policy headers. Nothing in this file may re-derive what these define —
// the det resize/DB parameters, the rec width buckets, the rec batch ladder, and
// the CTC collapse are all cross-backend policy. A private copy here is exactly
// how the Apple rec-ladder clamping bug happened (plan, DEDUPLICATION RULE 2).
#include "turbo_ocr/analysis/classification/cls_config.h"          // cls canvas/thresh/norm/flip
#include "turbo_ocr/base/geometry/perspective.h"        // compute_crop_transform
#include "turbo_ocr/core/norm_params.h"                 // norm::imagenet_bgr/rec/layout
#include "turbo_ocr/analysis/detection/det_config.h"               // det resize + DB params
#include "turbo_ocr/analysis/layout/layout_postfilter.h"
#include "turbo_ocr/analysis/layout/picodet_decode.h"              // decode_picodet_rows
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/recognition/ctc_decode.h"             // ctc_greedy_decode, dict
#include "turbo_ocr/analysis/recognition/rec_batching.h"           // plan_rec_batches, matrix
#include "turbo_ocr/analysis/recognition/rec_geometry.h"           // rec widths / buckets

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace turbo_ocr::amd {

namespace {

using backend::DType;
using backend::DeviceTensor;
using backend::OutputLease;

// Rec canvas height (mirrors kRecImageH in the NVIDIA path).
constexpr int kRecImageH = 48;
// Cls canvas + threshold + norm + flip come from the SHARED classification
// header. They used to be four private constants here — one of FIVE copies of
// the same three numbers, and the place AMD's `>=` threshold comparison and its
// hand-rolled cyclic quad rotation diverged from every other backend.
using turbo_ocr::classification::kClsImageH;
using turbo_ocr::classification::kClsImageW;


// Grow a cached device scratch buffer, EXCEPTION-SAFELY.
//
// Every one of these caches used to be `if (p) free(p); p = allocate(n); cap = n;`
// — which was fine while HIP_CHECK aborted the process on any error. It no
// longer does (it throws now, matching NVIDIA's recoverable CUDA_CHECK), and a
// throwing allocator would leave `p` holding a FREED pointer with `cap` still
// claiming the old size: the next request on this POOLED, long-lived stage
// would either skip the regrow (cap looks big enough) and write through a
// dangling pointer, or free it a second time.
//
// Releasing and CLEARING before the call that can throw makes the failure state
// simply "empty", which the next ensure() regrows correctly.
template <class T>
void grow_scratch(backend::IDeviceAllocator *alloc, T *&ptr, std::size_t &cap,
                  std::size_t want_bytes) {
  if (want_bytes <= cap) return;
  if (ptr) alloc->free(ptr);
  ptr = nullptr;
  cap = 0;                       // ordered BEFORE allocate(): see above
  ptr = static_cast<T *>(alloc->allocate(want_bytes));
  cap = want_bytes;
}

// Host (pinned) twin of grow_scratch, same ordering contract.
template <class T>
void grow_scratch_host(backend::IDeviceAllocator *alloc, T *&ptr,
                       std::size_t &cap, std::size_t want_bytes) {
  if (want_bytes <= cap) return;
  if (ptr) alloc->free_host(ptr);
  ptr = nullptr;
  cap = 0;
  ptr = static_cast<T *>(alloc->allocate_host(want_bytes));
  cap = want_bytes;
}

// Build a single-input warmup variant [batch, 3, h, w].
MIGraphXEngine::ShapeVariant nchw_variant(int batch, int h, int w) {
  return MIGraphXEngine::ShapeVariant{
      {{static_cast<std::int64_t>(batch), 3, static_cast<std::int64_t>(h),
        static_cast<std::int64_t>(w)}}};
}

} // namespace

// ===========================================================================
// RocmDetector
// ===========================================================================
struct RocmDetector::Impl {
  StageDeps deps;
  MIGraphXEngine engine;
  // SHARED det policy, read once at load(): resize limits (incl. the DET_*
  // env overrides) and the DB thresholds. NOT re-derived here — a private
  // max_side/box_thresh in this file would silently give AMD a different
  // detection recall from every other backend.
  detection::DetResizeParams resize_ = detection::kDetResizeDefault;
  detection::DbParams db_ = detection::kDbDefaults;

  // Cached device scratch (grown to the largest canvas seen).
  float *d_input = nullptr;      // [3*H*W]
  std::uint8_t *d_bitmap = nullptr; // [H*W]
  std::size_t input_cap = 0, bitmap_cap = 0;

  explicit Impl(StageDeps d) : deps(d), engine(d.device_id) {}
  ~Impl() {
    if (d_input) deps.alloc->free(d_input);
    if (d_bitmap) deps.alloc->free(d_bitmap);
  }
  void ensure(int W, int H) {
    std::size_t in = static_cast<std::size_t>(3) * W * H * sizeof(float);
    std::size_t bm = static_cast<std::size_t>(W) * H * sizeof(std::uint8_t);
    grow_scratch(deps.alloc, d_input, input_cap, in);
    grow_scratch(deps.alloc, d_bitmap, bitmap_cap, bm);
  }
};

RocmDetector::RocmDetector(StageDeps deps, std::string)
    : p_(std::make_unique<Impl>(deps)) {}
RocmDetector::~RocmDetector() = default;

bool RocmDetector::load(const std::string &model_path) {
  p_->engine.set_fp16(true); // det runs FP16 on the CUDA path
  // Shared policy + its env overrides, resolved once.
  p_->resize_ = detection::read_det_resize();
  p_->db_ = detection::read_db_params();
  ready_ = p_->engine.load(model_path);
  // NOTE (on-hardware, det shape explosion): unlike rec, the det canvas is a
  // FUNCTION OF THE PAGE ASPECT (compute_det_resize returns per-image /32 dims),
  // so there is no small ladder to pre-compile — a mixed corpus can present
  // dozens of distinct (H,W). MIGraphX compiles statically, so each new canvas
  // is one hot-path compile (logged loudly by the engine) and thereafter a cache
  // hit. Two real fixes to evaluate on hardware, in order of preference:
  //   1. MIGraphX dynamic dimensions (onnx_options dyn-dim / dynamic batch) for
  //      the det input, giving ONE program for all canvases;
  //   2. the persisted .mxr cache (engine TODO), so the compiles are paid once
  //      per fleet rather than once per process.
  // Do NOT "fix" this by forcing a single fixed det canvas: the resize policy is
  // shared, and changing it here alone would give AMD different recall from
  // every other backend.
  return ready_;
}

std::vector<turbo_ocr::Box>
RocmDetector::run(const ImageView &img, int orig_h, int orig_w,
                  DeviceQueue &queue) {
  if (!ready_ || img.empty())
    return {};

  // SHARED resize policy (PaddleOCR resize_image_type0 + max_side cap + /32
  // rounding), identical to the CUDA and CPU detectors.
  const auto [H, W] = detection::compute_det_resize(orig_h, orig_w, p_->resize_);
  p_->ensure(W, H);

  // SHARED det normalization: ImageNet mean/std over pixel/255, BGR-POSITIONAL.
  // This file used to build the params by hand and never set `order`, so it kept
  // the struct default (RGB) while every other backend feeds det BGR planes —
  // R and B swapped on AMD detection only. Use the factory; never retype it.
  p_->deps.kernels->resize_normalize(img, p_->d_input, W, H,
                                     backend::norm::imagenet_bgr(), queue);

  // Forward pass. MIGraphX output arrives as a device lease (pred map).
  std::vector<DeviceTensor> ins = {DeviceTensor{
      p_->engine.input_names().empty() ? "x" : p_->engine.input_names()[0],
      p_->d_input, backend::DeviceKind::Hip, DType::F32, p_->deps.device_id,
      {1, 3, H, W}}};
  std::vector<OutputLease> leases;
  if (!p_->engine.run(ins, {}, leases, queue) || leases.empty())
    return {};
  const float *d_pred = static_cast<const float *>(leases[0].data);

  // Threshold -> bitmap, then GPU CCL + Euclidean unclip -> resized-coord boxes.
  // Thresholds come from the SHARED DbParams, so AMD binarizes and filters
  // exactly where CUDA/CPU do.
  p_->deps.kernels->threshold(d_pred, p_->d_bitmap, W, H, 1, p_->db_.thresh,
                              queue);
  // SHARED thresholds -> seam params via the one conversion (kernels.h). The
  // remaining fields (expand clamp, sliver limits, component budget) now default
  // from detection::kDbDefaults / the shared geometry limits rather than from
  // this file's old private values.
  const auto db = backend::db_post_params(p_->db_, /*oriented=*/true);
  auto boxes = p_->deps.kernels->db_postprocess(d_pred, p_->d_bitmap, W, H, db,
                                                queue);

  // Rescale from resized canvas back to original image coordinates.
  const float sx = static_cast<float>(orig_w) / W;
  const float sy = static_cast<float>(orig_h) / H;
  for (auto &b : boxes)
    for (auto &pt : b.pts) {
      pt[0] = std::clamp(static_cast<int>(std::lround(pt[0] * sx)), 0, orig_w - 1);
      pt[1] = std::clamp(static_cast<int>(std::lround(pt[1] * sy)), 0, orig_h - 1);
    }
  return boxes;
}

// ===========================================================================
// RocmRecognizer
// ===========================================================================
struct RocmRecognizer::Impl {
  StageDeps deps;
  MIGraphXEngine engine;
  std::vector<std::string> dict;

  float *d_batch = nullptr;      // [B*3*H*W]
  float *d_M_invs = nullptr;     // [B*9]
  int *d_crop_w = nullptr;       // [B]
  int *d_idx = nullptr;          // [B*seq]
  float *d_scores = nullptr;     // [B*seq]
  float *h_M_invs = nullptr;     // pinned
  int *h_crop_w = nullptr;       // pinned
  int *h_idx = nullptr;          // pinned
  float *h_scores = nullptr;     // pinned
  std::size_t batch_cap = 0, bm_cap = 0, out_cap = 0;

  explicit Impl(StageDeps d) : deps(d), engine(d.device_id) {}
  ~Impl() {
    auto f = [&](void *p) { if (p) deps.alloc->free(p); };
    auto fh = [&](void *p) { if (p) deps.alloc->free_host(p); };
    f(d_batch); f(d_M_invs); f(d_crop_w); f(d_idx); f(d_scores);
    fh(h_M_invs); fh(h_crop_w); fh(h_idx); fh(h_scores);
  }
  void ensure_batch(int B, int W) {
    std::size_t need = static_cast<std::size_t>(B) * 3 * kRecImageH * W * sizeof(float);
    grow_scratch(deps.alloc, d_batch, batch_cap, need);
    std::size_t bm = static_cast<std::size_t>(B);
    // Four buffers keyed on ONE capacity: each grows through the exception-safe
    // helper, and bm_cap is published only after all four have succeeded — a
    // throw part-way leaves the grown ones valid and bm_cap unchanged, so the
    // next call simply regrows the rest.
    if (bm > bm_cap) {
      std::size_t c_dM = 0, c_dW = 0, c_hM = 0, c_hW = 0;
      grow_scratch(deps.alloc, d_M_invs, c_dM, bm * 9 * sizeof(float));
      grow_scratch(deps.alloc, d_crop_w, c_dW, bm * sizeof(int));
      grow_scratch_host(deps.alloc, h_M_invs, c_hM, bm * 9 * sizeof(float));
      grow_scratch_host(deps.alloc, h_crop_w, c_hW, bm * sizeof(int));
      bm_cap = bm;
    }
  }
  void ensure_out(std::size_t elems) {
    if (elems > out_cap) {
      std::size_t c_di = 0, c_ds = 0, c_hi = 0, c_hs = 0;
      grow_scratch(deps.alloc, d_idx, c_di, elems * sizeof(int));
      grow_scratch(deps.alloc, d_scores, c_ds, elems * sizeof(float));
      grow_scratch_host(deps.alloc, h_idx, c_hi, elems * sizeof(int));
      grow_scratch_host(deps.alloc, h_scores, c_hs, elems * sizeof(float));
      out_cap = elems;
    }
  }
};

RocmRecognizer::RocmRecognizer(StageDeps deps)
    : p_(std::make_unique<Impl>(deps)) {}
RocmRecognizer::~RocmRecognizer() = default;

bool RocmRecognizer::load(const std::string &model_path) {
  p_->engine.set_fp16(true);
  if (!(ready_ = p_->engine.load(model_path)))
    return false;

  // PERFORMANCE GATE: pre-compile every (width, batch) program the recognizer
  // can ever ask for, HERE, at load — so run() is only ever a cache hit. The
  // matrix is the SHARED one (recognition::rec_shape_matrix over
  // kRecWidthBuckets x the element-budgeted batch ladder), so the AMD program
  // set, the TRT profile set and the Apple executable set are generated from one
  // table and cannot drift.
  //
  // HONEST COST: this is O(tens) of MIGraphX graph compiles at startup and will
  // take real wall-clock time on first boot. That is the deliberate trade the
  // gate demands (never compile during a request); the persisted .mxr cache in
  // the engine TODO is what makes it a one-time cost per fleet.
  const auto matrix = recognition::rec_shape_matrix(
      recognition::kRecWidthBuckets, kRecImageH);
  std::vector<MIGraphXEngine::ShapeVariant> variants;
  variants.reserve(matrix.size());
  for (const auto &[w, b] : matrix)
    variants.push_back(nchw_variant(b, kRecImageH, w));
  const std::size_t ok = p_->engine.warmup(variants);
  if (ok != variants.size())
    std::fprintf(stderr,
                 "[RocmRecognizer] warmup compiled %zu/%zu (width,batch) "
                 "programs; the missing ones will compile on first use\n",
                 ok, variants.size());
  return ready_;
}

bool RocmRecognizer::load_dict(const std::string &dict_path) {
  // SHARED loader: it establishes the label vector's exact layout —
  // index 0 == "blank", then the file's tokens, then a trailing " ". The CTC
  // decoder indexes label_list[c] directly against that layout. A local
  // line-reader here (as this file used to have) reintroduces the off-by-one
  // that the shared pair exists to prevent.
  p_->dict.clear();
  p_->dict.push_back("blank");
  if (!recognition::load_label_dict(dict_path, p_->dict)) {
    p_->dict.clear();
    return false;
  }
  return p_->dict.size() > 1;
}

std::vector<backend::RecResult>
RocmRecognizer::run(const ImageView &img, const std::vector<turbo_ocr::Box> &boxes,
                    DeviceQueue &queue) {
  const int nboxes = static_cast<int>(boxes.size());
  dropped_crops_ = 0; // per-run; see last_dropped_crops()
  if (!ready_ || nboxes == 0 || img.empty())
    return std::vector<backend::RecResult>(std::max(0, nboxes));

  // ---- SHARED routing + batching policy -----------------------------------
  // group_by_width_bucket + snap_batch + chunking are ONE shared implementation
  // (recognition::plan_rec_batches). This backend supplies no routing opinion at
  // all: it receives whole (bucket, offset, count, batch) chunks and executes
  // them. That is what keeps a bucketing fix applied once from ever missing AMD,
  // and it is also the fast shape: one warp + one forward + one argmax per
  // chunk, no per-crop virtual dispatch.
  std::vector<std::vector<int>> bucket_lists;
  const auto plan = recognition::plan_rec_batches(
      boxes, kRecImageH, recognition::kRecWidthBuckets, bucket_lists);

  std::vector<backend::RecResult> out(nboxes);
  hipStream_t s = hip_stream_of(queue);

  for (const auto &chunk : plan) {
    const int bucket_w = recognition::kRecWidthBuckets[chunk.bucket];
    const int B = chunk.batch;   // the STATIC batch the program is compiled for
    const int n = chunk.count;   // real crops; slots [n, B) are padding
    const auto &idxs = bucket_lists[chunk.bucket];

    p_->ensure_batch(B, bucket_w);

    // Per-crop inverse transform + content width. Padding slots get an identity
    // transform and width 0, so warp_crops writes normalized zeros there (the
    // kernel's x >= crop_width branch) — never a stale previous crop.
    for (int i = 0; i < B; ++i) {
      float *m = p_->h_M_invs + i * 9;
      if (i < n) {
        const auto ct = turbo_ocr::compute_crop_transform(
            boxes[idxs[chunk.offset + i]], kRecImageH,
            recognition::kMaxRecWidth);
        std::copy(ct.M_inv, ct.M_inv + 9, m);
        p_->h_crop_w[i] = std::min(
            std::max(ct.crop_width, recognition::kMinRecWidth), bucket_w);
      } else {
        for (int k = 0; k < 9; ++k) m[k] = (k % 4 == 0) ? 1.0f : 0.0f;
        p_->h_crop_w[i] = 0;
      }
    }
    HIP_CHECK(hipMemcpyAsync(p_->d_M_invs, p_->h_M_invs,
                             (std::size_t)B * 9 * sizeof(float),
                             hipMemcpyHostToDevice, s));
    HIP_CHECK(hipMemcpyAsync(p_->d_crop_w, p_->h_crop_w,
                             (std::size_t)B * sizeof(int),
                             hipMemcpyHostToDevice, s));

    // SHARED rec normalization ((pixel/255 - 0.5)/0.5, RGB) — the factory, not
    // the struct default, so the intent is explicit at the call site.
    p_->deps.kernels->warp_crops(img, p_->d_M_invs, p_->d_crop_w, p_->d_batch, B,
                                 kRecImageH, bucket_w,
                                 backend::norm::rec_norm(), queue);

    std::vector<DeviceTensor> ins = {DeviceTensor{
        p_->engine.input_names().empty() ? "x" : p_->engine.input_names()[0],
        p_->d_batch, backend::DeviceKind::Hip, DType::F32, p_->deps.device_id,
        {B, 3, kRecImageH, bucket_w}}};
    std::vector<OutputLease> leases;
    if (!p_->engine.run(ins, {}, leases, queue) || leases.empty()) {
      // Entries stay pre-sized and empty, so the returned length still equals
      // boxes.size() — report the loss through the SHARED seam instead.
      // Count the REAL crops (n), not the padded static batch B: padding slots
      // never held text, and intel/apple already count n — a per-backend split
      // here silently mis-calibrates any alert threshold on this metric by 2x.
      dropped_crops_ += n;
      TOCR_LOG_ERROR_RL("rocm rec forward failed; dropping crops", "crops", n);
      continue;
    }

    // logits lease shape: [B, seq, classes]
    const auto &sh = leases[0].shape;
    if (sh.size() != 3) {
      dropped_crops_ += n; // real crops, not the padded batch (see above)
      TOCR_LOG_ERROR_RL("rocm rec logits rank unexpected; dropping crops",
                        "crops", n, "rank", static_cast<int>(sh.size()));
      continue;
    }
    const int seq = static_cast<int>(sh[1]);
    const int classes = static_cast<int>(sh[2]);
    const float *d_logits = static_cast<const float *>(leases[0].data);

    p_->ensure_out(static_cast<std::size_t>(B) * seq);
    p_->deps.kernels->argmax(d_logits, p_->d_idx, p_->d_scores, B, seq, classes,
                             queue);
    // Only the REAL crops' argmax rows are copied back; the padding tail stays
    // on the device and is never decoded.
    HIP_CHECK(hipMemcpyAsync(p_->h_idx, p_->d_idx,
                             (std::size_t)n * seq * sizeof(int),
                             hipMemcpyDeviceToHost, s));
    HIP_CHECK(hipMemcpyAsync(p_->h_scores, p_->d_scores,
                             (std::size_t)n * seq * sizeof(float),
                             hipMemcpyDeviceToHost, s));
    HIP_CHECK(hipStreamSynchronize(s));

    // SHARED greedy CTC collapse — the same function the CUDA path calls after
    // its on-GPU argmax, over the same "blank at index 0" label layout that
    // load_dict() built. No local collapse loop lives in this backend.
    for (int i = 0; i < n; ++i)
      out[idxs[chunk.offset + i]] = recognition::ctc_greedy_decode(
          p_->h_idx + (std::size_t)i * seq, p_->h_scores + (std::size_t)i * seq,
          seq, p_->dict);
  }
  return out;
}

// ===========================================================================
// RocmClassifier (text-line 0/180 angle; flips boxes in place)
// ===========================================================================
struct RocmClassifier::Impl {
  StageDeps deps;
  MIGraphXEngine engine;
  float *d_batch = nullptr;
  float *d_M_invs = nullptr;
  int *d_crop_w = nullptr;
  float *h_M_invs = nullptr;
  int *h_crop_w = nullptr;
  std::size_t batch_cap = 0, bm_cap = 0;
  explicit Impl(StageDeps d) : deps(d), engine(d.device_id) {}
  ~Impl() {
    auto f = [&](void *p) { if (p) deps.alloc->free(p); };
    auto fh = [&](void *p) { if (p) deps.alloc->free_host(p); };
    f(d_batch); f(d_M_invs); f(d_crop_w); fh(h_M_invs); fh(h_crop_w);
  }
  void ensure(int B) {
    std::size_t need = (std::size_t)B * 3 * kClsImageH * kClsImageW * sizeof(float);
    grow_scratch(deps.alloc, d_batch, batch_cap, need);
    // Through the SAME exception-safe helpers the recognizer uses. This site
    // was the one remaining `free(p); p = allocate(n);` pair — the exact shape
    // grow_scratch exists to forbid. HIP_CHECK throws now, so a failing
    // allocate left d_M_invs/d_crop_w/h_* holding FREED pointers while bm_cap
    // still described the old, larger capacity: the next request on this
    // pooled, long-lived stage either wrote through the dangling pointers or
    // freed them a second time.
    if ((std::size_t)B > bm_cap) {
      std::size_t c_dM = 0, c_dW = 0, c_hM = 0, c_hW = 0;
      grow_scratch(deps.alloc, d_M_invs, c_dM, (std::size_t)B * 9 * sizeof(float));
      grow_scratch(deps.alloc, d_crop_w, c_dW, (std::size_t)B * sizeof(int));
      grow_scratch_host(deps.alloc, h_M_invs, c_hM, (std::size_t)B * 9 * sizeof(float));
      grow_scratch_host(deps.alloc, h_crop_w, c_hW, (std::size_t)B * sizeof(int));
      bm_cap = B;
    }
  }
};

RocmClassifier::RocmClassifier(StageDeps deps)
    : p_(std::make_unique<Impl>(deps)) {}
RocmClassifier::~RocmClassifier() = default;

bool RocmClassifier::load(const std::string &model_path) {
  if (!(ready_ = p_->engine.load(model_path)))
    return false;
  // Same gate as the recognizer: pre-compile the batch rungs at the (fixed) cls
  // canvas so no request pays a compile. The rungs come from the SHARED ladder,
  // budgeted for the cls crop size — not a private list.
  std::vector<MIGraphXEngine::ShapeVariant> variants;
  for (int b : recognition::batch_ladder_for_width(kClsImageW, kClsImageH))
    variants.push_back(nchw_variant(b, kClsImageH, kClsImageW));
  const std::size_t ok = p_->engine.warmup(variants);
  if (ok != variants.size())
    std::fprintf(stderr,
                 "[RocmClassifier] warmup compiled %zu/%zu batch programs\n", ok,
                 variants.size());
  return ready_;
}

void RocmClassifier::run(const ImageView &img, std::vector<turbo_ocr::Box> &boxes,
                        DeviceQueue &queue) {
  const int n = static_cast<int>(boxes.size());
  if (!ready_ || n == 0 || img.empty())
    return;
  // Snap the demand to a pre-compiled batch rung (SHARED ladder) and pad; a raw
  // n would be a new static shape almost every request, i.e. a graph compile per
  // request. Crops [n, B) are padding and their outputs are ignored.
  //
  // NOTE: chunking above the top rung is not needed here because the classifier
  // is called with one page's boxes and the top rung at the 160x80 cls canvas is
  // large; if a page ever exceeds it the tail simply runs in a second call from
  // the shared pipeline. If that shows up in profiling, chunk with the same
  // recognition::plan_rec_batches used by the recognizer rather than inventing a
  // second chunker here.
  const auto rungs = recognition::batch_ladder_for_width(kClsImageW, kClsImageH);
  const int B = recognition::snap_batch(n, rungs);
  p_->ensure(B);
  hipStream_t s = hip_stream_of(queue);

  for (int i = 0; i < B; ++i) {
    float *m = p_->h_M_invs + i * 9;
    if (i < n) {
      const auto ct = turbo_ocr::compute_crop_transform(boxes[i], kClsImageH,
                                                        kClsImageW);
      std::copy(ct.M_inv, ct.M_inv + 9, m);
      p_->h_crop_w[i] = std::min(ct.crop_width, kClsImageW);
    } else {
      for (int k = 0; k < 9; ++k) m[k] = (k % 4 == 0) ? 1.0f : 0.0f;
      p_->h_crop_w[i] = 0;
    }
  }
  HIP_CHECK(hipMemcpyAsync(p_->d_M_invs, p_->h_M_invs, B * 9 * sizeof(float),
                           hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(p_->d_crop_w, p_->h_crop_w, B * sizeof(int),
                           hipMemcpyHostToDevice, s));

  // BUG FIX (third occurrence of this exact bug — Intel had it, Apple had a
  // variant): the text-line orientation classifier does NOT take ImageNet
  // normalization. Despite the PP-LCNet_x0_25 backbone, the shipped export is
  // trained with REC's (pixel/127.5 - 1); see classification::cls_norm() and the
  // MEASURED note in src/backends/nvidia/stages/paddle_cls.cpp. ImageNet input is the
  // wrong distribution => mis-classified 180-degree lines => reversed text, on
  // AMD only. The value now comes from the SHARED header so this cannot recur.
  p_->deps.kernels->warp_crops(img, p_->d_M_invs, p_->d_crop_w, p_->d_batch, B,
                               kClsImageH, kClsImageW,
                               classification::cls_norm(), queue);

  std::vector<DeviceTensor> ins = {DeviceTensor{
      p_->engine.input_names().empty() ? "x" : p_->engine.input_names()[0],
      p_->d_batch, backend::DeviceKind::Hip, DType::F32, p_->deps.device_id,
      {B, 3, kClsImageH, kClsImageW}}};
  std::vector<OutputLease> leases;
  if (!p_->engine.run(ins, {}, leases, queue) || leases.empty())
    return;

  // Output [B,2]: probabilities for {0°, 180°}. Copy back only the REAL rows.
  const auto &sh = leases[0].shape;
  const int ncls = sh.size() >= 2 ? static_cast<int>(sh.back()) : 2;
  std::vector<float> h(static_cast<std::size_t>(n) * ncls);
  HIP_CHECK(hipMemcpyAsync(h.data(), leases[0].data, h.size() * sizeof(float),
                           hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));

  int flipped = 0;
  for (int b = 0; b < n; ++b) {
    const float p0 = h[b * ncls + 0];
    const float p180 = ncls > 1 ? h[b * ncls + 1] : 0.0f;
    // SHARED decision + SHARED rotation. This site used `>=` where every other
    // backend uses `>` (flipping a hair more lines at exactly 0.9), and rebuilt
    // the quad cyclically where the others swap the diagonals — equivalent only
    // by luck of the corner order.
    if (classification::should_flip_180(p0, p180)) {
      classification::flip_quad_180(boxes[b]);
      ++flipped;
    }
  }
  // IClassifier::run returns void, so this count has no caller. Kept and LOGGED
  // rather than deleted: "how many crops came in upside down" is the only signal
  // that the orientation classifier is doing anything, and a silently-discarded
  // counter is how a dead cls stage goes unnoticed. Same treatment as the Apple
  // and Intel arms.
  if (flipped > 0)
    TOCR_LOG_DEBUG("amd cls flipped boxes 180", "flipped", flipped, "boxes", n);
}

// ===========================================================================
// RocmLayout (PP-DocLayoutV3, multi-IO image + im_shape + scale_factor)
// ===========================================================================
struct RocmLayout::Impl {
  StageDeps deps;
  MIGraphXEngine engine;
  float *d_input = nullptr;      // [3*S*S]
  float *d_im_shape = nullptr;   // [2]
  float *d_scale = nullptr;      // [2]
  int S = 800;
  explicit Impl(StageDeps d) : deps(d), engine(d.device_id) {}
  ~Impl() {
    if (d_input) deps.alloc->free(d_input);
    if (d_im_shape) deps.alloc->free(d_im_shape);
    if (d_scale) deps.alloc->free(d_scale);
  }
};

RocmLayout::RocmLayout(StageDeps deps) : p_(std::make_unique<Impl>(deps)) {}
RocmLayout::~RocmLayout() = default;

bool RocmLayout::load(const std::string &model_path) {
  ready_ = p_->engine.load(model_path);
  if (ready_) {
    const int S = p_->S;
    // Layout runs at ONE canvas, so its ladder is a single multi-IO variant
    // (image + im_shape + scale_factor). Pre-compile it so the first page does
    // not pay the graph compile.
    p_->engine.warmup({MIGraphXEngine::ShapeVariant{
        {{1, 3, S, S}, {1, 2}, {1, 2}}}});
    p_->d_input = static_cast<float *>(p_->deps.alloc->allocate((std::size_t)3 * S * S * sizeof(float)));
    p_->d_im_shape = static_cast<float *>(p_->deps.alloc->allocate(2 * sizeof(float)));
    p_->d_scale = static_cast<float *>(p_->deps.alloc->allocate(2 * sizeof(float)));
  }
  return ready_;
}

std::vector<turbo_ocr::layout::LayoutBox>
RocmLayout::run(const ImageView &img, int orig_h, int orig_w,
                float score_threshold, DeviceQueue &queue) {
  if (!ready_ || img.empty())
    return {};
  const int S = p_->S;
  hipStream_t s = hip_stream_of(queue);

  // SHARED layout normalization: pixel/255 (mean 0, std 1), BGR planes.
  // NOTE: this site used to set letterbox = true. No backend honours letterbox
  // (see the PARAMETER CONTRACT in kernels.h) — CUDA's baked layout
  // preprocessor STRETCHES, and the im_shape/scale_factor pair below is derived
  // from that stretch, so a letterboxed AMD input with stretch-derived scale
  // factors would have mis-mapped every layout box. The shared factory sets
  // letterbox = false, matching the coordinate math directly below.
  p_->deps.kernels->resize_normalize(img, p_->d_input, S, S,
                                     backend::norm::layout_norm(), queue);

  // im_shape = resized (S,S); scale_factor = resized/original (PP-DocLayoutV3
  // maps detections back to original coords internally).
  const float im_shape[2] = {static_cast<float>(S), static_cast<float>(S)};
  const float scale[2] = {static_cast<float>(S) / orig_h,
                          static_cast<float>(S) / orig_w};
  HIP_CHECK(hipMemcpyAsync(p_->d_im_shape, im_shape, sizeof(im_shape),
                           hipMemcpyHostToDevice, s));
  HIP_CHECK(hipMemcpyAsync(p_->d_scale, scale, sizeof(scale),
                           hipMemcpyHostToDevice, s));

  // Bind the 3 inputs by their model names (order from input_names()).
  const auto &names = p_->engine.input_names();
  auto name_at = [&](std::size_t i, const char *fallback) {
    return i < names.size() ? names[i] : std::string(fallback);
  };
  std::vector<DeviceTensor> ins = {
      DeviceTensor{name_at(0, "image"), p_->d_input, backend::DeviceKind::Hip,
                   DType::F32, p_->deps.device_id, {1, 3, S, S}},
      DeviceTensor{name_at(1, "im_shape"), p_->d_im_shape,
                   backend::DeviceKind::Hip, DType::F32, p_->deps.device_id, {1, 2}},
      DeviceTensor{name_at(2, "scale_factor"), p_->d_scale,
                   backend::DeviceKind::Hip, DType::F32, p_->deps.device_id, {1, 2}},
  };
  std::vector<OutputLease> leases;
  if (!p_->engine.run(ins, {}, leases, queue) || leases.empty())
    return {};

  // PP-DocLayoutV3 emits [N, 6|7] rows {class_id, score, x0, y0, x1, y1
  // [, read_order]} in ORIGINAL coords, plus a separate int32 COUNT tensor.
  //
  // This block used to decode the rows by hand and got three things wrong:
  // it rejected any tensor with cols < 6 while silently ignoring column 6
  // (read_order); it used det.shape[0] as the row count, which is
  // data-dependent and documented to go stale across repeated requests (it
  // silently dropped layout from every consecutive response on the TRT path,
  // and the same trap applies to MIGraphX); and it did no class-id range
  // check, so a garbage id indexes kLayoutLabels out of bounds downstream.
  // It now calls the SHARED decoder, which is Intel's correct implementation.
  const OutputLease &det = leases[0];
  const int rows_dim0 = det.shape.empty() ? 0 : static_cast<int>(det.shape[0]);
  const int stride = det.shape.size() >= 2 ? static_cast<int>(det.shape[1]) : 6;
  if (rows_dim0 <= 0 || stride < 6)
    return {};

  // Pull the count tensor when the export provides one (2nd rank-1 output).
  // Its value overrides rows_dim0 — see the shared decoder's contract.
  std::int32_t count_val = 0;
  const std::int32_t *count_ptr = nullptr;
  if (leases.size() >= 2 && leases[1].data) {
    HIP_CHECK(hipMemcpyAsync(&count_val, leases[1].data, sizeof(std::int32_t),
                             hipMemcpyDeviceToHost, s));
    count_ptr = &count_val;
  }

  // Copy the FULL NMS budget of rows, never `min(rows_dim0, budget)`. The
  // decoder's contract lets the count tensor OVERRIDE rows_dim0 — that is the
  // whole point of the count output, because shape[0] is data-dependent and
  // goes stale to a smaller value on repeated requests. Sizing this host
  // buffer from rows_dim0 while letting *count drive the loop meant that on
  // exactly those stale-shape requests (*count > rows_dim0) the shared decoder
  // walked past the end of `h` — a heap over-read. The rows tensor itself is
  // allocated at the budget regardless of how many detections the page
  // produced, so copying the budget is always in-bounds on the device side.
  const int rows_to_copy = turbo_ocr::layout::kPicodetMaxDet;
  std::vector<float> h(static_cast<std::size_t>(rows_to_copy) * stride);
  HIP_CHECK(hipMemcpyAsync(h.data(), det.data, h.size() * sizeof(float),
                           hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));

  // The SHARED postfilter (NMS + full-page-image drop + containment/merge-mode
  // reconciliation) must run on EVERY backend: CPU (ort_paddle_layout.cpp:271)
  // and NVIDIA (paddle_layout.cpp:223) already applied it, and these two arms
  // did not — so Intel/AMD returned raw overlapping boxes and their layout,
  // reading order and every downstream block/table decision diverged from the
  // other two on the same page. Generic policy is shared, never per backend.
  auto boxes = turbo_ocr::layout::decode_picodet_rows(
      h.data(), rows_to_copy, stride, count_ptr, score_threshold, orig_h, orig_w);
  return turbo_ocr::layout::postfilter_layout_boxes(std::move(boxes), orig_h,
                                                    orig_w);
}

} // namespace turbo_ocr::amd
