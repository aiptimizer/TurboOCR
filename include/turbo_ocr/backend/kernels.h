#pragma once

// IKernels — the device pre/post op set behind the existing
// src/backends/nvidia/kernels_cuda/kernels_cuda.h signatures, lifted to a device-agnostic
// interface. This is the finite, well-known set of operations the pipeline needs
// on-device around the model forward passes: decode, resize/normalize, ROI warp,
// threshold, DB post-process (connected components + unclip), argmax, and the
// fused table/layout region preprocessors.
//
// Each backend supplies its own implementation — CUDA .cu (NVIDIA, wrapped),
// Metal compute shaders (Apple), hipified .cu (AMD), SYCL/OpenCL (Intel),
// OpenCV on the host (CpuBackend). caps() reports, PER OP, whether the backend
// runs it natively on the device or falls back to the host (copy D2H, run
// OpenCV, copy H2D) — the graceful-degradation lever from the plan: e.g. Apple
// and Intel may run db_postprocess (CCL/JFA-unclip, which has no MPS/portable
// primitive) on the host until a native union-find is hand-written, while
// everything else stays resident.
//
// MEMORY CONVENTION: every raw pointer argument (const float*, uint8_t*, int*,
// …) and every ImageView points into the SAME device address space as
// `queue.device()`. They are caller-owned, pre-allocated device buffers — no op
// allocates per-call scratch on the hot path (matching the existing kernels.h
// contract, which threads caller-owned CudaPtr buffers). All work is enqueued on
// `queue`; device backends are async (sync the queue before reading host-side),
// the Host backend is synchronous.

// ===========================================================================
// PARAMETER CONTRACT — NormParams and DbPostParams are BINDING, not advisory.
//
// Before this rule existed, a backend could silently ignore a field and produce
// different pixels or different boxes with no error anywhere: AMD dropped
// NormParams::order (R/B swapped on det), NVIDIA SNIFFED params.mean to pick
// between two baked variants and discarded order/inv_std/inv_scale entirely,
// NVIDIA's db_postprocess ignored `oriented` and always emitted AABBs while the
// caller asked for rotated quads, and `letterbox` had four different meanings.
// Every one of those is a per-backend accuracy fork that no test can see,
// because the seam reports success.
//
// THE RULE, for every IKernels implementation:
//   1. HONOUR the field, or
//   2. declare it unsupported in caps().params, AND refuse the call loudly —
//      require_norm_supported() / require_db_supported() below assert in debug
//      and log-and-return-false in release; a void op then returns without
//      writing, and db_postprocess returns {}.
// NEVER silently substitute a different value. A caps() that over-claims defeats
// the whole design: the pipeline trusts caps() to decide placement/fallback.
//
// Adding a field to either struct means adding a ParamSupport flag for it and
// auditing every backend.
// ===========================================================================

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "turbo_ocr/backend/device_queue.h" // DeviceQueue
#include "turbo_ocr/backend/image_view.h"    // ImageView
#include "turbo_ocr/base/geometry/box.h"   // turbo_ocr::Box
#include "turbo_ocr/core/db_post_config.h" // kDbDefaults + geometry limits

namespace turbo_ocr::backend {

// Channel order of a normalized CHW tensor.
enum class ChannelOrder : int { BGR = 0, RGB = 1 };

// Normalization spec shared by resize_normalize / warp_crops. Output pixel is
// (pixel * inv_scale - mean[c]) * inv_std[c]. Defaults are REC's
// ((pixel/255 - 0.5)/0.5). NEVER write these literals at a call site — use the
// factories in turbo_ocr/core/norm_params.h (norm::rec_norm / cls_norm /
// imagenet_bgr / layout_norm), which are the single definition of each.
struct NormParams {
  float mean[3] = {0.5f, 0.5f, 0.5f};
  float inv_std[3] = {2.0f, 2.0f, 2.0f};
  float inv_scale = 1.0f / 255.0f;
  ChannelOrder order = ChannelOrder::RGB; // which source channel feeds plane 0
  bool letterbox = false; // preserve aspect ratio + pad (det/layout) vs stretch
};

// DB detection post-process parameters (CCL + Euclidean unclip).
//
// DEFAULTS COME FROM THE SHARED DETECTION CONFIG. They used to be
// box_thresh = 0.6 / unclip_ratio = 1.5 — AMD's old forked values, which no
// backend actually wants; any caller that default-constructed this silently got
// a stricter box filter and a wider unclip than detection::kDbDefaults
// (0.45 / 1.4), i.e. lower recall and fatter boxes than every configured path.
struct DbPostParams {
  // per-component mean-probability threshold
  float box_thresh = detection::kDbDefaults.box_thresh;   // 0.45
  // polygon-offset expansion ratio
  float unclip_ratio = detection::kDbDefaults.unclip_ratio; // 1.4
  float min_expand = detection::kMinExpand;   // 2.0 — floor on the unclip radius
  float max_expand = detection::kMaxExpand;   // 24.0 — ceiling on it
  // Pre-/post-unclip sliver filters, in the map's coordinate space.
  float min_box_side = detection::kMinBoxSide;             // 3.0
  float min_unclipped_side = detection::kMinUnclippedSide; // 5.0
  bool oriented = false;     // true => rotated min-area-rect quads; false => AABB
  int max_components = detection::kMaxDbComponents; // candidate budget
};

// Convert the shared detection DbParams (thresholds + env overrides, owned by
// turbo_ocr/analysis/detection/det_config.h) into the seam's DbPostParams. Every detector
// builds its DbPostParams this way so no backend can invent a threshold.
[[nodiscard]] inline DbPostParams
db_post_params(const detection::DbParams &p, bool oriented = false) noexcept {
  DbPostParams d;
  d.box_thresh = p.box_thresh;
  d.unclip_ratio = p.unclip_ratio;
  d.oriented = oriented;
  return d;
}

// WHICH parameterized op the contract guard is being asked about. The mean/std
// family is separately implementable per op — a backend can bake its constants
// into the full-frame preprocessor while its warp path takes real parameters
// (NVIDIA), or the exact reverse (Apple) — so ParamSupport carries one flag per
// path and require_norm_supported() must be TOLD which path it is guarding.
//
// Generic consults only `norm_mean_std`, so passing the path is PART OF
// IMPLEMENTING THE CONTRACT, not an optional refinement: a resize_normalize
// that omits it gets the Generic check and the backend's own
// norm_mean_std_full_frame is never read — the flag can be false while the op
// keeps running and substituting. NOT every backend passes a path yet. To audit,
// grep for `require_norm_supported(` and read every call with no NormPath
// argument as still-to-do (a backend reaching the guard through
// refuse_unbaked_norm passes the path there instead).
enum class NormPath : int { Generic, FullFrame, Warp };

// Which BINDING fields this backend can actually honour. Default: everything
// except letterbox, which no backend implements consistently today (see the
// note on IKernels::resize_normalize). A backend that CAN honour a field must
// say so; a backend that cannot must say so and refuse.
struct ParamSupport {
  bool norm_mean_std = true;      // arbitrary mean / inv_std / inv_scale
  // Same, but for the FULL-FRAME resize_normalize path specifically. NVIDIA's
  // full-frame preprocessor is two kernels with the det and layout constants
  // BAKED IN (the .cu lives in the main tree), so it can serve exactly those two
  // distributions and nothing else, while its warp path takes real parameters.
  // READ ONLY when the call site passes NormPath::FullFrame — declaring it false
  // in caps() enforces nothing on its own (see NormPath).
  bool norm_mean_std_full_frame = true;
  // Same, for warp_crops. Apple's warp.metal bakes rec's (v/127.5 - 1) in, so
  // that path serves exactly one distribution (which happens to be the one both
  // rec AND cls want — see classification::cls_norm). It used to take a
  // NormParams and ignore it wholesale.
  // READ ONLY when the call site passes NormPath::Warp — same caveat.
  bool norm_mean_std_warp = true;
  bool norm_channel_order = true; // NormParams::order
  bool norm_letterbox = false;    // NormParams::letterbox == true
  bool db_oriented = true;        // DbPostParams::oriented == true
  // DbPostParams::oriented == false. Separate flag because the two directions
  // are separately implementable: the host/Metal/SYCL fallbacks all delegate to
  // detection::extract_boxes_from_bitmap, which ALWAYS emits min-area-rect
  // quads and therefore cannot honour oriented=false; CUDA's JFA path emits
  // ONLY AABBs and cannot honour oriented=true. Both used to answer "sure" and
  // return the other thing.
  bool db_axis_aligned = true;
  bool db_expand_limits = true;   // min_expand / max_expand
  bool db_side_limits = true;     // min_box_side / min_unclipped_side
  bool db_max_components = true;  // max_components
};

// Loud refusal helper. Returns false when the params ask for something this
// backend declared unsupported: asserts in debug builds so it can never reach a
// release build unnoticed, and logs once per call in release so a production
// divergence is visible in the log rather than in the F1 number.
inline bool report_unhonoured(const char *op, const char *field) {
  std::fprintf(stderr,
               "[kernels] REFUSED %s: this backend cannot honour '%s' and the "
               "seam forbids silently substituting a different value. Either "
               "implement it or stop asking for it.\n",
               op, field);
  assert(false && "IKernels param not honoured — see stderr");
  return false;
}

// Forward declaration; KernelCaps carries the ParamSupport.
struct KernelCaps;

// Which fused region preprocessor to run (all read a sub-rect of a page image
// and emit a normalized CHW tensor sized to the model). Generalizes the
// cuda_fused_* family in kernels.h:
//   LayoutSubRect — pixel/255, BGR CHW, cell-detection sub-rect.
//   TableCls      — resize-short(256) -> center-crop(224) -> ImageNet -> BGR CHW.
//   SlanextBGR    — ResizeByLong(488) preserve-AR -> ImageNet -> bottom-right pad.
//   SlanextRGB    — 488 letterbox, RGB order + ImageNet + pad-0 (encoder-split).
enum class PreprocKind : int { LayoutSubRect, TableCls, SlanextBGR, SlanextRGB };

// MODEL INPUT GEOMETRY PER KIND — one definition, because preprocess_region's
// signature carries no target size and three implementations (host, CUDA, SYCL)
// were each hard-coding the same numbers. Three copies of a constant that MUST
// agree is one edit away from a backend that silently preprocesses to the wrong
// size — the tensor still binds, the model still runs, the output is just wrong.
struct PreprocGeometry {
  int target = 0;       // final square the model consumes (long side for Slanext)
  int resize_short = 0; // 0 = no resize-short/crop step; else the short side
};

[[nodiscard]] constexpr PreprocGeometry preproc_geometry(PreprocKind k) noexcept {
  switch (k) {
  case PreprocKind::LayoutSubRect: return {800, 0};
  case PreprocKind::TableCls:      return {224, 256};
  case PreprocKind::SlanextBGR:
  case PreprocKind::SlanextRGB:    return {488, 0};
  }
  return {};
}

struct Rect {
  int x = 0, y = 0, w = 0, h = 0;
};

// Per-op native/host-fallback report. `true` == the backend runs this op
// natively on its device; `false` == the backend copies to the host, runs a
// portable (OpenCV) implementation, and copies back. The pipeline uses this only
// for observability / stage placement — the op is ALWAYS callable regardless.
struct KernelCaps {
  DeviceKind device = DeviceKind::Host;
  bool decode_image = true;
  bool resize_normalize = true;
  bool warp_crops = true;
  bool threshold = true;
  bool db_postprocess = true; // CCL + unclip — first to fall back on Apple/Intel
  bool argmax = true;
  bool preprocess_region = true; // fused table/layout preproc

  // Which BINDING NormParams/DbPostParams fields this backend honours. See the
  // PARAMETER CONTRACT at the top of this header — over-claiming here is the
  // failure mode the contract exists to prevent.
  ParamSupport params{};
};

// Guards for the PARAMETER CONTRACT. Call at the top of every op that takes
// params; on false the op MUST return without doing anything (a void op writes
// nothing; db_postprocess returns {}). Both are constant-time.
// Exact equality of two normalization specs. Used by backends whose kernels bake
// their constants in: they can serve a FIXED SET of distributions, so they match
// against the shared factories and refuse anything else — instead of SNIFFING one
// field (NVIDIA tested `mean[0] == 0`) and silently misrouting a third variant.
[[nodiscard]] inline bool norm_equal(const NormParams &a,
                                     const NormParams &b) noexcept {
  for (int i = 0; i < 3; ++i)
    if (a.mean[i] != b.mean[i] || a.inv_std[i] != b.inv_std[i]) return false;
  return a.inv_scale == b.inv_scale && a.order == b.order &&
         a.letterbox == b.letterbox;
}

// `path` selects the per-op mean/std claim on top of the shared one: a backend
// whose kernel for THIS op bakes its constants in declares
// norm_mean_std_full_frame / norm_mean_std_warp = false, and this guard — not a
// hand-rolled check inside the backend — is what refuses. The Generic default
// exists only so an op with no per-path notion still compiles; an op that HAS
// one and leaves the argument off silently downgrades to `norm_mean_std`.
[[nodiscard]] inline bool require_norm_supported(const NormParams &p,
                                                 const ParamSupport &s,
                                                 const char *op,
                                                 NormPath path = NormPath::Generic) {
  if (p.letterbox && !s.norm_letterbox)
    return report_unhonoured(op, "NormParams::letterbox");
  if (!s.norm_channel_order && p.order != ChannelOrder::RGB)
    return report_unhonoured(op, "NormParams::order");
  if (!s.norm_mean_std)
    return report_unhonoured(op, "NormParams::mean/inv_std/inv_scale");
  if (path == NormPath::FullFrame && !s.norm_mean_std_full_frame)
    return report_unhonoured(
        op, "NormParams::mean/inv_std/inv_scale (this backend's full-frame "
            "preprocessor bakes its constants in)");
  if (path == NormPath::Warp && !s.norm_mean_std_warp)
    return report_unhonoured(
        op, "NormParams::mean/inv_std/inv_scale (this backend's warp path bakes "
            "its constants in)");
  return true;
}

// SHARED refusal for a backend whose kernel bakes its constants in and can
// therefore serve only a FIXED WHITELIST of distributions. Call it when the
// caller's params matched none of them: it runs the contract guard for `path`,
// so the ParamSupport flag (norm_mean_std_full_frame / norm_mean_std_warp) is
// what does the refusing, in ONE place, for every backend. If that guard passed
// anyway the backend's caps() is OVER-CLAIMING, which must still not become a
// silent substitution — so `baked_desc` is reported and the answer is the same.
// ALWAYS returns false, so the op can `return` (or `return {}`) on it.
inline bool refuse_unbaked_norm(const NormParams &p, const ParamSupport &s,
                                const char *op, NormPath path,
                                const char *baked_desc) {
  if (require_norm_supported(p, s, op, path))
    return report_unhonoured(op, baked_desc); // caps() over-claims — still refuse
  return false;
}

// The expand/side/max_components checks compare against the SHARED defaults:
// a backend that cannot implement a knob is only refusing callers who actually
// asked for something other than the shared value. `oriented` has no "unset"
// state, so both directions are checked unconditionally.
//
// CONTRACT NOTE — what makes the default-value pass-through sound: a backend
// declaring a knob false while serving default requests is legal ONLY when
// its baked behaviour EQUALS the shared default. That equality must be
// pinned, not asserted in a comment — kMaxGpuComponents is static_assert'd
// against detection::kMaxDbComponents in both device-kernel adapter TUs, and
// the side limits stopped being baked at all (both extract loops now gate on
// the caller's params and declare db_side_limits = true). Without a pin, a
// declared-false knob whose baked value drifts from the default is exactly
// the silent substitution this guard exists to prevent — it reported
// "honoured" for the only configuration that ever reached it.
[[nodiscard]] inline bool require_db_supported(const DbPostParams &p,
                                               const ParamSupport &s,
                                               const char *op) {
  if (p.oriented && !s.db_oriented)
    return report_unhonoured(op, "DbPostParams::oriented=true");
  if (!p.oriented && !s.db_axis_aligned)
    return report_unhonoured(op, "DbPostParams::oriented=false");
  if (!s.db_expand_limits &&
      (p.min_expand != detection::kMinExpand || p.max_expand != detection::kMaxExpand))
    return report_unhonoured(op, "DbPostParams::min_expand/max_expand");
  if (!s.db_side_limits &&
      (p.min_box_side != detection::kMinBoxSide ||
       p.min_unclipped_side != detection::kMinUnclippedSide))
    return report_unhonoured(op, "DbPostParams::min_box_side/min_unclipped_side");
  if (!s.db_max_components && p.max_components != detection::kMaxDbComponents)
    return report_unhonoured(op, "DbPostParams::max_components");
  return true;
}

// The device pre/post op set. One implementation per backend; obtained from
// Backend::make_kernels().
class IKernels {
public:
  virtual ~IKernels() = default;

  [[nodiscard]] virtual KernelCaps caps() const = 0;

  // Decode an encoded image (JPEG/PNG/…) directly into device memory (nvJPEG /
  // vImage+Metal / VAAPI / host stb). The returned ImageView is backed by
  // kernels-owned pool memory in `queue.device()` space, VALID UNTIL the next
  // decode_image() on this object (or until it is destroyed). Interleaved 8-bit
  // BGR, the pipeline's canonical decode format. Returns an empty ImageView on
  // failure.
  [[nodiscard]] virtual ImageView
  decode_image(const std::uint8_t *data, std::size_t len, DeviceQueue &queue) = 0;

  // Would decode_image() handle these bytes, or decline and leave them to the
  // host? A header sniff: no device work, no allocation, no queue — so callers
  // can ask before committing any resources (see Backend::can_device_decode for
  // why that ordering matters). Default false = always host-decode.
  [[nodiscard]] virtual bool can_decode_image(const std::uint8_t * /*data*/,
                                              std::size_t /*len*/) const {
    return false;
  }

  // Resize `src` into a normalized CHW float tensor `dst_chw` of dst_h x dst_w.
  // Generalizes cuda_fused_resize_normalize_det / _layout and the rec/cls
  // single-image path.
  //
  // BINDING (see the PARAMETER CONTRACT at the top): honour every NormParams
  // field or refuse via require_norm_supported(params, caps().params, ...,
  // NormPath::FullFrame) — the path argument is what makes
  // caps().params.norm_mean_std_full_frame load-bearing. A backend with a fixed
  // whitelist of baked distributions matches those first and hands every miss to
  // refuse_unbaked_norm().
  //
  // params.letterbox: the ONE field no backend honours today. CUDA's baked
  // layout preprocessor STRETCHES to the canvas and the layout stage's
  // coordinate rescale is derived from that stretch, so honouring letterbox on
  // one backend alone desynchronizes its boxes from every other backend's.
  // Callers therefore pass letterbox=false (norm::layout_norm() sets it), and
  // every backend declares caps().params.norm_letterbox = false so a caller that
  // does ask gets a loud refusal instead of a silent stretch. Making letterbox
  // real is a SHARED change: this contract + all backends + the layout
  // coordinate math, in one commit.
  virtual void resize_normalize(const ImageView &src, float *dst_chw, int dst_w,
                                int dst_h, const NormParams &params,
                                DeviceQueue &queue) = 0;

  // Batched perspective ROI warp + resize + normalize. For each of `batch`
  // crops, warp `src` by the 3x3 inverse `d_M_invs[i]` into slot i of
  // `d_dst_batch` [batch,3,dst_h,dst_w], padding to `d_crop_widths[i]`.
  // Generalizes cuda_batch_roi_warp — the residency win proven by the Apple POC.
  // BINDING: honour every NormParams field or refuse
  // (require_norm_supported(..., NormPath::Warp), which is what makes
  // caps().params.norm_mean_std_warp load-bearing; a baked warp path matches its
  // one distribution first and hands every miss to refuse_unbaked_norm()).
  // `letterbox` is meaningless here (each crop is warped to its own width) and
  // is ignored by contract, not by omission.
  virtual void warp_crops(const ImageView &src, const float *d_M_invs,
                          const int *d_crop_widths, float *d_dst_batch,
                          int batch_size, int dst_h, int dst_w,
                          const NormParams &params, DeviceQueue &queue) = 0;

  // Threshold a float probability map to a uint8 bitmap (255=fg,0=bg) over
  // batch_size*w*h elements. Generalizes cuda_threshold_to_u8 / _batch.
  virtual void threshold(const float *src, std::uint8_t *dst, int w, int h,
                         int batch_size, float thresh, DeviceQueue &queue) = 0;

  // DB post-process: connected-component labeling + per-component Euclidean
  // unclip + bbox/oriented-rect extraction over one map, returning HOST
  // page-coordinate boxes. Folds the whole cuda_gpu_ccl_detect / JFA-unclip /
  // extract chain into one op. `d_pred_map` is the raw probability map, `d_bitmap`
  // its threshold() result, both w x h in device memory. Boxes are in the map's
  // (resized) coordinate space; the caller rescales to original dims. This is the
  // op most likely to be a host fallback (caps().db_postprocess == false).
  //
  // BINDING: honour every DbPostParams field or refuse via
  // require_db_supported(params, caps().params, ...) and return {}. In
  // particular `oriented` changes the CROP GEOMETRY of every downstream
  // recognition — a backend that quietly returns AABBs to a caller asking for
  // rotated quads reads different text, and nothing reports it.
  [[nodiscard]] virtual std::vector<turbo_ocr::Box>
  db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap, int w,
                 int h, const DbPostParams &params, DeviceQueue &queue) = 0;

  // Per-timestep argmax for CTC decoding: for each [batch,seq] position over
  // num_classes, write the max-class index and its score. Generalizes
  // cuda_argmax. Outputs are device buffers (int[batch*seq], float[batch*seq]).
  virtual void argmax(const float *input_probs, int *output_indices,
                      float *output_scores, int batch_size, int seq_len,
                      int num_classes, DeviceQueue &queue) = 0;

  // Fused table/layout region preprocessor: read `rect` of `src` and emit the
  // model-sized normalized CHW tensor into `dst_chw`, per `kind`. Generalizes
  // cuda_fused_resize_normalize_layout(sub-rect) / cuda_fused_table_cls_pre /
  // cuda_fused_slanext_pre[_rgb].
  //
  // HONESTY NOTE (2026-08-03): implemented by all five backends, DISPATCHED BY
  // NOTHING — the NVIDIA table stages still call their cuda_fused_* kernels
  // directly, and no other stage has a device table path yet, so the
  // implementations (incl. table_kernels.hip, sycl_kernels) are currently
  // dead weight held for the planned device table stages (AMD/Intel bring-up
  // items). Routing the NVIDIA table stages through this seam is the step
  // that makes it real; until then do not build anything that branches on
  // caps().preprocess_region.
  virtual void preprocess_region(const ImageView &src, const Rect &rect,
                                 PreprocKind kind, float *dst_chw,
                                 DeviceQueue &queue) = 0;

  // Pre-size any HOST mirror buffers this implementation needs for the ops it
  // declares as fallbacks in caps(), for a probability map of up to
  // `max_map_pixels`. Called once at warmup.
  //
  // WHY THIS IS ON THE SEAM and not a vendor method: the performance gate
  // forbids allocation on the hot path, and EVERY backend that degrades an op
  // onto the host has the same need — a device backend staging a map back to
  // host memory sizes exactly the same buffer. It was previously a SyclKernels
  // member, which meant the stage could only be handed that concrete class;
  // that coupling is what forced the no-DPC++ build onto no-op stubs.
  //
  // NOT pure virtual: an implementation that allocates nothing on the host (the
  // host backend itself, or a fully device-resident one) correctly does nothing.
  virtual void reserve_host_fallback(std::size_t max_map_pixels) { (void)max_map_pixels; }
};

} // namespace turbo_ocr::backend
