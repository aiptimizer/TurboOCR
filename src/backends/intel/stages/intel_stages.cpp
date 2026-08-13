// Intel device stages — SYCL pre/post + OpenVINO forward pass, device-resident.
//
// TOOLCHAIN: the stage ORCHESTRATION in this file is toolchain-agnostic C++ (it
// only touches the seam types and the shared policy headers) and is
// syntax-checkable on any host; the device work happens inside SyclKernels and
// OpenVINOEngine. Real execution needs oneAPI + OpenVINO + an Intel GPU.
//
// Everything policy-shaped comes from the shared layer — see the header for the
// full list and the reason. Read that before adding a constant here.

#include "intel/stages/intel_stages.h"
#include "intel/stages/intel_stages_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_set>

#include "turbo_ocr/analysis/classification/cls_config.h"    // cls canvas/thresh/norm/flip
#include "turbo_ocr/base/env_utils.h"                     // env::* — every read is recorded
#include "turbo_ocr/base/geometry/box.h"          // sorted_boxes, Box
#include "turbo_ocr/base/geometry/perspective.h"  // compute_crop_transform
#include "turbo_ocr/base/log/logger.h"            // TOCR_LOG_*
#include "turbo_ocr/base/log/stage_profiler.h"    // prof::Scope — /profile breakdown
#include "turbo_ocr/core/norm_params.h"           // SHARED norm factories
#include "turbo_ocr/analysis/detection/det_config.h"         // read_det_resize/db, compute_det_resize
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/analysis/layout/layout_postfilter.h"
#include "turbo_ocr/analysis/layout/picodet_decode.h"        // decode_picodet_rows
#include "turbo_ocr/analysis/recognition/ctc_decode.h"       // ctc_greedy_decode, load_label_dict
#include "turbo_ocr/analysis/recognition/rec_batching.h"     // plan_rec_batches, rec_shape_matrix
#include "turbo_ocr/analysis/recognition/rec_geometry.h"     // kRecWidthBuckets, kMaxRecWidth

namespace turbo_ocr::intel {

namespace {

// Normalization comes from the SHARED factories (turbo_ocr/core/norm_params.h).
// These two used to be local copies of the same six floats found in seven other
// files — the fork that put ImageNet on the classifier three separate times.
using backend::norm::imagenet_bgr;

// Shared with intel_stages_structure.cpp (classifier + layout) — defined once
// in the internal header, not copied per TU.
using stagesdetail::tensor_name;

// TURBO_INTEL_DEBUG — ONE predicate, read from both the det chain and the rec
// batch path. It was two hand-rolled getenv copies with the same spelling of
// "set and not 0"; a debug switch that is on in one stage and off in another is
// worse than no switch, because the half-output reads as a complete one.
[[nodiscard]] bool intel_debug() {
  static const bool on = [] {
    const std::string e = env::env_or("TURBO_INTEL_DEBUG", "");
    return !e.empty() && e != "0";
  }();
  return on;
}

} // namespace

// ============================== Detector ====================================

struct IntelDetector::Impl {
  StageDeps d;
  bool ready = false;

  // SHARED detection policy (resize limits + DB thresholds), read once. Nothing
  // about the resize rule or the thresholds is restated here.
  detection::DetResizeParams resize{detection::kDetResizeDefault};
  detection::DbParams db{detection::kDbDefaults};

  // Close the canvas set when the engine JIT-compiles per shape (OpenVINO GPU
  // plugin — see caps().per_shape_jit and detection::snap_det_canvas_grid). Read
  // once at load(); run() letterboxes into the snapped canvas when set and
  // lazily compiles ONE static variant per canvas (built_canvases tracks
  // which). Static matters as much as closed: the dynamic variant's
  // shape-agnostic GPU kernels measured ~9x slower (143 vs 15.4 ms/img) even
  // with only two shapes in play.
  bool snap_canvas = false;
  std::unordered_set<std::uint64_t> built_canvases;

  // USM scratch, sized ONCE at load() to the largest canvas the shared resize
  // policy can emit (effective_det_max_side squared) and then reused. The
  // performance gate forbids per-request device allocation. d_canvas exists
  // only when snap_canvas: it holds the letterboxed engine input while d_input
  // keeps the contiguous resized content.
  backend::DeviceBuffer d_input, d_canvas, d_output, d_bitmap;
  // Host mirrors for the letterbox repack, allocated only when snap_canvas
  // AND the allocator is device-backed: sycl::malloc_device memory is not
  // host-addressable, so the repack must stage D2H -> host fill/copy -> H2D
  // instead of dereferencing device pointers on the host.
  std::vector<float> h_canvas, h_content;
  std::size_t cap_pixels = 0;

  explicit Impl(StageDeps deps) : d(std::move(deps)) {}
};

IntelDetector::IntelDetector(StageDeps deps)
    : impl_(std::make_unique<Impl>(std::move(deps))) {}
IntelDetector::~IntelDetector() = default;

bool IntelDetector::load(const std::string &model_path) {
  auto &I = *impl_;
  if (!I.d.engine->load(model_path))
    return false;

  I.resize = detection::read_det_resize();
  I.db = detection::read_db_params();

  // Worst-case canvas from the SHARED cap, so the buffers cover every resize the
  // policy can produce.
  const int max_side = detection::effective_det_max_side(I.resize);
  I.cap_pixels = static_cast<std::size_t>(max_side) * static_cast<std::size_t>(max_side);
  I.d_input = I.d.alloc->allocate_buffer(I.cap_pixels * 3 * sizeof(float));
  I.d_output = I.d.alloc->allocate_buffer(I.cap_pixels * sizeof(float));
  I.d_bitmap = I.d.alloc->allocate_buffer(I.cap_pixels);
  I.d.kernels->reserve_host_fallback(I.cap_pixels);

  // Canvas snapping (the resolved half of what used to be an "unresolved,
  // needs hardware" note here): measurement on the UHD 770 showed dynamic det
  // reshape IS expensive on the GPU plugin — 121 ms/img in-pipeline vs 15.4 ms
  // at a fixed shape, a per-new-shape kernel JIT. The fix went into the SHARED
  // layer as decided (detection::snap_det_canvas_grid, a 128-grid canvas ladder),
  // and this backend letterboxes into it whenever the engine says shapes are
  // compile-expensive. CPU keeps the exact per-image canvas: padding there is
  // pure wasted compute (measured det_infer 17 ms/pg with dynamic shapes).
  I.snap_canvas = I.d.engine->caps().per_shape_jit;
  if (I.snap_canvas) {
    I.d_canvas = I.d.alloc->allocate_buffer(I.cap_pixels * 3 * sizeof(float));
    if (I.d.alloc->has_device()) {
      I.h_canvas.resize(I.cap_pixels * 3);
      I.h_content.resize(I.cap_pixels * 3);
    }
  }
  I.ready = true;
  return true;
}

bool IntelDetector::is_ready() const noexcept { return impl_->ready; }

std::vector<turbo_ocr::Box>
IntelDetector::run(const backend::ImageView &img, int orig_h, int orig_w,
                   backend::DeviceQueue &queue) {
  auto &I = *impl_;
  if (!I.ready || img.empty() || orig_h <= 0 || orig_w <= 0)
    return {};

  // SHARED resize policy (PaddleOCR DetResizeForTest + the /32 rounding + the
  // max-side cap). Not restated here.
  const auto [rh, rw] = detection::compute_det_resize(orig_h, orig_w, I.resize);
  const std::size_t pixels = static_cast<std::size_t>(rh) * static_cast<std::size_t>(rw);
  if (pixels == 0 || pixels > I.cap_pixels)
    return {}; // the shared cap guarantees this cannot happen; fail closed

  auto *dst = static_cast<float *>(I.d_input.data());
  {
    prof::Scope _s(prof::DET_PRE);
    I.d.kernels->resize_normalize(img, dst, rw, rh, imagenet_bgr(), queue);
  }

  // Engine canvas: the exact content dims, or — when the engine JIT-compiles
  // per shape — the SHARED snapped canvas (detection::snap_det_canvas_grid) with the
  // content letterboxed top-left and the rest zero in NORMALIZED space (= the
  // per-channel mean pixel, which DB scores as background). Every coordinate
  // downstream of the engine is in canvas space, which CONTAINS content space
  // at the same scale, so the content-dims rescale at the end is unchanged.
  int eh = rh, ew = rw;
  const float *engine_in = dst;
  if (I.snap_canvas) {
    std::tie(eh, ew) = detection::snap_det_canvas_grid(rh, rw, I.resize);
    // One STATIC compile per canvas, first time it appears (the snapped ladder
    // keeps that to a handful; OV_CACHE_DIR persists them across restarts).
    // select() then binds the static variant instead of the dynamic one.
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(eh)) << 32) |
        static_cast<std::uint32_t>(ew);
    if (I.built_canvases.insert(key).second)
      (void)I.d.engine->prebuild({{1, 3, eh, ew}});
    if (eh != rh || ew != rw) {
      prof::Scope _s(prof::DET_PRE);
      queue.synchronize(); // content must be fully written before the repack
      const std::size_t canvas_elems = static_cast<std::size_t>(3) * eh * ew;
      if (!I.d.alloc->has_device()) {
        // Host buffers: repack in place, zero extra copies.
        auto *cv = static_cast<float *>(I.d_canvas.data());
        std::fill_n(cv, canvas_elems, 0.0f);
        for (int c = 0; c < 3; ++c)
          for (int y = 0; y < rh; ++y)
            std::copy_n(dst + (static_cast<std::size_t>(c) * rh + y) * rw, rw,
                        cv + (static_cast<std::size_t>(c) * eh + y) * ew);
        engine_in = cv;
      } else {
        // Device USM is NOT host-addressable (image_view.h declares L0
        // non-host-coherent): a raw std::fill_n/std::copy_n on these
        // pointers — which this branch used to do — is a host deref of
        // sycl::malloc_device memory. Stage through the host mirrors: one
        // D2H for the content, fill+row-copy on the host, one H2D for the
        // canvas.
        const std::size_t content_elems = static_cast<std::size_t>(3) * rh * rw;
        I.d.alloc->copy_d2h(I.h_content.data(), dst,
                            content_elems * sizeof(float), queue);
        queue.synchronize();
        std::fill(I.h_canvas.begin(),
                  I.h_canvas.begin() + static_cast<std::ptrdiff_t>(canvas_elems),
                  0.0f);
        for (int c = 0; c < 3; ++c)
          for (int y = 0; y < rh; ++y)
            std::copy_n(
                I.h_content.data() + (static_cast<std::size_t>(c) * rh + y) * rw,
                rw,
                I.h_canvas.data() + (static_cast<std::size_t>(c) * eh + y) * ew);
        I.d.alloc->copy_h2d(I.d_canvas.data(), I.h_canvas.data(),
                            canvas_elems * sizeof(float), queue);
        engine_in = static_cast<const float *>(I.d_canvas.data());
      }
    }
  }

  const auto &ins = I.d.engine->input_names();
  const auto &outs = I.d.engine->output_names();
  const std::vector<std::int64_t> in_shape{1, 3, eh, ew};
  std::vector<backend::DeviceTensor> in(1), out(1);
  in[0] = {tensor_name(ins, 0, "x"), const_cast<float *>(engine_in),
           I.d.alloc->has_device() ? backend::DeviceKind::L0
                                   : backend::DeviceKind::Host,
           backend::DType::F32, 0, in_shape};
  out[0] = {tensor_name(outs, 0, "sigmoid_0.tmp_0"), I.d_output.data(),
            in[0].space, backend::DType::F32, 0, {1, 1, eh, ew}};
  std::vector<backend::OutputLease> leases;
  const auto t_pre = std::chrono::steady_clock::now();
  bool ran = false;
  {
    prof::Scope _s(prof::DET_INFER);
    ran = I.d.engine->run(in, out, leases, queue);
  }
  const auto t_inf = std::chrono::steady_clock::now();

  auto *pred = static_cast<float *>(I.d_output.data());
  auto *bmp = static_cast<std::uint8_t *>(I.d_bitmap.data());

  // TURBO_INTEL_DEBUG=1 dumps the det chain's intermediate state. A detector
  // that returns zero boxes is indistinguishable end-to-end from one whose
  // engine failed, whose input was never written, or whose threshold ate
  // everything — this separates them without a debugger on a remote box.
  const bool dbg = intel_debug();
  if (dbg) {
    const auto *src = static_cast<const float *>(I.d_input.data());
    const std::size_t nc = static_cast<std::size_t>(rh) * rw;   // content px
    const std::size_t n = static_cast<std::size_t>(eh) * ew;    // canvas px
    float in_lo = 1e30f, in_hi = -1e30f;
    double in_sum = 0;
    for (std::size_t i = 0; i < nc * 3; ++i) {
      in_lo = std::min(in_lo, src[i]); in_hi = std::max(in_hi, src[i]);
      in_sum += src[i];
    }
    float lo = 1e30f, hi = -1e30f;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      lo = std::min(lo, pred[i]); hi = std::max(hi, pred[i]); sum += pred[i];
    }
    std::fprintf(stderr,
                 "[intel/det] canvas=%dx%d content=%dx%d engine_run=%d "
                 "in[%.4f..%.4f mean=%.4f] "
                 "pred[%.4f..%.4f mean=%.4f] thresh=%.3f caps.resize_normalize=%d\n",
                 ew, eh, rw, rh, ran ? 1 : 0, in_lo, in_hi, in_sum / (nc * 3.0),
                 lo, hi, sum / static_cast<double>(n), I.db.thresh,
                 I.d.kernels->caps().resize_normalize ? 1 : 0);
  }
  if (!ran)
    return {};

  prof::Scope _post(prof::DET_POST); // covers threshold + DB post + rescale/sort
  I.d.kernels->threshold(pred, bmp, ew, eh, 1, I.db.thresh, queue);

  // SHARED thresholds -> seam params through the ONE conversion (kernels.h).
  // `oriented` is now true: the host fallback delegates to
  // detection::extract_boxes_from_bitmap, which is minAreaRect-based and ALWAYS
  // returns rotated quads. This site used to ask for oriented=false and get
  // rotated quads anyway — exactly the silent substitution the PARAMETER
  // CONTRACT forbids. It now asks for what it actually receives.
  const auto dbp = backend::db_post_params(I.db, /*oriented=*/true);
  // Declared host fallback -> shared detection::extract_boxes_from_bitmap.
  // Canvas dims: the pad region is background by construction, so it yields no
  // boxes; the rescale below stays in CONTENT dims (same scale, top-left).
  auto boxes = I.d.kernels->db_postprocess(pred, bmp, ew, eh, dbp, queue);
  if (dbg) {
    std::size_t fg = 0;
    const std::size_t n = static_cast<std::size_t>(eh) * ew;
    for (std::size_t i = 0; i < n; ++i) fg += (bmp[i] != 0);
    std::fprintf(stderr,
                 "[intel/det] fg_px=%zu/%zu (%.2f%%) boxes=%zu | infer=%.1fms "
                 "post=%.1fms shape_misses=%zu\n",
                 fg, n, 100.0 * fg / n, boxes.size(),
                 std::chrono::duration<double, std::milli>(t_inf - t_pre).count(),
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_inf).count(),
                 I.d.engine->shape_misses());
  }

  // Map from the resize canvas back to original coordinates (the seam says the
  // caller of db_postprocess owns this rescale).
  const float sx = static_cast<float>(orig_w) / rw;
  const float sy = static_cast<float>(orig_h) / rh;
  for (auto &b : boxes)
    for (int k = 0; k < 4; ++k) {
      b[k][0] = std::clamp(static_cast<int>(std::lround(b[k][0] * sx)), 0, orig_w - 1);
      b[k][1] = std::clamp(static_cast<int>(std::lround(b[k][1] * sy)), 0, orig_h - 1);
    }
  turbo_ocr::sorted_boxes(boxes); // SHARED reading order
  return boxes;
}

// ============================= Recognizer ===================================

struct IntelRecognizer::Impl {
  StageDeps d;
  bool ready = false;

  int rec_h = 48;                       // REC_IMAGE_H; PaddleRec's rec_image_h_
  std::vector<std::string> labels;      // labels[0] == "blank"
  // Per-bucket static batch rungs actually built as OpenVINO variants. This is
  // a HARDWARE FACT (which artefacts exist), fed back into the SHARED planner —
  // exactly the `bucket_rungs` parameter plan_rec_batches takes.
  std::vector<std::vector<int>> bucket_rungs;
  // Model-reported [batch, seq, classes], so buffers are sized from the compiled
  // model instead of an assumed /8 stride. `classes_from_model` records whether
  // that number really came from a compiled artefact — the dict/class-head
  // consistency check below is only meaningful (and only fires) when it did.
  int max_seq = 0, classes = 0;
  bool classes_from_model = false;

  // USM scratch, sized once at load() for the largest prebuilt shape.
  backend::DeviceBuffer d_M, d_widths, d_batch, d_logits, d_idx, d_score;
  // Pinned host staging (also allocated once).
  std::vector<float> h_M;
  std::vector<int> h_widths;
  std::vector<int> h_idx;
  std::vector<float> h_score;

  explicit Impl(StageDeps deps) : d(std::move(deps)) {}
};

IntelRecognizer::IntelRecognizer(StageDeps deps)
    : impl_(std::make_unique<Impl>(std::move(deps))) {
  impl_->labels.push_back("blank"); // index 0 == CTC blank (PaddleRec contract)
}
IntelRecognizer::~IntelRecognizer() = default;

bool IntelRecognizer::load(const std::string &model_path) {
  auto &I = *impl_;
  if (!I.d.engine->load(model_path))
    return false;
  I.rec_h = env::env_int("REC_IMAGE_H", 48, 1, 4096);

  // ---- Warmup: build every (width, batch) artefact the SHARED planner can ask
  // for, so run() never compiles. The shape matrix itself is shared
  // (recognition::rec_shape_matrix over kRecWidthBuckets x kRecBatchLadder under
  // the shared element budget) — this backend only turns each shape into an
  // OpenVINO CompiledModel.
  //
  // OV_REC_MAX_PREBUILD_WIDTH caps how much of the ladder is compiled at boot
  // (compile_model is seconds-scale per shape without a warm ov::cache_dir).
  // Widths above the cap still WORK — they route to the dynamic variant — they
  // just are not pre-compiled. Tuning this is a hardware measurement, not a
  // guess; ship with OV_CACHE_DIR set so the second boot is nearly free.
  const int max_prebuild_w = env::env_int("OV_REC_MAX_PREBUILD_WIDTH", 1600, 1, 8192);

  // OV_REC_DYNAMIC_BATCH=1 compiles ONE artefact per WIDTH with a dynamic batch
  // dimension instead of one per (width,batch) rung. Width is what changes the
  // kernel shapes; batch is a dimension OpenVINO handles natively. Measured
  // cost of the full per-rung matrix on this part: 3.2 GB per replica (latency
  // hint) / 6.7 GB (throughput hint) for a 4.3 MB model, which OOM-kills any
  // multi-replica pool. Off by default so the proven per-rung path stays the
  // default until this is measured on more than one machine.
  const bool dyn_batch = env::env_enabled("OV_REC_DYNAMIC_BATCH");

  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<std::pair<int, int>> wb; // parallel (width,batch) for shape lookup
  if (dyn_batch) {
    for (const int w : recognition::kRecWidthBuckets) {
      if (w > max_prebuild_w)
        continue;
      // ONE probe artefact per width at the ladder's SMALLEST rung, purely to
      // read geometry. A dynamic-batch artefact reports its batch dimension as
      // dynamic, so [batch, seq, classes] read back from it cannot size the
      // logits/index scratch — the products come out non-positive and the
      // buffers end up unusable. The probe is static, so its seq/classes are
      // concrete, and seq/classes depend only on WIDTH (which is static in both
      // artefacts) — so they are exactly the numbers the dynamic artefact will
      // produce. Five extra small artefacts against 30 removed still leaves the
      // order-of-magnitude memory win.
      shapes.push_back({recognition::kRecBatchLadder.front(), 3, I.rec_h, w});
      wb.emplace_back(w, recognition::kRecBatchLadder.front());
      // The artefact that actually serves traffic: any batch at this width.
      shapes.push_back({-1, 3, I.rec_h, w});
      wb.emplace_back(w, -1);
    }
  } else {
    for (const auto &[w, b] :
         recognition::rec_shape_matrix(recognition::kRecWidthBuckets, I.rec_h)) {
      if (w > max_prebuild_w)
        continue;
      shapes.push_back({b, 3, I.rec_h, w});
      wb.emplace_back(w, b);
    }
  }
  I.d.engine->prebuild(shapes);

  // Record which rungs each bucket actually has an artefact for, and learn the
  // logits geometry from the compiled model rather than assuming a stride.
  const auto &outs = I.d.engine->output_names();
  const std::string out0 = tensor_name(outs, 0, "softmax_5.tmp_0");
  I.bucket_rungs.assign(recognition::kRecWidthBuckets.size(), {});
  std::size_t max_in_elems = 0, max_logit_elems = 0, max_steps = 0;
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    const auto os = I.d.engine->output_shape(shapes[i], out0);
    if (os.size() < 3)
      continue; // this shape did not compile; it will use the dynamic variant
    const auto [w, b] = wb[i];
    if (b < 0)
      continue; // dynamic artefact: geometry comes from its static probe twin
    for (std::size_t k = 0; k < recognition::kRecWidthBuckets.size(); ++k)
      if (recognition::kRecWidthBuckets[k] == w) {
        if (dyn_batch) {
          // One artefact serves every rung this width may legally use. Note it
          // is batch_ladder_for_width, NOT the whole kRecBatchLadder: the shared
          // element budget is what the scratch above is sized against, so
          // advertising a rung beyond it would let the planner submit a batch
          // the buffers cannot hold.
          I.bucket_rungs[k] = recognition::batch_ladder_for_width(w, I.rec_h);
        }
        else
          I.bucket_rungs[k].push_back(b);
      }
    const int seq = static_cast<int>(os[1]);
    const int cls = static_cast<int>(os[2]);
    I.max_seq = std::max(I.max_seq, seq);
    I.classes = std::max(I.classes, cls);
    I.classes_from_model = true;
    // In dynamic-batch mode the probe artefact's rung is the SMALLEST, but the
    // planner may submit any rung the bucket advertises — size scratch for the
    // largest submission the shared element budget permits at this width, or
    // run() will write past the end of a buffer sized for batch 4.
    const int size_b =
        dyn_batch ? recognition::batch_ladder_for_width(w, I.rec_h).back() : b;
    max_in_elems = std::max<std::size_t>(
        max_in_elems, static_cast<std::size_t>(size_b) * 3 * I.rec_h * w);
    max_logit_elems = std::max<std::size_t>(
        max_logit_elems, static_cast<std::size_t>(size_b) * seq * cls);
    max_steps = std::max<std::size_t>(max_steps, static_cast<std::size_t>(size_b) * seq);
  }
  for (auto &r : I.bucket_rungs)
    std::sort(r.begin(), r.end());

  // A bucket with NO prebuilt artefact still has to be routable. bucket_rungs is
  // the "which batch sizes may I submit for this width" input to the SHARED
  // planner (plan_rec_batches); handing it an empty set for a width the page
  // actually contains is not a degraded plan, it is an invalid one — that is why
  // OV_REC_MAX_PREBUILD_WIDTH below the full ladder crashed instead of falling
  // back.
  //
  // The answer is the BUDGET-CAPPED ladder for that width, exactly what the
  // prebuilt buckets advertise — NOT the full kRecBatchLadder this used to
  // assign. The dynamic variant accepts any batch, but the scratch below is
  // sized against the shared element budget: advertising rung 256 at width
  // 4000 (where the budget admits only {4,8}) let the planner submit a chunk
  // whose warp/logits writes ran hundreds of MB past the buffers.
  for (std::size_t k = 0; k < I.bucket_rungs.size(); ++k)
    if (I.bucket_rungs[k].empty())
      I.bucket_rungs[k] = recognition::batch_ladder_for_width(
          recognition::kRecWidthBuckets[k], I.rec_h);

  // Scratch must cover every chunk the planner may legally SUBMIT — i.e.
  // every bucket at its advertised (budget-capped) top rung — not only the
  // prebuilt shapes the loop above accumulated over. An un-prebuilt wide
  // bucket routes to the dynamic variant with a seq of ~w/8, and until this
  // second pass the buffers simply did not account for it.
  for (std::size_t k = 0; k < recognition::kRecWidthBuckets.size(); ++k) {
    const int w = recognition::kRecWidthBuckets[k];
    const int b = I.bucket_rungs[k].empty() ? recognition::kRecBatchLadder.front()
                                            : I.bucket_rungs[k].back();
    const int seq = std::max(I.max_seq, std::max(1, w / 8));
    max_in_elems = std::max<std::size_t>(
        max_in_elems, static_cast<std::size_t>(b) * 3 * I.rec_h * w);
    max_steps = std::max<std::size_t>(
        max_steps, static_cast<std::size_t>(b) * static_cast<std::size_t>(seq));
  }
  if (I.classes <= 0)
    // Nothing prebuilt (no OpenVINO, or the model refused every static
    // shape): the class count is unknowable here — load_dict() has not run
    // yet, so labels holds only "blank". Guess 2 for sizing; the run()-time
    // capacity guard turns any resulting shortfall into counted drops, never
    // an overflow.
    I.classes = std::max(2, static_cast<int>(I.labels.size()));
  if (I.max_seq <= 0) I.max_seq = std::max(1, recognition::kRecWidthBuckets.back() / 8);
  max_logit_elems = std::max<std::size_t>(
      max_logit_elems, max_steps * static_cast<std::size_t>(I.classes));

  const int max_batch = recognition::kRecBatchLadder.back();
  I.d_M = I.d.alloc->allocate_buffer(static_cast<std::size_t>(max_batch) * 9 * sizeof(float));
  I.d_widths = I.d.alloc->allocate_buffer(static_cast<std::size_t>(max_batch) * sizeof(int));
  I.d_batch = I.d.alloc->allocate_buffer(max_in_elems * sizeof(float));
  I.d_logits = I.d.alloc->allocate_buffer(max_logit_elems * sizeof(float));
  I.d_idx = I.d.alloc->allocate_buffer(max_steps * sizeof(int));
  I.d_score = I.d.alloc->allocate_buffer(max_steps * sizeof(float));
  I.h_M.resize(static_cast<std::size_t>(max_batch) * 9);
  I.h_widths.resize(static_cast<std::size_t>(max_batch));
  I.h_idx.resize(max_steps);
  I.h_score.resize(max_steps);

  I.ready = true;
  return true;
}

bool IntelRecognizer::load_dict(const std::string &dict_path) {
  auto &I = *impl_;
  // SHARED loader: prepends nothing (we already pushed "blank") and appends " ",
  // so class indices agree with every other backend by construction.
  if (!recognition::load_label_dict(dict_path, I.labels))
    return false;
  if (I.classes_from_model && static_cast<int>(I.labels.size()) != I.classes) {
    // Same guard PaddleRec applies: a dict that disagrees with the model's class
    // head silently garbles text, so refuse rather than run.
    return false;
  }
  return true;
}

bool IntelRecognizer::is_ready() const noexcept { return impl_->ready; }

std::vector<backend::RecResult>
IntelRecognizer::run(const backend::ImageView &img,
                     const std::vector<turbo_ocr::Box> &boxes,
                     backend::DeviceQueue &queue) {
  auto &I = *impl_;
  dropped_crops_ = 0; // per-run; see last_dropped_crops()
  const int n = static_cast<int>(boxes.size());
  if (!I.ready || n == 0 || img.empty())
    return {};

  std::vector<backend::RecResult> results(static_cast<std::size_t>(n));

  // ---- SHARED routing + chunking + rung choice, in one call. The bucket table,
  // the batch ladder, the element budget and the overflow rule all live in
  // rec_batching.h; `bucket_rungs` tells it which artefacts actually exist here.
  std::vector<std::vector<int>> lists;
  const auto plan = recognition::plan_rec_batches(
      boxes, I.rec_h, recognition::kRecWidthBuckets, lists,
      I.bucket_rungs.empty() ? std::span<const std::vector<int>>{}
                             : std::span<const std::vector<int>>(I.bucket_rungs));

  const auto &ins = I.d.engine->input_names();
  const auto &outs = I.d.engine->output_names();
  const std::string in0 = tensor_name(ins, 0, "x");
  const std::string out0 = tensor_name(outs, 0, "softmax_5.tmp_0");
  const backend::DeviceKind space = I.d.alloc->has_device()
                                        ? backend::DeviceKind::L0
                                        : backend::DeviceKind::Host;
  const backend::NormParams rec_norm; // defaults ARE rec's: (x/255 - 0.5)/0.5, RGB

  for (const auto &chunk : plan) {
    const int bucket_w = recognition::kRecWidthBuckets[chunk.bucket];
    const auto &idx_list = lists[chunk.bucket];

    // Logits geometry from the compiled model (never an assumed /8 stride).
    const std::vector<std::int64_t> in_shape{chunk.batch, 3, I.rec_h, bucket_w};
    auto os = I.d.engine->output_shape(in_shape, out0);
    const int seq = (os.size() >= 3) ? static_cast<int>(os[1])
                                     : std::max(1, bucket_w / 8);
    const int classes = (os.size() >= 3) ? static_cast<int>(os[2])
                                         : static_cast<int>(I.labels.size());

    // CAPACITY GUARD — the last line of defence for the load()-time sizing.
    // Anything that lets a chunk outgrow the preallocated scratch (a sizing
    // miss, a model whose dynamic-variant geometry disagrees with the /8
    // estimate, an unknown class count on the nothing-prebuilt path) must
    // become a counted drop, never a heap overflow behind warp/infer/argmax.
    const std::size_t need_in =
        static_cast<std::size_t>(chunk.batch) * 3 * I.rec_h * bucket_w * sizeof(float);
    const std::size_t need_steps =
        static_cast<std::size_t>(chunk.batch) * static_cast<std::size_t>(seq);
    const std::size_t need_logits =
        need_steps * static_cast<std::size_t>(classes) * sizeof(float);
    if (need_in > I.d_batch.bytes() || need_logits > I.d_logits.bytes() ||
        need_steps * sizeof(int) > I.d_idx.bytes() ||
        need_steps * sizeof(float) > I.d_score.bytes() ||
        need_steps > I.h_idx.size()) {
      dropped_crops_ += chunk.count;
      TOCR_LOG_ERROR_RL("intel rec chunk exceeds preallocated scratch; "
                        "dropping crops",
                        "batch", chunk.batch, "width", bucket_w, "seq", seq,
                        "classes", classes);
      continue;
    }


    // Per-crop inverse perspective + content width, from the SHARED transform
    // (which also handles the vertical-text corner rotation and the width
    // clamp). Slots beyond `count` get width 0 so warp_crops zero-fills them.
    {
      prof::Scope _s(prof::REC_PRE);
      for (int i = 0; i < chunk.batch; ++i) {
        if (i < chunk.count) {
          const auto ct = turbo_ocr::compute_crop_transform(
              boxes[static_cast<std::size_t>(idx_list[chunk.offset + i])],
              I.rec_h, bucket_w);
          I.h_widths[static_cast<std::size_t>(i)] = ct.crop_width;
          std::copy_n(ct.M_inv, 9,
                      I.h_M.begin() + static_cast<std::size_t>(i) * 9);
        } else {
          I.h_widths[static_cast<std::size_t>(i)] = 0;
          std::fill_n(I.h_M.begin() + static_cast<std::size_t>(i) * 9, 9, 0.0f);
        }
      }
      I.d.alloc->copy_h2d(I.d_M.data(), I.h_M.data(),
                          static_cast<std::size_t>(chunk.batch) * 9 * sizeof(float), queue);
      I.d.alloc->copy_h2d(I.d_widths.data(), I.h_widths.data(),
                          static_cast<std::size_t>(chunk.batch) * sizeof(int), queue);

      I.d.kernels->warp_crops(img, static_cast<float *>(I.d_M.data()),
                              static_cast<int *>(I.d_widths.data()),
                              static_cast<float *>(I.d_batch.data()), chunk.batch,
                              I.rec_h, bucket_w, rec_norm, queue);
    }

    std::vector<backend::DeviceTensor> in(1), out(1);
    in[0] = {in0, I.d_batch.data(), space, backend::DType::F32, 0, in_shape};
    out[0] = {out0, I.d_logits.data(), space, backend::DType::F32, 0,
              {chunk.batch, seq, classes}};
    std::vector<backend::OutputLease> leases;
    const auto t_rec0 = std::chrono::steady_clock::now();
    bool rec_ran = false;
    {
      prof::Scope _s(prof::REC_INFER);
      rec_ran = I.d.engine->run(in, out, leases, queue);
    }
    if (intel_debug())
      std::fprintf(stderr, "[intel/rec] batch_infer=%.1fms shape_misses=%zu\n",
                   std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_rec0).count(),
                   I.d.engine->shape_misses());
    if (!rec_ran) {
      // This chunk fails and its boxes keep their PRE-SIZED empty results, so
      // the returned vector is still exactly boxes.size() long and the
      // pipeline's under-return check structurally cannot see the loss —
      // which is why the count goes out through the SHARED seam
      // (backend::IRecognizer::last_dropped_crops) as well as the log.
      dropped_crops_ += chunk.count;
      TOCR_LOG_ERROR_RL("intel rec forward failed; dropping crops",
                        "crops", chunk.count);
      continue;
    }

    prof::Scope _dec(prof::REC_DECODE); // argmax + D2H + CTC, per chunk
    auto *d_idx = static_cast<int *>(I.d_idx.data());
    auto *d_scr = static_cast<float *>(I.d_score.data());
    I.d.kernels->argmax(static_cast<float *>(I.d_logits.data()), d_idx, d_scr,
                        chunk.batch, seq, classes, queue);

    const std::size_t steps = static_cast<std::size_t>(chunk.count) * seq;
    I.d.alloc->copy_d2h(I.h_idx.data(), d_idx, steps * sizeof(int), queue);
    I.d.alloc->copy_d2h(I.h_score.data(), d_scr, steps * sizeof(float), queue);
    queue.synchronize(); // results must be host-visible before decode

    // SHARED greedy CTC (blank collapse, repeat collapse, mean score).
    for (int j = 0; j < chunk.count; ++j) {
      const std::size_t off = static_cast<std::size_t>(j) * seq;
      results[static_cast<std::size_t>(idx_list[chunk.offset + j])] =
          recognition::ctc_greedy_decode(I.h_idx.data() + off,
                                         I.h_score.data() + off, seq, I.labels);
    }
  }
  return results;
}

} // namespace turbo_ocr::intel
