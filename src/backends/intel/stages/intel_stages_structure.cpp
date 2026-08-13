// Intel structure stages — classifier + layout. Split from intel_stages.cpp
// when that file crossed the 900-line gate (tools/checks/architecture.sh); the
// detector and recognizer stayed there. Same contract and same toolchain note
// as that file: the orchestration here is toolchain-agnostic C++ and is
// syntax-checkable on any host, while the device work happens inside
// SyclKernels and OpenVINOEngine.

#include "intel/stages/intel_stages.h"
#include "intel/stages/intel_stages_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "turbo_ocr/analysis/classification/cls_config.h"
#include "turbo_ocr/analysis/layout/layout_postfilter.h"
#include "turbo_ocr/analysis/layout/picodet_decode.h"
#include "turbo_ocr/analysis/recognition/rec_batching.h" // batch_ladder_for_width, snap_batch
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/base/geometry/perspective.h" // compute_crop_transform
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/base/log/stage_profiler.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/core/norm_params.h"

namespace turbo_ocr::intel {

// SHARED norm factory, as in intel_stages.cpp — never a local copy.
using backend::norm::layout_norm;
using stagesdetail::tensor_name;

// ============================= Classifier ===================================

struct IntelClassifier::Impl {
  StageDeps d;
  bool ready = false;
  // Cls canvas geometry from the SHARED classification header — not retyped.
  static constexpr int kH = classification::kClsImageH;
  static constexpr int kW = classification::kClsImageW;

  backend::DeviceBuffer d_M, d_widths, d_batch, d_out;
  std::vector<float> h_M, h_out;
  std::vector<int> h_widths;
  std::vector<int> rungs; // batch sizes actually prebuilt

  explicit Impl(StageDeps deps) : d(std::move(deps)) {}
};

IntelClassifier::IntelClassifier(StageDeps deps)
    : impl_(std::make_unique<Impl>(std::move(deps))) {}
IntelClassifier::~IntelClassifier() = default;

bool IntelClassifier::load(const std::string &model_path) {
  auto &I = *impl_;
  if (!I.d.engine->load(model_path))
    return false;

  // Same warmup discipline as rec, on the SHARED batch ladder.
  I.rungs = recognition::batch_ladder_for_width(Impl::kW, Impl::kH);
  std::vector<std::vector<std::int64_t>> shapes;
  shapes.reserve(I.rungs.size());
  for (int b : I.rungs)
    shapes.push_back({b, 3, Impl::kH, Impl::kW});
  I.d.engine->prebuild(shapes);

  const int max_b = I.rungs.empty() ? recognition::kRecBatchLadder.back()
                                    : I.rungs.back();
  I.d_M = I.d.alloc->allocate_buffer(static_cast<std::size_t>(max_b) * 9 * sizeof(float));
  I.d_widths = I.d.alloc->allocate_buffer(static_cast<std::size_t>(max_b) * sizeof(int));
  I.d_batch = I.d.alloc->allocate_buffer(
      static_cast<std::size_t>(max_b) * 3 * Impl::kH * Impl::kW * sizeof(float));
  I.d_out = I.d.alloc->allocate_buffer(static_cast<std::size_t>(max_b) * 2 * sizeof(float));
  I.h_M.resize(static_cast<std::size_t>(max_b) * 9);
  I.h_widths.resize(static_cast<std::size_t>(max_b));
  I.h_out.resize(static_cast<std::size_t>(max_b) * 2);

  I.ready = true;
  return true;
}

bool IntelClassifier::is_ready() const noexcept { return impl_->ready; }

void IntelClassifier::run(const backend::ImageView &img,
                         std::vector<turbo_ocr::Box> &boxes,
                         backend::DeviceQueue &queue) {
  auto &I = *impl_;
  const int n = static_cast<int>(boxes.size());
  if (!I.ready || n == 0 || img.empty())
    return;
  prof::Scope _s(prof::CLS);

  const auto &ins = I.d.engine->input_names();
  const auto &outs = I.d.engine->output_names();
  const std::string in0 = tensor_name(ins, 0, "x");
  const std::string out0 = tensor_name(outs, 0, "softmax_0.tmp_0");
  const backend::DeviceKind space = I.d.alloc->has_device()
                                        ? backend::DeviceKind::L0
                                        : backend::DeviceKind::Host;
  // SHARED cls normalization (== rec's). MEASURED, see cls_config.h — ImageNet
  // here is a regression that three backends have now shipped independently.
  const backend::NormParams cls_norm = classification::cls_norm();

  const int cap = I.rungs.empty() ? recognition::kRecBatchLadder.back()
                                  : I.rungs.back();
  int flipped = 0;

  for (int beg = 0; beg < n; beg += cap) {
    const int count = std::min(cap, n - beg);
    const int batch = I.rungs.empty()
                          ? count
                          : recognition::snap_batch(count, I.rungs);

    for (int j = 0; j < batch; ++j) {
      if (j < count) {
        // SHARED crop transform, target 80x160 (identical call to PaddleCls).
        const auto ct = turbo_ocr::compute_crop_transform(
            boxes[static_cast<std::size_t>(beg + j)], Impl::kH, Impl::kW);
        I.h_widths[static_cast<std::size_t>(j)] = ct.crop_width;
        std::copy_n(ct.M_inv, 9, I.h_M.begin() + static_cast<std::size_t>(j) * 9);
      } else {
        I.h_widths[static_cast<std::size_t>(j)] = 0;
        std::fill_n(I.h_M.begin() + static_cast<std::size_t>(j) * 9, 9, 0.0f);
      }
    }
    I.d.alloc->copy_h2d(I.d_M.data(), I.h_M.data(),
                        static_cast<std::size_t>(batch) * 9 * sizeof(float), queue);
    I.d.alloc->copy_h2d(I.d_widths.data(), I.h_widths.data(),
                        static_cast<std::size_t>(batch) * sizeof(int), queue);
    I.d.kernels->warp_crops(img, static_cast<float *>(I.d_M.data()),
                            static_cast<int *>(I.d_widths.data()),
                            static_cast<float *>(I.d_batch.data()), batch,
                            Impl::kH, Impl::kW, cls_norm, queue);

    std::vector<backend::DeviceTensor> in(1), out(1);
    in[0] = {in0, I.d_batch.data(), space, backend::DType::F32, 0,
             {batch, 3, Impl::kH, Impl::kW}};
    out[0] = {out0, I.d_out.data(), space, backend::DType::F32, 0, {batch, 2}};
    std::vector<backend::OutputLease> leases;
    if (!I.d.engine->run(in, out, leases, queue))
      continue;

    I.d.alloc->copy_d2h(I.h_out.data(), I.d_out.data(),
                        static_cast<std::size_t>(count) * 2 * sizeof(float), queue);
    queue.synchronize();

    for (int j = 0; j < count; ++j) {
      const float s0 = I.h_out[static_cast<std::size_t>(j) * 2 + 0];
      const float s180 = I.h_out[static_cast<std::size_t>(j) * 2 + 1];
      // SHARED decision + SHARED rotation (classification::should_flip_180 /
      // flip_quad_180) — one definition for every backend.
      if (classification::should_flip_180(s0, s180)) {
        classification::flip_quad_180(boxes[static_cast<std::size_t>(beg + j)]);
        ++flipped;
      }
    }
  }
  // IClassifier::run returns void, so this count has no caller. Kept and LOGGED
  // rather than deleted: "how many crops came in upside down" is the only signal
  // that the orientation classifier is doing anything, and a silently-discarded
  // counter is how a dead cls stage goes unnoticed. Same treatment as the Apple
  // arm (mps_stages.mm).
  if (flipped > 0)
    TOCR_LOG_DEBUG("intel cls flipped boxes 180", "flipped", flipped, "boxes", n);
}

// =============================== Layout =====================================

// PP-DocLayoutV3 IO, mirroring src/backends/nvidia/stages/paddle_layout.cpp exactly:
//   inputs : image [1,3,800,800] | im_shape [2] | scale_factor [2]
//   outputs: rows (N,7) = {class, score, x0, y0, x1, y1, read_order}
//            count (1,) int32 — AUTHORITATIVE row count (the rows tensor's
//                               first dim is data-dependent and unreliable)
//            mask (N,200,200)  — never read; left engine-internal
// paddle2onnx does not guarantee input order, so dispatch by NAME.
struct IntelLayout::Impl {
  StageDeps d;
  bool ready = false;
  static constexpr int kInput = 800;
  // NMS output budget — from the SHARED decoder header so the buffer sizing
  // here and the row clamp in decode_picodet_rows can never disagree.
  static constexpr int kMaxDet = turbo_ocr::layout::kPicodetMaxDet;

  std::string n_image, n_im_shape, n_scale, n_rows, n_count;
  backend::DeviceBuffer d_img, d_imshape, d_scale;

  explicit Impl(StageDeps deps) : d(std::move(deps)) {}
};

IntelLayout::IntelLayout(StageDeps deps)
    : impl_(std::make_unique<Impl>(std::move(deps))) {}
IntelLayout::~IntelLayout() = default;

bool IntelLayout::load(const std::string &model_path) {
  auto &I = *impl_;
  if (!I.d.engine->load(model_path))
    return false;

  // Dispatch inputs by name (paddle2onnx does not guarantee order); pick the
  // rank-2 outputs as rows/count in declaration order, as the TRT path does.
  for (const auto &n : I.d.engine->input_names()) {
    if (n == "im_shape")
      I.n_im_shape = n;
    else if (n == "scale_factor")
      I.n_scale = n;
    else if (I.n_image.empty() && n.find("image") != std::string::npos)
      I.n_image = n;
  }
  const auto &outs = I.d.engine->output_names();
  I.n_rows = tensor_name(outs, 0, "multiclass_nms3_0.tmp_0");
  I.n_count = tensor_name(outs, 1, "multiclass_nms3_0.tmp_2");
  if (I.n_image.empty() || I.n_im_shape.empty() || I.n_scale.empty())
    return false; // not a PP-DocLayoutV3 graph; disable cleanly

  // Layout has exactly ONE input shape, so the whole model prebuilds at load
  // and nothing is ever compiled on the hot path.
  I.d.engine->prebuild({{1, 3, Impl::kInput, Impl::kInput}});

  I.d_img = I.d.alloc->allocate_buffer(
      static_cast<std::size_t>(3) * Impl::kInput * Impl::kInput * sizeof(float));
  I.d_imshape = I.d.alloc->allocate_buffer(2 * sizeof(float));
  I.d_scale = I.d.alloc->allocate_buffer(2 * sizeof(float));

  I.ready = true;
  return true;
}

bool IntelLayout::is_ready() const noexcept { return impl_->ready; }

std::vector<turbo_ocr::layout::LayoutBox>
IntelLayout::run(const backend::ImageView &img, int orig_h, int orig_w,
                 float score_threshold, backend::DeviceQueue &queue) {
  auto &I = *impl_;
  if (!I.ready || img.empty() || orig_h <= 0 || orig_w <= 0)
    return {};
  prof::Scope _s(prof::LAYOUT);
  const int S = Impl::kInput;

  // Whole page stretched into the 800x800 canvas (pixel/255, BGR CHW).
  I.d.kernels->resize_normalize(img, static_cast<float *>(I.d_img.data()), S, S,
                                layout_norm(), queue);

  // PaddleX convention, copied verbatim from src/backends/nvidia/stages/paddle_layout.cpp:
  //   im_shape     = [resized_h, resized_w]  (always 800 x 800)
  //   scale_factor = [800/orig_h, 800/orig_w]
  // The PicoDet head applies these internally, so the emitted boxes are already
  // in ORIGINAL coordinates and need only a clamp. Getting this pair backwards
  // silently scales every layout box — hence "copied verbatim", not re-derived.
  const float imshape[2] = {static_cast<float>(S), static_cast<float>(S)};
  const float scale[2] = {static_cast<float>(S) / orig_h,
                          static_cast<float>(S) / orig_w};
  I.d.alloc->copy_h2d(I.d_imshape.data(), imshape, sizeof(imshape), queue);
  I.d.alloc->copy_h2d(I.d_scale.data(), scale, sizeof(scale), queue);

  const backend::DeviceKind space = I.d.alloc->has_device()
                                        ? backend::DeviceKind::L0
                                        : backend::DeviceKind::Host;
  std::vector<backend::DeviceTensor> in(3), out(2);
  in[0] = {I.n_image, I.d_img.data(), space, backend::DType::F32, 0, {1, 3, S, S}};
  in[1] = {I.n_im_shape, I.d_imshape.data(), space, backend::DType::F32, 0, {1, 2}};
  in[2] = {I.n_scale, I.d_scale.data(), space, backend::DType::F32, 0, {1, 2}};
  // data == nullptr: both of these have DATA-DEPENDENT shapes, so the engine
  // owns them and leases them back (see OpenVINOEngine's header). The ~48 MB
  // mask output is deliberately NOT listed — it stays inside OpenVINO and is
  // never copied.
  out[0] = {I.n_rows, nullptr, space, backend::DType::F32, 0, {}};
  out[1] = {I.n_count, nullptr, space, backend::DType::I32, 0, {}};
  std::vector<backend::OutputLease> leases;
  if (!I.d.engine->run(in, out, leases, queue))
    return {};

  const backend::OutputLease *rows = nullptr, *count = nullptr;
  for (const auto &l : leases) {
    if (l.name == I.n_rows)
      rows = &l;
    else if (l.name == I.n_count)
      count = &l;
  }
  if (!rows || !rows->data || rows->shape.size() < 2)
    return {};

  // SHARED PicoDet row decoder (turbo_ocr/analysis/layout/picodet_decode.h). This
  // decoder used to live inline here — it was the CORRECT one of the three
  // copies, so it was lifted verbatim into the shared header and AMD's (and any
  // future Apple) layout now calls the same code. Count-tensor precedence,
  // kPicodetMaxDet clamp, class-id range check, read_order and the truncating
  // (not lround) coordinate cast are all decided in exactly one place.
  // The SHARED postfilter (NMS + full-page-image drop + containment/merge-mode
  // reconciliation) must run on EVERY backend: CPU (ort_paddle_layout.cpp:271)
  // and NVIDIA (paddle_layout.cpp:223) already applied it, and these two arms
  // did not — so Intel/AMD returned raw overlapping boxes and their layout,
  // reading order and every downstream block/table decision diverged from the
  // other two on the same page. Generic policy is shared, never per backend.
  auto boxes = turbo_ocr::layout::decode_picodet_rows(
      static_cast<const float *>(rows->data), static_cast<int>(rows->shape[0]),
      static_cast<int>(rows->shape[1]),
      (count && count->data) ? static_cast<const std::int32_t *>(count->data)
                             : nullptr,
      score_threshold, orig_h, orig_w);
  return turbo_ocr::layout::postfilter_layout_boxes(std::move(boxes), orig_h,
                                                    orig_w);
}

} // namespace turbo_ocr::intel
