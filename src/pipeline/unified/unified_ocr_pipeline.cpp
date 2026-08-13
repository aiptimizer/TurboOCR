// UnifiedOcrPipeline — one orchestration over the Backend seam, single-image
// path. See the header for the design.
//
// The class spans four TUs, split when this file passed the 900-line ceiling
// tools/checks/architecture.sh enforces, along the seams its own banner comments
// already named:
//
//   unified_ocr_pipeline.cpp      construction, staging, det/cls/layout, the
//                                 single-image entry points, orientation,
//                                 infer_one            <- you are here
//   unified_pipeline_dispatch.cpp router + table/formula dispatch
//   unified_pipeline_batch.cpp    batch entry points + cross-image pipelining
//   unified_pipeline_stages.cpp   optional-stage bootstrap + warmup
//
// Nothing device-specific appears in any of them: an ImageView, a DeviceQueue
// and the stage interfaces are the whole vocabulary.

#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"
#include "turbo_ocr/pipeline/unified/staging_ring.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/analysis/classification/cls_options.h"      // cls_all_boxes_enabled
#include "turbo_ocr/analysis/formula/cjk_stats.h"                // cjk_stats
#include "turbo_ocr/base/env_utils.h"                // env::env_or
#include "turbo_ocr/base/errors.h"                   // BackendUnavailableError, ImageTooLargeError
// The SHARED decompression-bomb verdict — the same predicate and cap ordering
// the HTTP routes and the gRPC handlers use, so the guard behind run_encoded()
// cannot drift from the one in front of it.
#include "turbo_ocr/image/size_classify.h"
#include "turbo_ocr/base/log/logger.h"               // TOCR_LOG_*
#include "turbo_ocr/analysis/formula/formula_bundle_env.h"      // resolve_formula_bundle_env
#include "turbo_ocr/base/geometry/box.h"             // Box, sorted_boxes, is_vertical_box
#include "turbo_ocr/core/types.h"                    // OCRResultItem, kDropScore
#include "turbo_ocr/core/layout_types.h"             // LayoutBox
#include "turbo_ocr/pipeline/ocr_pipeline_detail.h"    // THE shared OCR result policy
#include "turbo_ocr/pipeline/reading_order_util.h"     // maybe_assign_reading_order
#include "turbo_ocr/core/router_types.h"             // TableResult / FormulaResult

namespace turbo_ocr::pipeline {

using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;

namespace {

// Shared OCR result policy — pipeline::detail::*. This used to be an
// anonymous-namespace COPY here, which drifted from the original: different
// warning text for the same condition, a different combine_recognition arity,
// and assign-instead-of-append (which silently replaced a correct diagnosis with
// a false one). One implementation now, in turbo_ocr_common. Generic policy is
// shared, never per pipeline.
//
// Only what this TU CALLS is named. flag_text_degraded/flag_dropped_crops are
// deliberately absent: combine_recognition calls them itself, so a using-decl
// for them here would be scaffolding that reads as a call site.
using turbo_ocr::pipeline::detail::combine_recognition;

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

UnifiedOcrPipeline::UnifiedOcrPipeline(backend::Backend &backend,
                                       backend::StageSet stages,
                                       std::unique_ptr<backend::DeviceQueue> queue)
    : backend_(backend),
      caps_(backend.caps()),
      allocator_(backend.allocator()),
      queue_(std::move(queue)),
      det_(std::move(stages.detector)),
      rec_(std::move(stages.recognizer)),
      cls_(std::move(stages.classifier)),
      layout_(std::move(stages.layout)),
      orient_(backend.make_orient_func()) {
  use_cls_ = stages.available.classifier && cls_ != nullptr;
  // CROSS-REQUEST DETECTION BATCHING. One batcher per backend, shared by every
  // replica — asked for here rather than injected by the bootstrap so that both
  // the server's pool and any other pool (turbo_bench's K replicas) pick it up
  // with no wiring. Returns nullptr unless the backend/env opted in, in which
  // case detect_() is a plain det_->run().
  det_batcher_ = shared_detection_batcher(caps_, det_.get());
}

UnifiedOcrPipeline::~UnifiedOcrPipeline() = default;

// ---------------------------------------------------------------------------
// Upload — device-agnostic. Host backend zero-copy-wraps the cv::Mat; a device
// backend allocates a contiguous buffer and stages one H2D copy.
// ---------------------------------------------------------------------------


// --- StagingRing: grow-only, per replica -----------------------------------
StagingRing::~StagingRing() {
  if (!alloc) return;
  for (auto &s : slots)
    if (s.host) alloc->free_host(s.host);
}

void StagingRing::reserve(std::size_t n) {
  if (n > slots.size()) slots.resize(n);
}

StagingRing::Lease
StagingRing::acquire(std::size_t bytes) {
  if (slots.empty()) slots.resize(kResidentStagingSlots);
  Slot &s = slots[cursor];
  cursor = (cursor + 1) % slots.size();   // advance so N+1 cannot clobber N
  if (bytes > s.host_cap) {
    // cap ordered BEFORE the allocate: a throwing/failing allocate must not
    // leave a stale capacity describing a pointer we no longer own.
    if (s.host) { alloc->free_host(s.host); s.host = nullptr; s.host_cap = 0; }
    s.host = alloc->allocate_host(bytes);
    s.host_cap = s.host ? bytes : 0;
  }
  if (bytes > s.dev_cap) {
    s.dev_cap = 0;                             // same ordering, same reason
    s.dev = alloc->allocate_buffer(bytes);     // old buffer released by assignment
    s.dev_cap = s.dev.data() ? bytes : 0;
  }
  return Lease{static_cast<std::uint8_t *>(s.host), s.dev.data()};
}

void StagingRing::trim(std::size_t keep) {
  if (slots.size() <= keep) return;
  for (std::size_t i = keep; i < slots.size(); ++i)
    if (slots[i].host) alloc->free_host(slots[i].host);
  slots.resize(keep);                     // DeviceBuffer dtor frees the device side
  if (cursor >= slots.size()) cursor = 0;
}

bool UnifiedOcrPipeline::stages_through_ring_() const noexcept {
  return caps_.device != backend::DeviceKind::Host && allocator_ != nullptr;
}

UnifiedOcrPipeline::Uploaded
UnifiedOcrPipeline::upload_image_(const cv::Mat &img) {
  Uploaded u;
  const int rows = img.rows;
  const int cols = img.cols;
  if (!stages_through_ring_()) {
    // Zero-copy: the caller's cv::Mat data is already host-addressable and
    // outlives this synchronous run.
    u.view = backend::ImageView{img.data, static_cast<std::size_t>(img.step),
                                rows, cols, backend::DeviceKind::Host};
    return u;
  }
  // Device backend: pack to contiguous BGR8 then stage one ordered H2D copy.
  // Both buffers come from the per-replica grow-only ring — NO per-request
  // allocate/free (see StagingRing in the header for why that mattered).
  const std::size_t bytes = static_cast<std::size_t>(rows) * cols * 3;
  staging_.alloc = allocator_.get();
  // Host and device come from the SAME slot: the pinned buffer is the source of
  // an in-flight async copy, so it is only free to be rewritten when the device
  // buffer it feeds is.
  const auto slot = staging_.acquire(bytes);
  if (!slot.dev || !slot.host) {  // genuinely failed — fail loud, never memcpy to null
    TOCR_LOG_ERROR("upload_image_: staging allocation failed");
    return u;                     // empty view; callers treat as a failed page
  }
  if (img.isContinuous()) {
    std::memcpy(slot.host, img.data, bytes);
  } else {
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * 3;
    for (int r = 0; r < rows; ++r)
      std::memcpy(slot.host + r * row_bytes, img.ptr(r), row_bytes);
  }
  allocator_->copy_h2d(slot.dev, slot.host, bytes, *queue_);
  u.view = backend::ImageView{slot.dev, static_cast<std::size_t>(cols) * 3,
                              rows, cols, caps_.device};
  return u;
}

// ---------------------------------------------------------------------------
// The ONE detection call site.
//
// With no batcher installed (the default, and always for a backend whose
// IDetector::max_batch_size() is 1) this is byte-for-byte the det_->run() call
// that used to be inlined at each site.
//
// With a batcher installed, this replica's detector and queue are LENT to the
// batcher: if this thread becomes the batch leader they run the coalesced
// submission, otherwise they are untouched and the thread simply waits for its
// slot. Either way the boxes returned are this image's boxes in this image's
// original coordinates.
//
// The device-visibility barrier below is why a cross-replica batch is safe: the
// page was staged onto THIS replica's queue by upload_image_, but the leader may
// run the forward pass on ANOTHER replica's queue. Draining our own lane first
// makes the pixels visible to any lane of the same device. It is a no-op on a
// synchronous (Host) backend and, on Apple, drains a lane whose only pending
// work is a unified-memory memcpy — but on a backend with a genuinely
// asynchronous H2D (CUDA) it is exactly the ordering the cross-lane hand-off
// requires. It is only paid when batching is on.
// ---------------------------------------------------------------------------

std::vector<Box> UnifiedOcrPipeline::detect_(const backend::ImageView &view,
                                             int orig_h, int orig_w) {
  if (!det_batcher_) return det_->run(view, orig_h, orig_w, *queue_);
  if (det_batcher_->coalescing() && queue_->is_async()) {
    queue_->flush();
    queue_->synchronize();
  }
  return det_batcher_->detect(*det_, *queue_, view, orig_h, orig_w);
}

// ---------------------------------------------------------------------------
// Layout lane. See the header for why this is one helper.
// ---------------------------------------------------------------------------

backend::LayoutFuture
UnifiedOcrPipeline::enqueue_layout_(const backend::ImageView &view, int orig_h,
                                    int orig_w) {
  if (!layout_ || !layout_->supports_async()) return {};
  if (!layout_queue_) layout_queue_ = backend_.make_queue();
  if (!layout_queue_) return {};
  // CROSS-LANE BARRIER. `view` was staged H2D on `queue_`; layout executes on
  // `layout_queue_`. Recorded HERE, at the current point of the main lane, and
  // waited on device-side — no host round-trip, and nothing later on `queue_`
  // (the next page's upload and detection, in the pipelined path) gets pulled
  // into the dependency.
  if (queue_->is_async() && layout_queue_->is_async()) {
    if (!upload_event_) upload_event_ = queue_->make_event();
    if (upload_event_) {
      queue_->record(*upload_event_);
      layout_queue_->wait(*upload_event_);
    }
  }
  return layout_->enqueue(view, orig_h, orig_w,
                          backend::kDefaultLayoutScoreThreshold,
                          *layout_queue_);
}

// ---------------------------------------------------------------------------
// Angle classification (ported from OcrPipeline::classify_angles_)
// ---------------------------------------------------------------------------

void UnifiedOcrPipeline::classify_angles_(const backend::ImageView &view,
                                          std::vector<Box> &boxes) {
  if (!use_cls_ || !cls_ || boxes.empty()) return;
  if (classification::cls_all_boxes_enabled()) {
    cls_->run(view, boxes, *queue_);  // flips 180° boxes in place
    return;
  }
  vertical_box_indices_.clear();
  for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
    if (turbo_ocr::is_vertical_box(boxes[i])) vertical_box_indices_.push_back(i);
  if (vertical_box_indices_.empty()) return;
  vertical_boxes_buf_.clear();
  vertical_boxes_buf_.reserve(vertical_box_indices_.size());
  for (int idx : vertical_box_indices_)
    vertical_boxes_buf_.push_back(boxes[idx]);
  cls_->run(view, vertical_boxes_buf_, *queue_);
  for (std::size_t j = 0; j < vertical_box_indices_.size(); ++j)
    boxes[vertical_box_indices_[j]] = vertical_boxes_buf_[j];
}

// ---------------------------------------------------------------------------
// Single-image entry points
// ---------------------------------------------------------------------------

std::vector<OCRResultItem> UnifiedOcrPipeline::run(const cv::Mat &img) {
  return run_with_layout(img).results;
}

OcrPipelineResult UnifiedOcrPipeline::run_with_layout(
    const cv::Mat &img, const RunFlags &flags,
    const backend_routing::RequestRouting &routing, bool defer_external) {
  if (img.empty()) return OcrPipelineResult{};
  Uploaded up = upload_image_(img);
  return run_from_view_(up.view, img.rows, img.cols, flags, routing,
                        defer_external);
}

// ENCODED-BYTES path. Decodes on-device when the backend has a decoder, which
// keeps the ~200 KB JPEG on the PCIe bus instead of the ~25 MB decoded page.
OcrPipelineResult UnifiedOcrPipeline::run_encoded(
    const std::uint8_t *data, std::size_t len, const RunFlags &flags,
    const backend_routing::RequestRouting &routing, bool defer_external) {
  if (!data || len == 0) return OcrPipelineResult{};

  // POST-DECODE DECOMPRESSION-BOMB GUARD. It lives HERE, not at the call sites,
  // because this is where the decode happens.
  //
  // The transports all sniff the encoded header before calling in
  // (reject_if_too_large_pre / grpc_pre_decode_dim_check), but that sniff cannot
  // parse every container — BMP and PNM in particular — so each of them ALSO
  // re-checks the decoded dimensions. That second check is the only thing
  // standing between a 60000x60000 BMP and a ~10 GB allocation.
  //
  // On the cv::Mat path the route still holds the decoded image and does it
  // itself. On THIS path it cannot: the whole point is that the route never
  // materializes the image. Leaving the guard at the call sites therefore
  // silently dropped it for every encoded caller — which is exactly what
  // happened when the gRPC image RPCs were moved onto this path. Putting it
  // behind the decode makes the protection travel with the decision instead of
  // needing to be remembered once per transport.
  const auto reject_oversize = [](int w, int h) {
    return decode::classify_image_size(w, h) != decode::ImageSizeVerdict::kOk;
  };

  if (caps_.native_image_decode) {
    if (!kernels_) kernels_ = backend_.make_kernels();
    if (kernels_) {
      const backend::ImageView v = kernels_->decode_image(data, len, *queue_);
      // Empty view = decoder declined (unsupported container, corrupt bytes).
      // Fall through to the host decoder rather than failing the request: the
      // seam documents an empty return as "failure", not as "the image is bad".
      if (v.data && v.rows > 0 && v.cols > 0) {
        // A device decoder can be handed a bomb too — nvJPEG will happily
        // allocate the full frame in VRAM.
        if (reject_oversize(v.cols, v.rows))
          throw turbo_ocr::ImageTooLargeError(
              "decoded image dimensions exceed the configured maximum");
        return run_from_view_(v, v.rows, v.cols, flags, routing, defer_external);
      }
    }
  }

  // Host decode + the normal upload path. This is what every backend without an
  // on-device decoder does, and it is byte-identical to calling the cv::Mat
  // overload directly.
  auto decode_fn = backend_.make_image_decoder();
  const cv::Mat img = decode_fn ? decode_fn(data, len) : cv::Mat{};
  // Same rule as make_infer_func's fallback: a failed decode is an ERROR, not
  // an empty page — returning an empty result answered a corrupt upload with
  // a clean 200 indistinguishable from a blank page.
  if (img.empty())
    throw turbo_ocr::ImageDecodeError("failed to decode image");
  if (reject_oversize(img.cols, img.rows))
    throw turbo_ocr::ImageTooLargeError(
        "decoded image dimensions exceed the configured maximum");
  return run_with_layout(img, flags, routing, defer_external);
}

// The shared core: everything after the page is resident, expressed against an
// ImageView so it does not care whether the pixels arrived via host decode +
// H2D or via an on-device decoder.
OcrPipelineResult UnifiedOcrPipeline::run_from_view_(
    const backend::ImageView &view, int rows, int cols, const RunFlags &flags,
    const backend_routing::RequestRouting &routing, bool defer_external) {
  OcrPipelineResult out;
  if (!view.data || rows <= 0 || cols <= 0) return out;
  const bool layout_active = flags.layout && layout_ != nullptr;

  // LAYOUT-ONLY (?text=0&layout=1): skip the whole det->cls->rec chain and run
  // just the layout model. This is the run the pre-seam CUDA server's
  // run_layout_and_structure performed; the unified pipeline used to lack it,
  // so validation rejected the request as "not implemented". No router / no
  // reading order: validation guarantees every text-consuming flag is off.
  if (!flags.text) {
    if (layout_active)
      out.layout = layout_->run(view, rows, cols,
                                backend::kDefaultLayoutScoreThreshold, *queue_);
    return out;
  }

  // Detection -> host boxes (the stage synchronizes internally as needed).
  // Routed through the shared cross-request batcher when one is installed.
  std::vector<Box> boxes = detect_(view, rows, cols);
  turbo_ocr::sorted_boxes(boxes);

  // LAYOUT OVERLAP. Layout depends only on the page, not on detection's boxes,
  // so it can run CONCURRENTLY with cls+rec instead of serialising in front of
  // them. The pre-seam CUDA pipeline did exactly this: layout was enqueued on
  // its own stream gated on the detection event and collected at the very end,
  // making the wall clock `upload + det + max(layout, cls+rec)` rather than
  // `upload + det + layout + cls + rec`.
  //
  // The layout LANE is a second DeviceQueue: submitting on `*queue_` would put
  // layout in the same ordered stream as recognition and overlap nothing. On a
  // host backend HostDeviceQueue is a synchronous no-op, so this costs nothing.
  //
  // An invalid future means the backend cannot overlap; layout then runs here,
  // blocking, exactly where it always did.
  backend::LayoutFuture layout_fut;
  if (layout_active) layout_fut = enqueue_layout_(view, rows, cols);
  if (layout_active && !layout_fut.valid()) {
    out.layout = layout_->run(view, rows, cols,
                              backend::kDefaultLayoutScoreThreshold, *queue_);
  }

  // Optional angle classification.
  classify_angles_(view, boxes);

  // Recognition -> (text, score) per box.
  std::vector<std::pair<std::string, float>> rec_results =
      rec_->run(view, boxes, *queue_);

  // Collect layout AFTER rec, so the two overlapped. `view` (and the staging
  // buffer behind it) outlives this point, which is what makes enqueue()'s
  // non-owning ImageView safe here. If rec_ throws, the future's destructor
  // still waits for the submission before `view` goes away.
  if (layout_fut.valid()) out.layout = layout_fut.collect();

  combine_recognition(out, boxes, rec_results, rec_->last_dropped_crops());

  // CUA router + table/formula dispatch (no-op on text-only pages).
  dispatch_router_(out, view, boxes, flags, routing, defer_external);

  maybe_assign_reading_order(flags.reading_order, out.results, out.layout,
                             out.reading_order);
  return out;
}

// ---------------------------------------------------------------------------
// Orientation + Tier-B ad-hoc inference
// ---------------------------------------------------------------------------

int UnifiedOcrPipeline::detect_orientation(const cv::Mat &bgr) {
  return orient_ ? orient_(bgr) : 0;
}

std::string
UnifiedOcrPipeline::infer_one(const cv::Mat &img, const std::string &modality,
                              const std::string &backend_name,
                              const backend_routing::BackendSpec *inline_spec) {
  if (img.empty()) return {};
  Uploaded up = upload_image_(img);
  const backend::ImageView &view = up.view;
  const int w = img.cols;
  const int h = img.rows;
  // The whole crop is the single region: corners [tl, tr, br, bl].
  const std::vector<Box> regions{Box{{{{0, 0}, {w, 0}, {w, h}, {0, h}}}}};

  if (modality == "table") {
    std::unique_ptr<backend::ITableRecognizer> transient;
    backend::ITableRecognizer *rec = nullptr;
    if (inline_spec) {
      transient = backend_.make_table_recognizer(*inline_spec);
      if (transient && transient->load()) rec = transient.get();
    } else {
      rec = pick_table_recognizer_(backend_name);
    }
    if (!rec || !rec->is_ready())
      throw turbo_ocr::BackendUnavailableError(
          "table backend '" + (inline_spec ? std::string("<inline>") : backend_name) +
          "' is unavailable (not loaded/ready)");
    auto r = rec->run(view, regions, /*page_ocr=*/{}, *queue_);
    return r.empty() ? std::string() : std::move(r[0].html);
  }
  if (modality == "formula") {
    std::unique_ptr<backend::IFormulaRecognizer> transient;
    backend::IFormulaRecognizer *rec = nullptr;
    if (inline_spec) {
      transient = backend_.make_formula_recognizer(*inline_spec);
      if (transient && transient->load_model_dir(inline_spec->model_path) &&
          transient->load_tokenizer(""))
        rec = transient.get();
    } else {
      rec = pick_formula_recognizer_(backend_name);
    }
    if (!rec || !rec->is_ready())
      throw turbo_ocr::BackendUnavailableError(
          "formula backend '" +
          (inline_spec ? std::string("<inline>") : backend_name) +
          "' is unavailable (not loaded/ready)");
    auto r = rec->run(view, regions, *queue_);
    if (!r.empty() && !r[0].ok)
      std::cerr << "[infer_one] formula backend degraded for /infer region "
                   "(transport/parse failure, not empty input)\n";
    return r.empty() ? std::string() : std::move(r[0].latex);
  }
  return {};
}

} // namespace turbo_ocr::pipeline
