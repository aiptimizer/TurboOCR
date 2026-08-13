// UnifiedOcrPipeline — batch entry points and cross-image stage pipelining.
//
// The multi-image schedules: run_batch/run_batch_with_layout, the
// enqueue/collect ring that overlaps image N's cls+rec with image N+1's
// detection, and the batched layout stage. The single-image path lives in
// unified_ocr_pipeline.cpp.
//
// Split out of unified_ocr_pipeline.cpp, which had reached 1000 lines — over the
// 900-line ceiling tools/checks/architecture.sh enforces. The seams were already
// named by the banner comments in that file; this is those seams made physical,
// not a new decomposition. All four TUs define members of the SAME class, so the
// header is unchanged and nothing outside this directory can tell the difference.

#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

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
#include "turbo_ocr/base/log/logger.h"               // TOCR_LOG_*
#include "turbo_ocr/base/geometry/box.h"             // Box, sorted_boxes, is_vertical_box
#include "turbo_ocr/core/types.h"                    // OCRResultItem, kDropScore
#include "turbo_ocr/pipeline/ocr_pipeline_detail.h"    // THE shared OCR result policy
#include "turbo_ocr/pipeline/reading_order_util.h"     // maybe_assign_reading_order

namespace turbo_ocr::pipeline {


using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;

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

std::vector<std::vector<OCRResultItem>>
UnifiedOcrPipeline::run_batch(const std::vector<cv::Mat> &imgs) {
  // Text-only entry point: every optional stage stays off (RunFlags defaults).
  auto outs = run_batch_with_layout(imgs, RunFlags{});
  std::vector<std::vector<OCRResultItem>> results;
  results.reserve(outs.size());
  for (auto &out : outs) results.push_back(std::move(out.results));
  return results;
}

std::vector<OcrPipelineResult> UnifiedOcrPipeline::run_batch_with_layout(
    const std::vector<cv::Mat> &imgs, const RunFlags &flags,
    const backend_routing::RequestRouting &routing) {
  if (imgs.empty()) return {};
  // LAYOUT-ONLY: no det/rec means no stage pipelining to win — each page is
  // one layout forward. run_with_layout short-circuits per page.
  if (!flags.text) {
    std::vector<OcrPipelineResult> outs;
    outs.reserve(imgs.size());
    for (const auto &img : imgs)
      outs.push_back(
          run_with_layout(img, flags, routing, /*defer_external=*/false));
    return outs;
  }
  if (imgs.size() == 1) {
    std::vector<OcrPipelineResult> single;
    single.push_back(
        run_with_layout(imgs[0], flags, routing, /*defer_external=*/false));
    return single;
  }

  // When the detector can overlap across images, the pipelined schedule beats
  // "upload everything, then batch every stage": it needs 2 pages resident
  // instead of N, and it starts recognizing page 0 while page 1 is still being
  // detected. Backends without async detection keep the batched path, which is
  // what their native run_batch/run_multi overrides exist for.
  if (supports_stage_pipelining()) return run_pipelined(imgs, flags, routing);

  const int n = static_cast<int>(imgs.size());

  // Upload every page; keep each Uploaded alive for the whole batch so device
  // buffers back the ImageViews through batched det/rec.
  //
  // ALL n pages are simultaneously live here — run_batch/run_multi are handed
  // the whole view array at once — so the staging ring must hold n distinct
  // slots. With the default 2 the ring aliased page i onto page i+2 and page 0
  // was detected and recognized from page 2's pixels.
  if (stages_through_ring_()) staging_.reserve(static_cast<std::size_t>(n));
  // The staging surplus must go back on EVERY path out of this function. The
  // trim used to be a plain call at the end, so any throwing stage unwound
  // past it and this replica kept all n page buffers (pinned host + device)
  // pinned forever — the ring is grow-only and nothing else shrinks it.
  struct TrimGuard {
    UnifiedOcrPipeline *p;
    ~TrimGuard() {
      if (p->stages_through_ring_()) p->staging_.trim(kResidentStagingSlots);
    }
  } trim_guard{this};
  std::vector<Uploaded> uploads;
  uploads.reserve(n);
  std::vector<backend::ImageView> views;
  std::vector<std::pair<int, int>> dims;
  views.reserve(n);
  dims.reserve(n);
  for (const auto &img : imgs) {
    uploads.push_back(upload_image_(img));
    // A failed upload (staging allocation failure) returns an EMPTY view by
    // contract, and the single-image path honours it (run_from_view_ bails on
    // !view.data). This path used to push the null view with the REAL page
    // dims — a backend then received {data=nullptr, rows>0} and dereferenced
    // it. Zero dims make every stage's own empty-input guard fire instead,
    // and the page comes back empty like any other failed page.
    if (!uploads.back().view.data) {
      TOCR_LOG_ERROR("batch: page upload failed; page will come back empty",
                     "rows", img.rows, "cols", img.cols);
      views.push_back(backend::ImageView{});
      dims.emplace_back(0, 0);
      continue;
    }
    views.push_back(uploads.back().view);
    dims.emplace_back(img.rows, img.cols);
  }

  // Batched detection (default impl loops run() when a backend has no native
  // batch; native-batch backends override run_batch).
  std::vector<std::vector<Box>> all_det = det_->run_batch(views, dims, *queue_);
  // stages.h states the contract: EXACTLY imgs.size() entries, in input order.
  // Indexing all_det[i] on a shorter vector is an out-of-bounds read, and this
  // is a PUBLIC entry point (the Python bindings and any pool reach it), so the
  // violation is caught here rather than trusted.
  if (all_det.size() != static_cast<std::size_t>(n)) {
    TOCR_LOG_ERROR("IDetector::run_batch violated its size contract",
                   "expected", n, "returned",
                   static_cast<long long>(all_det.size()));
    // RECOVER PER IMAGE, don't pad. A resize() fills the tail with EMPTY box
    // lists — and an empty list is indistinguishable downstream from a
    // genuinely blank page: flag_text_degraded is gated on num_boxes > 0, so
    // the client got a clean 200 for a page the detector never answered.
    // DetectionBatcher::submit_ handles the identical violation by re-running
    // each shorted slot individually; do the same here so the failure lands
    // on — and only on — the image that caused it.
    const std::size_t got = all_det.size();
    all_det.resize(static_cast<std::size_t>(n));
    for (std::size_t i = got; i < static_cast<std::size_t>(n); ++i)
      all_det[i] = det_->run(views[i], dims[i].first, dims[i].second, *queue_);
  }

  std::vector<backend::ImageCrops> crops(n);
  for (int i = 0; i < n; ++i) {
    turbo_ocr::sorted_boxes(all_det[i]);
    classify_angles_(views[i], all_det[i]);
    crops[i].img = views[i];
    crops[i].boxes = std::move(all_det[i]);
  }

  // Batched recognition across all pages.
  std::vector<std::vector<std::pair<std::string, float>>> all_rec =
      rec_->run_multi(crops, *queue_);
  // Same contract, same reason (stages.h: one result list per ImageCrops entry,
  // in input order). A short return here would index past the end below; the
  // tail images instead come back empty and combine_recognition's own
  // under-return check marks each of them text_degraded.
  if (all_rec.size() != static_cast<std::size_t>(n)) {
    TOCR_LOG_ERROR("IRecognizer::run_multi violated its size contract",
                   "expected", n, "returned",
                   static_cast<long long>(all_rec.size()));
    all_rec.resize(static_cast<std::size_t>(n));
  }

  std::vector<OcrPipelineResult> outs(n);
  for (int i = 0; i < n; ++i)
    combine_recognition(outs[i], crops[i].boxes, all_rec[i]);
  // run_multi() reports ONE drop count for the whole multi-image call, so it
  // cannot be attributed to a page. Reporting NOTHING was the wrong conclusion
  // from a sound premise (an earlier note here even said "left out
  // deliberately" while the comment above it argued the opposite): this file
  // holds everywhere that a thinner page must never read as a clean one, so
  // when ANY crop was dropped, EVERY page in the batch is flagged, with a
  // message that is honest about the attribution limit.
  if (const int batch_dropped = rec_ ? rec_->last_dropped_crops() : 0;
      batch_dropped > 0) {
    const std::string w =
        "text stage degraded: recognition dropped " +
        std::to_string(batch_dropped) + " crop(s) somewhere in this " +
        std::to_string(n) +
        "-image batch (per-page attribution is not available; they are not "
        "blank)";
    for (auto &o : outs) {
      o.text_degraded = true;
      o.text_warning = o.text_warning.empty() ? w : o.text_warning + "; " + w;
    }
  }

  if (flags.layout && layout_)
    run_batch_layout_stage_(crops, outs, flags, routing);

  // The surplus page buffers go back via TrimGuard above (destructor order:
  // after the last read of `views`/`crops`). Retaining all n would let one
  // 200-page batch pin 200 page buffers on this replica forever; retaining
  // kResidentStagingSlots keeps the single-image path allocation-free, which
  // is the whole point of the ring.
  return outs;
}

// ---------------------------------------------------------------------------
// Cross-image stage pipelining (SHARED). See the header for the contract.
// ---------------------------------------------------------------------------

bool UnifiedOcrPipeline::supports_stage_pipelining() const noexcept {
  // A coalescing batcher and the two-image enqueue/collect ring are two
  // schedules for the same stage and cannot both own detection: run_pipelined
  // keeps a future outstanding on THIS replica's detector, which is exactly the
  // detector the batcher may lend to another thread. When cross-request batching
  // is on it wins (it is the one that raises occupancy); the ring stays for the
  // unbatched case, unchanged.
  if (det_batcher_ && det_batcher_->coalescing()) return false;
  return det_ && det_->supports_async();
}

std::vector<OcrPipelineResult> UnifiedOcrPipeline::run_pipelined(
    const std::vector<cv::Mat> &imgs, const RunFlags &flags,
    const backend_routing::RequestRouting &routing) {
  const int n = static_cast<int>(imgs.size());
  std::vector<OcrPipelineResult> outs(n);
  if (n == 0) return outs;
  const bool layout_active = flags.layout && layout_ != nullptr;

  // TWO-DEEP RING of uploaded pages: while the current image is recognized out
  // of one slot, the next image's detection prefetches into the OTHER slot. A
  // slot is only overwritten one full iteration after the image it held
  // finished recognition — which is what keeps the non-owning ImageView handed
  // to the detector valid for the whole enqueue->collect gap.
  //
  // The cursor alternates over PROCESSED (non-empty) images, NOT over the image
  // index. Keying the slot on `i & 1` looked equivalent and is not: empty pages
  // are skipped, so a single empty page mid-batch makes the prefetch target the
  // SAME slot the current image is about to be read from. Concretely, for
  // [full, EMPTY, full] the i=0 iteration prefetched image 2 into ring[2&1] ==
  // ring[0] and then recognized image 0's boxes against ring[0] — i.e. image 0
  // was recognized from image 2's pixels.
  Uploaded ring[2];

  int first = 0;
  while (first < n && imgs[first].empty()) ++first;  // empty pages: empty result
  if (first >= n) return outs;

  int slot = 0;
  ring[slot] = upload_image_(imgs[first]);
  backend::BoxesFuture det_fut = det_->enqueue(
      ring[slot].view, imgs[first].rows, imgs[first].cols, *queue_);

  for (int i = first; i < n; ++i) {
    if (imgs[i].empty()) continue;
    // The slot holding THIS image; the prefetch below claims the other one.
    const int cur_slot = slot;
    std::vector<Box> boxes = det_fut.collect();
    turbo_ocr::sorted_boxes(boxes);

    // Layout goes onto its own lane FIRST, before the prefetch below pushes the
    // next page's upload+detection onto `queue_`. The order matters: the
    // cross-lane barrier inside enqueue_layout_ records at the CURRENT point of
    // `queue_`, so submitting layout after the prefetch would make the layout
    // lane wait on image i+1's detection as well and overlap nothing.
    backend::LayoutFuture layout_fut;
    if (layout_active)
      layout_fut = enqueue_layout_(ring[cur_slot].view, imgs[i].rows,
                                   imgs[i].cols);

    // Kick the NEXT image's detection before doing any of THIS image's
    // remaining work. On a backend with supports_async() this returns as soon
    // as the work is submitted, so the device chews image i+1's detection while
    // the host drives image i's classification + recognition (on Apple that
    // includes the Neural Engine, a genuinely parallel unit). On a synchronous
    // backend enqueue() simply runs it here — same results, same order.
    int nxt = i + 1;
    while (nxt < n && imgs[nxt].empty()) ++nxt;
    if (nxt < n) {
      const int nxt_slot = 1 - cur_slot; // never the slot we are about to read
      ring[nxt_slot] = upload_image_(imgs[nxt]);
      det_fut = det_->enqueue(ring[nxt_slot].view, imgs[nxt].rows,
                              imgs[nxt].cols, *queue_);
      slot = nxt_slot;
    }

    const backend::ImageView &view = ring[cur_slot].view;
    // Blocking layout stays HERE, after the prefetch: on a backend with an async
    // detector but a synchronous layout (Apple) this is what lets image i's
    // layout run on the host while image i+1 detects on the device.
    if (layout_active && !layout_fut.valid())
      outs[i].layout = layout_->run(view, imgs[i].rows, imgs[i].cols,
                                    backend::kDefaultLayoutScoreThreshold,
                                    *queue_);
    classify_angles_(view, boxes);
    std::vector<std::pair<std::string, float>> rec_results =
        rec_->run(view, boxes, *queue_);
    // Collected after rec, so the two overlapped — same schedule as the
    // single-image path. `view` outlives this point (its ring slot is only
    // reused a full iteration later), which is what keeps the non-owning
    // ImageView handed to enqueue() valid across the gap.
    if (layout_fut.valid()) outs[i].layout = layout_fut.collect();
    combine_recognition(outs[i], boxes, rec_results, rec_->last_dropped_crops());
    dispatch_router_(outs[i], view, boxes, flags, routing,
                     /*defer_external=*/false);
    maybe_assign_reading_order(flags.reading_order, outs[i].results,
                               outs[i].layout, outs[i].reading_order);
  }
  return outs;
}

void UnifiedOcrPipeline::run_batch_layout_stage_(
    const std::vector<backend::ImageCrops> &crops,
    std::vector<OcrPipelineResult> &outs, const RunFlags &flags,
    const backend_routing::RequestRouting &routing) {
  const int n = static_cast<int>(crops.size());
  if (n == 0) return;
  // SOFTWARE-PIPELINED over the batch: page i+1's layout is submitted to the
  // layout lane the moment page i's has been collected, so it executes while
  // the host decodes page i's boxes and runs page i's table/formula dispatch.
  //
  // Never TWO outstanding: the seam is a single-slot contract (one set of device
  // scratch per stage), so the enqueue for i+1 comes strictly after the collect
  // for i. A backend that cannot overlap returns an invalid future from
  // enqueue_layout_ and takes the blocking run() below, unchanged.
  backend::LayoutFuture fut =
      enqueue_layout_(crops[0].img, crops[0].img.rows, crops[0].img.cols);
  for (int i = 0; i < n; ++i) {
    const backend::ImageView &view = crops[i].img;
    if (fut.valid()) {
      outs[i].layout = fut.collect();
    } else {
      outs[i].layout =
          layout_->run(view, view.rows, view.cols,
                       backend::kDefaultLayoutScoreThreshold, *queue_);
    }
    if (i + 1 < n)
      fut = enqueue_layout_(crops[i + 1].img, crops[i + 1].img.rows,
                            crops[i + 1].img.cols);
    dispatch_router_(outs[i], view, crops[i].boxes, flags, routing,
                     /*defer_external=*/false);
  }
  for (auto &out : outs)
    maybe_assign_reading_order(flags.reading_order, out.results, out.layout,
                               out.reading_order);
}


} // namespace turbo_ocr::pipeline
