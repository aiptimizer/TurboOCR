// gRPC RecognizeBatch: per-slot isolation, GPU dispatcher + CPU fanout.
#include "turbo_ocr/service/capability/proto_capability_bridge.h"
#include "turbo_ocr/image/size_classify.h"
#include "turbo_ocr/service/grpc/grpc_service.h"

namespace turbo_ocr::server {
namespace {

// One process-wide permit pool for EXTRA batch workers. Deliberately a
// function-static behind an accessor rather than a static local inside the
// (template) fanout helper below: a static local in a function TEMPLATE is
// one-per-instantiation, so a second call site passing a different lambda
// type would silently mint a second pool and double the global worker
// ceiling. The accessor makes the "exactly one pool" guarantee independent of
// how many bodies the fanout is instantiated with.
[[nodiscard]] std::counting_semaphore<1024> &extra_worker_permits() {
  static std::counting_semaphore<1024> sem{static_cast<std::ptrdiff_t>(
      env::env_int("GRPC_BATCH_GLOBAL_WORKERS", 16, 1, 1024))};
  return sem;
}

// Bounded shared-counter fanout over slots [0, n) with `num_workers` threads.
// (Shared counter, NOT work stealing: every worker pulls the next index off one
// atomic. Work stealing means per-worker deques with cross-worker theft, which
// has materially different contention behaviour.)
// `slot_fn(i)` is the per-slot body; it runs concurrently for distinct i and
// must not throw (the RPC bodies below swallow per-slot failures themselves,
// because a slot fault is a tagged empty slot, never a batch abort).
//
// Extracted from the RPC body so the worker-budget policy lives in ONE named
// place: the invariant it protects — N concurrent RPCs must not spawn
// N*grpc_batch_workers threads, since unlike the HTTP WorkPool this pool is
// per-RPC — is easy to break when it is buried inline in a 300-line handler.
// Returns only after every worker has joined, so slot_fn may capture
// request-scoped state by reference.
template <typename SlotFn>
void run_bounded_fanout(int n, int num_workers, SlotFn &&slot_fn) {
  std::atomic<int> next_idx{0};
  const auto worker_fn = [&]() {
    while (true) {
      const int i = next_idx.fetch_add(1);
      if (i >= n) break;
      slot_fn(i);
    }
  };
  // RAII permit: acquired before the jthread is constructed, released on
  // scope exit even when jthread construction throws — an OS thread-
  // creation failure must not leak the permit and erode the pool.
  struct Permit {
    bool held = extra_worker_permits().try_acquire();
    ~Permit() {
      if (held) extra_worker_permits().release();
    }
    // Declaring only a destructor leaves copy construction/assignment
    // implicitly defaulted, and a copy would duplicate held==true so BOTH
    // objects release — over-crediting the process-wide semaphore and eroding
    // exactly the ceiling this type exists to hold. The vector below only ever
    // value-initializes in place, so deleting all four is free.
    Permit() = default;
    Permit(const Permit &) = delete;
    Permit &operator=(const Permit &) = delete;
    Permit(Permit &&) = delete;
    Permit &operator=(Permit &&) = delete;
  };
  // permits BEFORE workers: destruction runs in reverse declaration order, so
  // the jthreads join first and the permits release only after every worker
  // has exited. Declared the other way round, the permits would return to the
  // global pool while these workers still run, letting a concurrent RPC
  // oversubscribe the worker bound.
  std::vector<Permit> permits(
      static_cast<size_t>(std::max(0, num_workers - 1)));
  std::vector<std::jthread> workers;
  workers.reserve(static_cast<size_t>(num_workers));
  workers.emplace_back(worker_fn);  // guaranteed worker, no permit
  for (int w = 1; w < num_workers; ++w) {
    if (!permits[static_cast<size_t>(w - 1)].held) break;
    workers.emplace_back(worker_fn);
  }
}

// Pre-decode admission for one batch: per-image size caps plus the aggregate
// decoded-pixel budget, both evaluated from the header sniff so a bomb is
// refused before any decode cost. Verdicts only — the caller owns the wire
// tags, because the gRPC per-slot codes (IMAGE_TOO_LARGE /
// BATCH_PIXELS_EXCEEDED) are SCREAMING_CASE while the HTTP batch route's
// per-slot strings are snake_case; sharing the policy must not merge the two
// wire contracts.
//
// `device_decode_path` is true when the pipeline decodes for us
// (EncodedInferFunc) and slots therefore never land in the host `imgs` array:
// such slots must not count against the host-OOM aggregate budget, which bounds
// only up-front host-decoded slots. Their own per-image dim/area caps still
// apply, so an individually oversized image is still rejected.
//
// It is a whole-path flag, not a per-image JPEG sniff. The sniff it replaced
// existed because only JPEG had a device decoder (nvJPEG) — the backend seam
// now decodes whatever the vendor supports and falls back to the host decoder
// inside the pipeline, so format is no longer what decides where decode happens.
void admit_batch_slots(const ocr::OCRBatchRequest &request,
                       bool device_decode_path,
                       std::vector<bool> &too_large,
                       std::vector<bool> &budget_exceeded) {
  const int n = request.images_size();
  int64_t cumulative_pixels = 0;
  const int64_t batch_pixel_budget = decode::max_batch_pixels();
  for (int i = 0; i < n; ++i) {
    auto *p = reinterpret_cast<const unsigned char *>(request.images(i).data());
    const size_t blen = request.images(i).size();
    const bool device_decoded = device_decode_path;
    if (auto d = decode::peek_image_dimensions(p, blen)) {
      // Shared size verdict (decode/size_classify.h) — the same predicate and
      // ordering the HTTP batch route and single Recognize use. gRPC collapses
      // kDimTooLarge and kPixelsTooLarge into the one IMAGE_TOO_LARGE slot tag
      // it has always emitted, so only the classification is shared, not the
      // message.
      if (decode::classify_image_size(d->width, d->height) !=
          decode::ImageSizeVerdict::kOk) {
        too_large[i] = true;
      } else if (!device_decoded) {
        const int64_t pix = static_cast<int64_t>(d->width) * d->height;
        if (cumulative_pixels + pix > batch_pixel_budget) {
          too_large[i] = true;
          budget_exceeded[i] = true;
        } else {
          cumulative_pixels += pix;
        }
      }
    }
  }
}

} // namespace

grpc::Status OCRServiceImpl::RecognizeBatch(grpc::ServerContext *ctx,
                            const ocr::OCRBatchRequest *request,
                            ocr::OCRBatchResponse *response) {
  int n = request->images_size();
  if (n == 0)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "EMPTY_BATCH", "Empty images array");
  // Cap before the O(n) per-slot vectors + n response sub-messages below —
  // an unbounded repeated images field is a memory-amplification OOM lever.
  if (n > max_batch_images_)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "BATCH_TOO_LARGE",
        std::format("images has {} entries, max is {}", n, max_batch_images_));

  backend_routing::RequestRouting routing;
  if (auto err = grpc_validate_routing(ctx, request->route_table(),
                                       request->route_formula(), &routing);
      err)
    return *err;
  if (request->det_batch_num() != 0)
    // Deprecated, never honored (batching is managed internally). Warn so a
    // client tuning it learns it is inert instead of chasing a phantom knob.
    TOCR_LOG_WARN_RL("OCRBatchRequest.det_batch_num is deprecated and "
                     "ignored (batching is managed internally)",
                     "det_batch_num", request->det_batch_num());

  // The shared gate (see recognize_rpc.cpp). RecognizeBatch has no layout_only
  // field — batching is a full-OCR path — so it is never a text=0 request.
  InferOptions opts;
  if (auto r = parse_proto_options(*request, /*layout_only=*/false, loaded_,
                                   &opts);
      !r.error.empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      r.error_code.c_str(), r.error);
  const bool want_reading_order = opts.want_reading_order;
  const bool want_blocks = opts.want_blocks;
  const bool want_layout = opts.want_layout;
  const bool want_tables = opts.want_tables;
  const bool want_formulas = opts.want_formulas;
  if (auto err = grpc_check_structure_backends(
          ctx, opts.requested, loaded_, mode_ == GrpcResponseMode::json_bytes,
          want_blocks, /*raw_layout=*/request->layout()); err)
    return *err;

  // Per-slot oversize handling: an oversized image is dropped to an empty
  // slot (0 detections), NOT a whole-RPC abort — one decompression-bomb in
  // a batch must not deny service to every co-batched valid image. Mirrors
  // the per-slot contract of HTTP /ocr/batch and of this handler's own
  // decode-failure path. Pre-decode header sniff refuses bombs before any
  // decode cost, and an aggregate decoded-pixel budget bounds the batch as a
  // whole (admit_batch_slots above). Mirrors batch_check_dims_pre on the HTTP
  // /ocr/batch routes, keeping all three batch surfaces in lockstep.
  std::vector<bool> too_large(n, false);
  // Distinguish per-image oversize from aggregate-budget overflow: the
  // latter means the images are individually fine and the client should
  // split the batch — the HTTP routes emit distinct tags for exactly this
  // reason, and the gRPC slot error mirrors them.
  std::vector<bool> budget_exceeded(n, false);
  // When the pipeline can decode for us, NO slot is host-decoded up front, so
  // none of them consume the host-OOM aggregate budget (their per-image dim/area
  // caps still apply, so an individually oversized image is still refused).
  const bool device_decode_path = encoded_infer_fn_ != nullptr;
  admit_batch_slots(*request, device_decode_path, too_large, budget_exceeded);

  // On the encoded path the bytes go to the pipeline untouched; otherwise decode
  // here and ship the materialized cv::Mat.
  std::vector<cv::Mat> imgs(n);
  for (int i = 0; i < n; ++i) {
    if (too_large[i] || device_decode_path) continue;
    imgs[i] = grpc_decode_image(request->images(i));
  }

  // Post-decode safety net for residual formats we don't sniff (BMP/PNM).
  // Same shared verdict as the pre-decode sniff (decode/size_classify.h), so
  // the two stages cannot drift apart on cap ordering; the slot tag stays the
  // gRPC-specific IMAGE_TOO_LARGE assigned where entries are built.
  for (int i = 0; i < n; ++i) {
    if (imgs[i].empty()) continue;
    if (decode::classify_image_size(imgs[i].cols, imgs[i].rows) !=
        decode::ImageSizeVerdict::kOk) {
      too_large[i] = true;
      imgs[i].release();  // drop to empty slot
    }
  }

  // Does slot i still have work? ONE predicate for both paths: on the encoded
  // path every admitted slot is still encoded bytes (imgs is empty by
  // construction), on the host path a slot is live only once it decoded. The
  // two call sites below MUST agree — when they were written as separate
  // is_jpeg/imgs expressions, a slot could be counted valid by one and marked
  // empty by the other.
  const auto slot_has_work = [&](int i) {
    return !too_large[i] && (device_decode_path || !imgs[i].empty());
  };

  // Check we have at least one valid candidate. On the encoded path the bytes
  // are undecoded here — we trust the pre-decode dim sniff, and decode failures
  // surface per-slot below. A mixed batch proceeds (oversized slots emit empty
  // results via mark_empty_slot). Only an all-invalid batch aborts — and then
  // the code reflects WHY: if any slot was oversized, report
  // DIMENSIONS_TOO_LARGE (matching single Recognize) rather than the
  // misleading IMAGE_DECODE_FAILED.
  bool any_valid = false, any_too_large = false;
  for (int i = 0; i < n; ++i) {
    if (slot_has_work(i)) { any_valid = true; break; }
    if (too_large[i]) any_too_large = true;
  }
  if (!any_valid)
    return any_too_large
        ? grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                     "DIMENSIONS_TOO_LARGE",
                     "All images exceed the maximum dimension")
        : grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                     "IMAGE_DECODE_FAILED", "No valid images");

  // RepeatedPtrField is not thread-safe for concurrent add_*, so pre-allocate.
  response->set_total_images(n);
  std::vector<ocr::OCRResponse *> entries;
  entries.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    auto *e = response->add_batch_results();
    if (!slot_has_work(i))
      mark_empty_slot(e, budget_exceeded[i] ? "BATCH_PIXELS_EXCEEDED"
                         : too_large[i]     ? "IMAGE_TOO_LARGE"
                                            : "IMAGE_DECODE_FAILED");
    entries.push_back(e);
  }

  // ONE fanout for every backend: a bounded jthread pool (run_bounded_fanout
  // above owns the worker-budget policy) whose slot body calls the pipeline
  // synchronously. Worker count is per-RPC; the global ceiling lives in the
  // fanout helper.
  //
  // This replaced a second, dispatcher-specific implementation of the same loop
  // that submitted futures to a `dispatcher_` member deleted with the CUDA
  // pipeline — it had stopped compiling on the only configure that builds it.
  //
  // ONE PROPERTY WAS LOST WITH IT, deliberately and not silently: that loop
  // enforced a whole-BATCH deadline (the old per-service timeout across all
  // slots, so a wedged device could not block n * timeout) and tagged expired
  // slots INFERENCE_TIMEOUT. The synchronous path has no future to time out
  // on, so the shared REQUEST_TIMEOUT_MS bounds only the pool ACQUIRE — a
  // wedged stage hangs the slot. That is the same gap tracked for the lease
  // pool generally (work_pool.h "the pool's deadline also covers QUEUEING
  // only"); re-adding a per-slot execution deadline belongs there, once, rather
  // than here for one transport.
  const int num_workers = std::min(n, grpc_batch_workers_);
  run_bounded_fanout(n, num_workers, [&](int i) {
    if (!slot_has_work(i)) return;
    try {
      auto out = device_decode_path
                     ? run_infer_encoded(
                           reinterpret_cast<const std::uint8_t *>(
                               request->images(i).data()),
                           request->images(i).size(), want_layout,
                           want_reading_order, want_tables, want_formulas,
                           routing)
                     : run_infer(imgs[i], want_layout, want_reading_order,
                                 want_tables, want_formulas, routing);
      fill_response(entries[i], out, want_blocks);
    } catch (const turbo_ocr::ImageTooLargeError &) {
      // An oversized image that slipped past the pre-decode sniff surfaces here
      // (the decoder rejects it). Tag it like the single Recognize path does,
      // not the generic INFERENCE_ERROR.
      mark_empty_slot(entries[i], "IMAGE_TOO_LARGE");
    } catch (const turbo_ocr::ImageDecodeError &) {
      // Only reachable on the encoded path: the host path decoded up front and
      // a failure there already marked the slot IMAGE_DECODE_FAILED. Same tag
      // either way, so a client cannot tell which path served it.
      mark_empty_slot(entries[i], "IMAGE_DECODE_FAILED");
    } catch (const std::exception &e) {
      TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error", e.what());
      mark_empty_slot(entries[i], "INFERENCE_ERROR");
    } catch (...) {
      TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error",
                        "unknown exception");
      mark_empty_slot(entries[i], "INFERENCE_ERROR");
    }
  });
  return grpc::Status::OK;
}

} // namespace turbo_ocr::server
