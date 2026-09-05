// gRPC RecognizeBatch: per-slot isolation, GPU dispatcher + CPU fanout.
#include "turbo_ocr/grpc/grpc_service.h"

namespace turbo_ocr::server {

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

  if (auto err = grpc_check_layout_request(ctx, request->layout(),
          request->reading_order() || request->as_blocks() ||
          request->tables() || request->formulas(),
          layout_available_); err)
    return *err;
  bool want_layout = request->layout();
  bool want_reading_order = request->reading_order();
  const bool want_blocks = request->as_blocks();
  const bool want_tables = request->tables();
  const bool want_formulas = request->formulas();
  if (want_blocks) {
    want_reading_order = true;
    want_layout = true;
  }
  if (want_reading_order || want_tables || want_formulas) want_layout = true;
  if (auto err = grpc_check_structure_backends(ctx, want_tables, want_formulas,
          table_available_, formula_available_,
          mode_ == GrpcResponseMode::json_bytes,
          request->layout(), request->as_blocks()); err)
    return *err;

  // Per-slot oversize handling: an oversized image is dropped to an empty
  // slot (0 detections), NOT a whole-RPC abort — one decompression-bomb in
  // a batch must not deny service to every co-batched valid image. Mirrors
  // the per-slot contract of HTTP /ocr/batch and of this handler's own
  // decode-failure path. Pre-decode header sniff refuses bombs before any
  // decode cost.
  const int dim_cap = decode::max_image_dim();
  // Aggregate decoded-pixel budget: the per-image caps below bound a single
  // slot, but every non-JPEG slot is decoded up front and held alive at once,
  // so a batch of highly-compressible bomb PNGs can still OOM the host. Tag
  // sniffable slots once the running sum would exceed the budget so they are
  // never decoded. Mirrors batch_check_dims_pre on the HTTP /ocr/batch routes
  // (image_routes.cpp / cpu_main.cpp), keeping all three batch surfaces in
  // lockstep.
  int64_t cumulative_pixels = 0;
  const int64_t batch_pixel_budget = decode::max_batch_pixels();
  std::vector<bool> too_large(n, false);
  // Distinguish per-image oversize from aggregate-budget overflow: the
  // latter means the images are individually fine and the client should
  // split the batch — the HTTP routes emit distinct tags for exactly this
  // reason, and the gRPC slot error mirrors them.
  std::vector<bool> budget_exceeded(n, false);
  for (int i = 0; i < n; ++i) {
    auto *p = reinterpret_cast<const unsigned char *>(request->images(i).data());
    const size_t blen = request->images(i).size();
    // A JPEG on the GPU path decodes on the DEVICE (grpc_jpeg_decode_and_infer)
    // and never lands in the host `imgs` array, so it must not count against
    // the host-OOM aggregate budget — that budget bounds only up-front
    // host-decoded (PNG/BMP) slots. Its own per-image dim/area caps below
    // still apply, so an individually oversized JPEG is still rejected.
    bool device_jpeg = false;
#ifndef USE_CPU_ONLY
    device_jpeg = dispatcher_ && decode::NvJpegDecoder::is_jpeg(p, blen);
#endif
    if (auto d = decode::peek_image_dimensions(p, blen)) {
      if (d->width > dim_cap || d->height > dim_cap ||
          decode::exceeds_pixel_cap(d->width, d->height)) {
        too_large[i] = true;
      } else if (!device_jpeg) {
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

  // JPEGs decode inside the dispatcher lambda (see grpc_jpeg_decode_and_infer);
  // PNG/other decode here on CPU and ship the materialized cv::Mat.
  std::vector<cv::Mat> imgs(n);
  std::vector<bool> is_jpeg(n, false);
  for (int i = 0; i < n; ++i) {
    if (too_large[i]) continue;
    const auto &bytes = request->images(i);
    const auto *p = reinterpret_cast<const unsigned char *>(bytes.data());
#ifndef USE_CPU_ONLY
    if (dispatcher_ && decode::NvJpegDecoder::is_jpeg(p, bytes.size())) {
      is_jpeg[i] = true;
      continue; // decode happens inside the dispatcher lambda
    }
#endif
    imgs[i] = grpc_decode_image(bytes);
  }

  // Post-decode safety net for residual formats we don't sniff (BMP/PNM).
  for (int i = 0; i < n; ++i) {
    if (imgs[i].empty()) continue;
    if (imgs[i].cols > dim_cap || imgs[i].rows > dim_cap ||
        decode::exceeds_pixel_cap(imgs[i].cols, imgs[i].rows)) {
      too_large[i] = true;
      imgs[i].release();  // drop to empty slot
    }
  }

  // Check we have at least one valid candidate. JPEGs are still encoded
  // bytes at this point — we trust the pre-decode dim sniff and decode
  // failures will surface per-slot below. A mixed batch proceeds (oversized
  // slots emit empty results via mark_empty_slot). Only an all-invalid
  // batch aborts — and then the code reflects WHY: if any slot was
  // oversized, report DIMENSIONS_TOO_LARGE (matching single Recognize)
  // rather than the misleading IMAGE_DECODE_FAILED.
  bool any_valid = false, any_too_large = false;
  for (int i = 0; i < n; ++i) {
    if (is_jpeg[i] || !imgs[i].empty()) { any_valid = true; break; }
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
    if (!is_jpeg[i] && imgs[i].empty())
      mark_empty_slot(e, budget_exceeded[i] ? "BATCH_PIXELS_EXCEEDED"
                         : too_large[i]     ? "IMAGE_TOO_LARGE"
                                            : "IMAGE_DECODE_FAILED");
    entries.push_back(e);
  }

#ifndef USE_CPU_ONLY
  if (dispatcher_) {
    std::vector<std::future<pipeline::OcrPipelineResult>> futs(n);
    for (int i = 0; i < n; ++i) {
      try {
        if (is_jpeg[i]) {
          futs[i] = grpc_jpeg_decode_and_infer(
              *dispatcher_, request->images(i), want_layout,
              want_reading_order, want_tables, want_formulas, routing);
        } else if (!imgs[i].empty()) {
          cv::Mat img_owned = std::move(imgs[i]);
          futs[i] = dispatcher_->submit(
              [img_owned = std::move(img_owned), want_layout,
               want_reading_order, want_tables, want_formulas,
               routing](auto &e) {
                return e.pipeline->run_with_layout(
                    img_owned, e.stream, want_layout, want_reading_order,
                    routing, /*defer_external=*/false,
                    want_tables, want_formulas);
              });
        }
      } catch (const turbo_ocr::PoolExhaustedError &e) {
        return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "SERVER_BUSY", e.what());
      }
    }
    // Single overall deadline for the whole batch (C4): the slots run
    // concurrently on the dispatcher, so the per-request window applies to
    // the batch as a whole, not to each slot in turn (otherwise a wedged GPU
    // could block up to n * request_timeout_ms). Each .get() waits only the
    // time remaining until that one deadline. timeout<=0 means disabled, so
    // block (matches submit_for_default's disabled path).
    const bool batch_deadline_on = request_timeout_ms_ > 0;
    const auto batch_deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(batch_deadline_on ? request_timeout_ms_ : 0);
    for (int i = 0; i < n; ++i) {
      if (!futs[i].valid()) continue;
      try {
        // A wedged slot is abandoned and tagged empty rather than hanging the
        // whole batch RPC. The submit lambdas above own their inputs by
        // value, so an abandoned future is safe. TimeoutError derives from
        // std::exception, so the per-slot catch below marks it empty.
        pipeline::OcrPipelineResult out;
        if (batch_deadline_on) {
          long remaining_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  batch_deadline - std::chrono::steady_clock::now())
                  .count();
          if (remaining_ms <= 0) {
            // Batch deadline already elapsed (an earlier slot consumed the
            // window): abandon this slot empty. Must NOT call
            // get_with_timeout(fut, 0) — 0 means "disabled" there and would
            // block on future.get() forever, hanging the whole RPC.
            mark_empty_slot(entries[i], "INFERENCE_TIMEOUT");
            continue;
          }
          out = pipeline::get_with_timeout(futs[i], remaining_ms);
        } else {
          out = futs[i].get();
        }
        fill_response(entries[i], out, want_blocks);
      } catch (const turbo_ocr::ImageTooLargeError &) {
        // An oversized JPEG that slipped past the pre-decode sniff surfaces
        // here (the device decoder rejects it). Tag it like the single
        // Recognize path does, not the generic INFERENCE_ERROR.
        mark_empty_slot(entries[i], "IMAGE_TOO_LARGE");
      } catch (const turbo_ocr::TimeoutError &) {
        // get_with_timeout expired mid-slot: same condition as the
        // deadline-already-elapsed arm above, so it must carry the same
        // tag — two timed-out slots in one batch reporting different codes
        // would misattribute the timeout to an inference fault.
        mark_empty_slot(entries[i], "INFERENCE_TIMEOUT");
      } catch (const turbo_ocr::GpuDecodeError &e) {
        // The device decoder faulted on this slot: tagged with its own
        // (retryable) code, never re-decoded on the CPU.
        TOCR_LOG_ERROR_RL("gRPC batch GPU decode failed", "index", i, "error", e.what());
        mark_empty_slot(entries[i], "GPU_DECODE_FAILED");
      } catch (const std::exception &e) {
        TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error", e.what());
        mark_empty_slot(entries[i], "INFERENCE_ERROR");
      } catch (...) {
        mark_empty_slot(entries[i], "INFERENCE_ERROR");
      }
    }
    return grpc::Status::OK;
  }
#endif

  // CPU-only fanout: bounded jthread pool, each thread calls run_infer
  // (which is synchronous through the InferFunc on this build).
  // grpc_batch_workers_ bounds ONE RPC; without a process-wide ceiling, N
  // concurrent RPCs would spawn N*grpc_batch_workers_ threads (resource-
  // exhaustion vector, unlike the HTTP WorkPool which is globally bounded).
  // Every RPC keeps one guaranteed worker (progress under contention);
  // EXTRA workers beyond that first one draw a permit from this shared
  // pool of exactly GRPC_BATCH_GLOBAL_WORKERS permits.
  static std::counting_semaphore<1024> extra_worker_permits{
      static_cast<std::ptrdiff_t>(
          env::env_int("GRPC_BATCH_GLOBAL_WORKERS", 16, 1, 1024))};
  const int num_workers = std::min(n, grpc_batch_workers_);
  std::atomic<int> next_idx{0};
  {
    const auto worker_fn = [&]() {
      while (true) {
        const int i = next_idx.fetch_add(1);
        if (i >= n) break;
        if (imgs[i].empty()) continue;
        try {
          auto out = run_infer(imgs[i], want_layout, want_reading_order,
                               want_tables, want_formulas, routing);
          fill_response(entries[i], out, want_blocks);
        } catch (const std::exception &e) {
          TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error", e.what());
          mark_empty_slot(entries[i], "INFERENCE_ERROR");
        } catch (...) {
          mark_empty_slot(entries[i], "INFERENCE_ERROR");
        }
      }
    };
    // RAII permit: acquired before the jthread is constructed, released on
    // scope exit even when jthread construction throws — an OS thread-
    // creation failure must not leak the permit and erode the pool.
    struct Permit {
      bool held = extra_worker_permits.try_acquire();
      ~Permit() {
        if (held) extra_worker_permits.release();
      }
    };
    // permits BEFORE workers: destruction runs in reverse declaration
    // order, so the jthreads join first and the permits release only after
    // every worker has exited. Declared the other way round, the permits
    // would return to the global pool while these workers still run,
    // letting a concurrent RPC oversubscribe the worker bound.
    std::vector<Permit> permits(static_cast<size_t>(
        std::max(0, num_workers - 1)));
    std::vector<std::jthread> workers;
    workers.reserve(static_cast<size_t>(num_workers));
    workers.emplace_back(worker_fn);  // guaranteed worker, no permit
    for (int w = 1; w < num_workers; ++w) {
      if (!permits[static_cast<size_t>(w - 1)].held) break;
      workers.emplace_back(worker_fn);
    }
  }
  return grpc::Status::OK;
}

} // namespace turbo_ocr::server
