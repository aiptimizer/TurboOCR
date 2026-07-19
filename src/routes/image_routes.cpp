#include "turbo_ocr/routes/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/backend_error.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/output/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/server/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

namespace turbo_ocr::routes {

namespace {

// 504 response for a per-request inference deadline (C4). The dispatcher
// abandons the still-running task and throws turbo_ocr::TimeoutError; map it
// to the single shared kInferenceTimeout code so HTTP and gRPC stay in lockstep.
[[nodiscard]] drogon::HttpResponsePtr timeout_response() {
  using server::ErrorCode;
  return server::error_response(
      static_cast<drogon::HttpStatusCode>(
          server::error_http_status(ErrorCode::kInferenceTimeout)),
      server::error_code_str(ErrorCode::kInferenceTimeout).data(),
      "Inference exceeded the configured request deadline");
}

// L3 strict-query-params (opt-in). Default OFF preserves the historical
// lenient behavior (unknown params silently ignored). When
// TURBO_OCR_STRICT_QUERY_PARAMS=1, an unrecognized query parameter is a 400
// INVALID_PARAMETER instead. Read once; cached for the process lifetime.
[[nodiscard]] bool strict_query_params_enabled() noexcept {
  static const bool v = server::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  return v;
}

// When strict mode is on, reject the request if any query parameter is not in
// `allowed`. Returns true (and invokes `callback`) on rejection. No-op (returns
// false) when strict mode is off, so default behavior is byte-identical.
// NOTE: Drogon's getParameters() merges x-www-form-urlencoded body fields with
// the query string; these routes take JSON / binary / multipart-file bodies
// (no plain form fields), so the map here is the query string only.
[[nodiscard]] bool reject_unknown_query_params(
    const drogon::HttpRequestPtr &req,
    std::initializer_list<std::string_view> allowed,
    std::function<void(const drogon::HttpResponsePtr &)> &callback) {
  if (!strict_query_params_enabled()) return false;
  for (const auto &kv : req->getParameters()) {
    const std::string &key = kv.first;
    bool known = false;
    for (auto a : allowed)
      if (a == key) { known = true; break; }
    if (!known) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          std::format("Unknown query parameter '{}' "
                      "(TURBO_OCR_STRICT_QUERY_PARAMS=1)", key)));
      return true;
    }
  }
  return false;
}

// --- /ocr/raw: GPU-direct JPEG decode, Wuffs PNG ---
void register_ocr_raw_route_gpu(server::WorkPool &pool,
                                 pipeline::PipelineDispatcher &dispatcher,
                                 const server::ImageDecoder &decode,
                                 bool nvjpeg_available,
                                 bool layout_available,
                                 bool table_available,
                                 bool formula_available) {
  // Per-request routing override (Tier-A) validation set: computed ONCE from
  // the same routing config the pipeline loaded, so route-layer validation and
  // the pipeline's recognizer registry never drift (both via
  // routing::routable_backend_names). Captured by value into the handler.
  const auto rtbl = routing::load_routing_config();
  const std::set<std::string> valid_table   = routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = routing::routable_backend_names(rtbl, "formula");
  // Fail-loud availability comes from the warmed pipeline (single source of
  // truth), threaded down from main(); not re-derived from config here.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available,
       valid_table, valid_formula, table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    if (req->body().empty()) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return;
    }

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    // Tier-A: ?route_table=NAME&route_formula=NAME — validate against the
    // registered backend names; unknown => 400. Empty => route default.
    if (auto r = server::validate_routing_override(
            req->getParameter("route_table"), req->getParameter("route_formula"),
            valid_table, valid_formula, &opts.routing_override);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "tables", "formulas", "text",
                  "route_table", "route_formula"}, callback))
      return;
    if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/raw", [&] {
        const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
        size_t len = req->body().size();

        // Configurable cap (MAX_IMAGE_DIM, default 16384, bounds [64, 65535]).
        // Same env var as /ocr/pixels — single knob across all image routes.
        const int kMaxImageDim = decode::max_image_dim();
        auto reject_too_large = [&](int w, int h) {
          cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
              std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                          w, h, kMaxImageDim, kMaxImageDim)));
        };
        auto reject_too_many_pixels = [&](int w, int h) {
          cb(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
              std::format("Image area {}x{} exceeds maximum of {} pixels",
                          w, h, decode::max_image_pixels())));
        };

        // Pre-decode header sniff (PNG IHDR / JPEG SOFn). Refuses
        // decompression bombs (per-side AND total pixel area) without ever
        // calling the decoder.
        if (auto d = turbo_ocr::decode::peek_image_dimensions(data, len)) {
          if (d->width > kMaxImageDim || d->height > kMaxImageDim) {
            reject_too_large(d->width, d->height);
            return;
          }
          if (decode::exceeds_pixel_cap(d->width, d->height)) {
            reject_too_many_pixels(d->width, d->height);
            return;
          }
        }

        // JPEG with nvJPEG: submit GPU-direct decode + infer as one work item.
        // Layout-only requests (?text=0) skip this fast path — the generic
        // branch below dispatches them to run_layout_only.
        if (opts.want_text && nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
          // C4: capture `req` by value so the JPEG bytes (data/len point into
          // req->body()) stay alive if the future is abandoned on timeout and
          // the task keeps running after this handler returns.
          pipeline::OcrPipelineResult out;
          try {
          out = dispatcher.submit_for_default(
              [req, data, len, opts, kMaxImageDim](auto &e) {
            auto &nvjpeg = e.get_nvjpeg();
            auto [w, h] = nvjpeg.get_dimensions(data, len);
            // Decompression-bomb guard for JPEGs the pre-decode header sniff
            // couldn't parse (progressive / unusual SOF markers): nvJPEG's
            // own dims are authoritative — reject before allocating GPU
            // memory for them.
            if (w > kMaxImageDim || h > kMaxImageDim)
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image dimensions {}x{} exceed maximum of {}x{}",
                  w, h, kMaxImageDim, kMaxImageDim));
            if (decode::exceeds_pixel_cap(w, h))
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image area {}x{} exceeds maximum of {} pixels",
                  w, h, decode::max_image_pixels()));
            if (w > 0 && h > 0) {
              auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
              if (nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream)) {
                turbo_ocr::GpuImage gpu_img{.data = d_buf, .step = pitch, .rows = h, .cols = w};
                try {
                  // defer_external=true: remote formula/table crops are
                  // submitted non-blocking here (GPU worker), awaited later in
                  // finalize_deferred() off the worker. Local backends ignore it.
                  return e.pipeline->run_with_layout(gpu_img, e.stream,
                                                     opts.want_layout,
                                                     opts.want_reading_order,
                                                     opts.routing_override,
                                                     /*defer_external=*/true,
                                                     opts.want_tables,
                                                     opts.want_formulas);
                } catch (const std::exception &ex) {
                  // Don't silently re-run the whole request on CPU (~2x work
                  // invisibly): rate-limited log so a persistent GPU-path fault
                  // is visible. Falls through to CPU decode + standard path below.
                  TOCR_LOG_ERROR_RL("nvJPEG GPU fast-path failed; CPU fallback",
                                    "error", ex.what());
                }
              }
            }
            cv::Mat img = nvjpeg.decode(data, len);
            if (img.empty()) {
              if (len <= static_cast<size_t>(INT_MAX))
                img = cv::imdecode(
                    cv::Mat(1, static_cast<int>(len), CV_8UC1,
                            const_cast<unsigned char *>(data)),
                    cv::IMREAD_COLOR);
            }
            if (img.empty())
              throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
            // CPU-fallback bomb guard: nvjpeg.get_dimensions returns {0,0}
            // for headers it can't parse (so the dims check above was a
            // no-op), and the sniff is 64KB-bounded — re-check the decoded
            // size, same post-decode net as the non-JPEG branch.
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim)
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image dimensions {}x{} exceed maximum of {}x{}",
                  img.cols, img.rows, kMaxImageDim, kMaxImageDim));
            if (decode::exceeds_pixel_cap(img.cols, img.rows))
              throw turbo_ocr::ImageTooLargeError(std::format(
                  "Image area {}x{} exceeds maximum of {} pixels",
                  img.cols, img.rows, decode::max_image_pixels()));
            return e.pipeline->run_with_layout(img, e.stream,
                                               opts.want_layout,
                                               opts.want_reading_order,
                                               opts.routing_override,
                                               /*defer_external=*/true,
                                               opts.want_tables,
                                               opts.want_formulas);
          });
          } catch (const turbo_ocr::TimeoutError &) {
            cb(timeout_response());
            return;
          }
          // Await + assemble any deferred VLM crops OFF the GPU worker (here on
          // the work-pool thread). No-op when nothing was deferred.
          pipeline::finalize_deferred(out);
          cb(server::json_response(emit_pipeline_result_json(out, opts.want_blocks)));
          return;
        }

        // Non-JPEG (PNG, etc.) or nvJPEG not available
        cv::Mat img = decode(data, len);
        if (img.empty()) {
          cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
          return;
        }
        // Post-decode safety net for formats we didn't sniff (BMP/TIFF/WEBP).
        if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
          reject_too_large(img.cols, img.rows);
          return;
        }
        if (decode::exceeds_pixel_cap(img.cols, img.rows)) {
          reject_too_many_pixels(img.cols, img.rows);
          return;
        }

        // C4: capture `img` BY VALUE (cv::Mat copy is a cheap refcount bump)
        // so an abandoned-on-timeout task never dereferences the freed local.
        pipeline::OcrPipelineResult out;
        try {
          out = dispatcher.submit_for_default([img, opts](auto &e) {
            if (!opts.want_text)
              return e.pipeline->run_layout_only(img, e.stream);
            return e.pipeline->run_with_layout(img, e.stream,
                                                opts.want_layout,
                                                opts.want_reading_order,
                                                opts.routing_override,
                                                /*defer_external=*/true,
                                                opts.want_tables,
                                                opts.want_formulas);
          });
        } catch (const turbo_ocr::TimeoutError &) {
          cb(timeout_response());
          return;
        }
        pipeline::finalize_deferred(out);
        cb(server::json_response(emit_pipeline_result_json(out, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}

// --- /ocr/batch helpers ----------------------------------------------------
//
// The batch flow goes: base64 decode → pre-decode dim sniff → image decode
// (nvJPEG batch path for JPEGs when available, fallback per-image decoder
// otherwise) → post-decode dim guard → pipeline submit. Each stage tags
// per-slot errors so the response keeps a 1:1 mapping with the input array.

// One slot of the batch response: the full per-page pipeline result
// (text + layout + tables/formulas + reading order when requested).
struct BatchItem {
  pipeline::OcrPipelineResult out;
};

// Stage 1: base64 decode every input slot. Empty inputs short-circuit to
// "empty" so we don't pay decode cost on slots the caller never filled in.
void batch_decode_base64(const std::vector<std::string> &b64_strings,
                          std::vector<std::string> &raw_bytes,
                          std::vector<std::string> &errors) {
  size_t n = b64_strings.size();
  for (size_t i = 0; i < n; ++i) {
    const auto &b64 = b64_strings[i];
    if (b64.empty()) {
      errors[i] = "empty";
      continue;
    }
    raw_bytes[i] = base64_decode(b64);
    if (raw_bytes[i].empty()) errors[i] = "base64_decode_failed";
  }
}

// Stage 2: header-sniff (PNG/JPEG) every still-valid slot and reject
// oversized inputs before paying decode cost. Same MAX_IMAGE_DIM env as
// /ocr/raw and /ocr/pixels — but errors are per-slot, not whole-request 400s.
void batch_check_dims_pre(const std::vector<std::string> &raw_bytes,
                           int max_image_dim,
                           std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  // Aggregate decoded-pixel budget: the per-image cap below bounds one slot,
  // but the route holds every decoded image alive at once, so a batch of
  // highly-compressible bomb images can still OOM the host. Tag sniffable
  // slots once the running sum would exceed the budget so they are never
  // decoded. Covers every compressible bomb vector (PNG/JPEG/WebP/TIFF/GIF);
  // residual uncompressed formats (BMP/PNM) are body-cap-bounded and fall to
  // the per-image post-decode cap.
  int64_t cumulative_pixels = 0;
  const int64_t batch_pixel_budget = decode::max_batch_pixels();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    if (auto d = turbo_ocr::decode::peek_image_dimensions(
            reinterpret_cast<const unsigned char *>(raw.data()), raw.size())) {
      if (d->width > max_image_dim || d->height > max_image_dim) {
        errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                                 d->width, d->height,
                                 max_image_dim, max_image_dim);
      } else if (decode::exceeds_pixel_cap(d->width, d->height)) {
        errors[i] = std::format("pixels_too_large ({}x{} > {} px)",
                                 d->width, d->height, decode::max_image_pixels());
      } else {
        const int64_t pix = static_cast<int64_t>(d->width) * d->height;
        if (cumulative_pixels + pix > batch_pixel_budget) {
          errors[i] = std::format("batch_pixels_exceeded (batch sum > {} px)",
                                   batch_pixel_budget);
        } else {
          cumulative_pixels += pix;
        }
      }
    }
  }
}

// Stage 3: decode pixels. JPEGs go through nvJPEG batch decode (when
// available and we have ≥2 of them); everything else runs through the
// per-image fallback decoder.
void batch_decode_images(const std::vector<std::string> &raw_bytes,
                          bool nvjpeg_available,
                          const server::ImageDecoder &decode,
                          std::vector<cv::Mat> &imgs,
                          std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  std::vector<size_t> jpeg_indices;
  std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

  thread_local NvJpegDecoder tl_nvjpeg;
  const int kMaxImageDim = decode::max_image_dim();
  if (nvjpeg_available) {
    for (size_t i = 0; i < n; ++i) {
      if (!errors[i].empty()) continue;
      const auto &raw = raw_bytes[i];
      if (raw.size() >= 2 &&
          static_cast<unsigned char>(raw[0]) == 0xFF &&
          static_cast<unsigned char>(raw[1]) == 0xD8) {
        // Bomb guard: nvJPEG's own header dims are authoritative — reject
        // oversized JPEGs per-slot BEFORE batch_decode allocates a host Mat
        // from them (the pre-decode sniff is only 64KB-bounded).
        const auto *p = reinterpret_cast<const unsigned char *>(raw.data());
        auto [jw, jh] = tl_nvjpeg.get_dimensions(p, raw.size());
        if (jw > kMaxImageDim || jh > kMaxImageDim) {
          errors[i] = "dimensions_too_large";
          continue;
        }
        if (decode::exceeds_pixel_cap(jw, jh)) {
          errors[i] = "pixels_too_large";
          continue;
        }
        jpeg_indices.push_back(i);
        jpeg_buffers.emplace_back(p, raw.size());
      }
    }
  }

  if (jpeg_buffers.size() >= 2) {
    auto batch_mats = tl_nvjpeg.batch_decode(jpeg_buffers);
    for (size_t j = 0; j < jpeg_indices.size(); ++j)
      imgs[jpeg_indices[j]] = std::move(batch_mats[j]);
  }

  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    if (!imgs[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    imgs[i] = decode(
        reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    if (imgs[i].empty()) errors[i] = "decode_failed";
  }
}

// Stage 4: post-decode safety net for residual formats we don't header-sniff
// (BMP/PNM). Releases the image so it doesn't get fed to the pipeline.
void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors) {
  size_t n = imgs.size();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    if (imgs[i].cols > max_image_dim || imgs[i].rows > max_image_dim) {
      errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                               imgs[i].cols, imgs[i].rows,
                               max_image_dim, max_image_dim);
      imgs[i].release();
    } else if (decode::exceeds_pixel_cap(imgs[i].cols, imgs[i].rows)) {
      errors[i] = std::format("pixels_too_large ({}x{} > {} px)",
                               imgs[i].cols, imgs[i].rows,
                               decode::max_image_pixels());
      imgs[i].release();
    }
  }
}

// Stage 5: run the pipeline on every slot that survived decode. Chunks
// into kMaxBatch-sized batches through run_batch_with_layout — both with
// and without layout the batched path applies (batched det/rec), so
// ?layout=1 no longer falls back to serial single-image inference.
//
// C4: the submitted lambda is SELF-CONTAINED. It owns the decoded images
// (moved in) and writes results into the packaged_task's own return value,
// never into request-scoped state. So if the future is abandoned on timeout
// (TimeoutError), the still-running task can finish safely against memory it
// owns; the caller scatters the returned results back only after the future
// resolves in time. Results are chunk-local (indexed 0..valid count); the
// caller maps them to absolute slots via valid_indices.
void batch_run_pipeline(pipeline::PipelineDispatcher &dispatcher,
                         std::vector<cv::Mat> valid_imgs,
                         const std::vector<size_t> &valid_indices,
                         bool want_layout,
                         const server::InferOptions &opts,
                         std::vector<BatchItem> &all_items,
                         std::vector<std::string> &errors) {
  if (valid_imgs.empty()) return;

  // Self-contained per-valid-image output. PoolExhaustedError propagates out
  // of the lambda (rethrown by future.get() -> 503 at the route); every other
  // exception is tagged per-slot here, never escaping to taint other slots.
  struct BatchOutput {
    std::vector<pipeline::OcrPipelineResult> outs;
    std::vector<std::string> slot_errors;
  };

  BatchOutput result;
  try {
    result = dispatcher.submit_for_default(
        [imgs = std::move(valid_imgs), want_layout, opts](auto &e) mutable {
      constexpr size_t kMaxBatch = 8;
      const size_t total = imgs.size();
      BatchOutput o;
      o.outs.resize(total);
      o.slot_errors.assign(total, std::string{});
      for (size_t offset = 0; offset < total; offset += kMaxBatch) {
        size_t end = std::min(offset + kMaxBatch, total);
        // Per-chunk try wraps chunk construction too: a failure here taints
        // ONLY this chunk's slots, never the completed results of earlier
        // chunks (whose slot_errors are empty == success, the same sentinel a
        // blanket outer-catch would wrongly overwrite).
        try {
          std::vector<cv::Mat> chunk(
              std::make_move_iterator(imgs.begin() + offset),
              std::make_move_iterator(imgs.begin() + end));
          try {
            auto chunk_results = e.pipeline->run_batch_with_layout(
                chunk, e.stream, want_layout, opts.want_reading_order,
                opts.want_tables, opts.want_formulas);
            for (size_t j = 0; j < chunk_results.size(); ++j)
              o.outs[offset + j] = std::move(chunk_results[j]);
          } catch (const std::exception &) {
            // One degenerate image (1×1 / corrupt-decoded Mat) poisons the
            // whole batched chunk. Retry its images individually through
            // run_with_layout, whose internal guard degrades exactly the
            // bad slot to an empty result and resets the stream — per-slot
            // isolation, same as the single-image routes.
            for (size_t j = 0; j < chunk.size(); ++j) {
              try {
                o.outs[offset + j] = e.pipeline->run_with_layout(
                    chunk[j], e.stream, want_layout, opts.want_reading_order,
                    /*routing=*/{}, /*defer_external=*/false,
                    opts.want_tables, opts.want_formulas);
              } catch (const turbo_ocr::PoolExhaustedError &) {
                throw;  // -> 503 at the route, never a per-slot tag
              } catch (const std::exception &ex) {
                // Don't leak raw CUDA_CHECK text (absolute source paths) to
                // clients; log the detail, tag the slot with a stable code.
                TOCR_LOG_ERROR_RL("batch slot inference error",
                                  "slot", offset + j, "error", std::string_view(ex.what()));
                o.slot_errors[offset + j] = "inference_failed";
              }
            }
          }
        } catch (const turbo_ocr::PoolExhaustedError &) {
          throw;  // queue full mid-batch -> 503, don't swallow as per-slot
        } catch (const std::exception &ex) {
          // chunk construction (e.g. bad_alloc) failed: tag only THIS
          // chunk's still-empty slots, leaving completed chunks intact.
          TOCR_LOG_ERROR_RL("batch chunk error", "offset", offset,
                            "error", std::string_view(ex.what()));
          for (size_t j = offset; j < end; ++j)
            if (o.slot_errors[j].empty())
              o.slot_errors[j] = "inference_failed";
        }
      }
      return o;
    });
  } catch (const turbo_ocr::PoolExhaustedError &) {
    // GPU queue full at submit — no chunk ran. Propagate so the route's
    // run_with_error_handling turns it into 503 SERVER_BUSY, matching every
    // other inference route instead of a 200 with a fully-errored array.
    throw;
  } catch (const turbo_ocr::TimeoutError &) {
    // Per-request deadline tripped. Propagate so the route maps it to a single
    // 504 (the abandoned task keeps running against its own moved-in images).
    throw;
  } catch (const std::exception &ex) {
    // Submission-level failure before any chunk ran — tag every still-empty
    // slot so the caller knows their request didn't silently succeed (stable
    // code, detail to the log, no raw internal text to the client).
    TOCR_LOG_ERROR_RL("batch submission error", "error", std::string_view(ex.what()));
    for (size_t k : valid_indices)
      if (errors[k].empty()) errors[k] = "inference_failed";
    return;
  }

  // Scatter chunk-local results back to absolute slots. Only reached when the
  // future resolved in time, so the caller-scoped vectors are touched solely
  // here on the request thread.
  for (size_t j = 0; j < valid_indices.size(); ++j) {
    const size_t idx = valid_indices[j];
    if (!result.slot_errors[j].empty()) {
      if (errors[idx].empty()) errors[idx] = std::move(result.slot_errors[j]);
    } else {
      all_items[idx].out = std::move(result.outs[j]);
    }
  }
}

// Stage 6: serialize {batch_results, errors} JSON. Mirrors the CPU contract
// — null in the errors array for successful slots, an error string otherwise.
std::string batch_emit_json(std::vector<BatchItem> &all_items,
                             const std::vector<std::string> &errors,
                             bool want_layout,
                             bool want_blocks) {
  size_t n = all_items.size();
  std::string json_str;
  json_str.reserve(n * 1024);
  json_str += "{\"batch_results\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    if (want_layout) {
      // Full per-page emitter: text + layout (+ tables/formulas when the
      // CUA router fired, byte-identical to the legacy shape otherwise).
      json_str += emit_pipeline_result_json(all_items[i].out, want_blocks);
    } else {
      json_str += results_to_json(all_items[i].out.results);
    }
  }
  json_str += "],\"errors\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    const auto &e = errors[i];
    if (e.empty()) {
      json_str += "null";
    } else {
      json_str += '"';
      for (char c : e) {
        if (c == '"' || c == '\\') json_str += '\\';
        json_str += c;
      }
      json_str += '"';
    }
  }
  json_str += "]}";
  return json_str;
}

void register_ocr_batch_route_gpu(server::WorkPool &pool,
                                   pipeline::PipelineDispatcher &dispatcher,
                                   const server::ImageDecoder &decode,
                                   bool nvjpeg_available,
                                   bool layout_available,
                                   bool table_available,
                                   bool formula_available,
                                   int max_batch_images) {
  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/batch",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available,
       table_avail, formula_avail, max_batch_images](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "tables", "formulas"}, callback))
      return;
    // Layout-only (?text=0) is a single-image feature; the batched det/rec
    // path has no layout-only equivalent. Reject instead of silently running
    // full OCR against the caller's stated intent.
    if (!opts.want_text) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          "text=0 is not supported on /ocr/batch; send single images to "
          "/ocr/raw?text=0 instead"));
      return;
    }
    if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    auto json = req->getJsonObject();
    if (!json) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
      return;
    }
    if (!json->isMember("images") || !(*json)["images"].isArray()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
      return;
    }

    auto &images_json = (*json)["images"];
    size_t n = images_json.size();
    if (n == 0) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
      return;
    }
    // Cap before the O(n) per-slot allocations + n*1KB JSON reserve below —
    // an unbounded images[] (tens of millions of tiny strings in a <100MB
    // body) is otherwise a memory-amplification OOM lever.
    if (n > static_cast<size_t>(max_batch_images)) {
      callback(server::error_response(drogon::k400BadRequest, "BATCH_TOO_LARGE",
          std::format("images array has {} entries, max is {}", n, max_batch_images)));
      return;
    }

    // asString() throws Json::LogicError on a non-string element ({}/[]);
    // guard the type so a malformed slot becomes an empty string the per-slot
    // decode path tags, instead of an unhandled exception / 500.
    auto b64_strings = std::make_shared<std::vector<std::string>>(n);
    for (size_t i = 0; i < n; ++i) {
      const auto &el = images_json[static_cast<int>(i)];
      if (el.isString())
        (*b64_strings)[i] = el.asString();
    }

    server::submit_work(pool, std::move(callback),
        [b64_strings, n, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      const bool want_layout = opts.want_layout;
      server::run_with_error_handling(cb, "/ocr/batch", [&] {
        // Per-slot error tags for the response. Empty string == success.
        // Mirrors the CPU /ocr/batch contract: callers see which slots
        // failed (empty / decode_failed / dimensions_too_large) instead
        // of getting a silently-zero result list.
        std::vector<std::string> errors(n);
        std::vector<std::string> raw_bytes(n);

        batch_decode_base64(*b64_strings, raw_bytes, errors);

        const int kMaxImageDim = decode::max_image_dim();
        batch_check_dims_pre(raw_bytes, kMaxImageDim, errors);

        std::vector<cv::Mat> imgs(n);
        batch_decode_images(raw_bytes, nvjpeg_available, decode, imgs, errors);
        batch_check_dims_post(imgs, kMaxImageDim, errors);

        std::vector<cv::Mat> valid_imgs;
        std::vector<size_t> valid_indices;
        valid_imgs.reserve(n);
        valid_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
          if (errors[i].empty() && !imgs[i].empty()) {
            valid_imgs.push_back(std::move(imgs[i]));
            valid_indices.push_back(i);
          }
        }

        std::vector<BatchItem> all_items(n);
        // C4: hand ownership of the decoded images to the submitted task so an
        // abandoned-on-timeout run never touches this frame's vectors.
        try {
          batch_run_pipeline(dispatcher, std::move(valid_imgs),
                              std::move(valid_indices),
                              want_layout, opts, all_items, errors);
        } catch (const turbo_ocr::TimeoutError &) {
          cb(timeout_response());
          return;
        }

        cb(server::json_response(batch_emit_json(all_items, errors, want_layout, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}

// --- /ocr/pixels: raw BGR pixel data, zero decode overhead ---
void register_ocr_pixels_route_gpu(server::WorkPool &pool,
                                    pipeline::PipelineDispatcher &dispatcher,
                                    bool layout_available,
                                    bool table_available,
                                    bool formula_available) {
  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&pool, &dispatcher, layout_available, table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "tables", "formulas", "text",
                  "width", "height", "channels"}, callback))
      return;
    if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

    // Dimensions: query params (preferred) with the legacy X-* headers as a
    // v2.3-compat fallback; conflict -> 400. Shared with the CPU handler.
    const auto dims = server::resolve_pixel_dims(req);
    if (!dims.ok()) {
      callback(server::error_response(drogon::k400BadRequest,
          dims.error_code.c_str(), dims.error));
      return;
    }
    const int width = dims.width, height = dims.height, channels = dims.channels;
    const bool used_legacy_dim_header = dims.used_legacy_header;

    // Configurable via MAX_IMAGE_DIM (default 16384). Read once on first
    // request and cached for the process lifetime.
    const int kMaxPixelDim = decode::max_image_dim();
    if (width > kMaxPixelDim || height > kMaxPixelDim) {
      callback(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
          std::format("Dimensions {}x{} exceed maximum of {}x{}", width, height, kMaxPixelDim, kMaxPixelDim)));
      return;
    }
    if (decode::exceeds_pixel_cap(width, height)) {
      callback(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
          std::format("Image area {}x{} exceeds maximum of {} pixels",
                      width, height, decode::max_image_pixels())));
      return;
    }

    size_t expected = static_cast<size_t>(width) * height * channels;
    if (req->body().size() != expected) {
      callback(server::error_response(drogon::k400BadRequest, "BODY_SIZE_MISMATCH",
          std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                      expected, width, height, channels, req->body().size())));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, width, height, channels, opts,
         used_legacy_dim_header](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/pixels", [&] {
        cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                    const_cast<char *>(req->body().data()));
        // The pipeline is BGR-only: a 1-channel Mat reaches the GPU upload
        // with step == cols, trips the degenerate-input guard, and comes
        // back empty. Expand grayscale to BGR up front.
        if (channels == 1)
          cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        // C4: a 3-channel `img` is a non-owning view into req->body(), so the
        // task must hold both `img` (by value) and `req` (keeps the pixel
        // buffer alive) to survive an abandoned-on-timeout run.
        pipeline::OcrPipelineResult out;
        try {
          out = dispatcher.submit_for_default([img, req, opts](auto &e) {
            if (!opts.want_text)
              return e.pipeline->run_layout_only(img, e.stream);
            return e.pipeline->run_with_layout(img, e.stream,
                                                opts.want_layout,
                                                opts.want_reading_order,
                                                opts.routing_override,
                                                /*defer_external=*/false,
                                                opts.want_tables,
                                                opts.want_formulas);
          });
        } catch (const turbo_ocr::TimeoutError &) {
          cb(timeout_response());
          return;
        }
        auto resp = server::json_response(
            emit_pipeline_result_json(out, opts.want_blocks));
        if (used_legacy_dim_header)
          server::stamp_pixel_dim_deprecation(resp);
        cb(resp);
      });
    });
  }, {drogon::Post});
}

// --- /ocr/markdown: faithful Markdown export ---
void register_ocr_markdown_route_gpu(server::WorkPool &pool,
                                     pipeline::PipelineDispatcher &dispatcher,
                                     const server::ImageDecoder &decode,
                                     bool layout_available,
                                     bool table_available,
                                     bool formula_available) {
  drogon::app().registerHandler(
      "/ocr/markdown",
      [&pool, &dispatcher, &decode, layout_available, table_available,
       formula_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "EMPTY_BODY", "Empty body"));
          return;
        }
        if (!layout_available) {
          callback(server::error_response(drogon::k400BadRequest,
              "LAYOUT_DISABLED",
              "/ocr/markdown requires the layout model (do not start with "
              "DISABLE_LAYOUT=1)"));
          return;
        }
        // Always self-contained base64 data: URIs over HTTP. The legacy ?embed=0 file-ref
        // mode has render_markdown_with_assets write crop PNGs to the server CWD — which the
        // client can never retrieve and which is a swallowed no-op under a read-only root
        // filesystem (the markdown would then reference nonexistent files). Not served here.
        const bool embed = true;
        if (auto p = req->getParameter("embed"); p == "0" || p == "false")
          TOCR_LOG_WARN("/ocr/markdown embed=0 (file-ref) is not supported over "
                        "HTTP; returning self-contained data-URIs instead");

        // Faithful export gates structure on what the server actually loaded:
        // request table/formula recognition only when their backends exist, so
        // a text-only server produces honest text markdown rather than silently
        // dropping table/formula sections it claimed (via the hardcoded flags)
        // to have attempted. Matches /capabilities and the fail-loud routes.
        const bool md_want_tables = table_available;
        const bool md_want_formulas = formula_available;
        server::submit_work(pool, std::move(callback),
            [req, &dispatcher, &decode, embed, md_want_tables,
             md_want_formulas](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/markdown", [&] {
            const auto *data =
                reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();
            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest,
                  "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            const int kMaxImageDim = decode::max_image_dim();
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
              cb(server::error_response(drogon::k400BadRequest,
                  "DIMENSIONS_TOO_LARGE", "Image dimensions exceed maximum"));
              return;
            }
            if (decode::exceeds_pixel_cap(img.cols, img.rows)) {
              cb(server::error_response(drogon::k400BadRequest,
                  "PIXELS_TOO_LARGE", "Image area exceeds maximum pixel count"));
              return;
            }

            pipeline::OcrPipelineResult out;
            try {
              out = dispatcher.submit_for_default(
                  [img, md_want_tables, md_want_formulas](auto &e) {
                return e.pipeline->run_with_layout(
                    img, e.stream, /*want_layout=*/true,
                    /*want_reading_order=*/true, /*routing=*/{},
                    /*defer_external=*/true,
                    md_want_tables, md_want_formulas);
              });
            } catch (const turbo_ocr::TimeoutError &) {
              cb(timeout_response());
              return;
            }
            pipeline::finalize_deferred(out);

            turbo_ocr::assign_layout_ids(out.results, out.layout);
            std::string md = turbo_ocr::output::render_markdown_with_assets(
                out, img, /*base_dir=*/".", /*embed_images=*/embed);

            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k200OK);
            // no-silent-failure: the markdown body intentionally drops failed/garbage regions
            // (so a degraded stage is invisible in it). Surface degradation in a header so a
            // caller can detect a configured-but-failed stage rather than seeing a clean page.
            if (out.text_degraded || out.table_degraded || out.formula_degraded) {
              std::string warn;
              auto add = [&](bool d, const char *name, const std::string &w) {
                if (!d) return;
                if (!warn.empty()) warn += "; ";
                warn += name;
                if (!w.empty()) warn += ":" + w;
              };
              add(out.text_degraded, "text", out.text_warning);
              add(out.table_degraded, "table", out.table_warning);
              add(out.formula_degraded, "formula", out.formula_warning);
              resp->addHeader("X-OCR-Degraded", warn);
            }
            resp->setBody(std::move(md));
            resp->setContentTypeString("text/markdown; charset=utf-8");
            cb(resp);
          });
        });
      }, {drogon::Post});
}

// --- POST /infer (Tier-B): run ONE crop through a chosen backend ----------
// Body JSON: { "image": "<base64>", "modality": "table"|"formula",
//              "backend": "<registry-name>"  |  { inline BackendSpec } }
// Inline kind:openai (operator-supplied base_url => SSRF surface) is REJECTED
// unless TURBO_ALLOW_ADHOC_BACKENDS=1. Additive: touches no existing route.
void register_infer_route_gpu(server::WorkPool &pool,
                              pipeline::PipelineDispatcher &dispatcher,
                              const server::ImageDecoder &decode) {
  const auto rtbl = routing::load_routing_config();
  const std::set<std::string> valid_table   = routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = routing::routable_backend_names(rtbl, "formula");
  drogon::app().registerHandler(
      "/infer",
      [&pool, &dispatcher, &decode, valid_table, valid_formula](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON body"));
          return;
        }
        if (!json->isMember("image") || !(*json)["image"].isString() ||
            (*json)["image"].asString().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Missing 'image' (base64)"));
          return;
        }
        if (json->isMember("modality") && !(*json)["modality"].isString()) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'modality' must be a string ('table' or 'formula')"));
          return;
        }
        const std::string modality =
            json->isMember("modality") ? (*json)["modality"].asString() : "";
        if (modality != "table" && modality != "formula") {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'modality' must be 'table' or 'formula'"));
          return;
        }
        if (!json->isMember("backend")) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "Missing 'backend' (a registered backend name or an inline spec object)"));
          return;
        }

        // Resolve backend: named (registry) or inline spec.
        std::string backend_name;
        std::optional<routing::BackendSpec> inline_spec;
        const auto &be = (*json)["backend"];
        const auto &valid = (modality == "table") ? valid_table : valid_formula;
        if (be.isString()) {
          backend_name = be.asString();
          if (valid.find(backend_name) == valid.end()) {
            callback(server::error_response(drogon::k400BadRequest, "ROUTING_UNKNOWN_OVERRIDE",
                std::format("backend '{}' is not a configured {} backend (see /capabilities)",
                            backend_name, modality)));
            return;
          }
        } else if (be.isObject()) {
          // SSRF gate: an inline openai endpoint lets the caller name an
          // arbitrary base_url. Off by default; the operator opts in per-deploy.
          if (be.isMember("kind") && be["kind"].isString() &&
              be["kind"].asString() == "openai" &&
              !server::env_enabled("TURBO_ALLOW_ADHOC_BACKENDS")) {
            callback(server::error_response(drogon::k403Forbidden, "ADHOC_BACKENDS_DISABLED",
                "inline kind:openai backends are disabled; set TURBO_ALLOW_ADHOC_BACKENDS=1 "
                "to allow operator-supplied endpoint URLs (SSRF surface)"));
            return;
          }
          Json::StreamWriterBuilder w;
          w["indentation"] = "";
          const std::string spec_text = Json::writeString(w, be);
          try {
            inline_spec = routing::parse_inline_backend(modality, spec_text);
          } catch (const routing::RoutingConfigError &e) {
            // .what() begins with the ROUTING_* code; surface it verbatim.
            std::string msg = e.what();
            std::string code = "ROUTING_BAD_KIND";
            if (auto c = msg.find(':'); c != std::string::npos) code = msg.substr(0, c);
            callback(server::error_response(drogon::k400BadRequest, code.c_str(), msg));
            return;
          }
          // Reject inline kind:local: building a local engine on the request
          // thread spins up ORT-CUDA sessions + hundreds of MB of cudaMalloc,
          // unguarded — a resource-exhaustion vector. Local backends must be
          // named (already loaded at startup); only kind:openai (a cheap HTTP
          // client, gated by TURBO_ALLOW_ADHOC_BACKENDS above) may be inline.
          if (inline_spec && inline_spec->kind == routing::Kind::Local) {
            callback(server::error_response(drogon::k400BadRequest, "ADHOC_LOCAL_DISABLED",
                "inline kind:local backends are not allowed; name an already-loaded "
                "backend (see /capabilities) instead"));
            return;
          }
        } else {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'backend' must be a string (name) or an object (inline spec)"));
          return;
        }

        auto b64 = std::make_shared<std::string>((*json)["image"].asString());
        server::submit_work(pool, std::move(callback),
            [b64, &dispatcher, &decode, modality, backend_name, inline_spec](
                server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/infer", [&] {
            std::string bytes = turbo_ocr::base64_decode(*b64);
            if (bytes.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
              return;
            }
            // Decompression-bomb guard, mirroring reject_if_too_large_pre/post
            // on the other image routes: a header sniff before decode, then a
            // post-decode check for formats the sniff can't parse. Without it a
            // 60000x60000 PNG decodes to a ~10 GB host Mat and OOMs the worker.
            const int kMaxImageDim = decode::max_image_dim();
            const auto *raw = reinterpret_cast<const unsigned char *>(bytes.data());
            if (auto d = turbo_ocr::decode::peek_image_dimensions(raw, bytes.size())) {
              if (d->width > kMaxImageDim || d->height > kMaxImageDim) {
                cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
                    std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                                d->width, d->height, kMaxImageDim, kMaxImageDim)));
                return;
              }
              if (decode::exceeds_pixel_cap(d->width, d->height)) {
                cb(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
                    std::format("Image area {}x{} exceeds maximum of {} pixels",
                                d->width, d->height, decode::max_image_pixels())));
                return;
              }
            }
            cv::Mat img = decode(raw, bytes.size());
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
              cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
                  std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                              img.cols, img.rows, kMaxImageDim, kMaxImageDim)));
              return;
            }
            if (decode::exceeds_pixel_cap(img.cols, img.rows)) {
              cb(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
                  std::format("Image area {}x{} exceeds maximum of {} pixels",
                              img.cols, img.rows, decode::max_image_pixels())));
              return;
            }
            std::string result;
            try {
              // Capture inline_spec by value into the GPU task; pass its address
              // (the worker builds the transient recognizer in its own context).
              result = dispatcher.submit_for_default(
                  [img, modality, backend_name, inline_spec](auto &e) {
                    return e.pipeline->infer_one(
                        img, e.stream, modality, backend_name,
                        inline_spec ? &*inline_spec : nullptr);
                  });
            } catch (const turbo_ocr::TimeoutError &) {
              cb(timeout_response());
              return;
            } catch (const turbo_ocr::BackendUnavailableError &e) {
              cb(server::error_response(drogon::k503ServiceUnavailable,
                                        "BACKEND_UNAVAILABLE", e.what()));
              return;
            }
            const char *key = (modality == "table") ? "html" : "latex";
            std::string body = "{\"modality\":\"" + modality + "\",\"" + key + "\":\"";
            turbo_ocr::detail::append_escaped_string(body, result);
            body += "\"}";
            cb(server::json_response(body));
          });
        });
      },
      {drogon::Post});
}

} // namespace

void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available,
                           bool table_available,
                           bool formula_available,
                           int max_batch_images) {
  register_ocr_raw_route_gpu(pool, dispatcher, decode, nvjpeg_available,
                             layout_available, table_available, formula_available);
  register_ocr_batch_route_gpu(pool, dispatcher, decode, nvjpeg_available,
                               layout_available, table_available, formula_available,
                               max_batch_images);
  register_ocr_pixels_route_gpu(pool, dispatcher, layout_available,
                                table_available, formula_available);
  register_ocr_markdown_route_gpu(pool, dispatcher, decode, layout_available,
                                  table_available, formula_available);
  register_infer_route_gpu(pool, dispatcher, decode);
}

} // namespace turbo_ocr::routes
