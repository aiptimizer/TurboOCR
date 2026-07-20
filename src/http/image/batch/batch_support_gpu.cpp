#include "turbo_ocr/http/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/size_classify.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "../image_internal.h"
#include "batch_internal.h"

namespace turbo_ocr::routes::batchdetail {

void batch_decode_images(const std::vector<std::string> &raw_bytes,
                          bool nvjpeg_available,
                          const server::ImageDecoder &decode,
                          std::vector<cv::Mat> &imgs,
                          std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  std::vector<size_t> jpeg_indices;
  std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

  thread_local NvJpegDecoder tl_nvjpeg;
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
        const auto v = decode::classify_image_size(jw, jh);
        if (v == decode::ImageSizeVerdict::kDimTooLarge) {
          errors[i] = "dimensions_too_large";
          continue;
        }
        if (v == decode::ImageSizeVerdict::kPixelsTooLarge) {
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
                opts.want_tables, opts.want_formulas, opts.routing_override);
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
                    opts.routing_override, /*defer_external=*/false,
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

} // namespace turbo_ocr::routes::batchdetail
