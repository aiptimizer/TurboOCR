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
#include "turbo_ocr/decode/cpu_image_decode.h"
#include "turbo_ocr/decode/jpeg_codec.h"
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
  for (size_t i = 0; i < raw_bytes.size(); ++i) {
    if (!errors[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    const auto *p = reinterpret_cast<const unsigned char *>(raw.data());
    if (nvjpeg_available && decode::looks_like_jpeg(p, raw.size()))
      continue;  // decoded on the replica in batch_run_pipeline
    imgs[i] = decode(p, raw.size());
    if (imgs[i].empty()) errors[i] = "decode_failed";
  }
}

// Stage 4: post-decode safety net for residual formats we don't header-sniff

void batch_run_pipeline(pipeline::PipelineDispatcher &dispatcher,
                         std::vector<std::string> raw_bytes,
                         std::vector<cv::Mat> imgs,
                         bool nvjpeg_available,
                         bool want_layout,
                         const server::InferOptions &opts,
                         std::vector<BatchItem> &all_items,
                         std::vector<std::string> &errors) {
  const size_t n = raw_bytes.size();
  // Slots this task has to produce: everything not already tagged.
  std::vector<size_t> slots;
  for (size_t i = 0; i < n; ++i)
    if (errors[i].empty()) slots.push_back(i);
  if (slots.empty()) return;

  // Self-contained per-slot output (absolute slot index). PoolExhaustedError
  // propagates out of the lambda (rethrown by future.get() -> 503 at the
  // route); every other exception is tagged per slot here, never escaping to
  // taint other slots.
  struct BatchOutput {
    std::vector<pipeline::OcrPipelineResult> outs;
    std::vector<std::string> slot_errors;
  };

  BatchOutput result;
  try {
    result = dispatcher.submit_for_default(
        [raw = std::move(raw_bytes), imgs = std::move(imgs), slots,
         nvjpeg_available, want_layout, opts](auto &e) mutable {
      BatchOutput o;
      o.outs.resize(raw.size());
      o.slot_errors.assign(raw.size(), std::string{});

      // JPEG slots: decoded here, on the replica, with its own decoder. One
      // batched nvJPEG call for the set; each image carries its own status.
      std::vector<size_t> jpeg_slots;
      std::vector<std::pair<const unsigned char *, size_t>> jpeg_bufs;
      for (size_t i : slots) {
        if (!imgs[i].empty()) continue;
        const auto *p = reinterpret_cast<const unsigned char *>(raw[i].data());
        if (nvjpeg_available && decode::looks_like_jpeg(p, raw[i].size())) {
          jpeg_slots.push_back(i);
          jpeg_bufs.emplace_back(p, raw[i].size());
        } else {
          o.slot_errors[i] = "decode_failed";  // nothing decoded it upstream
        }
      }
      if (!jpeg_bufs.empty()) {
        auto &nvjpeg = e.get_nvjpeg();
        std::vector<decode::NvJpegDecoder::HostDecode> decoded;
        if (nvjpeg.available()) {
          decoded = jpeg_bufs.size() == 1
                        ? std::vector<decode::NvJpegDecoder::HostDecode>{
                              nvjpeg.decode(jpeg_bufs[0].first, jpeg_bufs[0].second, e.stream)}
                        : nvjpeg.batch_decode(jpeg_bufs, e.stream);
        } else {
          decoded.assign(jpeg_bufs.size(), {});  // status Failed: no device decoder
          for (auto &d : decoded) d.status = decode::JpegDecodeStatus::Unsupported;
        }
        for (size_t j = 0; j < jpeg_slots.size(); ++j) {
          const size_t i = jpeg_slots[j];
          cv::Mat img;
          switch (decoded[j].status) {
            case decode::JpegDecodeStatus::Ok:
              img = std::move(decoded[j].image);
              break;
            case decode::JpegDecodeStatus::Unsupported: {
              // Outside the hardware decoder's format support: the replica's
              // hybrid GPU backend next, the host codec only if that refuses too.
              auto &hy = e.get_nvjpeg_hybrid();
              if (hy.available()) {
                auto hd = hy.decode(jpeg_bufs[j].first, jpeg_bufs[j].second, e.stream);
                if (hd.status == decode::JpegDecodeStatus::Failed) {
                  TOCR_LOG_ERROR_RL("batch slot GPU decode failed (hybrid)", "slot", i,
                                    "nvjpeg_status", hd.nvjpeg_status);
                  o.slot_errors[i] = "gpu_decode_failed";
                  continue;
                }
                if (hd.status == decode::JpegDecodeStatus::Ok) { img = std::move(hd.image); break; }
              }
              img = decode::decode_cpu_fallback(jpeg_bufs[j].first, jpeg_bufs[j].second);
              break;
            }
            case decode::JpegDecodeStatus::Failed:
              TOCR_LOG_ERROR_RL("batch slot GPU decode failed", "slot", i,
                                "nvjpeg_status", decoded[j].nvjpeg_status);
              o.slot_errors[i] = "gpu_decode_failed";
              continue;
          }
          if (img.empty()) { o.slot_errors[i] = "decode_failed"; continue; }
          switch (decode::classify_image_size(img.cols, img.rows)) {
            case decode::ImageSizeVerdict::kDimTooLarge: o.slot_errors[i] = "dimensions_too_large"; continue;
            case decode::ImageSizeVerdict::kPixelsTooLarge: o.slot_errors[i] = "pixels_too_large"; continue;
            case decode::ImageSizeVerdict::kOk: break;
          }
          imgs[i] = std::move(img);
        }
      }

      // Everything decodable now sits in imgs[]; run it in chunks.
      std::vector<size_t> valid;
      for (size_t i : slots)
        if (o.slot_errors[i].empty() && !imgs[i].empty()) valid.push_back(i);

      constexpr size_t kMaxBatch = 8;
      for (size_t offset = 0; offset < valid.size(); offset += kMaxBatch) {
        const size_t end = std::min(offset + kMaxBatch, valid.size());
        // Per-chunk try wraps chunk construction too: a failure here taints
        // ONLY this chunk's slots, never the completed results of earlier
        // chunks (whose slot_errors are empty == success, the same sentinel a
        // blanket outer-catch would wrongly overwrite).
        try {
          std::vector<cv::Mat> chunk;
          chunk.reserve(end - offset);
          for (size_t k = offset; k < end; ++k) chunk.push_back(std::move(imgs[valid[k]]));
          try {
            auto chunk_results = e.pipeline->run_batch_with_layout(
                chunk, e.stream, want_layout, opts.want_reading_order,
                opts.want_tables, opts.want_formulas, opts.routing_override);
            for (size_t j = 0; j < chunk_results.size(); ++j)
              o.outs[valid[offset + j]] = std::move(chunk_results[j]);
          } catch (const std::exception &) {
            // One degenerate image (1×1 / corrupt-decoded Mat) poisons the
            // whole batched chunk. Retry its images individually through
            // run_with_layout, whose internal guard degrades exactly the
            // bad slot to an empty result and resets the stream — per-slot
            // isolation, same as the single-image routes.
            for (size_t j = 0; j < chunk.size(); ++j) {
              const size_t i = valid[offset + j];
              try {
                o.outs[i] = e.pipeline->run_with_layout(
                    chunk[j], e.stream, want_layout, opts.want_reading_order,
                    opts.routing_override, /*defer_external=*/false,
                    opts.want_tables, opts.want_formulas);
              } catch (const turbo_ocr::PoolExhaustedError &) {
                throw;  // -> 503 at the route, never a per-slot tag
              } catch (const std::exception &ex) {
                // Don't leak raw CUDA_CHECK text (absolute source paths) to
                // clients; log the detail, tag the slot with a stable code.
                TOCR_LOG_ERROR_RL("batch slot inference error",
                                  "slot", i, "error", std::string_view(ex.what()));
                o.slot_errors[i] = "inference_failed";
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
          for (size_t k = offset; k < end; ++k)
            if (o.slot_errors[valid[k]].empty())
              o.slot_errors[valid[k]] = "inference_failed";
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
    for (size_t i : slots)
      if (errors[i].empty()) errors[i] = "inference_failed";
    return;
  }

  // Scatter results back to absolute slots. Only reached when the future
  // resolved in time, so the caller-scoped vectors are touched solely here on
  // the request thread.
  for (size_t i : slots) {
    if (!result.slot_errors[i].empty()) {
      if (errors[i].empty()) errors[i] = std::move(result.slot_errors[i]);
    } else {
      all_items[i].out = std::move(result.outs[i]);
    }
  }
}

} // namespace turbo_ocr::routes::batchdetail
