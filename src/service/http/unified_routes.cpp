// unified_routes.cpp — see unified_routes.h.
//
// PARITY NOTE (/ocr/batch): this is a line-for-line port of the now-DELETED
// src/http/image/batch/batch_route_cpu.cpp — baseline
// `git show HEAD:src/http/image/batch/batch_route_cpu.cpp` — with the pool type swapped from
// pipeline::CpuPipelinePool to pipeline::UnifiedPipelinePool. Identical:
//   * the EndpointSpec / validate_request gate (routing=kBuildRoutingSupport,
//     layout_only_allowed=false)
//   * every 400 error code and message: INVALID_JSON ("Invalid JSON" /
//     "Missing images array"), EMPTY_BATCH, BATCH_TOO_LARGE (same std::format)
//   * the non-string-slot tolerance (empty string -> per-slot "empty" tag)
//   * the shared per-slot stages batch_decode_base64 / batch_check_dims_pre /
//     host decode / batch_check_dims_post, and their per-slot error strings
//   * the bounded jthread fan-out: one guaranteed worker + BATCH_FANOUT_GLOBAL_
//     WORKERS-capped extras, workers capped at pool_size
//   * "pool_exhausted" tagging when a worker cannot lease a pipeline
//   * per-slot exception tagging with e.what() / "unknown"
//   * the {batch_results, errors} body from batch_emit_json
// The ONLY behavioural differences, both deliberate:
//   1. inference goes through UnifiedOcrPipeline::run_batch_with_layout in
//      chunks of 8 (the batched det/rec path the GPU batch route already used)
//      instead of one image per call, with a per-image run_with_layout retry on
//      chunk failure so slot isolation is preserved exactly;
//   2. opts.routing_override is forwarded into the pipeline (the GPU batch route
//      does this; CpuOcrPipeline simply had no parameter for it). The route
//      validates ?route_table=/?route_formula= against the SAME name sets /ocr
//      uses (server::routing_name_sets()) — passing empty sets here would not be
//      a no-op, it would 400 every legal override with "names no configured
//      table backend". On a USE_CPU_ONLY build kBuildRoutingSupport is
//      kUnsupported, so the sets are never consulted and any override is
//      rejected with the honest CPU-build reason instead.

#include "turbo_ocr/service/http/unified_routes.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/image/image_config.h"     // decode::max_image_dim()
#include "turbo_ocr/service/validation/request_gate.h"

// Read-only include of the main tree's shared batch stages (batch_common.cpp).
// Reusing them is the whole point: the per-slot caps and the JSON emitter are
// device-neutral policy and must have exactly ONE implementation.
#include "image/batch/batch_internal.h"

namespace turbo_ocr::routes {

namespace {

// Same chunk size as the GPU batch route (batch_support_gpu.cpp): big enough to
// amortize the batched det/rec launch, small enough that one poisoned image only
// costs a retry of 8.
constexpr std::size_t kMaxBatch = 8;

// THE ADMISSION HALF of /ocr/batch: parse the JSON envelope, run the shared
// validation gate, cap the batch, and collect the base64 slots.
//
// Separated from the work half because they fail in different places and at
// different times. Everything here is a synchronous HTTP 400 on the event loop
// before any work is queued; everything after runs on a WorkPool thread, where
// a bad slot is a per-item error inside a 200 rather than a status code. Reading
// the two as one 240-line lambda hid which side of that line a given check was
// on.
//
// Returns nullopt having ALREADY answered `callback`.
struct BatchRequest {
  server::InferOptions opts;
  std::shared_ptr<std::vector<std::string>> b64;
};

std::optional<BatchRequest>
admit_batch_request(const drogon::HttpRequestPtr &req,
                    const capability::CapabilityMask &loaded,
                    const server::RoutingNameSets &routes, int max_batch_images,
                    server::DrogonCallback &callback) {
  server::InferOptions opts;
  server::EndpointSpec spec;
  spec.routing = server::kBuildRoutingSupport;
  spec.layout_only_allowed = false;
  // Hoisted ABOVE validate_request so body flags are classified. Without
  // the body, `{"images":[...], "layout":true}` was accepted with HTTP 200
  // and the layout silently dropped: query_only_params() parses only
  // req->query(), so the key was never seen — no X-Ignored-Params header,
  // no strict-mode 400. /ocr already passes it (ocr_base64_route.cpp:50);
  // this was the one body endpoint left behind by the change that exists
  // to remove exactly this class of silent-accept.
  auto json = req->getJsonObject();
  if (!server::validate_request(req, spec, loaded, routes.table,
                                routes.formula, &opts, callback,
                                /*allow_image_only=*/false,
                                json ? json.get() : nullptr))
    return std::nullopt;

  if (!json) {
    callback(server::error_response(server::ErrorCode::kInvalidJson, "Invalid JSON"));
    return std::nullopt;
  }
  if (!json->isMember("images") || !(*json)["images"].isArray()) {
    callback(server::error_response(server::ErrorCode::kInvalidJson, "Missing images array"));
    return std::nullopt;
  }

  auto &images_json = (*json)["images"];
  size_t n = images_json.size();
  if (n == 0) {
    callback(server::error_response(server::ErrorCode::kEmptyBatch, "Empty images array"));
    return std::nullopt;
  }
  // Cap before O(n) per-slot allocations (see GPU route): an unbounded
  // images[] is a memory-amplification OOM lever.
  if (n > static_cast<size_t>(max_batch_images)) {
    callback(server::error_response(server::ErrorCode::kBatchTooLarge,
                                    std::format("images array has {} entries, max is {}", n,
                                                max_batch_images)));
    return std::nullopt;
  }

  // Collect the raw base64 strings; decoding happens in the shared
  // batch_decode_base64 stage on the worker. asString() throws
  // Json::LogicError on a non-string element ({}/[]), so guard the
  // type — a malformed slot becomes an empty string the per-slot
  // stages tag, not a crash.
  auto b64_strings = std::make_shared<std::vector<std::string>>(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &el = images_json[static_cast<int>(i)];
    if (el.isString())
      (*b64_strings)[i] = el.asString();
  }
  return BatchRequest{std::move(opts), std::move(b64_strings)};
}

} // namespace



void register_ocr_batch_route_unified(
    server::WorkPool &work_pool,
    std::shared_ptr<pipeline::UnifiedPipelinePool> pool, int pool_size,
    const server::ImageDecoder &decode, const capability::CapabilityMask &loaded, int max_batch_images) {
  // Computed once at registration (it reads the routing config) and captured by
  // value, exactly as register_ocr_base64_route does for /ocr.
  const server::RoutingNameSets routes = server::routing_name_sets();
  drogon::app().registerHandler(
      "/ocr/batch",
      [&work_pool, pool, pool_size, &decode, loaded, max_batch_images, routes](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto admitted = admit_batch_request(req, loaded, routes,
                                            max_batch_images, callback);
        if (!admitted) return;
        const server::InferOptions opts = std::move(admitted->opts);
        auto b64_strings = std::move(admitted->b64);
        const size_t n = b64_strings->size();


        server::submit_work(work_pool, std::move(callback),
            [b64 = std::move(b64_strings), n, pool, pool_size, &decode,
             opts](server::DrogonCallback &cb) {
         server::run_with_error_handling(cb, "/ocr/batch", [&] {
          using namespace batchdetail;
          const bool want_layout = opts.want_layout;
          const int kMaxImageDim = decode::max_image_dim();

          // Shared per-slot stages (batch_common.cpp): base64 decode +
          // pre-decode dim sniff / area caps / aggregate pixel budget —
          // identical to both existing batch routes by construction.
          std::vector<std::string> raw_bytes(n), errors(n);
          std::vector<BatchItem> batch_items(n);
          batch_decode_base64(*b64, raw_bytes, errors);
          batch_check_dims_pre(raw_bytes, kMaxImageDim, errors);

          // Image decode through the BACKEND's decoder (nvJPEG on CUDA, vImage
          // on Apple, OpenCV on the host) — the one genuinely per-vendor step,
          // and it is already behind server::ImageDecoder.
          std::vector<cv::Mat> imgs(n);
          for (size_t i = 0; i < n; ++i) {
            if (!errors[i].empty()) continue;
            imgs[i] = decode(
                reinterpret_cast<const unsigned char *>(raw_bytes[i].data()),
                raw_bytes[i].size());
            if (imgs[i].empty()) errors[i] = "decode_failed";
          }
          batch_check_dims_post(imgs, kMaxImageDim, errors);

          // Work index list: slots that survived all checks.
          std::vector<size_t> valid_indices;
          valid_indices.reserve(n);
          for (size_t i = 0; i < n; ++i)
            if (errors[i].empty() && !imgs[i].empty())
              valid_indices.push_back(i);

          // Chunk cursor, in UNITS OF kMaxBatch images (the CPU route's cursor
          // was per-image; the claim protocol is otherwise identical).
          std::atomic<size_t> next_chunk{0};
          const size_t chunk_count =
              (valid_indices.size() + kMaxBatch - 1) / kMaxBatch;
          int num_workers = valid_indices.empty()
              ? 0
              : std::min({static_cast<int>(chunk_count), pool_size,
                          static_cast<int>(valid_indices.size())});
          {
            // Every request keeps ONE guaranteed worker (progress under
            // contention); extra fan-out workers draw from a process-wide
            // permit pool so N concurrent /ocr/batch requests can't create
            // N*pool_size OS threads (most would only block on
            // pool->acquire() anyway). Same bound the gRPC batch path applies.
            static std::counting_semaphore<4096> extra_batch_permits{
                static_cast<std::ptrdiff_t>(env::env_int(
                    "BATCH_FANOUT_GLOBAL_WORKERS", 64, 1, 4096))};
            struct Permit {
              bool held = false;
              ~Permit() {
                if (held) extra_batch_permits.release();
              }
              // A copy of a held==true permit would double-release the global
              // semaphore on destruction, over-crediting the ceiling this type
              // holds. Safe today only because the vector never resizes — i.e.
              // correct by accident, the exact hazard the gRPC twin
              // (recognize_batch_rpc.cpp) deletes these to prevent. Delete
              // them here too; the vector value-initializes in place.
              Permit() = default;
              Permit(const Permit &) = delete;
              Permit &operator=(const Permit &) = delete;
              Permit(Permit &&) = delete;
              Permit &operator=(Permit &&) = delete;
            };
            std::vector<Permit> permits(
                static_cast<size_t>(std::max(0, num_workers - 1)));
            std::vector<std::jthread> threads;
            threads.reserve(num_workers);
            const auto worker_body = [&]() {
              // Lease the pipeline outside the per-chunk loop — pool
              // exhaustion is a fatal worker-level error (no point trying
              // again), so it tags every remaining slot.
              std::optional<pipeline::UnifiedPipelinePool::Lease> lease;
              try {
                lease.emplace(pool->acquire());
              } catch (const std::exception &) {
                TOCR_LOG_ERROR("Batch worker pool exhausted", "route",
                               "/ocr/batch");
                // Tag every UNCLAIMED valid slot as failed so callers see it.
                while (true) {
                  size_t c = next_chunk.fetch_add(1);
                  if (c >= chunk_count) break;
                  const size_t start = c * kMaxBatch;
                  const size_t end =
                      std::min(start + kMaxBatch, valid_indices.size());
                  for (size_t k = start; k < end; ++k)
                    errors[valid_indices[k]] = "pool_exhausted";
                }
                return;
              }
              auto &pipe = lease->pipeline();
              while (true) {
                size_t c = next_chunk.fetch_add(1);
                if (c >= chunk_count) break;
                const size_t start = c * kMaxBatch;
                const size_t end =
                    std::min(start + kMaxBatch, valid_indices.size());

                std::vector<cv::Mat> chunk;
                chunk.reserve(end - start);
                for (size_t k = start; k < end; ++k)
                  chunk.push_back(imgs[valid_indices[k]]); // Mat header copy

                // The route's request options become the pipeline's own flag
                // POD here; the batched call and the per-image retry below share
                // the SAME value, so the fallback path cannot drift from it.
                const pipeline::RunFlags run_flags{
                    .layout = want_layout,
                    .reading_order = opts.want_reading_order,
                    .tables = opts.want_tables,
                    .formulas = opts.want_formulas};
                try {
                  auto results = pipe.run_batch_with_layout(
                      chunk, run_flags, opts.routing_override);
                  for (size_t j = 0; j < results.size() && start + j < end; ++j)
                    batch_items[valid_indices[start + j]].out =
                        std::move(results[j]);
                } catch (const std::exception &e) {
                  // One degenerate image poisons the whole batched chunk.
                  // Retry the chunk image-by-image so ONLY the bad slot is
                  // tagged — the same per-slot isolation the CPU route got
                  // for free by never batching.
                  TOCR_LOG_ERROR("Batch chunk error, retrying per-image",
                                 "route", "/ocr/batch", "chunk", c, "error",
                                 std::string_view(e.what()));
                  for (size_t k = start; k < end; ++k) {
                    const size_t idx = valid_indices[k];
                    try {
                      batch_items[idx].out = pipe.run_with_layout(
                          imgs[idx], run_flags, opts.routing_override,
                          /*defer_external=*/false);
                    } catch (const std::exception &ex) {
                      TOCR_LOG_ERROR("Batch image error", "route", "/ocr/batch",
                                     "image_index", idx, "error",
                                     std::string_view(ex.what()));
                      errors[idx] = ex.what();
                    } catch (...) {
                      TOCR_LOG_ERROR("Batch image error: unknown", "route",
                                     "/ocr/batch", "image_index", idx);
                      errors[idx] = "unknown";
                    }
                  }
                } catch (...) {
                  for (size_t k = start; k < end; ++k) {
                    TOCR_LOG_ERROR("Batch image error: unknown", "route",
                                   "/ocr/batch", "image_index",
                                   valid_indices[k]);
                    errors[valid_indices[k]] = "unknown";
                  }
                }
              }
            };
            // First worker is guaranteed; each subsequent one needs a permit.
            if (num_workers > 0) threads.emplace_back(worker_body);
            for (int w = 1; w < num_workers; ++w) {
              Permit &pm = permits[static_cast<size_t>(w - 1)];
              pm.held = extra_batch_permits.try_acquire();
              if (!pm.held) break;
              threads.emplace_back(worker_body);
            }
          }  // jthreads auto-join here (permits release after)

          cb(server::json_response(
              batch_emit_json(batch_items, errors, want_layout,
                              opts.want_blocks)));
         });
        });
      },
      {drogon::Post});
}

void register_backend_capabilities_route(const backend::BackendCaps &caps,
                                         int pool_size) {
  // Built ONCE at startup — the answer is immutable for the process lifetime.
  std::string body = "{\"backend\":\"";
  detail::append_escaped_string(body, caps.name);
  body += "\",\"device\":\"";
  detail::append_escaped_string(body,
                                std::string(backend::device_kind_name(caps.device)));
  body += "\",\"async\":";
  body += caps.async ? "true" : "false";
  body += ",\"native_image_decode\":";
  body += caps.native_image_decode ? "true" : "false";
  body += ",\"supports_batch\":";
  body += caps.supports_batch ? "true" : "false";
  body += ",\"pool_size\":";
  body += std::to_string(pool_size);
  body += ",\"available_backends\":[";
  bool first = true;
  for (const auto n : backend::available_backends()) {
    if (!first) body += ',';
    first = false;
    body += '"';
    detail::append_escaped_string(body, std::string(n));
    body += '"';
  }
  body += "]}";

  drogon::app().registerHandler(
      "/capabilities/backend",
      [body = std::move(body)](
          const drogon::HttpRequestPtr &,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        callback(server::json_response(body));
      },
      {drogon::Get});
}

} // namespace turbo_ocr::routes
