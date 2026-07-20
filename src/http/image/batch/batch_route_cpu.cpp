#include "turbo_ocr/http/cpu_image_routes.h"

#include <atomic>
#include <format>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/log/stage_profiler.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/validation/pixel_dims.h"
#include "turbo_ocr/validation/request_gate.h"

using turbo_ocr::base64_decode;

#include "batch_internal.h"

namespace turbo_ocr::routes {

void register_ocr_batch_route_cpu(server::WorkPool &work_pool,
                                  pipeline::CpuPipelinePool &pool,
                                  int pool_size,
                                  const server::ImageDecoder &decode,
                                  bool layout_available,
                                  bool table_available,
                                  bool formula_available,
                                  int max_batch_images) {
  drogon::app().registerHandler(
      "/ocr/batch",
      [&work_pool, &pool, pool_size, &decode, layout_available,
       table_available, formula_available, max_batch_images](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        server::InferOptions opts;
        server::EndpointSpec spec;
        spec.routing = server::kBuildRoutingSupport;
        spec.layout_only_allowed = false;
        if (!server::validate_request(req, spec, layout_available,
                                      table_available, formula_available,
                                      /*valid_route_table=*/{},
                                      /*valid_route_formula=*/{}, &opts,
                                      callback))
          return;

        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "INVALID_JSON", "Invalid JSON"));
          return;
        }
        if (!json->isMember("images") || !(*json)["images"].isArray()) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "INVALID_JSON",
                                          "Missing images array"));
          return;
        }

        auto &images_json = (*json)["images"];
        size_t n = images_json.size();
        if (n == 0) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "EMPTY_BATCH",
                                          "Empty images array"));
          return;
        }
        // Cap before O(n) per-slot allocations (see GPU route): an unbounded
        // images[] is a memory-amplification OOM lever.
        if (n > static_cast<size_t>(max_batch_images)) {
          callback(server::error_response(
              drogon::k400BadRequest, "BATCH_TOO_LARGE",
              std::format("images array has {} entries, max is {}", n,
                          max_batch_images)));
          return;
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

        server::submit_work(work_pool, std::move(callback),
            [b64 = std::move(b64_strings), n, &pool, pool_size, &decode,
             opts](server::DrogonCallback &cb) {
         server::run_with_error_handling(cb, "/ocr/batch", [&] {
          using namespace batchdetail;
          const bool want_layout = opts.want_layout;
          const int kMaxImageDim = decode::max_image_dim();

          // Shared per-slot stages (batch_common.cpp): base64 decode +
          // pre-decode dim sniff / area caps / aggregate pixel budget —
          // identical to the GPU batch route by construction.
          std::vector<std::string> raw_bytes(n), errors(n);
          std::vector<BatchItem> batch_items(n);
          batch_decode_base64(*b64, raw_bytes, errors);
          batch_check_dims_pre(raw_bytes, kMaxImageDim, errors);

          // CPU image decode (the GPU route decodes via nvJPEG where
          // possible; this is the backend-specific stage).
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

          std::atomic<size_t> next_valid{0};
          int num_workers = valid_indices.empty()
              ? 0
              : std::min(static_cast<int>(valid_indices.size()), pool_size);
          {
            // Every request keeps ONE guaranteed worker (progress under
            // contention); extra fan-out workers draw from a process-wide
            // permit pool so N concurrent /ocr/batch requests can't create
            // N*pool_size OS threads (most would only block on
            // pool.acquire() anyway). Same bound the gRPC batch path applies.
            static std::counting_semaphore<4096> extra_batch_permits{
                static_cast<std::ptrdiff_t>(env::env_int(
                    "BATCH_FANOUT_GLOBAL_WORKERS", 64, 1, 4096))};
            struct Permit {
              bool held = false;
              ~Permit() {
                if (held) extra_batch_permits.release();
              }
            };
            std::vector<Permit> permits(
                static_cast<size_t>(std::max(0, num_workers - 1)));
            std::vector<std::jthread> threads;
            threads.reserve(num_workers);
            const auto worker_body = [&]() {
              // Acquire the pool handle outside the per-image loop — pool
              // exhaustion is a fatal worker-level error (no point trying
              // again), so it tags every remaining slot.
              std::unique_ptr<decltype(pool.acquire())> handle_holder;
              try {
                handle_holder =
                    std::make_unique<decltype(pool.acquire())>(pool.acquire());
              } catch (const turbo_ocr::PoolExhaustedError &) {
                TOCR_LOG_ERROR("Batch worker pool exhausted", "route",
                               "/ocr/batch");
                // Tag every UNCLAIMED valid slot as failed so callers see it.
                while (true) {
                  size_t k = next_valid.fetch_add(1);
                  if (k >= valid_indices.size()) break;
                  errors[valid_indices[k]] = "pool_exhausted";
                }
                return;
              }
              auto &handle = *handle_holder;
              while (true) {
                size_t k = next_valid.fetch_add(1);
                if (k >= valid_indices.size()) break;
                size_t idx = valid_indices[k];
                // Per-image try/catch so one image failing does NOT leave
                // all later slots silently empty with HTTP 200.
                try {
                  batch_items[idx].out = handle->run_with_layout(
                      imgs[idx], want_layout, opts.want_reading_order,
                      opts.want_tables, opts.want_formulas);
                } catch (const std::exception &e) {
                  TOCR_LOG_ERROR("Batch image error", "route", "/ocr/batch",
                                 "image_index", idx, "error",
                                 std::string_view(e.what()));
                  errors[idx] = e.what();
                } catch (...) {
                  TOCR_LOG_ERROR("Batch image error: unknown", "route",
                                 "/ocr/batch", "image_index", idx);
                  errors[idx] = "unknown";
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

} // namespace turbo_ocr::routes
