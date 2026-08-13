// make_infer_func — the InferFunc factories over a UnifiedPipelinePool. The
// pool itself lives in unified_pipeline_pool.cpp (see that file's header for
// why they are separate TUs).

#include "turbo_ocr/pipeline/unified/make_infer_func.h"

#include "turbo_ocr/analysis/classification/doc_orientation_common.h" // rotate_upright
#include "turbo_ocr/base/log/logger.h"

#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "turbo_ocr/base/errors.h"                  // ImageTooLargeError
#include "turbo_ocr/image/size_classify.h"          // decode::classify_image_size
#include "turbo_ocr/pipeline/pipeline_result.h"     // finalize_deferred
#include "turbo_ocr/core/infer_result.h"          // InferResult, from_pipeline_result
#include "turbo_ocr/core/infer_options.h"      // InferOptions (complete type,
                                                    // drogon-free — see that header)

namespace turbo_ocr::pipeline {

namespace {

// Option derivation, lease scope and deferral drain are identical for both
// InferFuncs; only how the page reaches the pipeline differs. One copy so the
// two cannot drift (they already had two copies of the want_* derivation).
//
// LEASE SCOPE IS THE DEVICE WORK, NOT THE REQUEST: finalize_deferred() awaits a
// remote VLM over HTTP, and holding a replica across a network round-trip gives
// away throughput. Releasing early is safe by construction — submit_async
// completes its D2H and encodes every crop before returning, and
// PendingExternal holds parser snapshots with no pointer back into the
// recognizer.
template <class RunOnPipeline>
server::InferResult run_leased(UnifiedPipelinePool &pool,
                               const server::InferOptions &opts,
                               RunOnPipeline &&run) {
  // Tables/formulas/blocks all consume layout-detected regions, so any of them
  // implies layout. Parser-built options already carry this (query_options.h
  // step 4); the chains are defence-in-depth for the hand-built ones in
  // grpc/service_core.cpp and pipeline/job/pdf_job.cpp, whose `requested` is
  // empty.
  const RunFlags flags{
      .layout = opts.want_layout || opts.want_tables || opts.want_formulas ||
                opts.want_blocks,
      .reading_order = opts.want_reading_order || opts.want_blocks,
      .tables = opts.want_tables,
      .formulas = opts.want_formulas,
      .text = opts.want_text};

  OcrPipelineResult out;
  {
    auto lease = pool.acquire();
    out = run(lease.pipeline(), flags);
  }
  finalize_deferred(out);
  return server::from_pipeline_result(std::move(out));
}

} // namespace

namespace {

// Honour `opts.want_autorotate` HERE, at the seam every transport funnels
// through. The flag was parsed and availability-gated by validation on every
// route, and then no image transport ever applied it — /ocr/raw, /ocr,
// /ocr/pixels and /ocr/batch all returned byte-identical output with and
// without ?autorotate=1 (measured: a 180° page answered 27 garbage words
// either way while detect_orientation classified it correctly). The 90°/270°
// cases only LOOKED half-fixed because vertical-looking lines go through the
// per-line 0/180 classifier. Rotating a header-local Mat keeps the caller's
// image untouched (cv::rotate allocates a fresh buffer). The gRPC/stream
// image branches keep their own pre-rotation; after it this detect returns 0,
// so nothing double-rotates.
inline cv::Mat maybe_autorotate(UnifiedOcrPipeline &p,
                                const server::InferOptions &opts,
                                const cv::Mat &img) {
  if (!opts.want_autorotate || !p.has_doc_ori()) return img;
  cv::Mat page = img;
  const int deg = p.detect_orientation(page);
  TOCR_LOG_INFO("autorotate", "detected_deg", deg);
  if (deg) classification::rotate_upright(page, deg);
  return page;
}

} // namespace

server::InferFunc make_infer_func(std::shared_ptr<UnifiedPipelinePool> pool) {
  if (!pool || pool->size() == 0) return {};
  return [pool](const cv::Mat &img,
                const server::InferOptions &opts) -> server::InferResult {
    return run_leased(*pool, opts, [&](UnifiedOcrPipeline &p, const RunFlags &f) {
      return p.run_with_layout(maybe_autorotate(p, opts, img), f,
                               opts.routing_override,
                               /*defer_external=*/true);
    });
  };
}

server::EncodedInferFunc
make_encoded_infer_func(std::shared_ptr<UnifiedPipelinePool> pool,
                        backend::Backend &backend) {
  if (!pool || pool->size() == 0) return {};
  // Built ONCE: constructing it per request would put an allocation in front of
  // every image, and it is stateless.
  server::ImageDecoder host_decode = backend.make_image_decoder();
  backend::Backend *bk = &backend;
  return [pool, host_decode, bk](
             const std::uint8_t *data, std::size_t len,
             const server::InferOptions &opts) -> server::InferResult {
    // WHERE the decode happens decides throughput, not whether it is on-device.
    // run_encoded() falls back to a host decode when the vendor decoder declines
    // the container — but by then it holds a lease, pinning one of a handful of
    // replicas for the whole CPU decode. So sniff first: only bytes the backend
    // will really decode on-device go in encoded, everything else is decoded
    // here, outside the lease.
    // Autorotate needs the decoded page on the host to rotate it, so an
    // autorotate request takes the host-decode path even when the backend
    // could device-decode — correctness over the decode shortcut, and only
    // for requests that opted in.
    if (!opts.want_autorotate && bk->can_device_decode(data, len))
      return run_leased(*pool, opts, [&](UnifiedOcrPipeline &p, const RunFlags &f) {
        return p.run_encoded(data, len, f, opts.routing_override,
                             /*defer_external=*/true);
      });

    const cv::Mat img = host_decode ? host_decode(data, len) : cv::Mat{};
    // A failed decode is an ERROR, not an empty page. Returning {} here made
    // a truncated/corrupt upload answer 200 {"results":[]} — byte-identical
    // to a genuinely blank page — on every shipping backend (only NVIDIA
    // overrides can_device_decode, so CPU/Apple always take this branch), and
    // it made the routes' own kImageDecodeFailed branches dead code: this
    // throw is the ONLY producer for the ImageDecodeError catch chains that
    // already exist on all three transports.
    if (img.empty())
      throw turbo_ocr::ImageDecodeError("failed to decode image");
    // POST-DECODE BOMB GUARD — required HERE, not only inside run_encoded().
    // The routes deleted their own post-decode check on the strength of "the
    // guard moved behind run_encoded()", but this fallback (taken by every
    // backend whose can_device_decode() declines — the default) bypasses
    // run_encoded entirely. Without this line, a ~2 KB BMP declaring
    // 60000x60000 sails past the pre-decode sniff (which cannot parse BMP)
    // and materializes a ~10 GB frame straight into inference.
    if (decode::classify_image_size(img.cols, img.rows) !=
        decode::ImageSizeVerdict::kOk)
      throw turbo_ocr::ImageTooLargeError(
          "decoded image dimensions exceed the configured maximum");
    return run_leased(*pool, opts, [&](UnifiedOcrPipeline &p, const RunFlags &f) {
      return p.run_with_layout(maybe_autorotate(p, opts, img), f,
                               opts.routing_override,
                               /*defer_external=*/true);
    });
  };
}

server::InferOneFunc
make_infer_one_func(std::shared_ptr<UnifiedPipelinePool> pool) {
  if (!pool || pool->size() == 0) return {};
  return [pool](const cv::Mat &img, const std::string &modality,
                const std::string &backend_name,
                const void *inline_spec) -> std::string {
    auto lease = pool->acquire();
    return lease.pipeline().infer_one(
        img, modality, backend_name,
        static_cast<const backend_routing::BackendSpec *>(inline_spec));
  };
}

} // namespace turbo_ocr::pipeline
