#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/pipeline/pipeline_result.h"
#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif
#include "turbo_ocr/server/server_types.h"

// Internal to src/http/image/: the /ocr/batch GPU pipeline stages
// (ocr_batch_support_gpu.cpp) used by the registrar TU
// (ocr_batch_route_gpu.cpp). Stage order: base64 decode -> pre-decode dim
// sniff -> image decode -> post-decode caps -> batched inference -> JSON.
namespace turbo_ocr::routes::batchdetail {

// Per-slot result. Cardinality always equals the caller's images[] so
// batch_results[i]/errors[i] correlate; failed slots are tagged, not dropped.
struct BatchItem {
  pipeline::OcrPipelineResult out;
};

void batch_decode_base64(const std::vector<std::string> &b64_strings,
                          std::vector<std::string> &raw_bytes,
                          std::vector<std::string> &errors);

void batch_check_dims_pre(const std::vector<std::string> &raw_bytes,
                           int max_image_dim,
                           std::vector<std::string> &errors);

#ifndef USE_CPU_ONLY
// Host decode of the NON-JPEG slots on the work thread (PNG via Wuffs, the
// rest via cv::imdecode). With nvJPEG available, JPEG slots are left empty
// here and decoded on the replica inside batch_run_pipeline; without it they
// are host formats like the rest.
void batch_decode_images(const std::vector<std::string> &raw_bytes,
                          bool nvjpeg_available,
                          const server::ImageDecoder &decode,
                          std::vector<cv::Mat> &imgs,
                          std::vector<std::string> &errors);
#endif


void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors);

#ifndef USE_CPU_ONLY
// One dispatcher task for the whole batch: JPEG slots (empty `imgs[i]` with a
// JPEG in `raw_bytes[i]`) are decoded on the replica with its own decoder and
// size-checked there, then everything runs through the chunked batch
// inference. Slots already tagged in `errors` are skipped. The task owns
// `raw_bytes` and `imgs` (moved in) so an abandoned-on-timeout run never
// touches the caller's frame. Per-slot failures land in `errors`
// (`gpu_decode_failed`, `decode_failed`, `dimensions_too_large`,
// `pixels_too_large`, `inference_failed`); pool exhaustion and the request
// deadline propagate.
void batch_run_pipeline(pipeline::PipelineDispatcher &dispatcher,
                         std::vector<std::string> raw_bytes,
                         std::vector<cv::Mat> imgs,
                         bool nvjpeg_available,
                         bool want_layout,
                         const server::InferOptions &opts,
                         std::vector<BatchItem> &all_items,
                         std::vector<std::string> &errors);
#endif


std::string batch_emit_json(std::vector<BatchItem> &all_items,
                             const std::vector<std::string> &errors,
                             bool want_layout,
                             bool want_blocks);

} // namespace turbo_ocr::routes::batchdetail
