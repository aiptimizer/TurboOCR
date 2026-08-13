#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/service/server/server_types.h"

// Internal to src/service/http/image/: the /ocr/batch GPU pipeline stages
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

// (batch_decode_images was declared here, gated on USE_CPU_ONLY. It had no
// definition and no caller anywhere in the tree — it went with the CUDA batch
// route that used to host both. Its `nvjpeg_available` parameter is the same
// per-vendor-decoder special case the backend seam now handles behind
// EncodedInferFunc.)


void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors);



std::string batch_emit_json(std::vector<BatchItem> &all_items,
                             const std::vector<std::string> &errors,
                             bool want_layout,
                             bool want_blocks);

} // namespace turbo_ocr::routes::batchdetail
