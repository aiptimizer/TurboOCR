#pragma once

#include <cstddef>

#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/backend_routing/routing_config.h"

// GPU build only. The nvJPEG GPU-direct decode-and-infer body shared by the
// HTTP /ocr/raw route and the gRPC Recognize JPEG path — previously two
// near-verbatim copies that could drift.
namespace turbo_ocr::pipeline {

struct GpuPipelineEntry;

struct JpegRunOpts {
  bool want_layout = false;
  bool want_reading_order = false;
  bool want_tables = false;
  bool want_formulas = false;
  backend_routing::RequestRouting routing;
  // HTTP awaits deferred VLM crops off the worker (finalize_deferred);
  // the gRPC path blocks synchronously.
  bool defer_external = false;
  bool layout_only = false;
  // /ocr/raw logs a rate-limited error when the GPU zero-copy fast path
  // fails before the CPU retry; the gRPC path stays silent (legacy shape).
  bool log_fastpath_errors = false;
};

// Runs ON a dispatcher worker thread (call inside submit/submit_for_default):
// nvJPEG header dims + bomb guard → GPU-direct decode + inference, falling
// back to CPU decode + re-guard + inference. Throws ImageTooLargeError /
// ImageDecodeError; both transports' catch chains map them. `data`/`len`
// must be owned by the submitted lambda (timeout-abandon safety).
[[nodiscard]] OcrPipelineResult
decode_jpeg_and_run(GpuPipelineEntry &e, const unsigned char *data, size_t len,
                    const JpegRunOpts &opts);

} // namespace turbo_ocr::pipeline
