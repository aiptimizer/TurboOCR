#pragma once

#include <cstddef>

#include <opencv2/core.hpp>

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
};

// Runs ON a dispatcher worker thread (call inside submit/submit_for_default):
// nvJPEG header dims + bomb guard → GPU-direct decode + inference. The host
// codec is used only for JPEG variants nvJPEG does not decode (progressive,
// arithmetic, 12-bit, CMYK) and for layout-only requests; a GPU decoder
// fault throws GpuDecodeError (503 / UNAVAILABLE) instead of being retried
// on the CPU, so a broken device is never hidden behind slower requests.
// Also throws ImageTooLargeError / ImageDecodeError; every transport's catch
// chain maps all three. `data`/`len` must be owned by the submitted lambda
// (timeout-abandon safety).
[[nodiscard]] OcrPipelineResult
decode_jpeg_and_run(GpuPipelineEntry &e, const unsigned char *data, size_t len,
                    const JpegRunOpts &opts);

// Runs ON a dispatcher worker thread: decode a JPEG with the replica's own
// decoder into a host image, for the few consumers that need host pixels
// (/infer crops, /ocr/batch slots). Same rules as above: a device fault
// throws GpuDecodeError, an unsupported bitstream takes the host codec,
// anything undecodable throws ImageDecodeError. Never bomb-guarded here;
// callers apply their own size policy to the result.
[[nodiscard]] cv::Mat
decode_jpeg_on_replica(GpuPipelineEntry &e, const unsigned char *data, size_t len);

} // namespace turbo_ocr::pipeline
