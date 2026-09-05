#pragma once

// Service-boundary function aliases as a leaf header: registrar headers name
// these in signatures only (by reference), so forward declarations suffice —
// including the full InferResult/InferOptions stacks here would drag
// serialization + validation into every route header (the server_types.h
// umbrella exists for TUs that want that).
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace cv {
class Mat;
}

namespace turbo_ocr::server {

struct InferResult;
struct InferOptions;

/// Image decoder: (raw_bytes_ptr, length) -> cv::Mat
using ImageDecoder =
    std::function<cv::Mat(const unsigned char *data, size_t len)>;

/// Inference function: given cv::Mat + feature flags, run OCR pipeline.
using InferFunc = std::function<InferResult(const cv::Mat &, const InferOptions &)>;
/// JPEG-bytes inference: decode on the replica that runs inference (GPU-direct,
/// no host pixel buffer) and run the pipeline. Empty on builds without a device
/// decoder; routes then decode on the host and call InferFunc. The bytes are
/// shared so an abandoned (timed-out) task never reads freed memory.
using JpegInferFunc = std::function<InferResult(
    std::shared_ptr<const std::string> jpeg, const InferOptions &)>;

/// Orientation detector: rendered page -> clockwise rotation deg (0/90/180/270).
/// Empty/unset when the doc-orientation model isn't loaded (autorotate off).
using OrientFunc = std::function<int(const cv::Mat &)>;

} // namespace turbo_ocr::server
