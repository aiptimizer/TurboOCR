#pragma once

#include <cstddef>
#include <cstdint>

// Service-boundary function aliases as a leaf header: registrar headers name
// these in signatures only (by reference), so forward declarations suffice —
// including the full InferResult/InferOptions stacks here would drag
// serialization + validation into every route header (the server_types.h
// umbrella exists for TUs that want that).
#include <functional>
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

// ENCODED-BYTES inference. The cv::Mat InferFunc above forces a HOST decode at
// the transport boundary, which is why a backend with an on-device decoder
// (nvJPEG, vImage) could never use it: by the time the pipeline sees the page it
// is already a decoded Mat, so the ~200 KB JPEG has become a ~25 MB H2D.
//
// A route that still has the encoded bytes should prefer this; it falls back to
// the host decoder inside the pipeline when the backend has no device decode, so
// it is safe to call unconditionally and only the backends that benefit change
// behaviour. Null when the server was built without a pipeline pool (the
// legacy/offline drivers), so callers must check before use.
using EncodedInferFunc =
    std::function<InferResult(const std::uint8_t *, std::size_t,
                              const InferOptions &)>;

/// Orientation detector: rendered page -> clockwise rotation deg (0/90/180/270).
/// Empty/unset when the doc-orientation model isn't loaded (autorotate off).
using OrientFunc = std::function<int(const cv::Mat &)>;

/// SINGLE-CROP inference against ONE named (or inline) backend — what POST
/// /infer exposes. Distinct from InferFunc because it runs no OCR pipeline: it
/// hands one already-cropped region straight to a table/formula recognizer and
/// returns that backend's raw output (HTML for "table", LaTeX for "formula").
///
/// (const cv::Mat &crop, modality, backend_name, inline_spec_or_null) -> string
///
/// `inline_spec` is a `const backend_routing::BackendSpec *` behind a void* so
/// this leaf header stays free of the routing headers — every other alias here
/// is likewise nameable without dragging its stack in. The one call site casts
/// it back; the pipeline's infer_one takes the typed pointer.
using InferOneFunc =
    std::function<std::string(const cv::Mat &, const std::string &modality,
                              const std::string &backend_name,
                              const void *inline_spec)>;

} // namespace turbo_ocr::server
