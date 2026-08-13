#pragma once
#include <stdexcept>
#include <string>

namespace turbo_ocr {

class OcrError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class ModelLoadError : public OcrError {
  using OcrError::OcrError;
};

class InferenceError : public OcrError {
  using OcrError::OcrError;
};

// NOTE (removed): CudaError and HipError. base/ is defined as the foundation
// with ZERO domain knowledge, and two vendor-named exception types is the
// clearest possible breach of that — a header that must not mention OCR was
// naming CUDA and HIP. They now live in the arms that throw them
// (nvidia/support/cuda_check.h, amd/support/hip_check.h), still deriving from
// OcrError so every existing catch site is unchanged: nothing anywhere catches
// either type by name.
//
// The rule they encode moved with them and is worth keeping: an ordinary
// device-runtime error is RECOVERABLE and throws — the per-request handlers
// degrade and the server survives — and only a STICKY fault terminates the
// process. AMD once diverged, calling std::abort() on every HIP error while
// NVIDIA threw, so one bad kernel launch killed the whole server on that arm
// alone.

class PoolExhaustedError : public OcrError {
public:
  PoolExhaustedError() : OcrError("Pipeline pool exhausted (timeout)") {}
  explicit PoolExhaustedError(const std::string &msg) : OcrError(msg) {}
};

// A blocking wait (e.g. dispatcher.submit_for_default()) exceeded its
// per-request deadline. Mirrors PoolExhaustedError so routes/gRPC map it to a
// stable timeout status (504 / DEADLINE_EXCEEDED). Lives here (not a GPU-only
// header) so the CPU build's catch clauses see it too.
class TimeoutError : public OcrError {
public:
  TimeoutError() : OcrError("Inference timed out") {}
  explicit TimeoutError(const std::string &msg) : OcrError(msg) {}
};

class ImageDecodeError : public OcrError {
  using OcrError::OcrError;
};

// Decoded/declared image dimensions exceed MAX_IMAGE_DIM. Distinct from
// ImageDecodeError so the routes can map it to the stable DIMENSIONS_TOO_LARGE
// code (same as the pre/post-decode sniffs) instead of IMAGE_DECODE_FAILED.
class ImageTooLargeError : public OcrError {
  using OcrError::OcrError;
};

class PdfRenderError : public OcrError {
  using OcrError::OcrError;
};

// An EXPLICITLY requested table/formula backend (named registry pick or inline
// spec) was null or not ready — distinct from "no table/formula in this image".
// Routes map it to 503 BACKEND_UNAVAILABLE so a failed-to-load backend can't
// masquerade as an empty-but-successful result. Kept outside the OcrError
// hierarchy (plain runtime_error, as it has always been) so its handling never
// couples to the OcrError catch clauses.
struct BackendUnavailableError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace turbo_ocr
