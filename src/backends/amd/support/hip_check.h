#pragma once

// HIP_CHECK — the AMD mirror of common/cuda/cuda_check.h's CUDA_CHECK, and it
// must mirror its POLICY, not just its name: an ordinary device-runtime error
// THROWS (recoverable — the per-request and per-region handlers already degrade
// gracefully and the server survives), and only a STICKY fault terminates the
// process. This file previously called std::abort() on EVERY HIP error, so a
// single bad launch killed the whole server on the AMD arm while the identical
// failure on NVIDIA became a 5xx. Generic policy is shared, never forked per
// backend.

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>

#include "turbo_ocr/base/errors.h"

namespace turbo_ocr::amd {

// The HIP twin of nvidia::CudaError. Both live in their own arm rather than in
// base/errors.h, which is defined as having zero domain knowledge and so must
// not name a vendor. Separate types only so a log can name the runtime; the
// policy above is identical by rule.
class HipError : public turbo_ocr::OcrError {
  using turbo_ocr::OcrError::OcrError;
};

// A "sticky" HIP error poisons the process's HIP context permanently — the
// exact mirror of cuda_check.h's is_sticky_cuda_error, over the HIP codes that
// correspond 1:1 to the CUDA list (codes CUDA defines but HIP does not are
// simply absent). Recovering needs a fresh process; the only safe response is
// fail-fast so the orchestrator restarts the pod. Until this existed, the
// header PROMISED the two-tier policy while the impl threw for every error —
// a real sticky fault was caught as recoverable and the poisoned context kept
// serving garbage.
[[nodiscard]] inline bool is_sticky_hip_error(hipError_t err) noexcept {
  switch (err) {
  case hipErrorIllegalAddress:      // 700 — cudaErrorIllegalAddress
  case hipErrorLaunchTimeOut:       // 702 — cudaErrorLaunchTimeout
  case hipErrorContextIsDestroyed:  // 709 — cudaErrorContextIsDestroyed
  case hipErrorLaunchFailure:       // 719 — cudaErrorLaunchFailure
  case hipErrorECCNotCorrectable:   // 214 — cudaErrorECCUncorrectable
    return true;
  default:
    return false;
  }
}

// Inspect the most recent HIP error WITHOUT clearing it, and if it is sticky,
// log FATAL and terminate immediately. std::_Exit avoids atexit/destructors
// that would themselves issue poisoned HIP calls and hang. No-op when the
// context is healthy or holds only a recoverable error, which the caller then
// surfaces as a 5xx. Same call-site shape as abort_on_sticky_cuda_fault.
inline void abort_on_sticky_hip_fault(const char *where) noexcept {
  const hipError_t err = hipPeekAtLastError();
  if (err == hipSuccess || !is_sticky_hip_error(err))
    return;
  std::cerr << std::format(
                   "FATAL: sticky HIP fault at {} - {} ({}); the HIP context "
                   "is poisoned, exiting so the orchestrator restarts the pod",
                   where, hipGetErrorString(err), static_cast<int>(err))
            << std::endl;
  std::_Exit(EXIT_FAILURE);
}

inline void hip_check_impl(hipError_t err, const char *expr, const char *file,
                           int line) {
  if (err != hipSuccess) {
    // Two tiers, matching the policy at the top of this file: a sticky fault
    // fail-fasts (the context is unrecoverable); everything else throws and
    // degrades to a 5xx.
    if (is_sticky_hip_error(err)) {
      std::cerr << std::format(
                       "FATAL: sticky HIP fault at {}:{} - {} ({}) [{}]; the "
                       "HIP context is poisoned, exiting so the orchestrator "
                       "restarts the pod",
                       file, line, hipGetErrorString(err),
                       hipGetErrorName(err), expr)
                << std::endl;
      std::_Exit(EXIT_FAILURE);
    }
    auto msg = std::format("HIP Error at {}:{} - {} ({}) [{}]", file, line,
                           hipGetErrorString(err), hipGetErrorName(err), expr);
    std::cerr << msg << '\n';
    throw HipError(msg);
  }
}

} // namespace turbo_ocr::amd

#define HIP_CHECK(expr)                                                         \
  ::turbo_ocr::amd::hip_check_impl((expr), #expr, __FILE__, __LINE__)
