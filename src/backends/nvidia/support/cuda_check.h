#pragma once
#include "turbo_ocr/base/errors.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <iostream>

namespace turbo_ocr::nvidia {
// A CUDA runtime failure. Lives HERE, in the arm that throws it, rather than in
// base/errors.h — base is the foundation with zero domain knowledge and must
// not name a vendor. Derives from OcrError, so the catch sites (which all catch
// OcrError, never this type by name) are unaffected.
//
// POLICY, shared with the AMD twin: an ordinary device-runtime error is
// RECOVERABLE and throws — the per-request handlers degrade and the server
// survives — and only a STICKY fault terminates the process.
class CudaError : public turbo_ocr::OcrError {
  using turbo_ocr::OcrError::OcrError;
};
} // namespace turbo_ocr::nvidia

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      auto msg = std::format("CUDA Error at {}:{} - {}", __FILE__, __LINE__,  \
                             cudaGetErrorString(err));                         \
      std::cerr << msg << '\n';                                               \
      throw turbo_ocr::nvidia::CudaError(msg);                                        \
    }                                                                          \
  } while (0)

namespace turbo_ocr {

// A "sticky" CUDA error is one that poisons the current process's CUDA context
// permanently: it cannot be cleared with cudaGetLastError() and every
// subsequent CUDA call on any stream/engine returns it (so any further
// inference would silently serve garbage). Recovering would require a fresh
// process, so the only safe response is to fail-fast and let the orchestrator
// restart the pod.
[[nodiscard]] inline bool is_sticky_cuda_error(cudaError_t err) noexcept {
  switch (err) {
  case cudaErrorIllegalAddress:
  case cudaErrorLaunchFailure:
  case cudaErrorLaunchTimeout:
  case cudaErrorHardwareStackError:
  case cudaErrorIllegalInstruction:
  case cudaErrorMisalignedAddress:
  case cudaErrorInvalidAddressSpace:
  case cudaErrorInvalidPc:
  case cudaErrorECCUncorrectable:
  case cudaErrorUnsupportedExecAffinity:
  case cudaErrorExternalDevice:
  case cudaErrorNvlinkUncorrectable:
  case cudaErrorContextIsDestroyed:
    return true;
  default:
    return false;
  }
}

// Inspect the most recent CUDA error WITHOUT clearing it (cudaPeekAtLastError),
// and if it is sticky, log FATAL and terminate the process immediately with a
// non-zero status. std::_Exit avoids running atexit/destructors that would
// themselves issue poisoned CUDA calls and hang. `where` identifies the call
// site in the log. No-op (returns) when the context is healthy or holds only a
// recoverable error, which the caller is then free to surface as a 5xx.
inline void abort_on_sticky_cuda_fault(const char *where) noexcept {
  const cudaError_t err = cudaPeekAtLastError();
  if (err == cudaSuccess || !is_sticky_cuda_error(err))
    return;
  std::cerr << std::format(
                   "FATAL: sticky CUDA fault at {} - {} ({}); the CUDA context "
                   "is poisoned, exiting so the orchestrator restarts the pod",
                   where, cudaGetErrorString(err), static_cast<int>(err))
            << std::endl;
  std::_Exit(EXIT_FAILURE);
}

} // namespace turbo_ocr
