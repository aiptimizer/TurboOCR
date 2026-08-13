#pragma once

// ep_options.h — the PURE execution-provider option policy behind
// engine::OrtEngine (src/onnx/ort_engine.cpp).
//
// Everything here is a function of its arguments plus the environment: no
// session, no ORT handle, no OrtEngine state, no I/O. That is the whole point.
// The per-provider appenders in ort_engine.cpp cannot run without an
// onnxruntime that actually ships the provider, but the DECISIONS they make —
// which OpenVINO device, whether precision=FP16 is safe on it, how
// OPENVINO_EP_OPTS overrides a computed key, what one malformed pair does to
// the rest of the string, how a device ordinal is resolved — are exactly the
// parts with the regression history (the OpenVINO device default, the
// device-scoped FP16 knob), and they are testable on ANY machine.
//
// They live in a header rather than ort_engine.cpp's anonymous namespace so a
// test TU can include them; ort_engine.cpp is the only production caller.

#include <charconv>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "turbo_ocr/backend/engine_mode.h" // backend::EpConfig
#include "turbo_ocr/base/env_utils.h"      // env::* — every read is recorded

namespace turbo_ocr::engine::ep_options {

// Thread count from `var`, defaulting to hardware_concurrency and never below
// 1. Used for BOTH pools an operator may size: the shared global ORT pool
// (ORT_GLOBAL_THREADS) and XNNPACK's own intra-op pool (ORT_NUM_THREADS). A
// non-numeric or non-positive value is IGNORED rather than clamped, so a typo
// falls back to the default instead of pinning the pool to a single thread.
inline int env_thread_count(const char *var) {
  int n = static_cast<int>(std::thread::hardware_concurrency());
  n = env::env_int(var, n, 1, 4096);
  if (n <= 0)
    n = 1; // hardware_concurrency() is permitted to answer 0
  return n;
}

// `s` as a WHOLE-STRING non-negative decimal ordinal, or -1 for anything else.
// Whole-string on purpose: EpConfig.device also carries OpenVINO device NAMES
// ("CPU", "GPU", "GPU.1", "NPU"), and a prefix parse would read "0GPU" as
// device 0. Only a string that is nothing but digits is an ordinal.
[[nodiscard]] inline int device_ordinal(std::string_view s) noexcept {
  int v = 0;
  const char *const first = s.data();
  const char *const last = s.data() + s.size();
  const auto r = std::from_chars(first, last, v);
  if (r.ec != std::errc{} || r.ptr != last || v < 0)
    return -1;
  return v;
}

// Device ordinal for the providers that take one (CUDA / MIGraphX / ROCm /
// DML). Three rungs, most specific first:
//
//   1. EpConfig.device_id — an explicit ordinal from the caller. The sentinel
//      is -1, NOT 0, so an explicit "device 0" beats a stale CUDA_DEVICE_ID in
//      the environment instead of being indistinguishable from "unset".
//   2. EpConfig.device parsed as an ordinal. This is the knob operators
//      actually set (TURBO_EP_DEVICE -> BackendConfig::ep.device, and the
//      Python `device=` argument): it is a STRING because OpenVINO wants a
//      device NAME, but every other provider wants an ordinal, and a
//      device-selection knob that works on one vendor arm and silently no-ops
//      on three is precisely the per-backend divergence this tree forbids.
//      A non-numeric value (an OpenVINO name) falls through untouched.
//   3. `var` from the environment (CUDA_DEVICE_ID / ROCM_DEVICE_ID / ...),
//      else 0.
[[nodiscard]] inline int device_id_for(const backend::EpConfig &ep,
                                       const char *var) {
  if (ep.device_id >= 0)
    return ep.device_id;
  if (const int ord = device_ordinal(ep.device); ord >= 0)
    return ord;
  return env::env_int(var, 0, 0, 4096);
}

// Merge a "k1=v1,k2=v2" provider-option string into `out`; later entries win
// over earlier ones AND over whatever the caller already computed. Entries with
// no '=' or an empty key are skipped rather than rejected, so one malformed
// pair does not discard the rest of the string.
inline void merge_option_string(const char *raw,
                                std::unordered_map<std::string, std::string> &out) {
  std::string spec(raw);
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string kv = spec.substr(pos, comma - pos);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && eq > 0)
      out[kv.substr(0, eq)] = kv.substr(eq + 1);
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
}

// Which OpenVINO devices accept precision=FP16.
//
// The knob is DEVICE-SCOPED, and getting that wrong does not degrade quietly —
// it fails EP load outright. Measured on a Core Ultra 7 265T:
//   [OpenVINO] Unsupported inference precision is selected.
//              CPU only supports FP32, ACCURACY.
// so precision=FP16 on the CPU plugin takes the whole provider down and the
// engine falls back reporting "openvino unavailable", which reads like a
// missing provider rather than a bad option. Only GPU/NPU take FP16; AUTO may
// resolve to CPU, so it is treated as unsafe too.
[[nodiscard]] inline bool device_takes_fp16(std::string_view dev) noexcept {
  return dev.rfind("GPU", 0) == 0 || dev.rfind("NPU", 0) == 0;
}

// The complete OpenVINO provider-option map, exactly as it is handed to
// AppendExecutionProvider("OpenVINO", ...).
//
// ORDER MATTERS and is the fix for a real hole: OPENVINO_EP_OPTS is merged
// BEFORE the device-scoped decisions are taken, so the two guardrails read the
// device that will actually be used. Merging last (as this did originally) let
// OPENVINO_EP_OPTS="device_type=GPU" reach the OpenVINO GPU plugin with the
// accuracy warning below never firing, and left OPENVINO_DEVICE=GPU +
// OPENVINO_EP_OPTS="device_type=CPU" carrying precision=FP16 onto the CPU
// plugin — i.e. both guardrails were bypassable by the one knob documented as
// "set values win". The escape hatch still wins over every computed value; it
// just no longer wins over the safety checks derived from it.
[[nodiscard]] inline std::unordered_map<std::string, std::string>
openvino_options(const backend::EpConfig &ep) {
  std::unordered_map<std::string, std::string> ov;

  // An explicit EpConfig.device wins over the env var; env remains the
  // control for a plain ORT_EP=openvino run.
  // DEVICE DEFAULT IS "CPU", NOT "AUTO" — deliberately, and it is a
  // correctness choice rather than a performance one.
  //
  // Measured on a Core Ultra 7 265T (FUNSD-50, PP-OCRv6 tiny det+rec):
  //     device=CPU  -> F1 85.78%  (exactly the MLAS/reference number)
  //     device=GPU  -> F1 62-67%  (detection loses ~23% of boxes on EVERY
  //                                page; recall 86.7% -> 53.9%)
  // The GPU deficit is NOT ours: identical code/EP/models on CPU reproduce
  // the reference bit-for-bit, and it survives disabling ORT graph fusions,
  // forcing FP32/ACCURACY precision, disabling Winograd convolution, and
  // eliminating dynamic shapes. It matches open, unresolved OpenVINO GPU
  // plugin issues for PaddleOCR-family models (openvinotoolkit/openvino
  // #29364, #28897) which Intel has not root-caused.
  //
  // "AUTO" was the old default and is unsafe here precisely because it MAY
  // select the GPU — that would hand an operator a silent 20-point accuracy
  // loss with no signal at all. Opting into GPU stays possible; it just has
  // to be explicit, and it says so.
  if (!ep.device.empty())
    ov["device_type"] = ep.device;
  else
    ov["device_type"] = env::env_or("OPENVINO_DEVICE", "CPU");

  // An OPERATOR-SET precision is passed through as asked (the caller named a
  // value; silently overriding it is not this layer's job — apply_openvino_ep()
  // warns when it cannot work on the chosen device).
  if (const std::string p = env::env_or("OPENVINO_PRECISION", ""); !p.empty())
    ov["precision"] = p;
  if (const std::string c = env::env_or("OPENVINO_CACHE_DIR", ""); !c.empty())
    ov["cache_dir"] = c;

  // Escape hatch: OPENVINO_EP_OPTS="k1=v1,k2=v2" is merged over everything
  // computed above, so any provider option this build does not model explicitly
  // is still reachable without a recompile (e.g. INFERENCE_PRECISION_HINT,
  // execution_mode, num_streams, load_config). Set values win.
  if (const std::string raw = env::env_or("OPENVINO_EP_OPTS", ""); !raw.empty())
    merge_option_string(raw.c_str(), ov);

  // OpenVINO is the one EP with a genuine precision switch: FP16 here costs no
  // graph build and no separate model file. DERIVED from ep.fp16, so it only
  // fills a key nobody set, and only on a device that can take it (above).
  const bool have_precision = ov.find("precision") != ov.end();
  if (ep.fp16 && !have_precision && device_takes_fp16(ov["device_type"]))
    ov["precision"] = "FP16";

  return ov;
}

} // namespace turbo_ocr::engine::ep_options
