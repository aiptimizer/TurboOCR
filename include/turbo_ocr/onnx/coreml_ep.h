#pragma once

// coreml_ep.h — the CoreML execution-provider environment policy, in ONE place.
//
// Three call sites append ORT's CoreML provider: OrtEngine (every det/rec/cls
// session), the CPU layout stage, and the form-field detector. Each had grown
// its own raw read of COREML_FLAGS plus a `strtoul`, and its own reading of
// DISABLE_COREML. ort_engine.cpp's own comment already argued the point —
// "two hand-rolled copies of the same getenv could disagree about which one
// happened" — while a third copy sat in src/analysis/layout/. A knob whose
// meaning depends on which stage read it is not a knob.
//
// Lives under include/turbo_ocr/onnx/ because it is ORT-provider policy, and
// src/analysis/* sits on the ONNX layer by the rule in src/README.md. It names
// no ORT type, so including it costs nothing.

#include <cstdint>
#include <cstdlib>
#include <string>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::engine {

/// COREML_FLAG_USE_CPU_AND_GPU — CPU + GPU, which is also the only setting that
/// lets ORT reach the Neural Engine. The default at every call site.
inline constexpr std::uint32_t kCoreMlCpuAndGpu = 0x020;

/// True when the operator has forced CoreML off (DISABLE_COREML=1), so a
/// provider regression can be ruled out in the field without a rebuild.
///
/// This is deliberately distinct from "the append failed": collapsing the two
/// makes "suppressed on request" and "CoreML is broken here" the same false,
/// which is the ambiguity the startup warning exists to remove.
[[nodiscard]] inline bool coreml_disabled_by_env() {
  return env::env_enabled("DISABLE_COREML");
}

/// COREML_FLAGS as ORT's provider flag word. Parsed base-0, so 0x020 and 32 are
/// both accepted — the constants are published in hex and operators paste them
/// that way. Malformed input keeps `def` rather than becoming 0, which would
/// silently mean "CPU only" and look exactly like CoreML working badly.
[[nodiscard]] inline std::uint32_t coreml_flags(std::uint32_t def = kCoreMlCpuAndGpu) {
  const std::string raw = env::env_or("COREML_FLAGS", "");
  if (raw.empty()) return def;
  char *end = nullptr;
  const unsigned long v = std::strtoul(raw.c_str(), &end, 0);
  if (end == raw.c_str() || *end != '\0') return def;
  return static_cast<std::uint32_t>(v);
}

} // namespace turbo_ocr::engine
