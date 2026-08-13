#pragma once

// PER-STAGE DEVICE PLACEMENT — one policy, every backend.
//
// A machine with more than one accelerator is the normal case now, not the
// exotic one: an Intel Core Ultra has a CPU, an iGPU and an NPU, and a desktop
// adds a discrete GPU on top. Pinning the whole pipeline to a single device
// leaves the rest idle, and the right device is not the same for every stage —
// measured on an Ultra 9 285K, det_tiny at 640x640 runs 5.8 ms on the CPU,
// 6.1 ms on the iGPU and 14.2 ms on the NPU, and recognition is a different
// shape again (small crops, high batch) so it does not follow detection.
//
// Two independent wins, and the second is the interesting one:
//   1. Put each stage where it is individually fastest.
//   2. Put stages on DIFFERENT devices so they overlap. A pipeline whose det
//      and rec both take 10 ms on the best device is slower end-to-end than one
//      where det takes 14 ms on the NPU *while* rec runs on the GPU — the
//      stages are pipelined across pages, so total throughput is bounded by the
//      slowest device, not by the sum.
//
// WHAT LIVES HERE is the naming and precedence — the part that must not be
// spelled differently per vendor. WHAT DOES NOT is the mapping from a device
// STRING to a vendor device handle: "NPU" means something to OpenVINO and
// nothing to Metal, so each backend parses the string it understands and
// ignores the rest. A backend that supports exactly one device correctly
// ignores all of this.
//
// PRECEDENCE, highest first:
//   1. <STAGE>_DEVICE   e.g. DET_DEVICE=NPU     — this stage only
//   2. the backend's own global (OV_DEVICE for Intel, CUDA_DEVICE_ID, ...)
//   3. the backend's built-in default
//
// So the existing single-device configuration keeps working untouched: with no
// <STAGE>_DEVICE set, every stage resolves to exactly what it resolved to
// before this header existed.

#include <string>
#include <string_view>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::backend {

// The pipeline stages that own an inference engine. Deliberately NOT
// capability::CapabilityId: that enum covers the OPTIONAL capabilities a server
// may or may not load (Layout, Table, Formula, DocOrientation), whereas
// detection and recognition are always present and have no capability bit. The
// two lists overlap at Layout and are answering different questions.
enum class StageKind { Detection, Recognition, Classification, Layout };

// Env prefix per stage, matching the convention already used by DET_MODEL /
// REC_ONNX / CLS_BATCH — a reader who knows DET_BOX_THRESH can guess
// DET_DEVICE.
[[nodiscard]] constexpr std::string_view stage_env_prefix(StageKind k) noexcept {
  switch (k) {
  case StageKind::Detection:      return "DET";
  case StageKind::Recognition:    return "REC";
  case StageKind::Classification: return "CLS";
  case StageKind::Layout:         return "LAYOUT";
  }
  return "DET";
}

[[nodiscard]] inline std::string stage_kind_name(StageKind k) {
  switch (k) {
  case StageKind::Detection:      return "detection";
  case StageKind::Recognition:    return "recognition";
  case StageKind::Classification: return "classification";
  case StageKind::Layout:         return "layout";
  }
  return "detection";
}

// The per-stage override, uppercased, or "" when unset.
//
// Returning the raw string rather than a parsed enum is the whole point: this
// header cannot know a vendor's device set without depending on every vendor
// header, which is exactly the layering the architecture check forbids. The
// caller parses what it understands.
[[nodiscard]] inline std::string stage_device_override(StageKind k) {
  std::string var(stage_env_prefix(k));
  var += "_DEVICE";
  std::string v = turbo_ocr::env::env_or(var.c_str(), "");
  for (auto &c : v)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return v;
}

} // namespace turbo_ocr::backend
