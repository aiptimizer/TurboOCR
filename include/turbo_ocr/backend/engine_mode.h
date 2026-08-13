#pragma once

// EngineMode — the SECOND selection axis, orthogonal to the vendor Backend.
//
// Every vendor can reach its silicon two ways, and they trade the same thing in
// opposite directions:
//
//   Native ("ultra")  the vendor's own graph engine — a built TensorRT engine,
//                     an exported MPSGraph, a compiled OpenVINO blob, a
//                     MIGraphX program. Fastest steady state, but it needs an
//                     artifact that has to be BUILT (minutes, per GPU/driver)
//                     or exported offline, and it is useless without one.
//
//   Onnx ("fast")     the plain .onnx file straight through that vendor's ONNX
//                     Runtime execution provider, fp16 where the provider can
//                     do it, with NO graph build at all. Slower steady state,
//                     but it starts in seconds on the models already on disk.
//
// This header is the ONE place the two paths are named and the ONE place the
// vendor -> execution-provider mapping lives. Before it existed the fast path
// was reachable only by setting ORT_EP by hand and asking for the "cpu"
// backend, which is why `--backend apple` could load nothing but an MPSGraph
// export and simply failed on a normal models/ tree.
//
// DEDUP RULE: the Onnx path is ONE implementation for every vendor (host
// pre/post + ORT with a different provider string). Nothing here is per-vendor
// except the table below.

#include <string>
#include <string_view>

namespace turbo_ocr::backend {

enum class EngineMode {
  // Prefer Native, fall back to Onnx when the native artefact is absent. The
  // fallback is LOUD (logged with the reason) but not fatal — a missing engine
  // build must not stop a server that can serve from the .onnx today.
  Auto,
  Native, // "ultra"  — require the vendor graph engine; fail if unavailable
  Onnx,   // "fast"   — require the ONNX/EP path; never build a graph
};

[[nodiscard]] inline std::string_view engine_mode_name(EngineMode m) noexcept {
  switch (m) {
  case EngineMode::Native: return "native";
  case EngineMode::Onnx:   return "onnx";
  case EngineMode::Auto:   break;
  }
  return "auto";
}

// Accepts the mode names plus the two speed-oriented aliases operators reach
// for first ("ultra" / "fast"). Unknown strings return Auto so a typo degrades
// to the safe policy rather than hard-failing a boot; callers that want to
// reject a typo should compare against the parsed name.
[[nodiscard]] inline EngineMode parse_engine_mode(std::string_view s) noexcept {
  if (s == "native" || s == "ultra" || s == "graph" || s == "trt")
    return EngineMode::Native;
  if (s == "onnx" || s == "fast" || s == "ort")
    return EngineMode::Onnx;
  return EngineMode::Auto;
}

// The ONNX Runtime execution provider that carries a vendor's FAST path.
// These are exactly the ORT_EP strings engine::OrtEngine understands
// (src/onnx/ort_engine.cpp::apply_execution_provider).
//
//   nvidia -> "cuda"      CUDA EP: the ONNX graph on the GPU, no TRT build
//   amd    -> "migraphx"  AMD's go-forward EP (the ROCm EP was removed at 1.23)
//   intel  -> "openvino"  CPU/iGPU/Arc/NPU, and the ONE EP with a real fp16
//                         switch (precision=FP16) rather than an fp16 model
//   apple  -> "coreml"    ANE + GPU; CoreML runs fp16 natively on both
//   cpu    -> ""          default MLAS; "" means "leave ORT_EP unset"
//
// A vendor with no entry has no fast path and returns "".
[[nodiscard]] inline std::string_view onnx_provider_for(std::string_view backend) noexcept {
  if (backend == "nvidia") return "cuda";
  if (backend == "amd")    return "migraphx";
  if (backend == "intel")  return "openvino";
  if (backend == "apple")  return "coreml";
  return {}; // cpu (and anything unknown): the default CPU provider
}

// How a provider gets fp16 — they do NOT agree, and pretending they do is how
// "fp16" silently becomes fp32:
//
//   Provider   — the EP has a precision knob, so ONE session option is enough
//                and the fp32 .onnx on disk is all you need (OpenVINO).
//   Native     — the EP already computes in fp16 on its accelerator regardless
//                of the file's declared dtype (CoreML on ANE/GPU).
//   Model      — the EP runs whatever the graph says, so fp16 requires an fp16
//                MODEL. We do not build graphs here, so this means: use a
//                sibling *.fp16.onnx when one is on disk, else stay fp32 and
//                say so (CUDA / DirectML / MIGraphX).
//   None       — no meaningful fp16 (plain CPU/MLAS).
enum class Fp16Support { None, Provider, Native, Model };

[[nodiscard]] inline Fp16Support fp16_support_for(std::string_view provider) noexcept {
  if (provider == "openvino") return Fp16Support::Provider;
  if (provider == "coreml")   return Fp16Support::Native;
  if (provider == "cuda" || provider == "dml" || provider == "migraphx" ||
      provider == "rocm")
    return Fp16Support::Model;
  return Fp16Support::None;
}

// Everything the shared ONNX engine needs to reproduce a vendor's fast path.
// Passed explicitly rather than read from env inside the engine so that ONE
// process can hold two backends on two providers (the multi-backend binary is
// the whole point), and so a unit test can pin a provider without mutating
// process-global state.
struct EpConfig {
  std::string provider;   // "" => default CPU/MLAS

  // Provider-specific target, and the ONE device knob operators actually set
  // (TURBO_EP_DEVICE, the Python `device=` argument). It is a string because
  // OpenVINO wants a device NAME (AUTO|CPU|GPU|NPU|GPU.1), but the ordinal
  // providers read it too: a value that is nothing but digits is taken as the
  // CUDA/ROCm/DML ordinal (src/onnx/ep_options.h::device_id_for). Device
  // selection is generic policy — it must not work on one vendor arm and
  // silently no-op on the rest.
  std::string device;

  // Explicit CUDA/ROCm/DML ordinal, overriding both `device` and the per-
  // provider env var. Sentinel is -1, NOT 0: with 0 an explicit "device 0" is
  // indistinguishable from "unset" and loses to a stale CUDA_DEVICE_ID in the
  // environment.
  int device_id = -1;

  bool fp16 = true;       // honour fp16 wherever this provider supports it

  [[nodiscard]] bool is_default_cpu() const noexcept {
    return provider.empty() || provider == "cpu";
  }
};

} // namespace turbo_ocr::backend
