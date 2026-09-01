#pragma once

// MIGraphXEngine — backend::IEngine over AMD MIGraphX (parse ONNX -> compile per
// gfx -> eval), with DEVICE-RESIDENT (hipMalloc) I/O. This is the AMD peer of
// TrtEngine / OrtSession(CUDA): the model forward pass runs on the GPU and reads
// its inputs from, and writes its outputs to, device memory — no host round-trip.
//
// Contract, expressed through caps() (see wf_engine.txt design):
//   io_space              = Hip   (inputs are hipMalloc device pointers)
//   async                 = true  (eval enqueued on the caller's hipStream via
//                                  MIGraphX run_async; caller syncs the queue)
//   caller_owns_outputs   = FALSE (MIGraphX eval ALLOCATES the output arguments
//                                  in its own device context and returns them;
//                                  we surface them as OutputLease device pointers
//                                  valid until the NEXT run() on this engine —
//                                  the same "engine-owned, next-call-invalidated"
//                                  lifetime as OrtEngine::infer_batch_view, but
//                                  in Hip space). Stage code feeds the lease's
//                                  device pointer straight into IKernels::argmax
//                                  etc., keeping data resident.
//   multi_io              = true  (layout: image + im_shape + scale_factor)
//   dynamic_shapes        = true  (accepts any shape; see SHAPE LADDER below —
//                                  MIGraphX itself compiles statically, so the
//                                  engine keeps a CACHE of statically-compiled
//                                  programs keyed by input shape)
//   thread_safe_concurrent= false (one program+context per thread, mirroring TRT;
//                                  RocmBackend::make_engine() hands out one each)
//   graph / has_profiles  = false (no CUDA-graph / TRT-profile analogs)
//
// The header is pImpl so translation units that merely hold an IEngine (stages,
// backend) never see <migraphx/migraphx.hpp> or <hip/*> — exactly how OrtSession
// hides ORT from nvcc TUs.
//
// ---------------------------------------------------------------------------
// SHAPE LADDER + THE PERFORMANCE GATE (read before changing load()/run())
// ---------------------------------------------------------------------------
// MIGraphX compiles a program for CONCRETE input shapes. A rec batch has a
// varying (batch, width) — so a naive engine would recompile inside run(), i.e.
// a multi-SECOND graph compile in the middle of a request. That is the exact
// failure the plan's PERFORMANCE GATE forbids ("no compilation/allocation in the
// hot path (executables cached by (width,batch) at warmup)").
//
// This engine therefore:
//   1. parse_onnx is deferred per-variant: load() records the model path and
//      compiles ONE program for the declared default shape;
//   2. `warmup(shapes)` compiles and caches the whole (batch x width) ladder the
//      stage will ever ask for, at STARTUP;
//   3. run() looks the compiled program up by the input shape signature — a hash
//      map hit, no allocation, no compile;
//   4. if a shape MISSES the cache, run() compiles it, but LOUDLY logs a
//      hot-path-compile warning naming the shape, so a missing warmup entry
//      shows up as a diagnosable log line rather than as a mystery latency
//      spike. Never let that warning stand in production — add the shape to the
//      stage's warmup ladder instead.
// The ladder itself is NOT invented here: the widths come from the SHARED
// recognition::kRecWidthBuckets table (rec_geometry.h), so AMD, NVIDIA/TRT
// profiles and the ORT path cannot drift apart.
//
// PER-GFX COMPILE CACHE (implemented in Impl::compile_variant): each compiled
// program is persisted with migraphx::save() to an .mxr under the shared
// ~/.cache/turbo-ocr dir (MIGRAPHX_ENGINE_CACHE overrides; "off" disables),
// keyed by (model path+size+mtime, shape signature, fp16, gcnArchName, hip
// driver+runtime version) — the same key discipline as the TRT engine cache.
// First start pays the 42 compiles; every later start is a migraphx::load().
//
// TODO(on-hardware):
//   * fp16. TrtEngine runs det/rec FP16. set_fp16(true) calls quantize_fp16;
//     validate per-stage accuracy against the CPU/CUDA golden before trusting it.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"

namespace turbo_ocr::amd {

using backend::DeviceQueue;
using backend::DeviceTensor;
using backend::EngineCaps;
using backend::IEngine;
using backend::OutputLease;

class MIGraphXEngine final : public IEngine {
public:
  explicit MIGraphXEngine(int device_id = 0);
  ~MIGraphXEngine() override;

  MIGraphXEngine(const MIGraphXEngine &) = delete;
  MIGraphXEngine &operator=(const MIGraphXEngine &) = delete;

  // One entry of the warmup ladder: the concrete dims of every input, in
  // input_names() order. Any shape the stage can ask run() for should appear
  // here, or run() will pay a compile on first use.
  struct ShapeVariant {
    std::vector<std::vector<std::int64_t>> input_dims;
    // Model input names aligned 1:1 with input_dims. REQUIRED for multi-input
    // models: the engine discovers names via get_parameter_shapes().names(),
    // whose order is NOT the graph's declared order, so positional pairing of
    // dims to discovered names mis-pins shapes (layout.onnx: im_shape pinned
    // to [1,3,800,800]). Empty is allowed for single-input models only.
    std::vector<std::string> input_names;
  };

  // Parse the ONNX at `model_path`, compile it for the current gfx target at the
  // model's declared input shape, and discover I/O names. Returns false on
  // parse/compile failure. Call warmup() next for any other shape you will use.
  [[nodiscard]] bool load(const std::string &model_path) override;

  // Compile and cache the given shape ladder. Call at STARTUP, after load().
  // Returns the number of variants successfully compiled. Every entry it
  // compiles is one graph compile that run() will not have to pay for.
  std::size_t warmup(const std::vector<ShapeVariant> &variants);

  // Optional: compile with fp16 quantization (call before load()/warmup()).
  void set_fp16(bool on) noexcept;

  // Diagnostics: how many times run() had to compile because a shape missed the
  // warmup cache. MUST be 0 in a warmed pipeline; a non-zero value is a
  // performance bug (a missing ladder entry), not a curiosity.
  [[nodiscard]] std::size_t hot_path_compiles() const noexcept;

  [[nodiscard]] EngineCaps caps() const override;
  [[nodiscard]] const std::vector<std::string> &input_names() const override;
  [[nodiscard]] const std::vector<std::string> &output_names() const override;

  [[nodiscard]] bool run(const std::vector<DeviceTensor> &inputs,
                         const std::vector<DeviceTensor> &outputs,
                         std::vector<OutputLease> &leases,
                         DeviceQueue &queue) override;

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

} // namespace turbo_ocr::amd
