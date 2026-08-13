#pragma once

// OpenVINOEngine — the Intel backend's IEngine (backend/engine.h), over
// OpenVINO Runtime. The Intel analogue of TrtEngine / OrtSession.
//
// ============================ PERFORMANCE SHAPE =============================
// The rebuild's performance gate says: "no compilation/allocation in the hot
// path (executables cached by (width,batch) at warmup, buffers pre-sized and
// reused)". OpenVINO's cost model makes that concrete:
//
//   * ov::Core::compile_model() is EXPENSIVE (graph transform + JIT/OCL build,
//     100 ms - seconds). It must happen at load()/warmup, never per request.
//   * A CompiledModel with STATIC shapes is materially faster than one left
//     dynamic on the GPU plugin (no per-call reshape, better kernel selection).
//   * ov::InferRequest is a heavyweight object owning its own scratch, and is
//     NOT thread-safe. One per engine instance per shape, created at warmup.
//
// So this engine keeps a `Variant` per PRIMARY-INPUT SHAPE:
//     Variant = { ov::CompiledModel, ov::InferRequest, staging buffers }
// built by prebuild() at load time from shapes the caller derives from the
// SHARED policy helpers (recognition::rec_shape_matrix over kRecWidthBuckets x
// kRecBatchLadder for rec; the cls ladder; the single layout canvas). run()
// then does a hash lookup and binds — it never compiles and never allocates.
//
// A shape that was not prebuilt falls back to ONE dynamic-shape Variant (kept
// deliberately, because unlike TensorRT the OpenVINO runtime handles dynamic
// dims natively) and increments shape_misses(). A non-zero shape_misses() after
// warmup is the signal that the prebuild matrix is wrong — it is exposed so a
// bring-up run can assert on it rather than silently paying reshape cost.
//
// ============================== caps() ======================================
//   io_space  = L0 when a RemoteContext over the allocator's Level Zero context
//               was established (USM pointers bindable in place, zero-copy);
//               Host otherwise (CPU plugin, or GPU plugin without interop). It
//               is reported HONESTLY after load(), and run() stages through
//               pre-allocated mirrors when a caller binds the other space, so a
//               degraded interop never corrupts memory.
//   async     = false. See the note on run() below — a correctness decision,
//               not an oversight.
//   caller_owns_outputs = true (we bind the caller's output buffers).
//     ONE documented extension, needed by PP-DocLayoutV3 and forbidden by
//     nothing in the seam: an entry of `outputs` whose `data == nullptr` means
//     "engine, you own this one — hand it back". The engine then leaves the
//     tensor to OpenVINO and returns it as an OutputLease (host-side, valid
//     until the next run() on this engine). That is how a stage reads a
//     DATA-DEPENDENT output shape (layout's NMS row count) without pre-sizing a
//     buffer, and how it avoids materialising outputs it never reads: layout's
//     ~48 MB mask is simply not mentioned, so OpenVINO keeps it internally and
//     nothing crosses the bus.
//   multi_io  = true  (layout is 3-in; bind by name).
//   dynamic_shapes = true.
//   graph / has_profiles = false; profiles()/graph() return nullptr. There is no
//               TRT-profile or CUDA-graph analogue and the seam explicitly says
//               a backend that lacks them must not fake them.
//
// OpenVINO headers stay behind a pImpl (like OrtSession) so TUs that only need
// the interface don't pull in <openvino/openvino.hpp>.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"

namespace turbo_ocr::intel {

class L0Allocator;

class OpenVINOEngine final : public backend::IEngine {
public:
  enum class DeviceType { CPU, GPU, NPU };

  // OV_DEVICE = CPU | GPU | NPU (case-insensitive). Anything else -> fallback.
  [[nodiscard]] static DeviceType device_from_env(DeviceType fallback = DeviceType::GPU);
  [[nodiscard]] static const char *device_name(DeviceType d) noexcept;

  // True when the OpenVINO runtime enumerates this device, i.e. it can actually
  // run inference on it. This is the availability gate; it is NOT the same
  // question as L0Allocator::has_device(), which asks whether ZERO-COPY
  // Level-Zero interop is possible. A GPU with no L0 context still runs — it
  // stages through host memory and caps() reports io_space = Host.
  [[nodiscard]] static bool device_available(DeviceType d);

  // `alloc` supplies the shared Level Zero context the RemoteContext is built
  // over, so USM device pointers are bindable as RemoteTensors. May be null (or
  // report has_device()==false) — then io_space is Host.
  OpenVINOEngine(DeviceType device, std::shared_ptr<L0Allocator> alloc);
  ~OpenVINOEngine() override;

  // --- IEngine ---------------------------------------------------------------
  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] backend::EngineCaps caps() const override;
  [[nodiscard]] const std::vector<std::string> &input_names() const override;
  [[nodiscard]] const std::vector<std::string> &output_names() const override;

  // Binds `inputs`/`outputs` on the Variant selected by inputs[0].shape and runs
  // the forward pass. SYNCHRONOUS: it returns only once the outputs are written.
  //
  // WHY SYNCHRONOUS (caps().async == false): OpenVINO's GPU plugin executes on
  // ITS OWN internal stream. Unless that stream is provably the same Level Zero
  // command queue as `queue`, an "async" return would be a data race — the
  // caller's contract is "sync the DeviceQueue, then read", and syncing a SYCL
  // queue says nothing about an OpenVINO request. Reporting async=true here
  // would be a silent correctness bug traded for a speculative win. run()
  // therefore (a) barriers `queue` so SYCL-written inputs have landed, (b)
  // infers, and (c) leaves outputs valid on return. Making this genuinely async
  // requires threading OV onto our L0 queue — README bring-up item 2; when that
  // is validated on hardware, flip this flag and drop the barrier.
  [[nodiscard]] bool run(const std::vector<backend::DeviceTensor> &inputs,
                         const std::vector<backend::DeviceTensor> &outputs,
                         std::vector<backend::OutputLease> &leases,
                         backend::DeviceQueue &queue) override;

  [[nodiscard]] backend::IProfiles *profiles() override { return nullptr; }
  [[nodiscard]] backend::IGraphCapture *graph() override { return nullptr; }

  // --- Warmup-only API (never called on the hot path) ------------------------
  //
  // Compile + cache one CompiledModel/InferRequest per entry of
  // `primary_input_shapes` (the shape of input 0; other inputs keep their model
  // shape). Callers pass shapes derived from the SHARED batching policy, so the
  // set of artefacts and the set of shapes the pipeline can produce are the same
  // list by construction. Returns the number of variants successfully built.
  std::size_t prebuild(const std::vector<std::vector<int64_t>> &primary_input_shapes);

  // Output shape the compiled model reports for `out_name` under a prebuilt
  // primary-input shape. This is how a stage SIZES its device buffers from the
  // model instead of guessing (rec sequence length, class-head width, layout
  // detection count). Empty when the shape was never prebuilt.
  [[nodiscard]] std::vector<int64_t>
  output_shape(const std::vector<int64_t> &primary_input_shape,
               const std::string &out_name) const;

  // Hot-path shapes that were not prebuilt (ran on the dynamic variant).
  // Should be 0 after a correct warmup; assert on it during bring-up.
  [[nodiscard]] std::size_t shape_misses() const noexcept;

  [[nodiscard]] bool is_loaded() const noexcept;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::intel
