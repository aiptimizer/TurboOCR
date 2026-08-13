#pragma once

// IEngine — the ONE inference seam over every vendor's model runner:
// TrtEngine (native TensorRT), OrtEngine (ORT-CPU/CoreML/XNNPACK/DNNL),
// OrtSession (ORT-CUDA), MPSGraph (Apple), MIGraphX (AMD), OpenVINO (Intel).
//
// Design principle (distilled from the wf_engine.txt contract audit): the four
// things that genuinely differ between these runtimes — MEMORY SPACE of I/O,
// SYNC vs ASYNC completion, OUTPUT OWNERSHIP, and the SHAPE/PROFILE/GRAPH model
// — are answered UP FRONT by caps() instead of silently assumed, and the two
// leak-prone optimizers (TRT multi-profiles, CUDA-graph capture) are quarantined
// behind optional query interfaces that return nullptr when absent. That keeps
// the common infer() surface small while letting each backend keep its exact
// native contract:
//   * TrtEngine / OrtSession(CUDA) — caller-owned DEVICE buffers, async on the
//     caller's DeviceQueue, no implicit sync, zero-copy (engine writes into the
//     caller's output buffer).
//   * OrtEngine — HOST buffers, synchronous, single-IO float32, output COPIED
//     out (surfaced as an OutputLease rather than a bare pointer).
//   * MPSGraph / MIGraphX / OpenVINO — their own device space (Metal/Hip/L0),
//     reported via caps().io_space.
//
// No device SDK types appear here; DeviceTensor is a pure non-owning binding
// descriptor and every buffer lives wherever caps().io_space says.

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "turbo_ocr/backend/device_queue.h" // DeviceQueue, DeviceKind
#include "turbo_ocr/backend/image_view.h"    // DeviceKind

namespace turbo_ocr::backend {

// Element type of a bound tensor. F32 covers det/rec/cls/layout; I64 covers the
// token/position inputs of the formula AR decoder; U8 covers raw decoded pixels.
enum class DType : int { F32, F16, I64, I32, U8 };

// A NON-OWNING binding descriptor. The memory space is EXPLICIT (never inferred
// from the bare pointer), so a backend validates `data` against its own
// allocator / ORT MemoryInfo before binding it in place (TRT setTensorAddress /
// ORT CreateTensor-over-buffer — zero-copy). `data` is caller-owned unless the
// tensor was handed back as part of an OutputLease.
struct DeviceTensor {
  std::string name;          // IO name (multi-IO backends bind by name)
  void *data = nullptr;      // pointer valid in `space`
  DeviceKind space = DeviceKind::Host;
  DType dtype = DType::F32;
  int device_id = 0;         // for Cuda/Hip/L0 multi-device hosts
  std::vector<int64_t> shape;
};

// An engine-issued handle to output memory the ENGINE owns — for copy-out /
// host-view backends (OrtEngine.infer returns an owned vector;
// infer_batch_view returns a transient view). Expressed honestly with a STATED
// invalidation point instead of a raw pointer. Valid only until the next infer()
// on the same engine (or until the engine is destroyed). Never used by backends
// where caps().caller_owns_outputs == true (TRT/ORT-CUDA write into the caller's
// bound output buffers, so there is no lease).
struct OutputLease {
  std::string name;
  const void *data = nullptr; // read-only; in `space`
  DeviceKind space = DeviceKind::Host;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
};

// Static, cheap capability probe — the heart of the design. Callers branch on
// this rather than assuming a memory/stream/graph/ownership model.
struct EngineCaps {
  DeviceKind io_space = DeviceKind::Host; // where I/O buffers MUST live
  bool async = false;              // true => results not ready until the queue
                                   //         is synced; DeviceQueue meaningful
  bool caller_owns_outputs = true; // true: caller provides output buffers;
                                   // false: results come back via OutputLease
  bool multi_io = false;           // false => single-IO index-0 only (OrtEngine)
  bool dynamic_shapes = true;      // per-call shapes vs fixed
  bool per_shape_jit = false;      // dynamic shapes WORK but every NEW shape
                                   // pays a kernel compile (OpenVINO GPU
                                   // plugin, MPSGraph specialization). Callers
                                   // with an open shape set should close it —
                                   // see detection::snap_det_canvas_grid.
  bool graph = false;              // any CUDA-graph capability present (graph())
  bool has_profiles = false;       // TRT multi-optimization-profile (profiles())
  bool thread_safe_concurrent = false; // one instance callable from many threads
                                       // (ORT: yes; TRT: NO — one per thread)
  std::set<DType> dtypes{DType::F32};
};

class IProfiles;     // TRT-only optimization-profile control
class IGraphCapture; // CUDA-graph capture (explicit-slot OR transparent)

// The unified inference engine. One implementation per vendor stage runtime.
class IEngine {
public:
  virtual ~IEngine() = default;

  // Load a model from an on-disk artefact (a .trt plan, an .onnx / ORT model, a
  // CoreML/MPSGraph package, a MIGraphX .mxr, an OpenVINO IR dir — backend
  // decides). Returns false on recoverable load failure. Backend-specific
  // construction knobs (device_id, enable_cuda_graph, EP flags, profile layout)
  // are read by the concrete backend from its own config/env, NOT unioned into
  // this signature (see wf_engine.txt: "there is no common constructor").
  [[nodiscard]] virtual bool load(const std::string &model_path) = 0;

  [[nodiscard]] virtual EngineCaps caps() const = 0;

  [[nodiscard]] virtual const std::vector<std::string> &input_names() const = 0;
  [[nodiscard]] virtual const std::vector<std::string> &output_names() const = 0;

  // Core inference. Binds `inputs` (and `outputs`, when
  // caps().caller_owns_outputs) in place — zero-copy — and enqueues the forward
  // pass on `queue`. When caps().caller_owns_outputs == false, `outputs` may be
  // empty and results are returned through `leases` (host-owned copies/views).
  //
  // `queue` orders the work: for caps().async backends the call returns before
  // completion and the caller syncs the queue (or waits an event) before reading
  // outputs; for synchronous backends the results are ready on return and the
  // Host queue is a no-op. Returns false on recoverable failure; a backend MAY
  // fail-fast internally on an unrecoverable device fault (documented per
  // backend — TRT aborts on a sticky CUDA fault, ORT returns false).
  [[nodiscard]] virtual bool run(const std::vector<DeviceTensor> &inputs,
                                 const std::vector<DeviceTensor> &outputs,
                                 std::vector<OutputLease> &leases,
                                 DeviceQueue &queue) = 0;

  // --- Optional capability interfaces, obtained by query (nullptr if absent) --
  // Keeps profile/graph machinery OFF the common path so a backend lacking them
  // is never forced to fake them.
  [[nodiscard]] virtual IProfiles *profiles() { return nullptr; }
  [[nodiscard]] virtual IGraphCapture *graph() { return nullptr; }
};

// TRT native only (caps().has_profiles). select_profile() clears and re-binds
// all tensor addresses internally (TRT semantics); set_input_shape() specifies a
// dynamic dimension before run(). Backends without profiles return nullptr from
// IEngine::profiles().
class IProfiles {
public:
  virtual ~IProfiles() = default;
  [[nodiscard]] virtual int num_profiles() const = 0;
  virtual void select_profile(int idx, DeviceQueue &queue) = 0;
  [[nodiscard]] virtual bool set_input_shape(const std::string &name,
                                             const std::vector<int64_t> &dims) = 0;
};

// Unifies the two CUDA-graph philosophies behind one door WITHOUT collapsing
// them; mode() tells the caller which lifecycle it got.
//   ExplicitSlot (TRT): record at WARMUP with fixed shape + fixed I/O addresses,
//       returns a slot (or -1 => caller falls back to plain run()); replay via
//       launch(slot). Never capture during traffic.
//   Transparent (ORT-CUDA): begin_capture() just fixes the persistent binding;
//       the FIRST subsequent run() captures and later ones replay automatically —
//       launch() is then a no-op returning the sentinel slot 0.
class IGraphCapture {
public:
  virtual ~IGraphCapture() = default;
  enum class Mode { ExplicitSlot, Transparent };
  [[nodiscard]] virtual Mode mode() const = 0;

  [[nodiscard]] virtual int begin_capture(const std::vector<DeviceTensor> &inputs,
                                          const std::vector<DeviceTensor> &outputs,
                                          DeviceQueue &queue) = 0;
  [[nodiscard]] virtual bool launch(int slot, DeviceQueue &queue) = 0;
  virtual void reset() = 0; // drop captured graph/binding before buffers change
  [[nodiscard]] virtual std::vector<int64_t> output_shape(int slot) const = 0;
};

} // namespace turbo_ocr::backend
