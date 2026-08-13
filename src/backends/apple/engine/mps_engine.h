#pragma once

// MpsEngine — the Apple IEngine (backend/engine.h). Runs det / rec / cls (and,
// later, layout) forward passes on the GPU via MPSGraph, built from the Python
// ONNX export (graph.json + weights.bin) by the proven translator in
// src/backends/apple/engine/mps_rec_build.h (~25 ops, bit-accurate to ORT; the
// standalone tools/probes/apple/mps_*.mm probes share it through a forwarding header).
// I/O is device-resident:
// inputs and outputs are caller-owned MTLBuffers (bound zero-copy as
// MPSGraphTensorData), and work is encoded onto the caller's MetalDeviceQueue's
// command buffer so a whole image's warp+rec+argmax can share ONE command buffer
// (the residency win — tools/probes/apple/mps_ocr.mm:151).
//
// caps(): io_space = Metal, async = true, caller_owns_outputs = true, static
// shapes (a compiled MPSGraphExecutable per batch size, cached), graph =
// transparent (MPSGraph fuses/plans internally — no external capture handle).
//
// Optional GPU argmax head: enable_argmax_head() appends reductionArgMaximum +
// reductionMaximum over the class axis so run() returns the tiny [B,T] token
// indices + scores instead of the full [B,T,C] logits — exactly the recognizer
// path in tools/probes/apple/mps_ocr.mm:119-120 that keeps only ~14 KB crossing to the host.
//
// load(model_path): model_path is the EXPORT DIRECTORY (graph.json + weights.bin),
// the artefact tools/modelgen/mps_export_rec.py produces.

#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"

namespace turbo_ocr::apple {

class MpsEngine final : public backend::IEngine {
public:
  MpsEngine();
  ~MpsEngine() override;

  [[nodiscard]] bool load(const std::string &model_path) override;

  [[nodiscard]] backend::EngineCaps caps() const override;

  [[nodiscard]] const std::vector<std::string> &input_names() const override {
    return input_names_;
  }
  [[nodiscard]] const std::vector<std::string> &output_names() const override {
    return output_names_;
  }

  [[nodiscard]] bool run(const std::vector<backend::DeviceTensor> &inputs,
                         const std::vector<backend::DeviceTensor> &outputs,
                         std::vector<backend::OutputLease> &leases,
                         backend::DeviceQueue &queue) override;

  // --- Apple extension: GPU argmax head -------------------------------------
  // When enabled, the compiled graph targets (arg-max index, max score) over
  // `class_axis` (default 2, the [B,T,C] class dim) instead of the raw logits;
  // run() then expects two output buffers (I32 indices, F32 scores) shaped [B,T].
  void enable_argmax_head(bool on, int class_axis = 2);

  // --- Apple extension: FP16 compute ----------------------------------------
  // Build the graph with FP16 scalars/intermediates (the feed and the final
  // output stay FP32 — mps_rec_build.h casts at both ends). This is a
  // DEVICE-SPECIFIC precision knob, not a policy: MPSGraph on Apple silicon runs
  // FP16 matmul/conv on dedicated hardware paths. Must be set BEFORE the first
  // prepare()/run(); flips the executable cache.
  void set_fp16(bool on);
  [[nodiscard]] bool fp16() const noexcept { return fp16_; }

  // Build + compile (or fetch the cached) executable for `batch`, updating
  // time_steps()/num_classes(), WITHOUT running. Lets a stage learn the output T
  // so it can size its argmax buffers before run() (mps_ocr.mm:131). Returns
  // false if the model isn't loaded.
  [[nodiscard]] bool prepare(int batch);

  // Model input {C, H, W} from the export template (batch stripped). Empty until
  // load(). Lets the detector/recognizer size input canvases (DETSZ, RECHxRECW).
  [[nodiscard]] std::vector<int> input_chw() const;

  // The recognition time-step count T of the loaded model (output dim 1), known
  // after the first build; -1 before. Lets the recognizer size its argmax
  // buffers without re-deriving the graph shape.
  [[nodiscard]] int time_steps() const noexcept { return time_steps_; }
  [[nodiscard]] int num_classes() const noexcept { return num_classes_; }

  // Opaque pimpl holding the ObjC handles (graph.json dict, weights, exe cache).
  // Public-but-incomplete so the .mm's build helper can name MpsEngine::Impl.
  struct Impl;

private:
  std::unique_ptr<Impl> p_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool argmax_head_ = false;
  bool fp16_ = false;
  int class_axis_ = 2;
  int time_steps_ = -1;
  int num_classes_ = -1;
};

} // namespace turbo_ocr::apple
