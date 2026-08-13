#pragma once

// AneRecEngine — a CoreML mlprogram recognizer executor pinned to the Apple
// NEURAL ENGINE.
//
// WHY THIS IS APPLE-LOCAL AND NOT A POLICY OBJECT
// -----------------------------------------------
// Apple silicon has a THIRD compute engine besides CPU and GPU. It is reachable
// only through CoreML, only from a precompiled .mlpackage, and only at the
// batch shapes that package enumerates. That is a hardware fact with no analogue
// on CUDA/HIP/L0, so the engine wrapper lives in src/backends/apple/.
//
// What is NOT decided here: which lines go to which width bucket, and which
// static batch a chunk runs at. Both come from the SHARED planner
// (include/turbo_ocr/analysis/recognition/rec_batching.h). This class only
// reports `batch_shapes()` — the shapes its package physically supports — and
// executes one chunk. MpsRecognizer feeds those shapes to the shared
// plan_rec_batches() as `bucket_rungs`, so the routing rule stays identical to
// the GPU path (and to CPU/NVIDIA).
//
// WHY IT MAKES THE PIPELINE FASTER: the GPU is the bottleneck (det + warp + rec
// all contend for it). Moving the NARROW rec buckets — the bulk of the crops —
// onto the ANE frees GPU time. With a replica pool the two engines overlap
// naturally: replica A's ANE predict runs while replica B's det/warp occupies
// the GPU. No extra threading inside the stage.
//
// I/O: the crops are already in a Metal SHARED-storage buffer, so the host
// pointer is the same memory the warp kernel wrote — the MLMultiArray wraps it
// zero-copy. Only the [B,T] indices/scores come back.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace turbo_ocr::apple {

// Opaque handle for an in-flight two-phase run (begin_run/finish_run below).
struct AneTicket;

class AneRecEngine {
public:
  AneRecEngine();
  ~AneRecEngine();
  AneRecEngine(const AneRecEngine &) = delete;
  AneRecEngine &operator=(const AneRecEngine &) = delete;

  // Compile (cached process-wide) + load `mlpackage_path` on CPU+NeuralEngine,
  // read its enumerated batch shapes, and probe T with one warm predict.
  [[nodiscard]] bool load(const std::string &mlpackage_path, int rec_h, int rec_w);

  // Batch sizes the package supports, ascending. Handed to the SHARED planner.
  [[nodiscard]] const std::vector<int> &batch_shapes() const noexcept { return shapes_; }
  [[nodiscard]] int time_steps() const noexcept { return T_; }
  [[nodiscard]] bool is_ready() const noexcept { return ready_; }

  // Run `batch` (must be one of batch_shapes()) rows of a contiguous
  // [batch,3,rec_h,rec_w] float32 buffer. Writes batch*T indices and scores.
  [[nodiscard]] bool run(const float *crops, int batch, std::int32_t *idx_out,
                         float *score_out);

  // Two-phase run: begin_run() enqueues on the pooled batching service and
  // returns immediately; finish_run() blocks until idx_out/score_out are
  // written and reports whether they are valid (false => blanked, treat the
  // chunk as dropped). This exists so a round with SEVERAL ANE buckets puts
  // them ALL in flight before waiting — one blocking run() per bucket put two
  // full service round-trips on the round's critical path. `crops` and the
  // output buffers must stay alive and untouched until finish_run() returns.
  [[nodiscard]] std::shared_ptr<AneTicket> begin_run(const float *crops,
                                                     int batch,
                                                     std::int32_t *idx_out,
                                                     float *score_out);
  [[nodiscard]] bool finish_run(const std::shared_ptr<AneTicket> &t);

private:
  struct Impl;
  Impl *p_;
  std::vector<int> shapes_;
  int T_ = 0;
  int h_ = 0, w_ = 0;
  bool ready_ = false;
};

} // namespace turbo_ocr::apple
