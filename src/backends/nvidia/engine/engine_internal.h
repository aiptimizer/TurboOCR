#pragma once

// Internals shared by the TRT engine-builder TUs (onnx_to_trt.cpp,
// trt_engine_cache.cpp, trt_profiles.cpp). Not a public API.

#include <string>

#include <NvInfer.h>

namespace turbo_ocr::engine::detail {

// Effective det profile MAX for the engine being built. The builder is
// process-level (no per-model ServerConfig handle), so it resolves the resize
// policy the same way the detector ctor does — read_det_resize() applies the
// per-model default base (kDetResizeDefault, max_side_limit=1280) and any
// DET_* env overrides, then effective_det_max_side() folds in DET_MAX_SIDE
// and clamps. The detector (paddle_det.cpp / ort_paddle_det.cpp) sizes its
// pinned input buffers from effective_det_max_side(read_det_resize(cfg.resize));
// because every catalog row uses kDetResizeDefault as its base, the two
// agree field-for-field, so engine profile MAX == runtime buffer MAX. They can
// only diverge if a future model ships a non-default max_side_limit AND no
// DET_MAX_SIDE/DET_MAX_SIDE_LIMIT env is set — when that happens, set the
// matching DET_MAX_SIDE_LIMIT env so both sides see it.
[[nodiscard]] int read_det_effective_max_side();

// OPT batch size for the det optimization profile. Profile range is [1,8];
// OPT picks which batch TRT tunes tactics for. Default 4 balances single-image
// (batch 1) latency against the dynamic-batch throughput path. Set 1 to favor
// single-image latency, up to 8 to favor batched throughput.
[[nodiscard]] int read_det_opt_batch();

// TensorRT builder optimization level: 0..5 (TRT 10 range). Higher = better
// kernel selection at the cost of build time. Default 5. Operators on small
// instances or with strict cold-start budgets can drop to 3 (build ~3-5×
// faster, runtime regression typically <5%).
[[nodiscard]] int read_trt_opt_level();
// TRT_FP16 (default 1). 0 = build fp32 engines — the escape hatch for graphs
// whose fp16 foreign-node compilation fails on this TRT (see trt_engine_cache.cpp).
[[nodiscard]] bool read_trt_fp16();

// Add the per-model-type optimization profile(s) to config. Returns false on
// an unexpected input tensor (multi-input models dispatch by name). The rec
// type may add extra static graph profiles + setMaxAuxStreams(0) when CUDA
// graphs are opted into.
[[nodiscard]] bool add_optimization_profiles(nvinfer1::IBuilder &builder,
                                             nvinfer1::INetworkDefinition &network,
                                             nvinfer1::IBuilderConfig &config,
                                             const std::string &type);

} // namespace turbo_ocr::engine::detail
