#pragma once

#include <memory>
#include <string>

#include <NvInfer.h>

// TensorRT engine loader.

//
// Each GPU pool worker builds its own OcrPipeline whose recognizers deserialize
// their OWN private ICudaEngine. Sharing one engine across the pool to save the
// N× weight VRAM is the "supported" TensorRT pattern, but it corrupts output
// under concurrency: N execution contexts enqueuing on one MULTI-PROFILE engine
// make rec decode text boxes to blank while single-profile det is unaffected
// (the "boxes found, text blank" failure — see engine_cache.cpp). One engine per
// context is the structurally-safe shape, so engines are deliberately never
// shared.
//
// load_engine() just deserializes a .trt blob behind a process-global, lazily
// created IRuntime that is captured in every engine's shared_ptr deleter, so the
// runtime outlives its engines regardless of static-destruction order.

namespace turbo_ocr::engine {

// Deserialize the engine at `trt_path` and return it. NOT cached or shared —
// every call yields a fresh, independent engine. Thread-safe. Returns nullptr on
// read/deserialize failure. The caller MUST create its own
// createExecutionContext() and never share that context across threads.
[[nodiscard]] std::shared_ptr<nvinfer1::ICudaEngine>
load_engine(const std::string &trt_path);

} // namespace turbo_ocr::engine
