#include "turbo_ocr/engine/trt/engine_loader.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

namespace turbo_ocr::engine {

namespace {

class CacheLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINFO)
      return;
    // TensorRT calls loggers from its internal worker threads; serialize the
    // chained cerr write so concurrent builds don't interleave/tear output.
    static std::mutex log_mu;
    std::lock_guard<std::mutex> lk(log_mu);
    std::cerr << "[TRT] " << msg << '\n';
  }
};

// The logger must outlive the IRuntime AND every engine deserialized from it
// (TensorRT requirement). An intentionally-leaked singleton guarantees that
// regardless of static-destruction order — a member of the static CacheState
// could be destroyed while an engine (holding the runtime alive via its deleter)
// still exists, leaving the live runtime with a dangling logger (use-after-free).
nvinfer1::ILogger &cache_logger() {
  static CacheLogger *logger = new CacheLogger();
  return *logger;
}

// The IRuntime is constructed on first use and shared via each engine's deleter,
// so it is guaranteed to outlive every engine deserialized from it regardless of
// static-destruction order (TensorRT requires the runtime to outlive its engines).
struct CacheState {
  std::mutex mu;
  std::shared_ptr<nvinfer1::IRuntime> runtime;
};

CacheState &state() {
  static CacheState s;
  return s;
}

std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.good())
    return {};
  const std::streampos pos = f.tellg();
  if (pos < 0)
    return {};
  std::vector<char> buf(static_cast<std::size_t>(pos));
  f.seekg(0, std::ios::beg);
  f.read(buf.data(), pos);
  return buf;
}

} // namespace

// Engines are ALWAYS private: every caller deserializes its own ICudaEngine.
//
// Sharing one engine across the pool's execution contexts is the "supported"
// TensorRT pattern, but in practice N contexts enqueuing concurrently on one
// MULTI-PROFILE engine corrupt output under load — rec (small/large profiles)
// decodes text boxes to blank while single-profile det is unaffected (the exact
// "det finds the boxes, rec blank" failure). One execution context per engine is
// the structurally-safe shape and matches the pre-cache design; the VRAM saved
// by sharing is irrelevant on a GPU-compute-bound stack. There is no concurrency-
// safe way to share that keeps the throughput, so sharing is not offered.
std::shared_ptr<nvinfer1::ICudaEngine>
load_engine(const std::string &trt_path) {
  // Read the engine blob OUTSIDE the cache lock — only runtime creation +
  // deserialize need serialization, so concurrent cold starts don't queue on
  // each other's disk I/O.
  const std::vector<char> blob = read_file(trt_path);
  if (blob.empty()) {
    std::cerr << "[TRT] Error loading engine file: " << trt_path << '\n';
    return nullptr;
  }

  CacheState &s = state();
  std::lock_guard<std::mutex> lock(s.mu);

  if (!s.runtime) {
    s.runtime.reset(nvinfer1::createInferRuntime(cache_logger()));
    if (!s.runtime) {
      std::cerr << "[TRT] Failed to create runtime for: " << trt_path << '\n';
      return nullptr;
    }
  }

  nvinfer1::ICudaEngine *raw =
      s.runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!raw) {
    std::cerr << "[TRT] Failed to deserialize engine: " << trt_path << '\n';
    return nullptr;
  }

  // Capture the shared runtime in the deleter so it outlives this engine no
  // matter the static-destruction order.
  return std::shared_ptr<nvinfer1::ICudaEngine>(
      raw, [rt = s.runtime](nvinfer1::ICudaEngine *p) { delete p; });
}

} // namespace turbo_ocr::engine
