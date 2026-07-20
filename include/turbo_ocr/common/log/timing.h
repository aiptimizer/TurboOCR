#pragma once

#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>

#include <cuda_runtime.h>

// Zero-overhead timing infrastructure.
// Enabled by setting ENABLE_TIMING=1 environment variable.
// Uses CUDA events for GPU timing (accurate to ~0.5us).

namespace turbo_ocr {

class PipelineTimer {
public:
  [[nodiscard]] static bool enabled() {
    static const bool val = [] {
      const char *env = std::getenv("ENABLE_TIMING");
      return env && env[0] == '1' && env[1] == '\0';
    }();
    return val;
  }

  PipelineTimer() = default;
  PipelineTimer(const PipelineTimer &) = delete;
  PipelineTimer &operator=(const PipelineTimer &) = delete;
  PipelineTimer(PipelineTimer &&) = delete;
  PipelineTimer &operator=(PipelineTimer &&) = delete;

  void init(cudaStream_t stream) {
    if (!enabled())
      return;
    stream_ = stream;
    cudaEventCreate(&evt_start_);
    cudaEventCreate(&evt_stop_);
    initialized_ = true;
  }

  ~PipelineTimer() noexcept {
    if (initialized_) {
      cudaEventDestroy(evt_start_);
      cudaEventDestroy(evt_stop_);
    }
  }

  // Record the start of a named stage on the GPU timeline
  void gpu_start(const char *name) {
    if (!enabled())
      return;
    current_name_ = name;
    cudaEventRecord(evt_start_, stream_);
  }

  // Record the end of a named stage and print elapsed time
  void gpu_stop() {
    if (!enabled())
      return;
    cudaEventRecord(evt_stop_, stream_);
    cudaEventSynchronize(evt_stop_);
    float ms = 0;
    cudaEventElapsedTime(&ms, evt_start_, evt_stop_);
    std::cout << std::format("[TIMING] {}: {} ms", current_name_, ms) << '\n';
    total_ms_ += ms;
  }

  // CPU wall-clock start
  void cpu_start(const char *name) {
    if (!enabled())
      return;
    current_name_ = name;
    cpu_start_ = std::chrono::high_resolution_clock::now();
  }

  // CPU wall-clock stop and print
  void cpu_stop() {
    if (!enabled())
      return;
    auto end = std::chrono::high_resolution_clock::now();
    float ms =
        std::chrono::duration<float, std::milli>(end - cpu_start_).count();
    std::cout << std::format("[TIMING] {}: {} ms", current_name_, ms) << '\n';
    total_ms_ += ms;
  }

  // Print total accumulated time
  void print_total() {
    if (!enabled())
      return;
    std::cout << std::format("[TIMING] TOTAL pipeline: {} ms", total_ms_) << '\n';
  }

  // Reset accumulated total
  void reset() { total_ms_ = 0; }

private:
  cudaStream_t stream_ = 0;
  cudaEvent_t evt_start_ = nullptr;
  cudaEvent_t evt_stop_ = nullptr;
  bool initialized_ = false;
  const char *current_name_ = "";
  float total_ms_ = 0;
  std::chrono::high_resolution_clock::time_point cpu_start_;
};

} // namespace turbo_ocr
