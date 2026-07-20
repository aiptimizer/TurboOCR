#pragma once

#include <chrono>
#include <concepts>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

#include "turbo_ocr/common/errors.h"

namespace turbo_ocr::pipeline {

/// A poolable pipeline must be destructible (we store unique_ptr<T>).
template <typename T>
concept Poolable = std::destructible<T>;

// ============================================================================
// PipelinePool<Pipeline> — thread-safe object pool with RAII scoped handles.
//
// Stores raw Pipeline* pointers internally; owns all pipeline instances.
// Supports an optional timeout (default: 30s). Pass std::chrono::seconds::max()
// for infinite blocking.
// ============================================================================
template <Poolable Pipeline>
class PipelinePool {
public:
  // RAII handle — auto-releases the pipeline back to the pool on destruction.
  class ScopedHandle {
  public:
    ScopedHandle(Pipeline *p, PipelinePool *pool) : pipeline_(p), pool_(pool) {}
    ~ScopedHandle() noexcept {
      if (pipeline_ && pool_)
        pool_->release(pipeline_);
    }

    ScopedHandle(const ScopedHandle &) = delete;
    ScopedHandle &operator=(const ScopedHandle &) = delete;

    ScopedHandle(ScopedHandle &&other) noexcept
        : pipeline_(other.pipeline_), pool_(other.pool_) {
      other.pipeline_ = nullptr;
      other.pool_ = nullptr;
    }
    ScopedHandle &operator=(ScopedHandle &&other) noexcept {
      if (this != &other) {
        if (pipeline_ && pool_)
          pool_->release(pipeline_);
        pipeline_ = other.pipeline_;
        pool_ = other.pool_;
        other.pipeline_ = nullptr;
        other.pool_ = nullptr;
      }
      return *this;
    }

    Pipeline *operator->() const { return pipeline_; }
    Pipeline &operator*() const { return *pipeline_; }
    [[nodiscard]] Pipeline *get() const { return pipeline_; }

  private:
    Pipeline *pipeline_;
    PipelinePool *pool_;
  };

  /// Construct pool from pre-built pipelines.
  /// @param pipelines  Vector of initialized (and optionally warmed-up) pipelines.
  /// @param timeout    Maximum time to wait in acquire(). Default 30s.
  /// @throws std::invalid_argument if pipelines is empty.
  PipelinePool(std::vector<std::unique_ptr<Pipeline>> pipelines,
               std::chrono::seconds timeout = std::chrono::seconds(30))
      : timeout_(timeout) {
    if (pipelines.empty()) [[unlikely]]
      throw std::invalid_argument("PipelinePool: cannot create pool with zero pipelines");
    for (auto &p : pipelines)
      pool_.push(p.release());
  }

  ~PipelinePool() noexcept {
    while (!pool_.empty()) {
      delete pool_.front();
      pool_.pop();
    }
  }

  PipelinePool(const PipelinePool &) = delete;
  PipelinePool &operator=(const PipelinePool &) = delete;
  PipelinePool(PipelinePool &&) = delete;
  PipelinePool &operator=(PipelinePool &&) = delete;

  /// Acquire a pipeline (blocks up to timeout). Returns RAII scoped handle.
  /// @throws turbo_ocr::PoolExhaustedError on timeout.
  [[nodiscard]] ScopedHandle acquire() {
    std::unique_lock lock(mutex_);
    if (timeout_ == std::chrono::seconds::max()) {
      cv_.wait(lock, [this] { return !pool_.empty(); });
    } else {
      if (!cv_.wait_for(lock, timeout_, [this] { return !pool_.empty(); }))
        throw turbo_ocr::PoolExhaustedError();
    }
    auto *p = pool_.front();
    pool_.pop();
    return ScopedHandle(p, this);
  }

  /// Bounded acquire for readiness/health probes: returns nullopt instead of
  /// throwing when no pipeline frees within `wait`. The pool's own timeout_
  /// may be infinite (CPU pool), so acquire() can never time out — a probe
  /// must use this so it can report NOT-ready rather than block a handler
  /// thread indefinitely under saturation.
  [[nodiscard]] std::optional<ScopedHandle>
  try_acquire_for(std::chrono::milliseconds wait) {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, wait, [this] { return !pool_.empty(); }))
      return std::nullopt;
    auto *p = pool_.front();
    pool_.pop();
    return ScopedHandle(p, this);
  }

  /// Release a pipeline back to the pool (called automatically by ScopedHandle).
  void release(Pipeline *p) {
    std::lock_guard lock(mutex_);
    pool_.push(p);
    cv_.notify_one();
  }

  [[nodiscard]] size_t size() const {
    std::lock_guard lock(mutex_);
    return pool_.size();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Pipeline *> pool_;
  std::chrono::seconds timeout_;
};

} // namespace turbo_ocr::pipeline
