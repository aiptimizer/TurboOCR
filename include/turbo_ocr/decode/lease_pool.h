#pragma once

// Bounded lease pool for resources that are expensive to create, cannot be
// shared between threads, and must never be created once per thread.
//
// The motivating case is the nvJPEG decoder: every instance pins ~190 MB of
// device memory and ~50 MB of host memory for the life of the process, and a
// `thread_local` instance on a 128-thread work pool therefore costs ~24 GB
// of VRAM once every thread has decoded one JPEG (GitHub #33). A pool of
// `capacity` instances, leased per decode, bounds that at capacity × cost
// regardless of how many threads exist.
//
// Semantics:
//  - Instances are constructed lazily by `factory` on first demand, never
//    beyond `capacity`, and are reused (LIFO) thereafter.
//  - `acquire()` blocks until an instance is free; `try_acquire_for()` gives
//    up after `timeout` so callers can fall back (e.g. to CPU decode) instead
//    of stalling a request behind a wedged decoder.
//  - A `Lease` is a move-only RAII handle; destroying it returns the instance
//    to the pool. The pool must outlive every lease (the server owns it for
//    the process lifetime).
//  - A factory that throws, or returns null, releases the reserved slot; the
//    pool never leaks capacity.
//
// Header-only and CUDA-free so it can be unit-tested with any T.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace turbo_ocr::decode {

template <class T>
class LeasePool {
 public:
  using Factory = std::function<std::unique_ptr<T>()>;

  class Lease {
   public:
    Lease() = default;
    Lease(Lease &&o) noexcept
        : pool_(std::exchange(o.pool_, nullptr)), obj_(std::move(o.obj_)) {}
    Lease &operator=(Lease &&o) noexcept {
      if (this != &o) {
        release();
        pool_ = std::exchange(o.pool_, nullptr);
        obj_ = std::move(o.obj_);
      }
      return *this;
    }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    ~Lease() { release(); }

    explicit operator bool() const noexcept { return obj_ != nullptr; }
    T &operator*() const noexcept { return *obj_; }
    T *operator->() const noexcept { return obj_.get(); }
    T *get() const noexcept { return obj_.get(); }

    // Return the instance to the pool now (idempotent).
    void release() noexcept {
      if (pool_ && obj_) pool_->put_back_(std::move(obj_));
      pool_ = nullptr;
      obj_.reset();
    }

   private:
    friend class LeasePool;
    Lease(LeasePool *pool, std::unique_ptr<T> obj) noexcept
        : pool_(pool), obj_(std::move(obj)) {}

    LeasePool *pool_ = nullptr;
    std::unique_ptr<T> obj_;
  };

  LeasePool(size_t capacity, Factory factory)
      : capacity_(capacity == 0 ? 1 : capacity), factory_(std::move(factory)) {
    idle_.reserve(capacity_);
  }

  LeasePool(const LeasePool &) = delete;
  LeasePool &operator=(const LeasePool &) = delete;

  // Block until an instance is available. Throws whatever `factory` throws.
  // Returns an empty Lease only if the factory returned null.
  [[nodiscard]] Lease acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      if (auto l = take_or_construct_(lk)) return std::move(*l);
      if (aborted_) return {};
      cv_.wait(lk);
    }
  }

  // Like acquire(), but gives up after `timeout` (nullopt). An empty Lease
  // inside the optional means the factory returned null.
  [[nodiscard]] std::optional<Lease>
  try_acquire_for(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      if (auto l = take_or_construct_(lk)) return l;
      if (aborted_) return Lease{};
      if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
        // One last look: a release may have raced the timeout.
        if (auto l = take_or_construct_(lk)) return l;
        return std::nullopt;
      }
    }
  }

  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] size_t created() const {
    std::lock_guard<std::mutex> lk(mu_);
    return created_;
  }
  [[nodiscard]] size_t idle() const {
    std::lock_guard<std::mutex> lk(mu_);
    return idle_.size();
  }

 private:
  // Caller holds `lk`. Returns an idle instance, or constructs a new one when
  // under capacity (factory runs with the lock RELEASED so a slow constructor
  // never stalls returns), or nullopt when the pool is exhausted.
  std::optional<Lease> take_or_construct_(std::unique_lock<std::mutex> &lk) {
    if (!idle_.empty()) {
      auto obj = std::move(idle_.back());
      idle_.pop_back();
      return Lease(this, std::move(obj));
    }
    if (created_ >= capacity_) return std::nullopt;
    ++created_;
    lk.unlock();
    std::unique_ptr<T> obj;
    try {
      obj = factory_();
    } catch (...) {
      lk.lock();
      --created_;
      cv_.notify_one();
      throw;
    }
    if (!obj) {
      lk.lock();
      --created_;
      // A factory that cannot produce an instance would make every waiter
      // spin on construction; mark the pool aborted so they return empty.
      aborted_ = true;
      cv_.notify_all();
      return Lease{};
    }
    lk.lock();
    return Lease(this, std::move(obj));
  }

  void put_back_(std::unique_ptr<T> obj) noexcept {
    {
      std::lock_guard<std::mutex> lk(mu_);
      idle_.push_back(std::move(obj));
    }
    cv_.notify_one();
  }

  const size_t capacity_;
  Factory factory_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<T>> idle_;
  size_t created_ = 0;
  bool aborted_ = false;
};

} // namespace turbo_ocr::decode
