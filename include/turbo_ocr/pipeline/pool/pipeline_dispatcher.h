#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <format>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "turbo_ocr/pipeline/pool/gpu_pipeline_pool.h"

// Header carries the templated submit API (must be inline); the worker loop,
// watchdog, and factories live in src/pipeline/pipeline_dispatcher.cpp.
namespace turbo_ocr::pipeline {

/// Wait for a future with a deadline. Returns the result if it resolves in
/// time, otherwise throws turbo_ocr::TimeoutError and ABANDONS the future:
/// the underlying task may still be running and writing into whatever it
/// captured. Callers MUST therefore submit work that owns its inputs by
/// value, so a timed-out task can safely outlive the request handler that
/// launched it. `timeout_ms <= 0` waits unbounded (legacy behaviour).
template <typename T>
T get_with_timeout(std::future<T> &future, long timeout_ms) {
  if (timeout_ms <= 0) return future.get();
  if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
      std::future_status::timeout)
    throw turbo_ocr::TimeoutError(std::format(
        "Inference exceeded the {} ms request deadline", timeout_ms));
  return future.get();
}

/// Work-queue dispatcher that keeps GPU pipelines permanently busy.
///
/// Each worker thread owns one GpuPipelineEntry and pulls tasks from a
/// shared FIFO queue.  Unlike the acquire/release PipelinePool pattern,
/// the GPU never idles waiting for HTTP round-trip overhead — while one
/// request's response is being serialised and sent, the worker is already
/// processing the next queued image.
class PipelineDispatcher {
public:
  /// `spec` records how each entry was built so a wedged entry can be rebuilt
  /// on its owning worker thread (see request_recycle). Pass {} to disable
  /// recycling (request_recycle becomes a no-op).
  explicit PipelineDispatcher(std::vector<std::unique_ptr<GpuPipelineEntry>> entries,
                              PipelineBuildSpec spec = {});

  ~PipelineDispatcher();

  PipelineDispatcher(const PipelineDispatcher &) = delete;
  PipelineDispatcher &operator=(const PipelineDispatcher &) = delete;

  /// Submit work that runs on a GPU worker thread.  Returns a future
  /// whose value is whatever the callable returns.
  ///
  /// The callable signature must be:  R fn(GpuPipelineEntry &)
  template <typename F>
  auto submit(F &&fn) -> std::future<std::invoke_result_t<F, GpuPipelineEntry &>> {
    return submit_with_deadline_(0, std::forward<F>(fn));  // no deadline (e.g. readiness probe)
  }

  /// submit() that stamps an absolute wall-clock deadline (ms; 0 = none) on the queued task,
  /// so a worker can skip it if the caller's deadline elapses while it waits in queue.
  template <typename F>
  auto submit_with_deadline_(long long deadline_ms, F &&fn)
      -> std::future<std::invoke_result_t<F, GpuPipelineEntry &>> {
    using R = std::invoke_result_t<F, GpuPipelineEntry &>;
    auto task = std::make_shared<std::packaged_task<R(GpuPipelineEntry &)>>(
        std::forward<F>(fn));
    auto future = task->get_future();
    {
      std::unique_lock lock(mutex_);
      if (queue_.size() >= max_queue_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (GPU queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push({[task = std::move(task)](GpuPipelineEntry &e) { (*task)(e); }, deadline_ms});
    }
    cv_.notify_one();
    return future;
  }

  /// Submit work and wait for it with a deadline. On timeout, throws
  /// turbo_ocr::TimeoutError and abandons the still-queued/running task.
  ///
  /// SAFETY: because the task may outlive this call, `fn` MUST capture every
  /// input it touches BY VALUE (no references/pointers into the caller's
  /// stack or request buffers). The result type R must likewise be safe to
  /// discard if the deadline elapses. `timeout_ms <= 0` waits unbounded.
  ///
  /// The callable signature must be:  R fn(GpuPipelineEntry &)
  template <typename F>
  auto submit_for(long timeout_ms, F &&fn)
      -> std::invoke_result_t<F, GpuPipelineEntry &> {
    const long long deadline = timeout_ms > 0 ? now_ms_() + timeout_ms : 0;
    auto future = submit_with_deadline_(deadline, std::forward<F>(fn));
    return get_with_timeout(future, timeout_ms);
  }

  /// Set the per-request deadline applied by submit_for_default and used by
  /// the stuck-worker watchdog. `ms <= 0` disables both (legacy unbounded
  /// blocking). Default 0; relaxed atomic, safe to set once at startup.
  void set_request_timeout_ms(long ms) noexcept {
    request_timeout_ms_.store(ms, std::memory_order_relaxed);
  }

  /// The configured per-request deadline (ms; 0 = unbounded). Lets callers that
  /// drive the dispatcher via raw submit() (e.g. the PDF orchestrator) bound
  /// their own future joins with the same deadline.
  [[nodiscard]] long request_timeout_ms() const noexcept {
    return request_timeout_ms_.load(std::memory_order_relaxed);
  }

  /// Submit + wait honouring the configured request deadline. When a positive
  /// timeout is set this is submit_for(timeout, fn) (throws TimeoutError on
  /// overrun); when disabled (0) it preserves today's submit(fn).get()
  /// unbounded-blocking behaviour exactly.
  ///
  /// Same SAFETY contract as submit_for: `fn` MUST own its inputs by value so
  /// an abandoned timed-out task can outlive the caller.
  template <typename F>
  auto submit_for_default(F &&fn)
      -> std::invoke_result_t<F, GpuPipelineEntry &> {
    long timeout = request_timeout_ms_.load(std::memory_order_relaxed);
    if (timeout > 0) return submit_for(timeout, std::forward<F>(fn));
    return submit(std::forward<F>(fn)).get();
  }

  [[nodiscard]] size_t worker_count() const noexcept { return workers_.size(); }

  /// Current GPU-queue depth, for the /metrics saturation gauge. Takes the
  /// same mutex as submit()/the worker loop so the read is consistent.
  [[nodiscard]] size_t queue_depth() const;

  /// Flag worker `worker_index`'s entry as wedged. Its owning worker rebuilds
  /// the OcrPipeline + CUDA stream from the build spec before its next task,
  /// rather than leaking the slot forever. Safe to call from any thread (e.g.
  /// a watchdog) and idempotent. No-op when the dispatcher was constructed
  /// without a build spec, or for an out-of-range index.
  void request_recycle(size_t worker_index) noexcept;

private:
  using WorkFn = std::function<void(GpuPipelineEntry &)>;
  // A queued task + the absolute wall-clock ms deadline (0 = none) by which its caller will
  // already have received a 504. Workers skip a task past its deadline rather than spend GPU
  // time on a result nobody will read — prevents goodput collapse / congestion under overload.
  struct QueuedWork { WorkFn fn; long long deadline_ms = 0; };

  static long long now_ms_() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void maybe_recycle_(GpuPipelineEntry &entry) noexcept;
  void watchdog_loop_();

  static constexpr std::chrono::milliseconds kWatchdogInterval{1000};
  // Extra slack beyond the request deadline before declaring a worker wedged,
  // so a task that finishes right at the deadline is never needlessly recycled.
  static constexpr long long kWatchdogGraceMs = 2000;

  std::queue<QueuedWork> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable watchdog_cv_;
  std::vector<std::thread> workers_;
  std::thread watchdog_;
  std::vector<std::shared_ptr<std::atomic<bool>>> recycle_flags_;
  std::vector<std::shared_ptr<std::atomic<long long>>> task_start_ms_;
  // Per-worker time (ms, 0 == none) the watchdog first requested a recycle for
  // the current task; gates the hard-kill so it only fires on a genuinely
  // unhonoured recycle (see watchdog_loop_).
  std::vector<std::shared_ptr<std::atomic<long long>>> recycle_requested_at_ms_;
  std::atomic<long> request_timeout_ms_{0};
  PipelineBuildSpec spec_;
  bool recycle_enabled_ = !spec_.det_model.empty();
  bool stop_{false};  // guarded by mutex_
  size_t max_queue_depth_ = 4096;
};

/// Factory: create, init, warmup GPU pipelines and wrap in a dispatcher.
// Build + warm exactly one pipeline (det/rec/cls + optional layout/doc_ori).
// Returns nullptr on a non-fatal init failure (logged).
[[nodiscard]] std::unique_ptr<GpuPipelineEntry> build_one_pipeline(
    int idx, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model,
    const std::string &layout_model, const std::string &doc_ori_model,
    const DetInferConfig &det_cfg);

[[nodiscard]] std::unique_ptr<PipelineDispatcher> make_pipeline_dispatcher(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "",
    const std::string &doc_ori_model = "",
    const DetInferConfig &det_cfg = {turbo_ocr::detection::kDetResizeDefault,
                                     turbo_ocr::detection::kDbDefaults});

} // namespace turbo_ocr::pipeline
