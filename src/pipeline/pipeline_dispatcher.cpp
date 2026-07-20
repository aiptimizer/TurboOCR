#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"

#include <cstdlib>
#include <format>
#include <iostream>

namespace turbo_ocr::pipeline {

PipelineDispatcher::PipelineDispatcher(
    std::vector<std::unique_ptr<GpuPipelineEntry>> entries,
    PipelineBuildSpec spec)
    : spec_(std::move(spec)) {
  size_t n = entries.size();
  workers_.reserve(n);
  recycle_flags_.reserve(n);
  // Per-worker task-start timestamp (steady_clock ms; 0 == idle). Read by the
  // watchdog, written by the owning worker around work(*entry); heap-allocated
  // shared_ptrs so the vector can't relocate atomics out from under either.
  task_start_ms_.reserve(n);
  recycle_requested_at_ms_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto flag = std::make_shared<std::atomic<bool>>(false);
    recycle_flags_.push_back(flag);
    auto start_ms = std::make_shared<std::atomic<long long>>(0);
    task_start_ms_.push_back(start_ms);
    // When the watchdog first requests a recycle for this worker's CURRENT
    // task it records the time here (0 == none requested). The hard-kill only
    // fires after a recycle was actually requested AND remained unhonoured for
    // PIPELINE_HARD_KILL_MS — i.e. the worker is wedged mid-CUDA and never
    // reaches the next loop top to consume the flag.
    auto req_at = std::make_shared<std::atomic<long long>>(0);
    recycle_requested_at_ms_.push_back(req_at);
    workers_.emplace_back([this, flag, start_ms, req_at,
                           entry = std::move(entries[i])]() mutable {
      while (true) {
        QueuedWork work;
        {
          std::unique_lock lock(mutex_);
          // Predicate form of wait() is atomic w.r.t. notify only when
          // the notifier holds the same mutex around the state change.
          // The dtor takes mutex_ before flipping stop_ and calling
          // notify_all() — see ~PipelineDispatcher.
          cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
          if (queue_.empty()) {
            if (stop_) return;
            continue;
          }
          work = std::move(queue_.front());
          queue_.pop();
        }
        // Recycle a suspect entry before reusing it. Only this worker may
        // touch its entry, so the rebuild is race-free w.r.t. inference.
        // A still-running timed-out task on the OLD stream is what made the
        // entry suspect; the new stream/pipeline is independent of it.
        if (flag->exchange(false))
          maybe_recycle_(*entry);
        // Clear any stale recycle-request time from a prior task BEFORE marking
        // this task started, so the watchdog never hard-kills a fresh task on a
        // previous task's timestamp.
        req_at->store(0, std::memory_order_relaxed);
        start_ms->store(now_ms_(), std::memory_order_relaxed);
        // Deadline-drop: skip work whose client deadline already elapsed while it waited in
        // queue (the caller has its 504) — running it only wastes GPU time and deepens
        // congestion under overload. work.fn still destructs, so its abandoned packaged_task
        // future gets a harmless broken_promise that nobody is waiting on.
        if (work.deadline_ms == 0 || now_ms_() <= work.deadline_ms)
          work.fn(*entry);
        start_ms->store(0, std::memory_order_relaxed);
        req_at->store(0, std::memory_order_relaxed);
        // A recoverable CUDA fault in the table/formula stage flags the
        // pipeline for rebuild; recycle the entry before its next task so the
        // slot self-heals instead of re-faulting on every later request.
        if (recycle_enabled_ && entry->pipeline &&
            entry->pipeline->consume_recycle_request())
          flag->store(true, std::memory_order_relaxed);
        else
          // M4: the task COMPLETED (this worker is healthy, not wedged — a wedged worker
          // never reaches here and is handled by the hard-kill path). Cancel any recycle the
          // watchdog requested purely because the task was SLOW: otherwise a slow-but-
          // successful request would trigger a wasteful full-pipeline rebuild + CUDA-graph
          // re-warm that pulls the slot out of service right when the box is already busy.
          flag->store(false, std::memory_order_relaxed);
      }
    });
  }
  // Single watchdog thread: flags workers whose in-flight task has overrun
  // the request deadline (plus grace) for recycle. Only meaningful once a
  // positive request_timeout_ms_ is set, so the scan early-outs otherwise.
  watchdog_ = std::thread([this] { watchdog_loop_(); });
}

PipelineDispatcher::~PipelineDispatcher() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  watchdog_cv_.notify_all();
  // Stop the watchdog before joining workers so it never flags a recycle on
  // a worker that is already shutting down.
  if (watchdog_.joinable()) watchdog_.join();
  for (auto &w : workers_)
    if (w.joinable()) w.join();
}

size_t PipelineDispatcher::queue_depth() const {
  std::lock_guard lock(mutex_);
  return queue_.size();
}

void PipelineDispatcher::request_recycle(size_t worker_index) noexcept {
  if (!recycle_enabled_ || worker_index >= recycle_flags_.size()) return;
  recycle_flags_[worker_index]->store(true);
  // Wake an idle worker so a slot with no queued work still recycles. The
  // owning worker re-checks its flag after every wakeup.
  cv_.notify_all();
}

// Rebuild `entry` from the stored spec; on failure log and leave the entry
// dead (pipeline == nullptr) — a subsequent task throws rather than touching
// a half-built pipeline, and the watchdog can request another recycle.
void PipelineDispatcher::maybe_recycle_(GpuPipelineEntry &entry) noexcept {
  try {
    entry.recycle(spec_);
  } catch (const std::exception &e) {
    std::cerr << std::format("[Dispatcher] entry recycle failed: {}\n",
                             e.what());
  }
}

// Scan workers ~once a second; flag any whose in-flight task has run longer
// than the request deadline plus a grace for recycle. The flag only takes
// effect at the worker's NEXT loop iteration — it cannot interrupt a CUDA
// call mid-flight (that wedge is caught by the H4 sticky-fault fail-fast).
void PipelineDispatcher::watchdog_loop_() {
  // Hard margin AFTER a recycle was actually requested before a wedged-mid-CUDA
  // worker is declared unrecoverable and the process exits. Env-configurable
  // (PIPELINE_HARD_KILL_MS) with a deliberately large default so it only ever
  // catches a true hang, not a merely slow task on a contended box.
  static const long long kHardKillMs = [] {
    const char *e = std::getenv("PIPELINE_HARD_KILL_MS");
    long long v = e ? std::atoll(e) : 600000;  // 10 min
    return v > 0 ? v : 600000;
  }();
  while (true) {
    {
      std::unique_lock lock(mutex_);
      if (watchdog_cv_.wait_for(lock, kWatchdogInterval,
                                [this] { return stop_; }))
        return;  // woken by shutdown
    }
    long timeout = request_timeout_ms_.load(std::memory_order_relaxed);
    if (timeout <= 0) continue;  // watchdog inert until a deadline is set
    long long deadline = static_cast<long long>(timeout) + kWatchdogGraceMs;
    long long now = now_ms_();
    for (size_t i = 0; i < task_start_ms_.size(); ++i) {
      long long started =
          task_start_ms_[i]->load(std::memory_order_relaxed);
      if (started == 0) continue;
      const long long overrun = now - started;
      const long long req_at =
          recycle_requested_at_ms_[i]->load(std::memory_order_relaxed);
      // Only hard-kill when a recycle was ACTUALLY requested for this task and
      // remained unhonoured for kHardKillMs — the worker is wedged mid-CUDA, so
      // the recycle flag (read only at the next loop top) will never be consumed
      // and the slot is unreclaimable in-process. Exit so the orchestrator
      // restarts the pod (mirrors the sticky-fault _Exit in cuda_check.h).
      if (req_at != 0 && now - req_at > kHardKillMs) {
        std::cerr << std::format(
                         "[Dispatcher] FATAL: worker {} still wedged {} ms after "
                         "a recycle request went unhonoured ({} ms total overrun, "
                         "stuck mid-CUDA); exiting so the orchestrator restarts "
                         "the pod\n",
                         i, now - req_at, overrun)
                  << std::flush;
        std::_Exit(EXIT_FAILURE);
      }
      // First crossing of the deadline+grace: request a recycle and record when,
      // so the hard-kill clock starts from the request, not from task start.
      if (overrun > deadline && req_at == 0) {
        request_recycle(i);
        recycle_requested_at_ms_[i]->store(now, std::memory_order_relaxed);
      }
    }
  }
}

// Build + warm exactly one pipeline (det/rec/cls + optional layout/doc_ori).
// Returns nullptr on a non-fatal init failure (logged).
std::unique_ptr<GpuPipelineEntry> build_one_pipeline(
    int idx, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model,
    const std::string &layout_model, const std::string &doc_ori_model,
    const DetInferConfig &det_cfg) {
  auto pipeline = std::make_unique<OcrPipeline>();
  if (!pipeline->init(det_model, rec_model, rec_dict, cls_model, det_cfg)) {
    std::cerr << std::format("[Dispatcher] Failed to init GPU pipeline {}\n", idx);
    return nullptr;
  }
  if (!layout_model.empty() && !pipeline->load_layout_model(layout_model))
    throw turbo_ocr::ModelLoadError(std::format(
        "[Dispatcher] Failed to load layout model for pipeline {}", idx));
  if (!doc_ori_model.empty())
    (void)pipeline->load_doc_ori_model(doc_ori_model); // soft-disable on fail
  maybe_load_router_models(*pipeline);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto entry = std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream);
  entry->pipeline->warmup_gpu(entry->stream); // grows buffers + bakes graphs
  return entry;
}

std::unique_ptr<PipelineDispatcher> make_pipeline_dispatcher(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model,
    const std::string &layout_model, const std::string &doc_ori_model,
    const DetInferConfig &det_cfg) {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Dispatcher] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    if (auto e = build_one_pipeline(i, det_model, rec_model, rec_dict,
                                    cls_model, layout_model,
                                    doc_ori_model, det_cfg))
      entries.push_back(std::move(e));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(std::format(
        "[Dispatcher] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Pipeline warmup complete ({} pipelines).\n",
                           entries.size());
  // Carry the build recipe so a watchdog can rebuild a wedged entry in place.
  PipelineBuildSpec spec{det_model,  rec_model,    rec_dict,      cls_model,
                         layout_model, doc_ori_model, det_cfg};
  return std::make_unique<PipelineDispatcher>(std::move(entries),
                                              std::move(spec));
}

} // namespace turbo_ocr::pipeline
