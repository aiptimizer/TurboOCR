#include "turbo_ocr/vlm/vlm_client.h"
#include "turbo_ocr/vlm/crop_pool.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "simdutf.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/log/logger.h"
#include "crop_pool_internal.h"

namespace turbo_ocr::vlm {

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

VLMCropPool &VLMCropPool::instance() {
    static VLMCropPool pool;
    return pool;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

VLMCropPool::VLMCropPool()
    : max_concurrency_(turbo_ocr::env::env_int("VLM_GLOBAL_CONCURRENCY", 50, 1, 4096)),
      max_retries_(turbo_ocr::env::env_int("VLM_MAX_RETRIES", 2, 0, 100)) {
    vlm::ensure_curl_init();
    // Pipe for waking the worker when new items are queued.
    // Both ends must be non-blocking: the reader uses drain_pipe() which loops
    // until EAGAIN, and the writer uses write() from submit() which must not
    // stall if the pipe buffer fills.
    int fds[2];
    if (pipe(fds) != 0) {
        TOCR_LOG_ERROR("VLMCropPool pipe() failed", "error", strerror(errno));
    } else {
        wake_rd_ = fds[0];
        wake_wr_ = fds[1];
        fcntl(wake_rd_, F_SETFL, fcntl(wake_rd_, F_GETFL, 0) | O_NONBLOCK);
        fcntl(wake_wr_, F_SETFL, fcntl(wake_wr_, F_GETFL, 0) | O_NONBLOCK);
    }

    multi_ = curl_multi_init();
    if (!multi_) {
        TOCR_LOG_ERROR("VLMCropPool curl_multi_init() failed");
        return;
    }
    // Limit open connections to avoid FD exhaustion; each handle is loopback
    // so one persistent connection per handle is fine.
    curl_multi_setopt(multi_, CURLMOPT_MAXCONNECTS, (long)max_concurrency_);

    worker_ = std::thread([this] { worker_loop(); });
    TOCR_LOG_INFO("VLMCropPool started", "max_concurrency", max_concurrency_,
                  "wake_rd", wake_rd_, "wake_wr", wake_wr_);
}

VLMCropPool::~VLMCropPool() {
    shutdown();
}

void VLMCropPool::shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) return;

    // Wake the worker so it observes stop_ and returns. This write races
    // nothing: the worker is the only reader and is still alive here.
    if (wake_wr_ >= 0) { char b = 0; (void)write(wake_wr_, &b, 1); }
    if (worker_.joinable()) worker_.join();

    if (wake_rd_ >= 0) { close(wake_rd_); wake_rd_ = -1; }

    // Drain any remaining pending items (not-confident: shutdown is a failure
    // to deliver, not a genuine empty result). Worker has joined, so the
    // worker-owned retry_queue_ is safe to touch here.
    for (auto &r : retry_queue_)
        if (r) resolve_request(*r, std::string{}, /*ok=*/false);
    retry_queue_.clear();

    // Close wake_wr_ AND drain the queue under queue_mu_: a concurrent submit()
    // takes the same lock, so it either observes stop_ and resolves in place
    // (never touching the fd) or completed its write to a still-open fd before
    // us — no write to a closed/recycled descriptor.
    std::lock_guard<std::mutex> lk(queue_mu_);
    if (wake_wr_ >= 0) { close(wake_wr_); wake_wr_ = -1; }
    while (!queue_.empty()) {
        resolve_request(*queue_.front(), std::string{}, /*ok=*/false);
        queue_.pop();
    }

    if (multi_) {
        curl_multi_cleanup(multi_);
        multi_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// submit()
// ---------------------------------------------------------------------------

std::future<std::string> VLMCropPool::submit(std::vector<uint8_t> png_bytes,
                                              std::string          prompt,
                                              std::string          model,
                                              int                  max_tokens,
                                              int                  timeout_s,
                                              std::string          base_url,
                                              std::string          api_key) {
    auto req = std::make_unique<CropRequest>();
    req->png_bytes  = std::move(png_bytes);
    req->prompt     = std::move(prompt);
    req->model      = std::move(model);
    req->max_tokens = max_tokens;
    req->timeout_s  = timeout_s;
    req->base_url   = std::move(base_url);
    req->api_key    = std::move(api_key);
    req->max_retries = max_retries_;
    // Cap total retry wall time to the caller's timeout budget: a single
    // attempt may consume the whole timeout, so anchor the retry deadline to
    // one timeout window from now. Past it, complete_handle stops retrying.
    req->deadline_ms = steady_now_ms() + 1000L * std::max(1, timeout_s);
    auto fut = req->result.get_future();

    {
        // The stop_ check must happen under queue_mu_: shutdown() sets stop_
        // and then drains queue_ under this same mutex, so a submit that wins
        // the lock AFTER the drain would otherwise park a request nobody ever
        // resolves — the caller's future would hang forever. Checking inside
        // the lock makes "queued" and "will be drained or processed" one
        // atomic decision. Empty resolve == degraded, surfaced upstream.
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (stop_.load()) {
            req->result.set_value(std::string());
            return fut;
        }
        queue_.push(std::move(req));
        // Wake the worker UNDER the lock: shutdown() closes wake_wr_ while
        // holding queue_mu_, so the fd is guaranteed open here (or stop_ is
        // set and we returned above).
        if (wake_wr_ >= 0) { char b = 0; (void)write(wake_wr_, &b, 1); }
    }
    queue_cv_.notify_one();
    return fut;
}

std::future<CropOutcome>
VLMCropPool::submit_with_status(std::vector<uint8_t> png_bytes,
                                std::string          prompt,
                                std::string          model,
                                int                  max_tokens,
                                int                  timeout_s,
                                std::string          base_url,
                                std::string          api_key) {
    auto req = std::make_unique<CropRequest>();
    req->png_bytes  = std::move(png_bytes);
    req->prompt     = std::move(prompt);
    req->model      = std::move(model);
    req->max_tokens = max_tokens;
    req->timeout_s  = timeout_s;
    req->base_url   = std::move(base_url);
    req->api_key    = std::move(api_key);
    req->max_retries = max_retries_;
    req->deadline_ms = steady_now_ms() + 1000L * std::max(1, timeout_s);
    req->want_status    = true;
    req->status_result  = std::make_shared<std::promise<CropOutcome>>();
    auto fut = req->status_result->get_future();

    {
        // stop_ checked under queue_mu_ — same shutdown-drain race as submit().
        std::lock_guard<std::mutex> lk(queue_mu_);
        if (stop_.load()) {
            req->status_result->set_value(CropOutcome{std::string(), false});
            return fut;
        }
        queue_.push(std::move(req));
        // Wake under the lock — see submit() for the fd-lifetime rationale.
        if (wake_wr_ >= 0) { char b = 0; (void)write(wake_wr_, &b, 1); }
    }
    queue_cv_.notify_one();
    return fut;
}

// Resolve whichever promise the caller is waiting on. submit() callers get the
// bare text; submit_with_status() callers get text + the success flag.
void VLMCropPool::resolve_request(CropRequest &req, std::string text, bool ok) {
    // set_value can only throw std::future_error if the promise was already
    // satisfied (a benign double-resolve race between retry/timeout paths) or
    // the future was abandoned. Either way the caller has its result or no
    // longer cares, so swallowing is the correct best-effort behavior here.
    if (req.want_status && req.status_result) {
        try { req.status_result->set_value(CropOutcome{std::move(text), ok}); }
        catch (...) { /* benign double-resolve: caller already has its result */ }
    } else {
        try { req.result.set_value(std::move(text)); }
        catch (...) { /* benign double-resolve: caller already has its result */ }
    }
}

} // namespace turbo_ocr::vlm
