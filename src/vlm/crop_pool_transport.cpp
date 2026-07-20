#include "turbo_ocr/common/encoding.h"
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

namespace {

// True for failures that are safe to retry on an idempotent POST: transient
// gateway/load-balancer codes (vLLM rolling restart) and clean connection-
// level errors where the request likely never reached the model. A 4xx or a
// clean 2xx (even empty) is NOT retried.
bool is_transient_failure(CURLcode rc, long http_status) {
    if (rc == CURLE_OK) {
        return http_status == 502 || http_status == 503 || http_status == 504;
    }
    switch (rc) {
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_COULDNT_CONNECT:
    case CURLE_GOT_NOTHING:      // server closed before any response
    case CURLE_RECV_ERROR:       // connection reset mid-response
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
        return true;
    default:
        return false;
    }
}

// Small jittered backoff so a fleet of crops hitting a restarting vLLM don't
// retry in lockstep. Bounded; the deadline check in complete_handle keeps the
// total within the per-request timeout budget.
long retry_backoff_ms(int attempt) {
    static thread_local std::mt19937 rng(std::random_device{}());
    const long base = 50L * (1L << std::min(attempt, 4)); // 50,100,200,400,800
    std::uniform_int_distribution<long> jitter(0, base / 2);
    return base + jitter(rng);
}

// Build JSON body for the chat/completions call.
std::string build_json_body(const std::vector<uint8_t> &png_bytes,
                             const std::string &prompt,
                             const std::string &model,
                             int max_tokens) {
    // Base64-encode the PNG.
    std::string b64 = base64_encode(png_bytes.data(), png_bytes.size());

    // Build content array for the user message.
    nlohmann::json image_block;
    image_block["type"] = "image_url";
    image_block["image_url"] = nlohmann::json::object();
    image_block["image_url"]["url"] = std::string("data:image/png;base64,") + b64;

    nlohmann::json text_block;
    text_block["type"] = "text";
    text_block["text"] = prompt;

    nlohmann::json content = nlohmann::json::array();
    content.push_back(std::move(image_block));
    content.push_back(std::move(text_block));

    nlohmann::json user_msg;
    user_msg["role"] = "user";
    user_msg["content"] = std::move(content);

    nlohmann::json messages = nlohmann::json::array();
    messages.push_back(std::move(user_msg));

    nlohmann::json body;
    body["model"] = model;
    body["max_tokens"] = max_tokens;
    body["temperature"] = 0.0;
    body["messages"] = std::move(messages);

    return body.dump();
}

// Extract the assistant's text content from a chat/completions response.
std::string parse_content(const std::string &body) {
    try {
        auto j = nlohmann::json::parse(body);
        return j.at("choices").at(0).at("message").at("content")
                 .get<std::string>();
    } catch (const std::exception &e) {
        // A 200-OK with an unexpected shape must not vanish silently: this pool
        // is the default VLM backend, so log loudly with a truncated body (as
        // the non-pool vlm_formula/vlm_table paths do). Caller derives failure
        // from the empty return; this only makes the cause visible.
        TOCR_LOG_ERROR_RL("VLMCropPool response parse failed", "error", e.what(), "body", body.substr(0, std::min<size_t>(body.size(), 200)));
        return {};
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Worker thread: curl_multi_poll loop
// ---------------------------------------------------------------------------

size_t VLMCropPool::write_cb(char *ptr, size_t sz, size_t nmemb, void *ud) {
    auto *ctx = static_cast<HandleCtx *>(ud);
    // Abort (short count -> CURLE_WRITE_ERROR) past the cap: a wedged or
    // hostile endpoint must not grow the buffer without bound for the whole
    // timeout window.
    constexpr size_t kMaxResponseBytes = 32u << 20;
    const size_t n = sz * nmemb;
    if (ctx->response.size() + n > kMaxResponseBytes) return 0;
    ctx->response.append(ptr, n);
    return n;
}

void VLMCropPool::complete_handle(CURL *easy, CURLcode rc) {
    HandleCtx *ctx = nullptr;
    curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
    if (!ctx) {
        curl_multi_remove_handle(multi_, easy);
        curl_easy_cleanup(easy);
        return;
    }

    long status = 0;
    if (rc == CURLE_OK) curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);

    // Move the request out before tearing down the handle so we can re-queue
    // it on a transient failure (or resolve its promise on success/exhaustion).
    std::unique_ptr<CropRequest> req = std::move(ctx->req);
    std::string response = std::move(ctx->response);

    curl_multi_remove_handle(multi_, easy);
    curl_slist_free_all(ctx->headers);
    curl_easy_cleanup(easy);
    delete ctx;
    in_flight_.fetch_sub(1, std::memory_order_relaxed);

    if (!req) return;

    const bool success = (rc == CURLE_OK && status >= 200 && status < 300);
    if (success) {
        resolve_request(*req, parse_content(response), /*ok=*/true);
        return;
    }

    if (rc == CURLE_OK)
        TOCR_LOG_ERROR_RL("VLMCropPool HTTP error", "status", status, "body", response.substr(0, 200));
    else
        TOCR_LOG_ERROR_RL("VLMCropPool curl error", "error", curl_easy_strerror(rc));

    // Retry only idempotent transient failures, while attempts and the wall-
    // clock budget both remain. A 4xx or a clean empty 200 falls through to a
    // not-confident resolution so downstream never stamps a false-positive.
    // Never retry once shutting down — drain resolves callers promptly.
    const bool can_retry = !stop_.load(std::memory_order_relaxed) &&
                           is_transient_failure(rc, status) &&
                           req->attempts <= req->max_retries &&
                           steady_now_ms() < req->deadline_ms;
    if (can_retry) {
        const long backoff = retry_backoff_ms(req->attempts);
        // Keep the backoff inside the remaining budget; skip retry if it would
        // blow past the deadline.
        if (steady_now_ms() + backoff < req->deadline_ms) {
            TOCR_LOG_WARN_RL("VLMCropPool retrying crop", "attempt", req->attempts, "max", req->max_retries, "backoff_ms", backoff);
            // Park (don't block the worker on backoff); promote_ready_retries()
            // re-dispatches once ready_ms passes.
            req->ready_ms = steady_now_ms() + backoff;
            retry_queue_.push_back(std::move(req));
            return;
        }
    }

    resolve_request(*req, std::string{}, /*ok=*/false);
}

void VLMCropPool::dispatch_one(std::unique_ptr<CropRequest> req) {
    req->attempts += 1;

    // Build the JSON body.
    std::string json_body;
    try {
        json_body = build_json_body(req->png_bytes, req->prompt,
                                    req->model, req->max_tokens);
    } catch (const std::exception &e) {
        TOCR_LOG_ERROR_RL("VLMCropPool json build error", "error", e.what());
        resolve_request(*req, std::string{}, /*ok=*/false);
        return;
    }

    // Own the ctx until it is successfully handed off to curl_multi;
    // any early-exit path below frees it deterministically and resolves
    // the promise, so the caller's future never hangs.
    auto ctx        = std::make_unique<HandleCtx>();
    ctx->post_body  = std::move(json_body);
    ctx->pool       = this;

    CURL *easy = curl_easy_init();
    if (!easy) {
        TOCR_LOG_ERROR_RL("VLMCropPool curl_easy_init() failed");
        resolve_request(*req, std::string{}, /*ok=*/false);
        return;
    }
    ctx->easy = easy;

    ctx->headers = nullptr;
    ctx->headers = curl_slist_append(ctx->headers, "Content-Type: application/json");
    ctx->headers = curl_slist_append(ctx->headers, "Accept: application/json");
    // Keep-alive on loopback: no cost, avoids TCP setup per request.
    ctx->headers = curl_slist_append(ctx->headers, "Connection: keep-alive");
    // Optional bearer auth for external endpoints (empty on the default
    // self-hosted/gateway-fronted path).
    if (!req->api_key.empty()) {
        const std::string auth = "Authorization: Bearer " + req->api_key;
        ctx->headers = curl_slist_append(ctx->headers, auth.c_str());
    }

    std::string url = req->base_url + "/v1/chat/completions";
    curl_easy_setopt(easy, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(easy, CURLOPT_POST,          1L);
    curl_easy_setopt(easy, CURLOPT_POSTFIELDS,    ctx->post_body.c_str());
    curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE, (long)ctx->post_body.size());
    curl_easy_setopt(easy, CURLOPT_HTTPHEADER,    ctx->headers);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT,       (long)req->timeout_s);
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL,      1L);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA,     ctx.get());
    curl_easy_setopt(easy, CURLOPT_PRIVATE,       ctx.get());
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(easy, CURLOPT_MAXREDIRS, 3L);  // bound redirect chains
    // HTTP/1.1 keep-alive is enough for loopback.
    curl_easy_setopt(easy, CURLOPT_HTTP_VERSION,  CURL_HTTP_VERSION_1_1);

    // Stash the request on the ctx so completion can retry or resolve it.
    ctx->req = std::move(req);

    CURLMcode mc = curl_multi_add_handle(multi_, easy);
    if (mc != CURLM_OK) {
        TOCR_LOG_ERROR_RL("VLMCropPool curl_multi_add_handle error", "error", curl_multi_strerror(mc));
        resolve_request(*ctx->req, std::string{}, /*ok=*/false);
        curl_slist_free_all(ctx->headers);
        curl_easy_cleanup(easy);
        return;  // ctx freed by unique_ptr; in_flight_ unchanged
    }
    // Ownership transferred to curl (recovered in complete_handle via
    // CURLINFO_PRIVATE, then deleted there); the raw pointer is intentionally
    // dropped since curl now owns it via CURLOPT_PRIVATE set above.
    (void)ctx.release();
    in_flight_.fetch_add(1, std::memory_order_relaxed);
}

void VLMCropPool::try_dispatch() {
    // Pop as many items as we have free capacity.
    while (in_flight_.load(std::memory_order_relaxed) < max_concurrency_) {
        std::unique_ptr<CropRequest> req;
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            if (queue_.empty()) break;
            req = std::move(queue_.front());
            queue_.pop();
        }
        dispatch_one(std::move(req));
    }
}

long VLMCropPool::promote_ready_retries() {
    if (retry_queue_.empty()) return -1;
    const long now = steady_now_ms();
    long soonest = -1;
    std::vector<std::unique_ptr<CropRequest>> due;
    auto it = retry_queue_.begin();
    while (it != retry_queue_.end()) {
        if ((*it)->ready_ms <= now) {
            due.push_back(std::move(*it));
            it = retry_queue_.erase(it);
        } else {
            const long wait = (*it)->ready_ms - now;
            if (soonest < 0 || wait < soonest) soonest = wait;
            ++it;
        }
    }
    if (!due.empty()) {
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (auto &r : due) queue_.push(std::move(r));
    }
    return soonest;
}

void VLMCropPool::worker_loop() {
    // Drain the wake pipe without blocking.
    auto drain_pipe = [this]() {
        if (wake_rd_ < 0) return;
        char buf[64];
        while (read(wake_rd_, buf, sizeof(buf)) > 0) {}
    };

    while (!stop_.load(std::memory_order_relaxed)) {
        // Promote any due retries back onto the queue, then dispatch up to the
        // concurrency limit.
        long next_retry_ms = promote_ready_retries();
        try_dispatch();

        int running = 0;
        CURLMcode mc = curl_multi_perform(multi_, &running);
        if (mc != CURLM_OK && mc != CURLM_CALL_MULTI_PERFORM) {
            TOCR_LOG_ERROR_RL("VLMCropPool curl_multi_perform error", "error", curl_multi_strerror(mc));
        }

        // Harvest completed transfers.
        int msgs_in_queue = 0;
        CURLMsg *msg;
        while ((msg = curl_multi_info_read(multi_, &msgs_in_queue)) != nullptr) {
            if (msg->msg == CURLMSG_DONE) {
                complete_handle(msg->easy_handle, msg->data.result);
                // After completing, we may have new capacity — try dispatching.
                try_dispatch();
            }
        }

        // Poll with a short timeout; the wake fd lets us wake early.
        // Use curl_multi_poll if available (libcurl ≥7.68); fall back to
        // curl_multi_wait otherwise.
        int numfds = 0;
        struct curl_waitfd extra_fd;
        extra_fd.fd      = wake_rd_;
        extra_fd.events  = CURL_WAIT_POLLIN;
        extra_fd.revents = 0;

        // 200 ms baseline; shrink to a pending retry's ready time so a parked
        // crop re-dispatches on schedule rather than up to 200 ms late.
        int timeout_ms = 200;
        if (next_retry_ms >= 0)
            timeout_ms = static_cast<int>(std::clamp<long>(next_retry_ms, 1, 200));
        curl_multi_poll(multi_, (wake_rd_ >= 0 ? &extra_fd : nullptr),
                        (wake_rd_ >= 0 ? 1u : 0u), timeout_ms, &numfds);

        if (extra_fd.revents & CURL_WAIT_POLLIN) drain_pipe();
    }

    // Drain remaining handles on shutdown.
    int running = 0;
    do {
        curl_multi_perform(multi_, &running);
        int msgs = 0;
        CURLMsg *msg;
        while ((msg = curl_multi_info_read(multi_, &msgs)) != nullptr) {
            if (msg->msg == CURLMSG_DONE)
                complete_handle(msg->easy_handle, msg->data.result);
        }
        if (running > 0) curl_multi_poll(multi_, nullptr, 0, 50, nullptr);
    } while (running > 0);
}

} // namespace turbo_ocr::vlm
