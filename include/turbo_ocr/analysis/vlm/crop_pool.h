#pragma once

// Global async VLM crop pool.
//
// Replaces the per-VLMFormula / per-VLMTable thread-per-crop model with a
// process-global pool that keeps up to VLM_GLOBAL_CONCURRENCY (default 50)
// CURL easy handles in flight simultaneously across ALL pipeline instances.
//
// Usage:
//   auto fut = VLMCropPool::instance().submit(png_bytes, prompt, model,
//                                             max_tokens, timeout_s);
//   std::string result = fut.get();   // blocks only the calling thread
//
// The pool owns a single worker thread that polls curl_multi, drains the
// completion queue, and fulfils promises. Application threads submit and
// wait on futures independently — no cross-page serialisation.
//
// Env vars:
//   VLM_GLOBAL_CONCURRENCY  (default 50)  max simultaneous CURL handles
//   VLM_PNG_THREADS         (default 4)   per-page PNG encode workers (unused here, exposed for callers)
//   VLM_MAX_RETRIES         (default 2)   extra attempts on a transient
//                                          (idempotent) endpoint failure

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>

namespace turbo_ocr::vlm {

// Resolved outcome of one crop. `text` is the parsed assistant content (may be
// legitimately empty on a clean 200); `ok` distinguishes that genuine empty
// from a failed/exhausted-retry transport so callers don't stamp a false-
// confident result. Carried only by submit_with_status(); submit() keeps the
// historical std::future<std::string> shape for existing callers.
struct CropOutcome {
    std::string text;
    bool        ok = false;
};

struct CropRequest {
    std::vector<uint8_t>               png_bytes;
    std::string                        prompt;
    std::string                        model;
    int                                max_tokens  = 512;
    int                                timeout_s   = 30;
    std::string                        base_url;          // includes http://host:port, no trailing slash
    std::string                        api_key;           // optional; empty => no Authorization header
    std::promise<std::string>          result;

    // Set when the caller used submit_with_status(): resolved instead of
    // `result`, and carries the success flag alongside the text.
    bool                                       want_status = false;
    std::shared_ptr<std::promise<CropOutcome>> status_result;

    // Retry bookkeeping (server-driven; only transient/idempotent failures
    // are retried, capped at max_retries within the timeout budget).
    int      attempts    = 0;   // attempts already made on this request
    int      max_retries = 0;   // extra attempts allowed beyond the first
    long     deadline_ms = 0;   // steady-clock ms after which no more retries
    long     ready_ms    = 0;   // steady-clock ms before which not (re)dispatched
};

class VLMCropPool {
public:
    // Process-global singleton.
    static VLMCropPool &instance();

    // Submit one crop for async processing. Never blocks the caller beyond
    // a brief queue-lock acquisition. Returns a future that resolves to the
    // extracted text/LaTeX/OTSL string, or empty on error.
    std::future<std::string> submit(std::vector<uint8_t> png_bytes,
                                    std::string          prompt,
                                    std::string          model,
                                    int                  max_tokens,
                                    int                  timeout_s,
                                    std::string          base_url,
                                    std::string          api_key = "");

    // Status-carrying variant: identical dispatch, but the future resolves to
    // a CropOutcome whose `ok` flag is false on a failed/exhausted-retry
    // transport (true on any 2xx, including a clean empty body). Additive;
    // submit() is unchanged for existing callers.
    std::future<CropOutcome> submit_with_status(std::vector<uint8_t> png_bytes,
                                                std::string          prompt,
                                                std::string          model,
                                                int                  max_tokens,
                                                int                  timeout_s,
                                                std::string          base_url,
                                                std::string          api_key = "");

    // Graceful shutdown: called by destructor. Waits for in-flight ops.
    void shutdown();

    int max_concurrency() const noexcept { return max_concurrency_; }

    VLMCropPool(const VLMCropPool &) = delete;
    VLMCropPool &operator=(const VLMCropPool &) = delete;

private:
    VLMCropPool();
    ~VLMCropPool();

    void worker_loop();

    // Per-handle state kept alive during the async operation.
    struct HandleCtx {
        CURL                      *easy      = nullptr;
        struct curl_slist         *headers   = nullptr;
        std::string                post_body;  // JSON body (kept alive for CURL)
        std::string                response;   // accumulates write callback data
        VLMCropPool               *pool       = nullptr;
        // Originating request, kept so a transient failure can re-queue the
        // crop without rebuilding it. Resolving the caller's promise is
        // deferred to the request once retries are exhausted (or on success).
        std::unique_ptr<CropRequest> req;
    };

    static size_t write_cb(char *ptr, size_t sz, size_t nmemb, void *ud);
    void complete_handle(CURL *easy, CURLcode rc);
    void dispatch_one(std::unique_ptr<CropRequest> req);
    void try_dispatch();  // pop from queue_ and add to multi if capacity allows
    // Move any due retry_queue_ entries back onto queue_; returns ms until the
    // next not-yet-ready retry (or -1 if none) so the poll timeout can shrink.
    long promote_ready_retries();
    // Resolve the caller's promise (status- or string-shaped) for `req`.
    static void resolve_request(CropRequest &req, std::string text, bool ok);

    int                          max_concurrency_;
    int                          max_retries_     = 0;
    CURLM                       *multi_           = nullptr;
    std::thread                  worker_;
    std::atomic<bool>            stop_{false};

    // Pending requests waiting for a free slot.
    std::mutex                   queue_mu_;
    std::condition_variable      queue_cv_;
    std::queue<std::unique_ptr<CropRequest>> queue_;

    // Transient-failure retries parked here with a steady-clock ready_ms;
    // promote_ready() moves any that are due back onto queue_ (worker thread
    // only — no lock needed). Avoids blocking the single worker on backoff.
    std::vector<std::unique_ptr<CropRequest>> retry_queue_;

    // Count of handles currently in the multi handle.
    std::atomic<int>             in_flight_{0};

    // The worker is woken from submit() by curl_multi_wakeup(multi_), which is
    // libcurl's own thread-safe interrupt for a blocked curl_multi_poll (>= 7.68).
    // This used to be a self-pipe registered as an extra curl_waitfd: that needs
    // pipe()/read()/fcntl and, on Windows, a SOCKET rather than a pipe handle —
    // so the mechanism was POSIX-only for a portability reason libcurl had
    // already solved. No fds to own, close, or drain.
};

} // namespace turbo_ocr::vlm
