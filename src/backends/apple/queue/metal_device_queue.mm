// MetalDeviceQueue / MetalDeviceEvent implementation (see metal_device_queue.h).

#import "apple/queue/metal_device_queue.h"
#import "apple/support/metal_common.h"
#import "apple/support/apple_contention.h"

#import <Foundation/Foundation.h>

#include "turbo_ocr/base/env_utils.h"

#include <atomic>
#include <cstdio>

namespace turbo_ocr::apple {

namespace {
// Host waits are bounded so a GPU fault / failed shader compile surfaces as an
// error instead of wedging a pipeline replica forever (a hung replica holds its
// pool Lease and cascades into an unbounded wait in every other worker).
std::uint64_t wait_timeout_ms() {
  static const std::uint64_t v = static_cast<std::uint64_t>(
      env::env_int("TURBO_APPLE_GPU_TIMEOUT_MS", 30000, 1, 24 * 60 * 60 * 1000));
  return v;
}
std::atomic<unsigned long long> g_cb_errors{0};
} // namespace

// A COMMAND BUFFER THAT FAILS IS SILENT OTHERWISE. Nothing in this backend ever
// read MTLCommandBuffer.error, so a GPU fault / device-oversubscription abort
// left every buffer that command buffer was supposed to write holding whatever
// was in it before — and the host read it as if it were this page's detection
// map or this page's CTC indices. Under multi-process load that is exactly the
// "whole page returned another page's transcript" signature. Attach a completion
// handler to every command buffer we commit so a failure is LOUD.
void attach_error_watch(id<MTLCommandBuffer> cb,
                        std::shared_ptr<std::atomic<unsigned long long>> sink) {
  if (!cb) return;
  // `sink` is captured BY VALUE: this handler can fire after the queue that
  // committed the buffer has been destroyed, so it must own a reference.
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    if (done.status == MTLCommandBufferStatusError || done.error) {
      g_cb_errors.fetch_add(1, std::memory_order_relaxed);
      if (sink) sink->fetch_add(1, std::memory_order_relaxed);
      NSLog(@"[apple] COMMAND BUFFER FAILED: status=%ld error=%@ — every device "
            @"buffer it was to write is now STALE/undefined",
            (long)done.status, done.error);
    }
  }];
}

unsigned long long command_buffer_error_count() {
  return g_cb_errors.load(std::memory_order_relaxed);
}

// --- MetalDeviceEvent -------------------------------------------------------

MetalDeviceEvent::~MetalDeviceEvent() { event_ = nil; }

void *MetalDeviceEvent::native_handle() const noexcept {
  return (__bridge void *)event_;
}

void MetalDeviceEvent::synchronize() {
  if (!event_ || signal_value == 0) return;
  // Was a bare `while (signaledValue < v) {}` busy-spin with a "/* yield */"
  // comment and no yield: at K=8 that burns 8 cores and, if the signalling
  // command buffer errors out (or was never committed — e.g. recorded inside an
  // open BatchScope), it hangs forever. waitUntilSignaledValue blocks properly
  // and gives up.
  if (![event_ waitUntilSignaledValue:signal_value timeoutMS:wait_timeout_ms()])
    NSLog(@"[apple] event wait timed out at value %llu (signalled %llu) — GPU "
          @"fault, or the signal was encoded on a command buffer that was never "
          @"committed", (unsigned long long)signal_value,
          (unsigned long long)event_.signaledValue);
}

bool MetalDeviceEvent::query() const noexcept {
  return !event_ || event_.signaledValue >= signal_value;
}

// --- MetalDeviceQueue -------------------------------------------------------

MetalDeviceQueue::MetalDeviceQueue()
    : cb_errors_(std::make_shared<std::atomic<unsigned long long>>(0)) {
  q_ = [mtl_device() newCommandQueue];
  timeline_ = [mtl_device() newSharedEvent];
}

MetalDeviceQueue::~MetalDeviceQueue() {
  // Drain anything still committed so buffers aren't freed under the GPU.
  synchronize();
  open_ = nil;
  last_committed_ = nil;
  timeline_ = nil;
  q_ = nil;
}

void *MetalDeviceQueue::native_handle() const noexcept {
  // The open command buffer while batching (device_queue.h contract), else the
  // command queue.
  return open_ ? (__bridge void *)open_ : (__bridge void *)q_;
}

MPSCommandBuffer *MetalDeviceQueue::acquire_cb() {
  if (open_) return open_;
  return [MPSCommandBuffer commandBufferFromCommandQueue:q_];
}

// EVERY commit goes through here so the timeline value monotonically tracks all
// submitted work. synchronize() then waits for the HIGHEST issued value rather
// than for one particular command buffer.
//
// Why that matters: MTLCommandQueue guarantees submission order, not completion
// order for concurrently executing buffers, and the old code waited on a single
// `last_committed_` that every submit/record/wait overwrote. With only one
// buffer ever outstanding it was accidentally correct; the moment two are in
// flight (which is the whole point of the async detector path) it would return
// while real work was still writing the buffers the host is about to read.
void MetalDeviceQueue::commit_(MPSCommandBuffer *cb) {
  TURBO_APPLE_STAT(q_commit);
  const std::uint64_t v = ++timeline_value_;
  id<MTLCommandBuffer> root = cb.rootCommandBuffer;
  [root encodeSignalEvent:timeline_ value:v];
  attach_error_watch(root, cb_errors_);
  [root commit];
  last_committed_ = root;
  if (v > submitted_value_) submitted_value_ = v;
}

void MetalDeviceQueue::submit_cb(MPSCommandBuffer *cb) {
  if (cb == open_) return; // committed at end_batch()
  commit_(cb);
}

void MetalDeviceQueue::begin_batch() {
  // Batches do not nest (device_queue.h). Open one command buffer that every
  // subsequent encoder appends to.
  TURBO_APPLE_STAT(q_new_cb);
  open_ = [MPSCommandBuffer commandBufferFromCommandQueue:q_];
  batch_open_ = true;
}

void MetalDeviceQueue::end_batch() {
  if (!open_) { batch_open_ = false; return; }
  commit_(open_);           // single submission — no wait
  open_ = nil;
  batch_open_ = false;
}

void MetalDeviceQueue::flush() {
  if (!open_) return;       // outside a batch every op already self-commits
  MPSCommandBuffer *cb = open_;
  open_ = [MPSCommandBuffer commandBufferFromCommandQueue:q_]; // batch stays open
  commit_(cb);
}

void MetalDeviceQueue::record(backend::DeviceEvent &ev) {
  auto &me = static_cast<MetalDeviceEvent &>(ev);
  const std::uint64_t v = ++timeline_value_;
  me.signal_value = v;
  // Encode the signal on whatever command buffer is (or will be) current. Inside
  // an open batch the signal rides the batch buffer and only fires at
  // end_batch()/flush() — so a host-side ev.synchronize() before then would
  // block (now: time out, not hang). Record outside a batch, or flush() first.
  MPSCommandBuffer *cb = acquire_cb();
  [cb.rootCommandBuffer encodeSignalEvent:timeline_ value:v];
  if (cb != open_) {
    [cb.rootCommandBuffer commit];
    last_committed_ = cb.rootCommandBuffer;
    if (v > submitted_value_) submitted_value_ = v;
  }
}

void MetalDeviceQueue::wait(const backend::DeviceEvent &ev) {
  const auto &me = static_cast<const MetalDeviceEvent &>(ev);
  // Wait on the event that was RECORDED, not on this queue's own timeline. The
  // old code encoded a wait on `timeline_`, so a cross-queue wait silently
  // waited on the wrong (own) timeline — device_queue.h's "wait on an event from
  // any queue of the same device" was unimplemented.
  id<MTLSharedEvent> target = me.raw();
  if (!target) return;
  MPSCommandBuffer *cb = acquire_cb();
  [cb.rootCommandBuffer encodeWaitForEvent:target value:me.signal_value];
  submit_cb(cb);
}

void MetalDeviceQueue::synchronize() {
  if (batch_open_) {
    // Contract violation (device_queue.h): the open batch is UNCOMMITTED, so
    // this can only wait on stale work and the caller would read the previous
    // page's results. Flush so the wait is at least meaningful, and say so.
    NSLog(@"[apple] DeviceQueue::synchronize() called with a batch OPEN — that "
          @"is a logic error (the open command buffer is not submitted). "
          @"Flushing it; close the BatchScope before synchronizing.");
    flush();
  }
  if (submitted_value_ == 0) return;
  TURBO_APPLE_STAT(q_sync_wait);
  if (![timeline_ waitUntilSignaledValue:submitted_value_
                               timeoutMS:wait_timeout_ms()])
    NSLog(@"[apple] queue synchronize timed out waiting for %llu (signalled "
          @"%llu) — GPU fault or lost command buffer",
          (unsigned long long)submitted_value_,
          (unsigned long long)timeline_.signaledValue);
}

// The checked twin of synchronize(). See the header for why every host read of
// reused device scratch must go through this.
bool MetalDeviceQueue::sync_ok(unsigned long long mark) {
  const std::uint64_t want = submitted_value_;
  if (batch_open_) flush();
  bool ok = true;
  if (submitted_value_ != 0) {
    TURBO_APPLE_STAT(q_sync_wait);
    if (![timeline_ waitUntilSignaledValue:submitted_value_
                                 timeoutMS:wait_timeout_ms()]) {
      NSLog(@"[apple] queue synchronize TIMED OUT waiting for %llu (signalled "
            @"%llu) — treating this page's device buffers as UNDEFINED",
            (unsigned long long)submitted_value_,
            (unsigned long long)timeline_.signaledValue);
      ok = false;
    }
  }
  // A failed command buffer may still signal the event (driver-dependent), so
  // the wait succeeding is NOT sufficient — the error counter is authoritative.
  const unsigned long long now =
      cb_errors_ ? cb_errors_->load(std::memory_order_relaxed) : 0;
  if (now != mark) {
    NSLog(@"[apple] %llu command buffer(s) FAILED on this queue while producing "
          @"this result (mark=%llu now=%llu, timeline want=%llu) — its device "
          @"buffers hold the PREVIOUS page's bytes; refusing to read them",
          now - mark, mark, now, (unsigned long long)want);
    ok = false;
  }
  return ok;
}

std::unique_ptr<backend::DeviceEvent> MetalDeviceQueue::make_event() {
  return std::make_unique<MetalDeviceEvent>(timeline_);
}

} // namespace turbo_ocr::apple
