// AneRecEngine implementation (see ane_rec_engine.h).

#import "apple/engine/ane_rec_engine.h"
#import "apple/support/apple_profile.h"
#import "apple/support/apple_contention.h"
#import "apple/support/coreml_compile.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "turbo_ocr/base/env_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace turbo_ocr::apple {
namespace {

// mlpackage -> compiled .mlmodelc URL, shared process-wide.
// [MLModel compileModelAtURL:] costs ~100 ms; a replica pool would otherwise pay
// it once per replica per bucket.
// .mlpackage compile cache — shared with every CoreML consumer in this
// backend (see support/coreml_compile.h).
NSURL *compiled_url(NSString *path) { return coreml_compiled_url(path); }



// ---------------------------------------------------------------------------
// AneBatchService — CROSS-REPLICA coalescing in front of one CoreML model.
//
// WHY IT EXISTS (measured): the Neural Engine reaches peak crops/s only at a
// FILLED batch, and switching between a package's enumerated shapes makes
// CoreML re-specialize the program (at W320: 14 ms for batch 48 vs 42 ms for
// batch 96 on the same crops). A single pipeline replica has ~30 narrow crops
// per page — never a full batch — so on its own it either wastes ANE compute on
// padding or pays per-predict overhead on a tiny batch (1764 crops/s at batch 16
// vs 3424 crops/s at batch 48).
//
// A REPLICA POOL has several pages in flight simultaneously. This service pools
// their crops: replicas submit and block, and a small number of worker threads
// assemble ONE fixed-shape batch out of whatever is pending and run a single
// predict. One shape => no re-specialization; pooled rows => filled batches.
//
// This is device mechanics, not policy. WHICH lines land in a width bucket and
// how many crops a chunk carries is still decided by the SHARED planner
// (recognition::plan_rec_batches); the service only decides how many pending
// submissions ride the same ANE program invocation.
// ---------------------------------------------------------------------------
struct AneReq {
  const float *crops = nullptr;
  int count = 0;
  std::int32_t *idx = nullptr;
  float *sc = nullptr;
  bool done = false;
  bool ok = false;   // false => idx/sc were NOT filled with this page's tokens
  std::chrono::steady_clock::time_point t_submit{}; // for contention accounting
};

// Blank CTC index. On any failure the submitter's scratch is filled with blanks
// (which ctc_greedy_decode collapses to an empty string) instead of being left
// holding the PREVIOUS page's tokens — the scratch buffers are allocated once at
// load() and reused for every page, so "leave it alone on error" means decoding
// stale text with plausible confidences that passes kDropScore silently.
constexpr std::int32_t kBlankIndex = 0;

class AneBatchService {
public:
  // One service per (mlpackage, chosen shape). nullptr if the package can't load.
  static AneBatchService *get(const std::string &pkg, int h, int w, int shape_idx);

  [[nodiscard]] int batch() const noexcept { return batch_; }
  [[nodiscard]] int time_steps() const noexcept { return T_; }

  // Two-phase submission. enqueue() places the request and returns at once;
  // wait_done() blocks until its tokens are written (or blanks them on
  // timeout). Splitting the phases is what lets MpsRecognizer put EVERY ANE
  // bucket of a round in flight before waiting on any of them — with one
  // blocking submit() per bucket, two buckets meant two serialized service
  // round-trips (~4.4 ms each, profiled) on the round's critical path.
  //
  // nullptr is returned only for an OVERSIZED request; the output buffers are
  // blanked and the caller must treat them as invalid.
  [[nodiscard]] std::shared_ptr<AneReq> enqueue(const float *crops, int count,
                                               std::int32_t *idx, float *sc) {
    if (count > batch_) {
      // HARD FAIL, never a silent clamp. The old worker clamped an oversized
      // request to `batch_` rows and the scatter loop stopped there too, leaving
      // the TAIL of the submitter's idx/sc scratch UNWRITTEN — i.e. holding a
      // previous page's tokens, which then decoded as real text. The shared
      // planner caps chunks at the pinned batch, so this is unreachable today;
      // make it loud if that invariant ever breaks.
      NSLog(@"[apple] ANE submit of %d rows exceeds the pinned batch %d — "
            @"rejecting (the shared planner must chunk at the batch shape)",
            count, batch_);
      std::fill(idx, idx + (size_t)count * T_, kBlankIndex);
      std::fill(sc, sc + (size_t)count * T_, 0.0f);
      return nullptr;
    }
    auto r = std::make_shared<AneReq>();
    r->crops = crops; r->count = count; r->idx = idx; r->sc = sc;
    r->t_submit = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    q_.push_back(r);
    TURBO_APPLE_BUMP(ane_queue_depth, (long long)q_.size());
    have_.notify_one();
    return r;
  }

  // Returns false when the request's tokens could NOT be produced (predict
  // failure or timeout). On timeout the output buffers are blanked here and
  // the queued request is defused (crops nulled) so a late worker pickup is a
  // gather no-op — the scatter-back into the caller's scratch still races a
  // 30 s-late worker in principle, exactly as the one-phase submit always did.
  [[nodiscard]] bool wait_done(const std::shared_ptr<AneReq> &r) {
    if (!r) return false; // enqueue() already blanked and logged
    TURBO_APPLE_STAT_N(ane_submit_wait, r->count);
    std::unique_lock<std::mutex> lk(mu_);
    // Bounded wait: a stalled ANE with 2 workers and 16 replicas would otherwise
    // pin every replica forever (each holding a pipeline-pool lease).
    if (!done_cv_.wait_for(lk, std::chrono::milliseconds(env::env_int("TURBO_APPLE_ANE_TIMEOUT_MS", 30000, 1, 24 * 60 * 60 * 1000)),
                           [&] { return r->done; })) {
      r->crops = nullptr; // the worker may still pick it up; make it a no-op
      NSLog(@"[apple] ANE submit timed out after %d rows — treating as failure",
            r->count);
      std::fill(r->idx, r->idx + (size_t)r->count * T_, kBlankIndex);
      std::fill(r->sc, r->sc + (size_t)r->count * T_, 0.0f);
      return false;
    }
    return r->ok;
  }

  // One-phase convenience: enqueue and wait. The warm-up probe uses this.
  [[nodiscard]] bool submit(const float *crops, int count, std::int32_t *idx,
                            float *sc) {
    return wait_done(enqueue(crops, count, idx, sc));
  }

private:
  bool start(const std::string &pkg, int h, int w, int shape_idx, int workers);
  void worker_loop(int wi);
  bool predict(MLModel *model, const float *rows, int nrows,
               std::int32_t *idx_out, float *sc_out);

  std::vector<std::thread> threads_;
  std::vector<MLModel *> models_; // one per worker; predict is serial per model
  std::mutex mu_;
  std::condition_variable have_;
  // Completion is broadcast on ONE service-wide cv rather than a per-request
  // one: a per-request cv lived on the submitter's STACK, which the two-phase
  // API cannot allow (the waiter's frame is not pinned between enqueue() and
  // wait_done()). Few waiters ever block here, so the broadcast is cheap.
  std::condition_variable done_cv_;
  std::deque<std::shared_ptr<AneReq>> q_;
  NSString *idx_key_ = nil, *score_key_ = nil;
  int batch_ = 0, T_ = 0, h_ = 0, w_ = 0;
  Stat *st_predict_ = nullptr, *st_qwait_ = nullptr; // per-width, contention only
  std::atomic<bool> stop_{false};
};

bool AneBatchService::start(const std::string &pkg, int h, int w, int shape_idx,
                            int workers) {
  @autoreleasepool {
    h_ = h;
    w_ = w;
    NSString *path = [NSString stringWithUTF8String:pkg.c_str()];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) return false;
    NSURL *c = compiled_url(path);
    if (!c) return false;
    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSError *err = nil;
    for (int i = 0; i < workers; ++i) {
      MLModel *m = [MLModel modelWithContentsOfURL:c configuration:cfg error:&err];
      if (!m) {
        NSLog(@"[apple] ANE load failed for %@: %@", path, err.localizedDescription);
        return false;
      }
      models_.push_back(m);
    }
    MLModel *m0 = models_[0];

    // Enumerated batch shapes the package physically supports; pin ONE of them.
    MLFeatureDescription *in = m0.modelDescription.inputDescriptionsByName[@"x"];
    if (!in || !in.multiArrayConstraint) return false;
    std::vector<int> shapes;
    for (NSArray<NSNumber *> *s in
         in.multiArrayConstraint.shapeConstraint.enumeratedShapes)
      if (s.count >= 1) shapes.push_back(s[0].intValue);
    if (shapes.empty()) shapes.push_back(in.multiArrayConstraint.shape[0].intValue);
    std::sort(shapes.begin(), shapes.end());
    shapes.erase(std::unique(shapes.begin(), shapes.end()), shapes.end());
    // KEEP shape_idx AT 0 (the SMALLEST enumerated shape). MEASURED: a predict
    // always runs the FULL pinned shape and its cost scales with that shape, so
    // any larger rung loses. FUNSD-50, K=24, interleaved A/B:
    //   default (W320=16, W480=8, W800=8)      105 img/s   4935 ANE rows/s
    //   TURBO_APPLE_ANE_SHAPE_IDX=1            64 img/s
    //   W480/W800 raised to 24 (W320 kept 16)  66 img/s    3110 ANE rows/s
    // F1 was 85.44% in every case, so this is purely a throughput knob.
    batch_ = shapes[std::min((size_t)std::max(shape_idx, 0), shapes.size() - 1)];
    if (contention_enabled()) {
      char nm[64];
      std::snprintf(nm, sizeof nm, "ane_predict_W%d_b%d", w_, batch_);
      st_predict_ = new Stat(strdup(nm));
      std::snprintf(nm, sizeof nm, "ane_qwait_W%d", w_);
      st_qwait_ = new Stat(strdup(nm));
    }
    if (contention_enabled()) {
      std::string all;
      for (int v : shapes) all += std::to_string(v) + " ";
      NSLog(@"[apple] ANE %s W%d: enumerated shapes [%s] pinned batch=%d workers=%d",
            pkg.c_str(), w, all.c_str(), batch_, workers);
    }

    for (NSString *k in m0.modelDescription.outputDescriptionsByName) {
      MLFeatureDescription *d = m0.modelDescription.outputDescriptionsByName[k];
      if (!d.multiArrayConstraint) continue;
      if (d.multiArrayConstraint.dataType == MLMultiArrayDataTypeInt32) idx_key_ = k;
      else score_key_ = k;
    }
    if (!idx_key_ || !score_key_) return false;

    // Warm predict at the pinned shape on every worker's model: learns T and
    // pays the ANE program-load cost outside the hot path.
    std::vector<float> zeros((size_t)batch_ * 3 * h_ * w_, 0.0f);
    for (size_t i = 0; i < models_.size(); ++i) {
      NSArray *shp = @[ @(batch_), @3, @(h_), @(w_) ];
      NSArray *strd = @[ @(3 * h_ * w_), @(h_ * w_), @(w_), @1 ];
      MLMultiArray *arr =
          [[MLMultiArray alloc] initWithDataPointer:zeros.data()
                                              shape:shp
                                           dataType:MLMultiArrayDataTypeFloat32
                                            strides:strd
                                        deallocator:nil
                                              error:&err];
      MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
          initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:arr]}
                       error:&err];
      id<MLFeatureProvider> o = [models_[i] predictionFromFeatures:fp error:&err];
      if (!o) {
        NSLog(@"[apple] ANE warm predict failed: %@", err.localizedDescription);
        return false;
      }
      if (T_ == 0) {
        MLMultiArray *idxA = [[o featureValueForName:idx_key_] multiArrayValue];
        T_ = idxA.shape.count >= 2 ? idxA.shape[1].intValue : 0;
      }
    }
    if (T_ <= 0) return false;

    for (int i = 0; i < workers; ++i)
      threads_.emplace_back([this, i] { worker_loop(i); });
    return true;
  }
}

bool AneBatchService::predict(MLModel *model, const float *rows, int nrows,
                              std::int32_t *idx_out, float *sc_out) {
  @autoreleasepool {
    NSError *err = nil;
    NSArray *shp = @[ @(nrows), @3, @(h_), @(w_) ];
    NSArray *strd = @[ @(3 * h_ * w_), @(h_ * w_), @(w_), @1 ];
    MLMultiArray *arr =
        [[MLMultiArray alloc] initWithDataPointer:(void *)rows
                                            shape:shp
                                         dataType:MLMultiArrayDataTypeFloat32
                                          strides:strd
                                      deallocator:nil
                                            error:&err];
    if (!arr) return false;
    MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:arr]}
                     error:&err];
    id<MLFeatureProvider> o = [model predictionFromFeatures:fp error:&err];
    if (!o) {
      NSLog(@"[apple] ANE predict W%d B%d failed: %@", w_, nrows,
            err.localizedDescription);
      return false;
    }
    MLMultiArray *idxA = [[o featureValueForName:idx_key_] multiArrayValue];
    MLMultiArray *scA = [[o featureValueForName:score_key_] multiArrayValue];
    const int T = T_;
    // getBytesWithHandler yields a SYNCED contiguous block whose row stride may
    // exceed T (CoreML pads rows) — recompute it from the byte count.
    [idxA getBytesWithHandler:^(const void *bytes, NSInteger size) {
      const std::int32_t *ip = (const std::int32_t *)bytes;
      const long rn = std::max(1L, (long)idxA.shape[0].integerValue);
      const long rs = (size / 4) / rn;
      for (int i = 0; i < nrows; ++i)
        for (int t = 0; t < T; ++t) idx_out[(size_t)i * T + t] = ip[i * rs + t];
    }];
    // The score head comes back FLOAT16 from the ANE while the index head is
    // INT32. Reading it as float32 yields garbage confidences that the
    // pipeline's drop-score filter silently deletes — the whole ANE half of the
    // page vanishes. Convert by the array's DECLARED dtype, never by assumption.
    const MLMultiArrayDataType sdt = scA.dataType;
    [scA getBytesWithHandler:^(const void *bytes, NSInteger size) {
      const long rn = std::max(1L, (long)scA.shape[0].integerValue);
      if (sdt == MLMultiArrayDataTypeFloat16) {
        const __fp16 *sp = (const __fp16 *)bytes;
        const long rs = (size / 2) / rn;
        for (int i = 0; i < nrows; ++i)
          for (int t = 0; t < T; ++t)
            sc_out[(size_t)i * T + t] = (float)sp[i * rs + t];
      } else {
        const float *sp = (const float *)bytes;
        const long rs = (size / 4) / rn;
        for (int i = 0; i < nrows; ++i)
          for (int t = 0; t < T; ++t) sc_out[(size_t)i * T + t] = sp[i * rs + t];
      }
    }];
    return true;
  }
}

void AneBatchService::worker_loop(int wi) {
  const size_t row_elems = (size_t)3 * h_ * w_;
  std::vector<float> stage((size_t)batch_ * row_elems, 0.0f);
  std::vector<std::int32_t> idx((size_t)batch_ * T_);
  std::vector<float> sc((size_t)batch_ * T_);
  MLModel *model = models_[wi];

  for (;;) {
    std::vector<std::shared_ptr<AneReq>> taken;
    int rows = 0;
    {
      std::unique_lock<std::mutex> lk(mu_);
      have_.wait(lk, [&] { return !q_.empty() || stop_.load(); });
      if (q_.empty()) return; // stopping
      // Take whatever is pending, up to a full batch, and go. Waiting for a
      // fuller batch is a MEASURED LOSS, not a missed opportunity: a request is
      // already ~9 rows against a 16/8-row pinned shape, so two of them almost
      // never fit together. A 2 ms fill window left rows/predict at 9.1, raised
      // per-predict latency 7.3 -> 11.6 ms and cost 91 -> 40 img/s at K=24.
      while (!q_.empty() && rows + q_.front()->count <= batch_) {
        rows += q_.front()->count;
        taken.push_back(q_.front());
        q_.pop_front();
      }
      if (taken.empty()) {
        // enqueue() rejects count > batch_, so the only way here is a request
        // that cannot fit even alone — impossible, but fail it rather than
        // truncate it into a partially-written buffer.
        auto r = q_.front();
        q_.pop_front();
        r->ok = false;
        r->done = true;
        done_cv_.notify_all();
        continue;
      }
    }

    // Gather the contributing rows into the pinned-shape staging batch. Every
    // taken request fits whole (submit() guarantees count <= batch_ and the
    // accumulate loop only takes what fits), so there is no truncation.
    int off = 0;
    for (auto &r : taken) {
      if (r->crops)
        std::memcpy(stage.data() + (size_t)off * row_elems, r->crops,
                    (size_t)r->count * row_elems * sizeof(float));
      off += r->count;
    }
    // How long the OLDEST taken request sat in the queue before a worker got to
    // it: the direct measure of the service being a throughput bottleneck.
    if (st_qwait_) {
      const auto now = std::chrono::steady_clock::now();
      for (auto &r : taken)
        st_qwait_->add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                           now - r->t_submit).count(), r->count);
    }
    bool ok;
    {
      TURBO_APPLE_PROF("ane.service.predict");
      TURBO_APPLE_STAT_N(ane_predict, rows); // units = USEFUL rows in this predict
      const auto tp0 = std::chrono::steady_clock::now();
      ok = predict(model, stage.data(), batch_, idx.data(), sc.data());
      if (st_predict_)
        st_predict_->add(
            (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tp0).count(), rows);
    }
    if (!ok)
      NSLog(@"[apple] ANE predict FAILED for %zu pooled request(s) at W%d — "
            @"blanking their tokens (was: silently scattering the previous "
            @"iteration's staging rows)", taken.size(), w_);
    // Scatter [T] rows back to each submitter and wake it. On failure write
    // BLANKS: `idx`/`sc` here are worker-local staging reused across iterations,
    // so scattering them unchecked hands one replica ANOTHER replica's tokens.
    off = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto &r : taken) {
        if (ok) {
          std::memcpy(r->idx, idx.data() + (size_t)off * T_,
                      (size_t)r->count * T_ * sizeof(std::int32_t));
          std::memcpy(r->sc, sc.data() + (size_t)off * T_,
                      (size_t)r->count * T_ * sizeof(float));
        } else {
          std::fill(r->idx, r->idx + (size_t)r->count * T_, kBlankIndex);
          std::fill(r->sc, r->sc + (size_t)r->count * T_, 0.0f);
        }
        off += r->count;
        r->ok = ok;
        r->done = true;
      }
      done_cv_.notify_all(); // once per batch — every taken request completed
    }
  }
}

AneBatchService *AneBatchService::get(const std::string &pkg, int h, int w,
                                      int shape_idx) {
  static std::mutex mu;
  static std::map<std::string, AneBatchService *> reg;
  const std::string key = pkg + "#" + std::to_string(shape_idx);
  std::lock_guard<std::mutex> lk(mu);
  auto it = reg.find(key);
  if (it != reg.end()) return it->second;
  auto *svc = new AneBatchService();
  const int workers = env::env_int("TURBO_APPLE_ANE_WORKERS", 2, 1, 64);
  if (!svc->start(pkg, h, w, shape_idx, workers)) {
    reg[key] = nullptr;
    return nullptr;
  }
  reg[key] = svc;
  return svc;
}

} // namespace

// --- AneRecEngine: a thin per-replica handle on the shared service -----------

struct AneRecEngine::Impl {
  AneBatchService *svc = nullptr;
};

AneRecEngine::AneRecEngine() : p_(new Impl) {}
AneRecEngine::~AneRecEngine() { delete p_; }

bool AneRecEngine::load(const std::string &mlpackage_path, int rec_h, int rec_w) {
  h_ = rec_h;
  w_ = rec_w;
  const int shape_idx = env::env_int("TURBO_APPLE_ANE_SHAPE_IDX", 0, 0, 4096);
  p_->svc = AneBatchService::get(mlpackage_path, rec_h, rec_w, shape_idx);
  if (!p_->svc) return false;
  // The service pins ONE batch shape, so that is the only batch the SHARED
  // planner may ask for; chunking above it is the planner's job.
  shapes_.assign(1, p_->svc->batch());
  T_ = p_->svc->time_steps();
  ready_ = T_ > 0;
  return ready_;
}

bool AneRecEngine::run(const float *crops, int batch, std::int32_t *idx_out,
                       float *score_out) {
  if (!ready_ || batch <= 0) return false;
  TURBO_APPLE_PROF("rec.ane(submit+wait)");
  return p_->svc->submit(crops, batch, idx_out, score_out);
}

// AneTicket is the engine-level handle over the service request: opaque in the
// header so mps_stages needs neither CoreML types nor the service definition.
struct AneTicket {
  std::shared_ptr<AneReq> req;
};

std::shared_ptr<AneTicket> AneRecEngine::begin_run(const float *crops,
                                                   int batch,
                                                   std::int32_t *idx_out,
                                                   float *score_out) {
  if (!ready_ || batch <= 0) return nullptr;
  auto t = std::make_shared<AneTicket>();
  t->req = p_->svc->enqueue(crops, batch, idx_out, score_out);
  // An enqueue rejection (oversized chunk) already blanked the buffers; hand
  // back a ticket with a null req so finish_run reports the failure.
  return t;
}

bool AneRecEngine::finish_run(const std::shared_ptr<AneTicket> &t) {
  if (!ready_ || !t) return false;
  TURBO_APPLE_PROF("rec.ane(wait)");
  return p_->svc->wait_done(t->req);
}

} // namespace turbo_ocr::apple
