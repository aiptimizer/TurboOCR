// MpsDetector / MpsRecognizer / MpsClassifier / MpsLayout implementation
// (see mps_stages.h). Mirrors tools/probes/apple/mps_ocr.mm's proven det + fused warp/rec path.

#import "apple/stages/mps_stages.h"
#import "apple/queue/metal_device_queue.h"
#import "apple/support/apple_profile.h"
#import "apple/support/apple_contention.h"
#import "apple/support/coreml_compile.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>

#include <filesystem>

#include "turbo_ocr/analysis/classification/cls_config.h"     // cls canvas/thresh/norm/flip
#include "turbo_ocr/base/env_utils.h"                      // env::* — every read is recorded
#include "turbo_ocr/base/geometry/perspective.h" // compute_crop_transform
#include "turbo_ocr/core/norm_params.h"            // SHARED norm factories
#include "turbo_ocr/core/db_post_config.h"      // shared DB geometry limits
#include "turbo_ocr/analysis/detection/det_config.h"          // read_db_params + env overrides
#include "turbo_ocr/analysis/detection/det_postprocess.h"    // extract_boxes_from_bitmap
#include "turbo_ocr/core/model_catalog.h"          // kV6DetConfig[Tiny] per-tier DB base
#include "apple/stages/stage_tier.h"
#include "turbo_ocr/analysis/recognition/ctc_decode.h"        // ctc_greedy_decode, load_label_dict
#include "turbo_ocr/analysis/recognition/rec_geometry.h"      // rec_input_width (bucket pick)
#include "turbo_ocr/analysis/recognition/rec_batching.h"      // SHARED bucket+batch planner

namespace turbo_ocr::apple {

// ===========================================================================
// MpsRecognizer — the proven fused warp -> rec -> argmax path
// ===========================================================================
MpsRecognizer::MpsRecognizer(std::shared_ptr<MetalAllocator> alloc,
                             std::string dict_path)
    : alloc_(std::move(alloc)), dict_path_(std::move(dict_path)) {}
MpsRecognizer::~MpsRecognizer() = default;

namespace {
bool file_exists(const std::string &p) {
  std::ifstream f(p);
  return f.good();
}
// An .mlpackage is a DIRECTORY, so ifstream is not a reliable probe for it.
bool path_exists(const std::string &p) {
  return [[NSFileManager defaultManager]
      fileExistsAtPath:[NSString stringWithUTF8String:p.c_str()]];
}
} // namespace

bool MpsRecognizer::load(const std::string &model_path) {
  // labels: {blank} + dict (matches OrtPaddleRec / mps_ocr.mm:53-54).
  labels_ = {"blank"};
  if (!turbo_ocr::recognition::load_label_dict(dict_path_, labels_)) {
    NSLog(@"[apple] rec dict load failed: %s", dict_path_.c_str());
    return false;
  }

  // Discover bucket export dirs. A direct export dir (contains graph.json) => a
  // single bucket; otherwise treat model_path as the BASE dir holding rec_b<W>
  // subdirs (mirrors tools/probes/apple/mps_ocr_funsd_bucket.mm's <rec_base_dir>).
  std::vector<std::string> dirs;
  if (file_exists(model_path + "/graph.json")) {
    dirs.push_back(model_path);
  } else {
    // DATA-DRIVEN bucket discovery: every rec_b<W> subdir under the base dir
    // that carries a graph.json becomes a bucket. The old hardcoded ladder
    // {320,480,800,1200,1600} padded a 500px line out to 800 (300 columns of
    // mid-gray) — the CPU/ORT recognizer snaps to a step-16 bucket and never
    // pads more than 15 columns (rec_geometry.h snap_width_step,
    // ort_paddle_rec.cpp REC_BUCKET_STEP=16). Finer exports close that
    // distribution gap without another code change: drop a new rec_b<W> export
    // in and it is picked up. TURBO_APPLE_REC_BUCKETS=w1,w2,... restricts the
    // set (A/B without moving files).
    std::vector<int> want;
    if (const std::string spec = env::env_or("TURBO_APPLE_REC_BUCKETS", ""); !spec.empty()) {
      const char *p = spec.c_str();
      while (*p) {
        char *end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) want.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
      }
    } else {
      // DEFAULT = the SHARED width table (recognition::kRecWidthBuckets), the
      // same ladder the TRT recognizer builds profiles for. Scanning the export
      // directory instead (the old behaviour) silently picked up all 42 exports
      // lying around and ran a 42-executable ladder at half the throughput —
      // and, worse, gave Apple a DIFFERENT ladder from every other backend.
      // TURBO_APPLE_REC_BUCKETS still overrides for A/B experiments.
      want.assign(turbo_ocr::recognition::kRecWidthBuckets.begin(),
                  turbo_ocr::recognition::kRecWidthBuckets.end());
    }
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    for (int w : want) {
      std::string d = model_path + "/rec_b" + std::to_string(w);
      if (file_exists(d + "/graph.json")) dirs.push_back(d);
    }
  }
  if (dirs.empty()) {
    NSLog(@"[apple] rec: no export (graph.json) found under %s", model_path.c_str());
    return false;
  }

  // GPU+ANE HYBRID SELECTION (hardware, not policy). Apple silicon has a third
  // compute engine reachable only through CoreML. Narrow buckets carry the bulk
  // of the crops and are the cheapest to run there, so they go to the ANE while
  // the GPU keeps det + warp + the wide buckets; with a replica pool the two
  // engines overlap for free. TURBO_APPLE_ANE_MAXW=<w> sets the split
  // (0 disables the ANE entirely); TURBO_APPLE_COREML_DIR points at the
  // rec_ane_<W>.mlpackage set. A bucket with no package falls back to MPSGraph.
  // 0 disables the ANE, so the lower bound is 0 rather than 1.
  int ane_maxw = env::env_int("TURBO_APPLE_ANE_MAXW", 800, 0, 8192);
  // DEFAULT LOOKUP ORDER for the package set, so a shipped model bundle is
  // self-contained: (1) TURBO_APPLE_COREML_DIR, an explicit override;
  // (2) <models>/coreml — `model_path` is the rec export base
  //     (<models>/rec_<tier>), so its parent is the models directory and a
  //     bundle dropped there (the apple_native_<tier> release asset layout)
  //     carries its own ANE packages;
  // (3) ~/.apple_ocr_ml/coreml — the bring-up machine's location, kept last so
  //     dev setups keep working. Before (2) existed, the DEFAULT pointed at
  //     the dev path only, and every wheel install ran GPU-only ANE-less
  //     native mode with nothing in the log but per-bucket fallbacks.
  std::string coreml_dir = env::env_or("TURBO_APPLE_COREML_DIR", "");
  if (coreml_dir.empty()) {
    std::error_code ec;
    const std::filesystem::path sibling =
        std::filesystem::path(model_path).parent_path() / "coreml";
    if (std::filesystem::is_directory(sibling, ec)) {
      coreml_dir = sibling.string();
    } else {
      // HOME is the process's ambient identity, not a TurboOCR knob, so it is
      // read raw — listing it in the startup override inventory would be noise.
      const char *home = std::getenv("HOME");  // pre-commit-allow-getenv (not a knob)
      coreml_dir = std::string(home ? home : "") + "/.apple_ocr_ml/coreml";
    }
  }

  // ---- ANE TIER RESOLUTION -------------------------------------------------
  // The ANE package encodes a FIXED output dictionary. PP-OCRv6's tiers do NOT
  // share one: tiny has 6906 classes, small/medium 18710. This code used to
  // compose `rec_ane_<W>.mlpackage` with NO tier component, so (a) the ~28
  // tiered packages on disk (rec_small_ane_<W>, rec_medium_ane_<W>) were
  // unreachable, and (b) pointing the backend at small/medium loaded the TINY
  // packages and decoded every crop against the wrong dictionary — garbage text,
  // silently, on the ANE buckets only.
  //
  // Tier comes from an explicit env override, else from the rec export path,
  // else from the dict path, else from the dictionary SIZE (the authoritative
  // signal: labels_ is {blank} + dict). If it cannot be determined, the ANE is
  // DISABLED rather than defaulting to tiny — a wrong dictionary is far worse
  // than losing the ANE, and the MPSGraph twin is the same export as the GPU rec
  // so the fallback is correct by construction.
  std::string tier = env::env_or("TURBO_APPLE_REC_TIER", "");
  if (tier.empty()) tier = tier_from_path(model_path);
  if (tier.empty()) tier = tier_from_path(dict_path_);
  if (tier.empty()) {
    // labels_ == {blank} + dict, so 6906 == keys_tiny.txt, 18710 == keys.txt.
    if (labels_.size() == 6906) tier = "tiny";
    else if (labels_.size() == 18710) tier = "small_or_medium";
  }
  // Last resort for the ONE genuinely ambiguous case. The medium export is
  // models/rec.onnx with models/keys.txt — no tier infix on either path — so
  // every signal above lands on "small_or_medium" and the ANE switches off,
  // costing ~9% throughput (measured 3.2 -> 3.5 img/s on FUNSD) on the very
  // tier the accuracy claims are made at. OCR_MODEL already names the tier, so
  // ask it.
  //
  // Only consulted when the dictionary has ALREADY narrowed the answer to
  // exactly {small, medium}, and only "small"/"medium" are accepted: OCR_MODEL
  // also takes PP-OCRv5 language entries (arabic/korean/...), and an explicit
  // REC_ONNX overrides OCR_MODEL independently, so it is a hint that breaks a
  // tie the dict already established — never a source of truth on its own. A
  // wrong dict here is silent garbage, which is why this stays this narrow.
  if (tier == "small_or_medium") {
    const std::string m = env::env_or("OCR_MODEL", "");
    if (m == "small" || m == "medium") tier = m;
  }
  // Package naming on disk: tiny is the historical un-infixed name; the other
  // tiers carry their tier as an infix. A per-tier SUBDIRECTORY layout
  // (<dir>/<tier>/rec_ane_<W>.mlpackage) is checked first so a clean install can
  // use it without another code change.
  auto ane_pkg_paths = [&](int width) {
    std::vector<std::string> out;
    const std::string w = std::to_string(width);
    if (!tier.empty() && tier != "small_or_medium") {
      out.push_back(coreml_dir + "/" + tier + "/rec_ane_" + w + ".mlpackage");
      out.push_back(coreml_dir + "/rec_" + (tier == "tiny" ? "" : tier + "_") +
                    "ane_" + w + ".mlpackage");
    }
    return out;
  };
  if (tier.empty() || tier == "small_or_medium") {
    NSLog(@"[apple] ANE DISABLED: cannot determine the rec tier for '%s' "
          @"(dict has %zu labels). An ANE package carries a FIXED dictionary; "
          @"loading the wrong tier's package silently decodes every crop against "
          @"the wrong dict. Set TURBO_APPLE_REC_TIER=tiny|small|medium to enable.",
          model_path.c_str(), labels_.size());
    ane_maxw = 0;
  }

  for (const auto &d : dirs) {
    Bucket b;
    b.engine = std::make_unique<MpsEngine>();
    if (!b.engine->load(d)) return false;
    b.engine->enable_argmax_head(true, /*class_axis*/ 2); // [B,T,C] -> [B,T]
    // FP16 rec compute (device-specific precision knob, not policy): the CRNN
    // rec head is numerically forgiving and Apple silicon runs FP16 conv/matmul
    // on a faster path. TURBO_APPLE_REC_FP16=0 falls back to FP32.
    b.engine->set_fp16(env::env_or("TURBO_APPLE_REC_FP16", "1") != "0");
    auto chw = b.engine->input_chw();
    if (chw.size() == 3) { rech_ = chw[1]; b.width = chw[2]; }
    buckets_.push_back(std::move(b));
  }
  std::sort(buckets_.begin(), buckets_.end(),
            [](const Bucket &x, const Bucket &y) { return x.width < y.width; });

  // EAGER SHAPE BUILD. The hot path must never compile an MPSGraph nor allocate:
  // the old code called prepare(n) with the page's exact per-bucket box count,
  // so every new count triggered a fresh compile mid-image (measured 7.5 ms/img
  // of pure compile). Every batch size the SHARED planner may ask for is built
  // here, and every buffer is sized for that bucket's largest batch.
  bucket_widths_.clear();
  for (auto &b : buckets_) bucket_widths_.push_back(b.width);
  bucket_rungs_.clear();
  for (auto &b : buckets_) {
    if (ane_maxw > 0 && b.width <= ane_maxw) {
      // TIER-AWARE package lookup. Only packages belonging to THIS tier are
      // considered; a missing package leaves the bucket on MPSGraph (same
      // export as the GPU rec => same dictionary) instead of silently reaching
      // for another tier's file.
      for (const auto &pkg : ane_pkg_paths(b.width)) {
        if (!path_exists(pkg)) continue;
        auto ane = std::make_unique<AneRecEngine>();
        if (!ane->load(pkg, rech_, b.width)) {
          NSLog(@"[apple] ANE package %s failed to load; bucket %d stays on "
                @"MPSGraph", pkg.c_str(), b.width);
          continue;
        }
        b.ane = std::move(ane);
        b.rungs = b.ane->batch_shapes(); // the package's ENUMERATED shapes
        b.T = b.ane->time_steps();
        b.engine.reset();                // the MPSGraph twin is dead weight
        break;
      }
      if (!b.ane)
        NSLog(@"[apple] no '%s'-tier ANE package for width %d under %s — bucket "
              @"stays on MPSGraph (NOT falling back to another tier's package)",
              tier.c_str(), b.width, coreml_dir.c_str());
    }
    if (!b.ane) {
      b.rungs = turbo_ocr::recognition::batch_ladder_for_width(b.width, rech_);
      for (int r : b.rungs)
        if (!b.engine->prepare(r)) return false;
      b.T = b.engine->time_steps();
    }
    if (b.rungs.empty() || b.T <= 0) return false;
    b.max_batch = b.rungs.back();
    bucket_rungs_.push_back(b.rungs);
    const size_t mb = (size_t)b.max_batch;
    b.h_buf = alloc_->allocate_buffer(mb * 9 * sizeof(float));
    b.cw_buf = alloc_->allocate_buffer(mb * sizeof(int));
    b.crops_buf =
        alloc_->allocate_buffer(mb * 3 * rech_ * b.width * sizeof(float));
    b.idx_buf = alloc_->allocate_buffer(mb * b.T * sizeof(int));
    b.max_buf = alloc_->allocate_buffer(mb * b.T * sizeof(float));
    if (!b.h_buf || !b.cw_buf || !b.crops_buf || !b.idx_buf || !b.max_buf)
      return false;
  }
  ready_ = true;
  return true;
}

std::vector<backend::RecResult>
MpsRecognizer::run(const backend::ImageView &img,
                   const std::vector<turbo_ocr::Box> &boxes,
                   backend::DeviceQueue &queue) {
  const int B = (int)boxes.size();
  dropped_crops_ = 0; // per-run; see last_dropped_crops()
  if (!ready_ || B == 0 || img.empty() || buckets_.empty()) return {};

  // SHARED policy: route every line to the smallest fitting width bucket and
  // pick the tightest STATIC batch >= that bucket's demand, chunked so one
  // submission never exceeds the shared element budget. No Apple-private ladder
  // (that fork is how the >1600px clamping bug survived on this path alone).
  const auto plan = turbo_ocr::recognition::plan_rec_batches(
      boxes, rech_, bucket_widths_, lists_, bucket_rungs_);

  // A bucket whose demand exceeds its top rung yields several chunks; those
  // chunks share the bucket's single scratch buffer, so they cannot ride the
  // same command buffer. Group chunks into ROUNDS (chunk #0 of every bucket,
  // then chunk #1, ...) — one command buffer + one host sync per round. On
  // FUNSD that is exactly one round for 49/50 pages.
  std::vector<int> seen(buckets_.size(), 0);
  std::vector<std::vector<turbo_ocr::recognition::RecBatchChunk>> rounds;
  for (const auto &c : plan) {
    const int r = seen[c.bucket]++;
    if ((int)rounds.size() <= r) rounds.emplace_back();
    rounds[r].push_back(c);
  }

  std::vector<backend::RecResult> out(B);
  // SHARED rec normalization ((v/255 - 0.5)/0.5 == v/127.5 - 1, RGB). The warp
  // shader bakes these constants today; passing the factory keeps the call site
  // honest and makes a future parameterized shader a one-line change.
  const backend::NormParams rec = backend::norm::rec_norm();
  // Per-chunk success. bk.idx_buf/max_buf are load()-time scratch reused for
  // every page, so a chunk whose forward pass failed must NOT be CTC-decoded —
  // it would produce the previous page's text with the previous page's
  // confidences, passing the drop-score filter with nothing logged.
  std::vector<char> chunk_ok;

  for (const auto &round : rounds) {
    {
      TURBO_APPLE_PROF("rec.homography(host)");
      TURBO_APPLE_STAT(rec_homography);
      for (const auto &c : round) {
        Bucket &bk = buckets_[c.bucket];
        auto *Hm = static_cast<float *>(bk.h_buf.data());
        auto *cw = static_cast<int *>(bk.cw_buf.data());
        for (int i = 0; i < c.count; ++i) {
          auto ct = turbo_ocr::compute_crop_transform(
              boxes[lists_[c.bucket][c.offset + i]], rech_, bk.width);
          for (int j = 0; j < 9; ++j) Hm[i * 9 + j] = ct.M_inv[j];
          cw[i] = std::min(ct.crop_width, bk.width);
        }
        // Pad rows of the static batch: content width 0 => warp_crops writes
        // the zero (mid-gray) canvas, so the padded rows are well-defined.
        for (int i = c.count; i < c.batch; ++i) {
          for (int j = 0; j < 9; ++j) Hm[i * 9 + j] = 0.0f;
          cw[i] = 0;
        }
      }
    }

    chunk_ok.assign(round.size(), 1);

    // Encode chunk ci's device work onto the CURRENT command buffer: every
    // bucket's crops are warped on the GPU (the warp is a Metal kernel either
    // way); only the WIDE (MPSGraph) buckets also run their rec here. GPU
    // buckets need the whole static batch warped (the executable's shape is
    // fixed); ANE buckets only need the REAL rows, because the batching
    // service copies exactly `count` rows into its own batch.
    auto encode_chunk = [&](std::size_t ci) {
      const auto &c = round[ci];
      Bucket &bk = buckets_[c.bucket];
      kernels_.warp_crops(img, static_cast<const float *>(bk.h_buf.data()),
                          static_cast<const int *>(bk.cw_buf.data()),
                          static_cast<float *>(bk.crops_buf.data()),
                          bk.engine ? c.batch : c.count, rech_, bk.width, rec,
                          queue);
      if (!bk.engine) return; // ANE bucket: warp only; predict is host-side
      backend::DeviceTensor in{
          .name = bk.engine->input_names().empty()
                      ? std::string{}
                      : bk.engine->input_names()[0],
          .data = bk.crops_buf.data(), .space = backend::DeviceKind::Metal,
          .dtype = backend::DType::F32,
          .shape = {c.batch, 3, rech_, bk.width}};
      backend::DeviceTensor out_idx{
          .name = "idx", .data = bk.idx_buf.data(),
          .space = backend::DeviceKind::Metal, .dtype = backend::DType::I32,
          .shape = {c.batch, bk.T, 1}};
      backend::DeviceTensor out_max{
          .name = "max", .data = bk.max_buf.data(),
          .space = backend::DeviceKind::Metal, .dtype = backend::DType::F32,
          .shape = {c.batch, bk.T, 1}};
      std::vector<backend::OutputLease> leases;
      if (!bk.engine->run({in}, {out_idx, out_max}, leases, queue)) {
        chunk_ok[ci] = 0;
        // Entries stay pre-sized and empty, so the returned length still
        // equals boxes.size() and the pipeline's under-return check cannot
        // see this — the count goes out through the SHARED seam
        // (backend::IRecognizer::last_dropped_crops) as well as the log.
        dropped_crops_ += c.count;
        NSLog(@"[apple] rec forward FAILED at W%d B%d — dropping %d line(s)",
              bk.width, c.batch, c.count);
      }
    };

    // CoreML predict for the ANE buckets, TWO-PHASE. begin puts every ANE
    // chunk in flight at once; finish collects them. One blocking run() per
    // bucket used to put ~2 serialized service round-trips (~4.4 ms each,
    // profiled) on the round's critical path. The warped crops are already in
    // host-visible (Metal SHARED) memory, so CoreML wraps them zero-copy —
    // but ONLY after the command buffer that warped them has been
    // checked-synced (a failed warp buffer would hand the ANE the previous
    // page's crops).
    std::vector<std::shared_ptr<AneTicket>> ane_tickets(round.size());
    auto begin_ane = [&](std::size_t ci) {
      const auto &c = round[ci];
      Bucket &bk = buckets_[c.bucket];
      if (!bk.ane || !chunk_ok[ci]) return;
      ane_tickets[ci] = bk.ane->begin_run(
          static_cast<const float *>(bk.crops_buf.data()), c.count,
          static_cast<std::int32_t *>(bk.idx_buf.data()),
          static_cast<float *>(bk.max_buf.data()));
    };
    auto finish_ane = [&](std::size_t ci) {
      const auto &c = round[ci];
      Bucket &bk = buckets_[c.bucket];
      if (!bk.ane || !chunk_ok[ci]) return;
      if (!bk.ane->finish_run(ane_tickets[ci])) {
        chunk_ok[ci] = 0;
        // Say so. These crops keep their PRE-SIZED empty entries, so the
        // page comes back the right length and the pipeline's under-return
        // check cannot see the loss — hence the SHARED seam count as well as
        // the log (an ANE failure otherwise looks like genuinely blank lines).
        dropped_crops_ += c.count;
        NSLog(@"[apple] rec ANE forward FAILED at W%d — dropping %d line(s)",
              bk.width, c.count);
      }
    };

    // Invalidate this round's still-good chunks on one side of the engine
    // split (or both). bk.idx_buf / bk.max_buf are allocated once per bucket
    // at load() and reused for every chunk of every page, so a chunk whose
    // command buffer failed at EXECUTION still holds the PREVIOUS page's CTC
    // indices — ctc_greedy_decode below would return that page's complete,
    // correct transcript for this one. Encode-level engine->run() cannot see
    // that, which is why a failed checked-sync lands here.
    enum class Side { Ane, Gpu, Both };
    auto invalidate = [&](Side side, const char *why) {
      int dropped = 0;
      for (std::size_t ci = 0; ci < round.size(); ++ci) {
        const bool is_ane = static_cast<bool>(buckets_[round[ci].bucket].ane);
        if (side == Side::Ane && !is_ane) continue;
        if (side == Side::Gpu && is_ane) continue;
        if (!chunk_ok[ci]) continue; // already counted at encode/predict time
        chunk_ok[ci] = 0;
        dropped_crops_ += round[ci].count;
        dropped += round[ci].count;
      }
      if (dropped > 0)
        NSLog(@"[apple] %s — dropping %d line(s) rather than decoding the "
              @"previous page's CTC indices", why, dropped);
    };

    bool round_has_ane = false, round_has_gpu = false;
    for (const auto &c : round)
      (buckets_[c.bucket].ane ? round_has_ane : round_has_gpu) = true;

    if (round_has_ane && round_has_gpu) {
      // HYBRID OVERLAP. The old shape — ONE command buffer (warp all + rec) →
      // checked sync → CoreML predict — ran the two engines SERIALLY within a
      // replica: profiled at ~8.0 ms ANE after ~4.8 ms GPU per round, with the
      // hoped-for cross-replica overlap measured at <10%. Split the submission
      // instead: (1) warp the narrow (ANE) buckets in their own small command
      // buffer and checked-sync it (sub-ms), (2) commit the wide buckets'
      // warp+rec+argmax buffer WITHOUT waiting, (3) run the CoreML predict on
      // the host while that buffer executes on the GPU, (4) checked-sync the
      // GPU. The two engines overlap inside one image; the round's rec cost
      // drops from (ane + gpu) toward max(ane, gpu).
      bool ane_warp_ok;
      {
        TURBO_APPLE_PROF("rec.anewarp(warp+sync)");
        TURBO_APPLE_STAT(rec_ane_warp);
        const unsigned long long mark0 = as_metal(queue).sync_mark();
        {
          backend::BatchScope batch(queue);
          for (std::size_t ci = 0; ci < round.size(); ++ci)
            if (buckets_[round[ci].bucket].ane) encode_chunk(ci);
        }
        // This sync MUST precede the GPU commit below: sync_ok() waits on the
        // queue's highest submitted timeline value, so committing the wide
        // buckets first would make this wait cover them too and re-serialize
        // the whole round.
        ane_warp_ok = as_metal(queue).sync_ok(mark0);
      }
      if (!ane_warp_ok)
        invalidate(Side::Ane, "rec ANE warp command buffer FAILED");
      // Every ANE chunk goes in flight BEFORE the GPU encode below: the pooled
      // service's workers predict while this thread is still encoding, so the
      // host-side encode (~2 ms) is covered by the ANE window too.
      for (std::size_t ci = 0; ci < round.size(); ++ci) begin_ane(ci);
      const unsigned long long gpu_mark = as_metal(queue).sync_mark();
      {
        TURBO_APPLE_PROF("rec.gpu(encode)");
        TURBO_APPLE_STAT(rec_gpu_encode);
        backend::BatchScope batch(queue);
        for (std::size_t ci = 0; ci < round.size(); ++ci)
          if (!buckets_[round[ci].bucket].ane) encode_chunk(ci);
      }
      {
        TURBO_APPLE_PROF("rec.ane(predict)");
        TURBO_APPLE_STAT(rec_ane_stage);
        for (std::size_t ci = 0; ci < round.size(); ++ci) finish_ane(ci);
      }
      {
        TURBO_APPLE_PROF("rec.gpu(sync)");
        TURBO_APPLE_STAT(rec_gpu_sync);
        if (!as_metal(queue).sync_ok(gpu_mark))
          invalidate(Side::Gpu, "rec round INVALIDATED by a failed command "
                                "buffer");
      }
    } else {
      // Single-engine round: ONE command buffer for the WHOLE round — every
      // bucket's warp AND (for wide buckets) its rec + GPU argmax. This is the
      // seam's BatchScope (device_queue.h begin/end_batch) doing what it
      // exists for — on Metal one MPSCommandBuffer spanning warp -> rec ->
      // argmax for all buckets, on CUDA/Host a no-op. The old code opened a
      // BatchScope and blocked on synchronize() PER BUCKET: ~5.8 commits
      // + 5.8 host round-trips per page at ~9 ms each.
      const unsigned long long round_mark = as_metal(queue).sync_mark();
      {
        TURBO_APPLE_PROF("rec.gpu(warp+rec+sync)");
        TURBO_APPLE_STAT(rec_gpu_total);
        {
          TURBO_APPLE_STAT(rec_gpu_encode);
          backend::BatchScope batch(queue);
          for (std::size_t ci = 0; ci < round.size(); ++ci) encode_chunk(ci);
        }
        {
          TURBO_APPLE_STAT(rec_gpu_sync);
          if (!as_metal(queue).sync_ok(round_mark))
            invalidate(Side::Both, "rec round INVALIDATED by a failed command "
                                   "buffer");
        }
      }
      if (round_has_ane) {
        TURBO_APPLE_PROF("rec.ane(predict)");
        TURBO_APPLE_STAT(rec_ane_stage);
        for (std::size_t ci = 0; ci < round.size(); ++ci) begin_ane(ci);
        for (std::size_t ci = 0; ci < round.size(); ++ci) finish_ane(ci);
      }
    }

    {
      TURBO_APPLE_PROF("rec.ctc(host)");
      TURBO_APPLE_STAT(rec_ctc);
      for (std::size_t ci = 0; ci < round.size(); ++ci) {
        const auto &c = round[ci];
        if (!chunk_ok[ci]) continue; // leaves out[] entries as ("", 0) -> dropped
        Bucket &bk = buckets_[c.bucket];
        const auto *idx = static_cast<const int *>(bk.idx_buf.data());
        const auto *sc = static_cast<const float *>(bk.max_buf.data());
        for (int i = 0; i < c.count; ++i)
          out[lists_[c.bucket][c.offset + i]] =
              turbo_ocr::recognition::ctc_greedy_decode(
                  idx + (size_t)i * bk.T, sc + (size_t)i * bk.T, bk.T, labels_);
      }
    }
  }
  return out;
}

// ===========================================================================
// MpsClassifier — text-line 0/180 angle (structural; validate golden)
// ===========================================================================
MpsClassifier::MpsClassifier(std::shared_ptr<MetalAllocator> alloc)
    : alloc_(std::move(alloc)) {}
MpsClassifier::~MpsClassifier() = default;

bool MpsClassifier::load(const std::string &model_path) {
  if (!engine_.load(model_path)) return false;
  engine_.enable_argmax_head(true, /*class_axis*/ 1); // [B,2] -> [B,1]
  auto chw = engine_.input_chw();
  if (chw.size() == 3) { clsh_ = chw[1]; clsw_ = chw[2]; }
  // Same SHARED static-batch policy as the recognizer, applied to the cls
  // canvas: pre-build one executable per rung and size the scratch for the top
  // rung, so a page's box count never triggers a compile or an allocation.
  rungs_ = turbo_ocr::recognition::batch_ladder_for_width(clsw_, clsh_);
  max_batch_ = rungs_.back();
  for (int r : rungs_)
    if (!engine_.prepare(r)) return false;
  const size_t mb = (size_t)max_batch_;
  h_buf_ = alloc_->allocate_buffer(mb * 9 * sizeof(float));
  cw_buf_ = alloc_->allocate_buffer(mb * sizeof(int));
  crops_buf_ = alloc_->allocate_buffer(mb * 3 * clsh_ * clsw_ * sizeof(float));
  idx_buf_ = alloc_->allocate_buffer(mb * sizeof(int));
  max_buf_ = alloc_->allocate_buffer(mb * sizeof(float));
  if (!h_buf_ || !cw_buf_ || !crops_buf_ || !idx_buf_ || !max_buf_) return false;
  ready_ = true;
  return true;
}

void MpsClassifier::run(const backend::ImageView &img,
                       std::vector<turbo_ocr::Box> &boxes,
                       backend::DeviceQueue &queue) {
  const int B = (int)boxes.size();
  if (!ready_ || B == 0 || img.empty()) return;

  auto *Hm = static_cast<float *>(h_buf_.data());
  auto *cw = static_cast<int *>(cw_buf_.data());
  const auto *idx = static_cast<const int *>(idx_buf_.data());
  const auto *scr = static_cast<const float *>(max_buf_.data());
  // SHARED cls normalization (== rec's (v/127.5 - 1), NOT ImageNet — see
  // classification::cls_norm(); three backends have shipped the ImageNet bug).
  const backend::NormParams norm = classification::cls_norm();
  // IClassifier::run returns void, so this count has no caller. Kept and LOGGED
  // rather than deleted: "how many crops came in upside down" is the only signal
  // that the orientation classifier is doing anything, and a silently-discarded
  // counter is how a dead cls stage goes unnoticed. Matches the NSLog above,
  // which already reports the failure case.
  int flipped = 0;

  // Chunk at the top rung, snapping each chunk to its tightest static batch
  // (recognition::snap_batch — the SHARED policy, same rungs the recognizer
  // uses). One command buffer + one host sync per chunk.
  for (int off = 0; off < B; off += max_batch_) {
    const int cnt = std::min(max_batch_, B - off);
    const int nb = turbo_ocr::recognition::snap_batch(cnt, rungs_);
    {
      TURBO_APPLE_PROF("cls.prepare(compile)");
      if (!engine_.prepare(nb)) return;
    }
    for (int i = 0; i < cnt; ++i) {
      auto ct = turbo_ocr::compute_crop_transform(boxes[off + i], clsh_, clsw_);
      for (int k = 0; k < 9; ++k) Hm[i * 9 + k] = ct.M_inv[k];
      cw[i] = std::min(ct.crop_width, clsw_);
    }
    for (int i = cnt; i < nb; ++i) { // pad rows -> zero canvas
      for (int k = 0; k < 9; ++k) Hm[i * 9 + k] = 0.0f;
      cw[i] = 0;
    }

    bool cls_ok = false;
    const unsigned long long cls_mark = as_metal(queue).sync_mark();
    {
      TURBO_APPLE_PROF("cls.gpu(warp+fwd+sync)");
      {
        backend::BatchScope batch(queue);
        kernels_.warp_crops(img, Hm, cw, static_cast<float *>(crops_buf_.data()),
                            nb, clsh_, clsw_, norm, queue);
        backend::DeviceTensor in{
            .name = engine_.input_names().empty() ? std::string{}
                                                  : engine_.input_names()[0],
            .data = crops_buf_.data(), .space = backend::DeviceKind::Metal,
            .dtype = backend::DType::F32, .shape = {nb, 3, clsh_, clsw_}};
        backend::DeviceTensor out_idx{
            .name = "idx", .data = idx_buf_.data(),
            .space = backend::DeviceKind::Metal, .dtype = backend::DType::I32,
            .shape = {nb, 1}};
        backend::DeviceTensor out_max{
            .name = "max", .data = max_buf_.data(),
            .space = backend::DeviceKind::Metal, .dtype = backend::DType::F32,
            .shape = {nb, 1}};
        std::vector<backend::OutputLease> leases;
        cls_ok = engine_.run({in}, {out_idx, out_max}, leases, queue);
      }
      // idx_buf_/max_buf_ are reused across pages exactly like the det/rec
      // scratch; an execution failure here would apply the PREVIOUS page's
      // rotation decisions to this page's crops.
      cls_ok = as_metal(queue).sync_ok(cls_mark) && cls_ok;
    }
    // idx_buf_/max_buf_ are load()-time scratch reused across pages: decoding a
    // failed forward pass would flip THIS page's boxes according to the PREVIOUS
    // page's angles. Leaving the boxes unflipped is the safe degradation.
    if (!cls_ok) {
      NSLog(@"[apple] cls forward FAILED — leaving %d box(es) unflipped", cnt);
      continue;
    }

    // Decision rule — IDENTICAL to OrtPaddleCls::run (src/analysis/classification/
    // ort_paddle_cls.cpp): flip only when the 180 class both WINS and clears
    // the confidence threshold (s180 > s0 && s180 > 0.9). cls.onnx ends in
    // Softmax, so the argmax head's max value IS s180 when idx == 1.
    for (int i = 0; i < cnt; ++i) {
      // SHARED decision + SHARED rotation (classification::*): identical to the
      // CPU/Intel/AMD sites, expressed once.
      if (classification::should_flip_180_argmax(idx[i], scr[i])) {
        classification::flip_quad_180(boxes[off + i]);
        ++flipped;
      }
    }
  }
  // DEBUG-GATED. IClassifier::run returns void, so this count has no caller;
  // it is kept because "how many crops came in upside down" is the only signal
  // that the orientation classifier is doing anything, and a silently-discarded
  // counter is how a dead cls stage goes unnoticed. But it fires on nearly every
  // page — the classifier only sees the vertical-looking boxes, which are
  // exactly the ambiguous ones — so an ungated NSLog here is one line per image
  // per replica in every benchmark and every production log. Behind the same
  // switch as the rest of this arm's diagnostics.
  if (flipped > 0 && turbo_ocr::apple::Profiler::enabled())
    NSLog(@"[apple] cls flipped %d of %d box(es) 180", flipped, B);
}


// ===========================================================================
// MpsLayout — STUB (multi-IO not yet supported)
// ===========================================================================
bool MpsLayout::load(const std::string & /*model_path*/) {
  // PP-DocLayoutV3 takes image + im_shape + scale_factor (3 inputs) and emits
  // detection rows in original coords. The single-input MPSGraph builder
  // (mps_rec_build.h) handles one placeholder; multi-IO + the layout post-decode
  // are a TODO(apple-layout-multi-io). Report unavailable so the pipeline
  // runs det+rec only.
  NSLog(@"[apple] layout stage not implemented on Apple yet "
        "(TODO(apple-layout-multi-io): MPSGraph multi-input feed)");
  return false;
}

std::vector<turbo_ocr::layout::LayoutBox>
MpsLayout::run(const backend::ImageView &, int, int, float, backend::DeviceQueue &) {
  return {};
}

} // namespace turbo_ocr::apple
