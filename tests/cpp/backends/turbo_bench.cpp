// turbo_bench — the ONE accuracy + throughput harness for EVERY backend.
//
// ============================================================================
// WHY ONE BINARY
// ============================================================================
// This replaces tests/cpp/backends/{funsd_unified_cpu.cpp, funsd_unified_apple.mm,
// funsd_unified_apple_conc.mm}: three copies of the same protocol, each pinned
// to one backend, none of whose numbers could be compared with another machine's
// because nothing recorded WHAT was run.
//
// A driver that only calls backend::make_backend(name) + UnifiedOcrPipeline
// needs NO Objective-C++ and no vendor header: every Metal/MPSGraph symbol is
// inside libturbo_ocr_backend_apple.a, behind the seam, and CUDA will be inside
// libturbo_ocr_backend_nvidia.a exactly the same way. So this is a plain .cpp
// that runs the IDENTICAL protocol on cpu / apple / nvidia / amd / intel:
//
//     make_backend(name) -> load_stages() x K -> K UnifiedOcrPipeline replicas
//     -> warmup (EXCLUDED from timing) -> timed run -> words JSON + metrics JSON
//
// ============================================================================
// MEASUREMENT DISCIPLINE (baked in, not optional — each cost us real time)
// ============================================================================
//  1. THROUGHPUT WINDOWS MUST BE >= 15 s. A short window is dominated by model
//     load and graph JIT; that is how a fabricated "288 img/s" reading was once
//     produced. Shorter windows FAIL unless --allow-short-window is passed.
//  2. WALL-CLOCK CROSS-CHECK. Two independent clocks measure the same timed
//     region (one steady_clock span vs the summed per-image latencies / K). A
//     >5% disagreement means the window contains work that is not per-image OCR
//     and the rate is untrustworthy — it FAILS LOUDLY. This is the check that
//     caught the 288 artifact. Self-test it with --selftest-skew-ms.
//  3. NEVER A THROUGHPUT NUMBER WITHOUT ITS ACCURACY. Every run scores its own
//     transcript against the FUNSD GT (bag-of-words F1, the same metric as
//     tools/bench/score_funsd.py) and prints them together. --no-score opts out and
//     says so loudly.
//  4. THERMAL DRIFT. Absolute numbers drift ~12% downward over a long session,
//     so a cross-session or cross-machine A-vs-B comparison of raw throughput is
//     untrustworthy. For any comparison use INTERLEAVED paired mode:
//     --ab cpu,apple runs A,B,A,B,... and reports the per-pair ratio.
//  5. PROVENANCE IN THE JSON. hostname, OS, chip, backend, device, model paths
//     AND THEIR SHA256, the image-set hash, pool size, thread count, and every
//     relevant env var. This is the part that makes a run on an NVIDIA box
//     comparable to a run on this M3 Max — without it, two F1 numbers are just
//     two numbers.
//
// ============================================================================
// USAGE
// ============================================================================
//   turbo_bench [--backend cpu|apple|nvidia|amd|intel|auto]
//               --images <dir> [--count N] [--tier tiny|small|medium]
//               [--threads K] [--repeat R] [--chunk C]
//               [--det P --rec P --keys P --cls P --layout P]
//               [--warmup N] [--out metrics.json] [--words words.json]
//               [--gt tests/benchmark/funsd_gt_words.json] [--no-score]
//               [--ab backendA,backendB] [--allow-short-window]
//               [--det-batch N] [--det-batch-delay-us U] [--ab-det-batch A,B]
//               [--ab-rounds N]
//               [--assert-f1 X] [--assert-throughput X]
//
// BACKWARD COMPATIBILITY: the retired funsd_unified_cpu CLI
//     funsd_unified_cpu <cache_dir> <N> <out.json> [--det ...]
// still works verbatim — leading positionals are read as
// <images-dir> <count> <words-json>. The build also emits shim scripts named
// funsd_unified_cpu / funsd_unified_apple / funsd_unified_apple_conc.

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "harness.h"
#include "turbo_ocr/base/log/stage_profiler.h"
#include "turbo_ocr/pipeline/unified/stage_batcher.h"
#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

using namespace turbo_ocr;
using namespace turbo_ocr::harness;

namespace {

struct RunConfig {
  std::string backend;
  std::string images;
  int count = 50;
  std::string tier = "tiny";
  int threads = 1;
  int repeat = 1;
  int chunk = 1;
  int warmup = 1;
  bool score = true;
  bool allow_short = false;
  double selftest_skew_ms = 0;
  std::string gt;
  // Cross-request detection batching (include/turbo_ocr/pipeline/unified/stage_batcher.h).
  //  -1 => leave it to env/backend advice (the default: normally OFF)
  //   0 => explicitly OFF (no batcher object at all)
  //   1 => instrument-only: detection runs inline exactly as with 0, but
  //        det ms/call is recorded — this is the honest "A" arm of an A/B
  //  >1 => coalescing, up to that many images per submission
  int det_batch = -1;
  int det_batch_delay_us = 0;
  // CORRECTNESS AUDIT (--words-all <file>). Normally only the FIRST pass over
  // the image set owns the transcript (`j < N`), so with --repeat R only 1/R of
  // the pages processed is ever compared against anything — and that pass
  // happens in the first second of the window, before any sustained-load effect
  // exists. That is why a cross-page mix-up under load can run for a whole
  // window and leave no trace in the scored output. With --words-all EVERY
  // processed page is recorded next to the index of the image it was supposed
  // to be, so a mix-up anywhere in the window is detectable.
  std::string words_all;
};

struct RunResult {
  std::string backend_name;
  backend::BackendCaps caps;
  ModelPaths models;
  Accuracy acc;
  TimingVerdict timing;
  double load_ms = 0;
  double warmup_ms = 0;
  long images_processed = 0;
  double p50_ms = 0, p95_ms = 0, mean_ms = 0;
  UtilStats gpu_util, ane_compiler;
  int replicas = 0;
  std::string images_sha;
  std::vector<std::vector<std::string>> words;
  // --words-all: one entry per PROCESSED page (all passes), each tagged with
  // the image index it should correspond to. See RunConfig::words_all.
  std::vector<std::vector<std::string>> words_all;
  std::vector<int> words_all_idx;
  std::string det_batch_json = "null";
  std::string det_batch_line;
  bool ok = false;
};

// ---------------------------------------------------------------------------
// The protocol. IDENTICAL for every backend — this function names no vendor.
// ---------------------------------------------------------------------------
RunResult run_one(const RunConfig &rc, const ImageSet &set, const Args &args) {
  RunResult r;
  auto b = open_backend(rc.backend == "auto" ? std::string() : rc.backend);
  if (!b) return r;
  r.caps = b->caps();
  r.backend_name = r.caps.name;
  r.images_sha = set.sha256;

  r.models = resolve_models(args, r.backend_name, rc.tier);
  auto cfg = to_config(r.models);

  const int K = rc.threads > 0 ? rc.threads : std::max(1, r.caps.recommended_pool_size);
  r.replicas = K;

  std::printf("\n=== backend '%s' (device=%s) | %d replica(s) | tier=%s ===\n",
              r.backend_name.c_str(), backend::device_kind_name(r.caps.device), K,
              rc.tier.c_str());
  std::printf("  det   : %s\n  rec   : %s\n  keys  : %s\n  cls   : %s\n",
              r.models.det.c_str(), r.models.rec.c_str(), r.models.keys.c_str(),
              r.models.cls.empty() ? "(disabled)" : r.models.cls.c_str());
  if (!r.models.layout.empty()) std::printf("  layout: %s\n", r.models.layout.c_str());

  // --- cross-request detection batching -------------------------------------
  // Configured BEFORE the replicas are constructed, because each replica asks
  // for the shared batcher in its constructor. --det-batch -1 (the default)
  // leaves the env/backend-advice path alone, so an untouched command line
  // behaves exactly as it did before this flag existed.
  if (rc.det_batch >= 0) {
    pipeline::DetBatchConfig bc;
    bc.enabled = rc.det_batch >= 1;
    bc.preferred_batch_size = std::max(1, rc.det_batch);
    bc.max_queue_delay_us = rc.det_batch_delay_us;
    pipeline::configure_detection_batching(bc);
    std::printf("  det-batch: %s\n",
                rc.det_batch == 0   ? "OFF (no batcher; det_->run() inline)"
                : rc.det_batch == 1 ? "instrument-only (inline, timed)"
                                    : "COALESCING");
  }

  // --- load: K independent replicas, each with its own stages + queue --------
  // (the production shape: one pipeline + one DeviceQueue per worker, see
  // BackendCaps::recommended_pool_size / pipeline/make_infer_func.h)
  auto t_load = clk::now();
  std::vector<std::unique_ptr<pipeline::UnifiedOcrPipeline>> pipes;
  for (int k = 0; k < K; ++k) {
    auto stages = b->load_stages(cfg);
    if (!stages.available.detector || !stages.available.recognizer) {
      std::fprintf(stderr,
                   "replica %d: load_stages gave det=%d rec=%d cls=%d — required "
                   "stage missing (check model paths)\n",
                   k, stages.available.detector, stages.available.recognizer,
                   stages.available.classifier);
      return r;
    }
    if (k == 0)
      std::printf("  stages: det=%d rec=%d cls=%d layout=%d\n",
                  stages.available.detector, stages.available.recognizer,
                  stages.available.classifier,
                  stages.available.optional.get(
                      capability::CapabilityId::Layout));
    auto q = b->make_queue();
    pipes.push_back(std::make_unique<pipeline::UnifiedOcrPipeline>(
        *b, std::move(stages), std::move(q)));
  }
  r.load_ms = ms_since(t_load);

  // --- warmup: EXCLUDED from the timed window -------------------------------
  // Every replica runs `--warmup N` real pages so lazy allocation, graph JIT and
  // first-touch faults are paid before the clock starts. This is exactly what
  // the short-window artifact was: warmup cost landing inside the measurement.
  auto t_warm = clk::now();
  for (int k = 0; k < K; ++k) {
    pipes[k]->warmup();
    for (int w = 0; w < rc.warmup && !set.imgs.empty(); ++w)
      (void)pipes[k]->run(set.imgs[w % set.imgs.size()]);
  }
  r.warmup_ms = ms_since(t_warm);
  std::printf("  load=%.0f ms  warmup=%.0f ms (both EXCLUDED from timing)\n",
              r.load_ms, r.warmup_ms);
  // Detection-batcher counters must describe the TIMED window only — warmup's
  // batch-1 submissions would otherwise drag the mean batch size down.
  auto det_batcher = pipeline::current_detection_batcher();
  if (det_batcher) det_batcher->reset_stats();
  // PROFILE_STAGES=1: the per-stage accumulators must describe the TIMED window
  // only — discard warmup's contribution (same rule as the batcher counters).
  if (prof::enabled()) (void)prof::dump_json_and_reset();

  // --- timed run ------------------------------------------------------------
  const int N = static_cast<int>(set.imgs.size());
  const long total = static_cast<long>(N) * rc.repeat;
  r.words.assign(N, {});
  std::vector<std::vector<double>> lat(K);
  std::atomic<long> next{0};
  // --words-all audit sink. Pre-sized to `total` and indexed by the work item
  // `j`, so the recording itself needs no lock and cannot itself mis-attribute
  // a page (a mutex-protected append would make the audit's own ordering a
  // suspect in exactly the investigation it exists to serve).
  const bool audit_all = !rc.words_all.empty();
  std::vector<std::vector<std::string>> all_words;
  if (audit_all) all_words.assign(static_cast<std::size_t>(total), {});

  // Device saturation is sampled DURING the timed window (on its own thread,
  // outside the measured work): a rate with no utilization next to it cannot
  // tell "hardware limit" from "device idling on the host".
  UtilSampler sampler;
  sampler.start();
  auto t_all = clk::now();
  {
    std::vector<std::thread> ths;
    ths.reserve(K);
    for (int k = 0; k < K; ++k) {
      ths.emplace_back([&, k] {
        std::vector<cv::Mat> chunk;
        for (;;) {
          const long j = next.fetch_add(rc.chunk);
          if (j >= total) break;
          const int cnt = static_cast<int>(std::min<long>(rc.chunk, total - j));
          auto t = clk::now();
          if (rc.chunk == 1) {
            const int i = static_cast<int>(j % N);
            auto res = pipes[k]->run(set.imgs[i]);
            lat[k].push_back(ms_since(t));
            if (j < N || audit_all) {
              std::vector<std::string> w;
              w.reserve(res.size());
              for (auto &it : res) w.push_back(it.text);
              if (audit_all) all_words[static_cast<std::size_t>(j)] = w;
              if (j < N) r.words[i] = std::move(w); // the first pass is scored
            }
          } else {
            chunk.clear();
            for (int c = 0; c < cnt; ++c) chunk.push_back(set.imgs[(j + c) % N]);
            auto outs = pipes[k]->run_batch(chunk);
            const double dt = ms_since(t);
            for (int c = 0; c < cnt; ++c) lat[k].push_back(dt / cnt);
            for (int c = 0; c < cnt && c < static_cast<int>(outs.size()); ++c) {
              if (j + c >= N) continue;
              const int i = static_cast<int>((j + c) % N);
              std::vector<std::string> w;
              w.reserve(outs[c].size());
              for (auto &it : outs[c]) w.push_back(it.text);
              r.words[i] = std::move(w);
            }
          }
        }
      });
    }
    for (auto &t : ths) t.join();
    if (audit_all) {
      r.words_all = std::move(all_words);
      r.words_all_idx.reserve(r.words_all.size());
      for (long j = 0; j < total; ++j)
        r.words_all_idx.push_back(static_cast<int>(j % N));
    }
    // SELF-TEST hook for discipline rule 2: this sleep is INSIDE the timed
    // window but is accounted for by no per-image latency — exactly the shape of
    // "model load landed inside the measurement", which is what produced the
    // bogus 288 img/s. The cross-check must fire. See README "proving the
    // cross-check works".
    if (rc.selftest_skew_ms > 0)
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<long>(rc.selftest_skew_ms)));
  }
  const double wall_ms = ms_since(t_all);
  sampler.stop();
  // Stage attribution for THIS timed window (totals in ms; divide by
  // images_processed for ms/img). Printed here, not in print_run, so the dump
  // resets exactly at the window edge and A/B arms cannot bleed into each other.
  if (prof::enabled())
    std::printf("  PROFILE stage_ms over %ld imgs: %s\n", total,
                prof::dump_json_and_reset().c_str());
  r.gpu_util = sampler.gpu();
  r.ane_compiler = sampler.ane_compiler();
  if (det_batcher) {
    r.det_batch_json = det_batcher->stats_json();
    r.det_batch_line = det_batcher->stats_line();
  }

  std::vector<double> all;
  for (auto &v : lat) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  if (!all.empty()) {
    double s = 0;
    for (double d : all) s += d;
    r.mean_ms = s / all.size();
    r.p50_ms = all[all.size() / 2];
    r.p95_ms = all[(all.size() * 95) / 100];
  }
  r.images_processed = total;
  r.timing = check_timing(wall_ms, all, K, total);

  // --- accuracy, ALWAYS reported next to throughput -------------------------
  if (rc.score) {
    r.acc = score_words(r.words, rc.gt);
    if (!r.acc.scored)
      std::printf("  !! could not score: GT '%s' unreadable — a throughput number "
                  "without accuracy is meaningless\n", rc.gt.c_str());
  }
  r.ok = true;
  return r;
}

void print_run(const RunResult &r, bool strict) {
  std::printf("\nRESULT backend=%s  images=%ld  replicas=%d\n", r.backend_name.c_str(),
              r.images_processed, r.replicas);
  if (r.acc.scored)
    std::printf("  ACCURACY   F1=%.2f%%  P=%.2f%%  R=%.2f%%  (pages=%d)\n",
                r.acc.f1 * 100, r.acc.precision * 100, r.acc.recall * 100, r.acc.pages);
  else
    std::printf("  ACCURACY   (not scored)\n");
  std::printf("  THROUGHPUT %.1f img/s   latency mean=%.0f p50=%.0f p95=%.0f ms\n",
              r.timing.rate, r.mean_ms, r.p50_ms, r.p95_ms);
  print_timing_verdict(r.timing, strict);
  print_utilization(r.gpu_util, r.ane_compiler);
  if (!r.det_batch_line.empty()) std::printf("  %s\n", r.det_batch_line.c_str());
}

std::string run_json(const RunResult &r, const RunConfig &rc) {
  std::vector<std::pair<std::string, std::string>> extra;
  extra.emplace_back("images_sha256", jstr(r.images_sha));
  extra.emplace_back("image_count", std::to_string(rc.count));
  extra.emplace_back("threads", std::to_string(r.replicas));
  extra.emplace_back("repeat", std::to_string(rc.repeat));
  extra.emplace_back("chunk", std::to_string(rc.chunk));
  extra.emplace_back("warmup_images", std::to_string(rc.warmup));
  extra.emplace_back("tier", jstr(rc.tier));
  std::string j = "{";
  j += "\"provenance\":" + provenance_json(r.caps, hash_models(r.models), extra);
  j += ",\"accuracy\":{\"scored\":" + std::string(r.acc.scored ? "true" : "false") +
       ",\"f1\":" + std::to_string(r.acc.f1) +
       ",\"precision\":" + std::to_string(r.acc.precision) +
       ",\"recall\":" + std::to_string(r.acc.recall) +
       ",\"pages\":" + std::to_string(r.acc.pages) + "}";
  j += ",\"throughput\":{\"img_per_s\":" + std::to_string(r.timing.rate) +
       ",\"img_per_s_accounted\":" + std::to_string(r.timing.rate_accounted) +
       ",\"window_s\":" + std::to_string(r.timing.window_s) +
       ",\"accounted_s\":" + std::to_string(r.timing.accounted_s) +
       ",\"wall_clock_skew\":" + std::to_string(r.timing.skew) +
       ",\"window_long_enough\":" + (r.timing.window_long_enough ? "true" : "false") +
       ",\"wall_clock_agrees\":" + (r.timing.wall_clock_agrees ? "true" : "false") +
       ",\"images\":" + std::to_string(r.images_processed) + "}";
  // SATURATION: the cheapest evidence of whether this run was device-bound.
  // NOTE the deliberate absence of any ANE utilization field — see harness.h.
  j += ",\"saturation\":{";
  j += "\"gpu_utilization_available\":" + std::string(r.gpu_util.have ? "true" : "false");
  if (r.gpu_util.have)
    j += ",\"gpu_utilization_pct\":{\"min\":" + std::to_string(r.gpu_util.min) +
         ",\"median\":" + std::to_string(r.gpu_util.median) +
         ",\"max\":" + std::to_string(r.gpu_util.max) +
         ",\"samples\":" + std::to_string(r.gpu_util.samples) + "}" +
         ",\"device_bound\":" + (r.gpu_util.median >= 90 ? "true" : "false");
  if (r.ane_compiler.have)
    j += ",\"ane_compiler_cpu_pct\":{\"median\":" + std::to_string(r.ane_compiler.median) +
         ",\"max\":" + std::to_string(r.ane_compiler.max) + "}";
  j += ",\"ane_utilization_pct\":null";
  j += "}";
  j += ",\"latency_ms\":{\"mean\":" + std::to_string(r.mean_ms) +
       ",\"p50\":" + std::to_string(r.p50_ms) +
       ",\"p95\":" + std::to_string(r.p95_ms) + "}";
  j += ",\"det_batch\":" + r.det_batch_json;
  j += ",\"load_ms\":" + std::to_string(r.load_ms);
  j += ",\"warmup_ms\":" + std::to_string(r.warmup_ms);
  j += "}";
  return j;
}

} // namespace

int main(int argc, char **argv) {
  Args args(argc, argv);
  auto bad = args.unknown({"backend", "images", "count", "tier", "threads", "repeat",
                           "chunk", "warmup", "out", "words", "gt", "no-score",
                           "allow-short-window", "ab", "det", "rec", "keys", "cls",
                           "layout", "assert-f1", "assert-throughput",
                           "selftest-skew-ms", "any-images", "help",
                           "det-batch", "det-batch-delay-us", "ab-det-batch",
                           "ab-rounds", "words-all"});
  if (!bad.empty()) {
    for (const auto &f : bad) std::fprintf(stderr, "unknown flag --%s\n", f.c_str());
    return 2;
  }
  if (args.has("help")) {
    std::printf("see the header comment of tests/cpp/backends/turbo_bench.cpp and "
                "tests/cpp/backends/README.md\n");
    return 0;
  }

  RunConfig rc;
  const auto &pos = args.positionals();
  // Backward-compatible positional form: <images-dir> <N> <words.json>
  rc.images = args.get("images", pos.size() > 0 ? pos[0] : std::string());
  rc.count = args.get_int("count", pos.size() > 1 ? std::atoi(pos[1].c_str()) : 50);
  std::string words_out = args.get("words", pos.size() > 2 ? pos[2] : std::string());
  rc.words_all = args.get("words-all");
  rc.backend = args.get("backend", "auto");
  rc.tier = args.get("tier", "tiny");
  rc.threads = args.get_int("threads", 1);
  rc.repeat = args.get_int("repeat", 1);
  rc.chunk = args.get_int("chunk", 1);
  rc.warmup = args.get_int("warmup", 1);
  rc.score = !args.get_bool("no-score", false);
  rc.allow_short = args.get_bool("allow-short-window", false);
  rc.selftest_skew_ms = args.get_double("selftest-skew-ms", 0);
  rc.gt = args.get("gt", default_gt_path());
  rc.det_batch = args.get_int("det-batch", -1);
  rc.det_batch_delay_us = args.get_int("det-batch-delay-us", 0);
  const std::string metrics_out = args.get("out");
  const std::string ab = args.get("ab");
  // Per-ARM detection batching for interleaved mode: --ab-det-batch 1,8 runs
  // arm A instrument-only and arm B coalescing at 8. This is what makes an
  // honest batcher-on-vs-off comparison possible on ONE machine that drifts up
  // to 40% between back-to-back runs: the two arms alternate inside one process,
  // so thermal state and background load hit both equally.
  const std::string ab_det_batch = args.get("ab-det-batch");

  if (rc.images.empty()) {
    std::fprintf(stderr,
                 "usage: %s --backend <name> --images <dir> [--count N] ...\n"
                 "       %s <images-dir> <N> <words.json>   (legacy form)\n",
                 argv[0], argv[0]);
    return 2;
  }

  std::printf("turbo_bench — one harness, every backend\n");
  print_thermal_warning();

  ImageSet set = load_images(rc.images, rc.count, args.get_bool("any-images", false));
  if (set.imgs.empty()) {
    std::fprintf(stderr, "no images loaded from %s\n", rc.images.c_str());
    return 2;
  }
  std::printf("images: %zu from %s (set sha256=%.16s...)\n", set.imgs.size(),
              rc.images.c_str(), set.sha256.c_str());

  // A run is a THROUGHPUT CLAIM when it repeats the set or asserts a rate; a
  // plain single-pass run is an ACCURACY run (50 FUNSD pages take ~11 s on CPU,
  // legitimately under the 15 s floor) and its img/s is reported as indicative
  // only. Only a throughput claim is hard-failed by discipline rules 1 and 2 —
  // and --allow-short-window downgrades even that to a warning, on purpose, so
  // an operator can override but never by accident.
  const bool throughput_claim =
      args.has("assert-throughput") || rc.repeat > 1 || !ab.empty();
  const bool strict = throughput_claim && !rc.allow_short;
  if (!throughput_claim)
    std::printf("mode: ACCURACY (single pass). The img/s below is indicative; for a "
                "publishable throughput number use --repeat R so the window is "
                ">= %.0f s.\n", kMinWindowSeconds);

  std::vector<RunResult> results;
  if (!ab.empty()) {
    // ---- INTERLEAVED PAIRED MODE (A,B,A,B,...) ------------------------------
    // The ONLY trustworthy way to compare two backends: thermal drift and
    // background load hit both halves of each pair equally.
    auto comma = ab.find(',');
    if (comma == std::string::npos) {
      std::fprintf(stderr, "--ab needs two backends: --ab cpu,apple\n");
      return 2;
    }
    const std::string A = ab.substr(0, comma), B = ab.substr(comma + 1);
    // --ab-rounds N separates "how many A,B pairs" from "how many passes per
    // arm". Without it, --repeat is the round count and each arm is ONE pass —
    // which on a fast backend is a sub-second window, i.e. exactly the
    // short-window artifact discipline rule 1 exists to prevent. With it,
    // --repeat keeps its normal meaning (passes per arm, sizing the window) and
    // the pairing repeats N times.
    const bool split_rounds = args.has("ab-rounds");
    const int rounds =
        std::max(1, split_rounds ? args.get_int("ab-rounds", 3)
                                 : args.get_int("repeat", 3));
    std::printf("\nINTERLEAVED PAIRED A/B: %s vs %s, %d rounds (A,B,A,B,...)\n",
                A.c_str(), B.c_str(), rounds);
    RunConfig a_cfg = rc, b_cfg = rc;
    a_cfg.backend = A; b_cfg.backend = B;
    if (!split_rounds) a_cfg.repeat = b_cfg.repeat = 1;
    std::string a_tag = A, b_tag = B;
    if (!ab_det_batch.empty()) {
      const auto c2 = ab_det_batch.find(',');
      if (c2 == std::string::npos) {
        std::fprintf(stderr, "--ab-det-batch needs two values: --ab-det-batch 1,8\n");
        return 2;
      }
      a_cfg.det_batch = std::atoi(ab_det_batch.substr(0, c2).c_str());
      b_cfg.det_batch = std::atoi(ab_det_batch.substr(c2 + 1).c_str());
      a_tag = A + "/detbatch=" + std::to_string(a_cfg.det_batch);
      b_tag = B + "/detbatch=" + std::to_string(b_cfg.det_batch);
      std::printf("ARM A: %s   ARM B: %s   (delay_us=%d)\n", a_tag.c_str(),
                  b_tag.c_str(), rc.det_batch_delay_us);
    }
    std::vector<double> ra, rb, fa, fb;
    for (int i = 0; i < rounds; ++i) {
      auto ares = run_one(a_cfg, set, args);
      auto bres = run_one(b_cfg, set, args);
      if (!ares.ok || !bres.ok) return 1;
      print_run(ares, false);
      print_run(bres, false);
      ra.push_back(ares.timing.rate); rb.push_back(bres.timing.rate);
      fa.push_back(ares.acc.f1); fb.push_back(bres.acc.f1);
      std::printf("pair %d: %s=%.1f img/s (F1 %.2f%%, gpu %.0f%%)  "
                  "%s=%.1f img/s (F1 %.2f%%, gpu %.0f%%)  ratio=%.3f\n",
                  i, a_tag.c_str(), ra.back(), fa.back() * 100,
                  ares.gpu_util.have ? ares.gpu_util.median : -1.0, b_tag.c_str(),
                  rb.back(), fb.back() * 100,
                  bres.gpu_util.have ? bres.gpu_util.median : -1.0,
                  rb.back() / std::max(1e-9, ra.back()));
      results.push_back(std::move(ares));
      results.push_back(std::move(bres));
    }
    double ma = 0, mb = 0, m_f1a = 0, m_f1b = 0;
    for (std::size_t i = 0; i < ra.size(); ++i) {
      ma += ra[i]; mb += rb[i]; m_f1a += fa[i]; m_f1b += fb[i];
    }
    ma /= ra.size(); mb /= rb.size(); m_f1a /= fa.size(); m_f1b /= fb.size();
    std::printf("\nPAIRED RESULT (%zu pairs): %s %.1f img/s @ F1 %.2f%%  |  %s %.1f "
                "img/s @ F1 %.2f%%  |  B/A = %.3f\n",
                ra.size(), a_tag.c_str(), ma, m_f1a * 100, b_tag.c_str(), mb,
                m_f1b * 100, mb / std::max(1e-9, ma));
    std::printf("Only this RATIO is comparable across sessions. Raw img/s from two "
                "different sessions or machines are NOT.\n");
    std::printf("(--words/--out, if given, describe the LAST run of the pairing: %s.)\n",
                b_tag.c_str());
  } else {
    auto res = run_one(rc, set, args);
    if (!res.ok) return 1;
    print_run(res, strict);
    results.push_back(std::move(res));
  }

  const RunResult &last = results.back();
  if (!words_out.empty()) {
    if (!write_words_json(words_out, last.words)) {
      std::fprintf(stderr, "cannot write %s\n", words_out.c_str());
      return 1;
    }
    std::printf("wrote transcript %s\n", words_out.c_str());
  }
  // --words-all: every page processed in the timed window, plus the index of
  // the image each one was supposed to be. A separate `<file>.idx` keeps the
  // words file in the ordinary words-json shape so the same scorer reads both.
  if (!rc.words_all.empty() && !last.words_all.empty()) {
    if (!write_words_json(rc.words_all, last.words_all)) {
      std::fprintf(stderr, "cannot write %s\n", rc.words_all.c_str());
      return 1;
    }
    std::ofstream f(rc.words_all + ".idx");
    for (int i : last.words_all_idx) f << i << "\n";
    std::printf("wrote per-pass audit transcript %s (+.idx), %zu pages\n",
                rc.words_all.c_str(), last.words_all.size());
  }
  if (!metrics_out.empty()) {
    std::ofstream f(metrics_out);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", metrics_out.c_str()); return 1; }
    f << run_json(last, rc) << "\n";
    std::printf("wrote metrics  %s\n", metrics_out.c_str());
  }

  // --- gates ----------------------------------------------------------------
  int rc_exit = 0;
  if (args.has("assert-f1")) {
    const double want = args.get_double("assert-f1", 0);
    if (!last.acc.scored || last.acc.f1 * 100.0 < want) {
      std::fprintf(stderr, "GATE FAILED: F1 %.2f%% < required %.2f%%\n",
                   last.acc.f1 * 100.0, want);
      rc_exit = 1;
    } else {
      std::printf("GATE OK: F1 %.2f%% >= %.2f%%\n", last.acc.f1 * 100.0, want);
    }
  }
  if (args.has("assert-throughput")) {
    const double want = args.get_double("assert-throughput", 0);
    if (last.timing.rate < want) {
      std::fprintf(stderr, "GATE FAILED: %.1f img/s < required %.1f img/s\n",
                   last.timing.rate, want);
      rc_exit = 1;
    } else {
      std::printf("GATE OK: %.1f img/s >= %.1f img/s\n", last.timing.rate, want);
    }
  }
  // Discipline rules 1 and 2 are hard failures whenever a throughput claim is
  // being made (i.e. unless --allow-short-window explicitly downgrades them).
  if (strict && (!last.timing.window_long_enough || !last.timing.wall_clock_agrees)) {
    std::fprintf(stderr,
                 "MEASUREMENT REJECTED: window_ok=%d wall_clock_ok=%d. The throughput "
                 "number above is NOT publishable. Re-run with a larger --repeat, or "
                 "pass --allow-short-window if you only want the accuracy number.\n",
                 static_cast<int>(last.timing.window_long_enough),
                 static_cast<int>(last.timing.wall_clock_agrees));
    rc_exit = 4;
  }
  return rc_exit;
}
