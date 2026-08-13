// turbo_golden — per-STAGE golden diff, parameterized by (backend, stage).
//
// ============================================================================
// WHAT THIS GENERALIZES
// ============================================================================
// tests/cpp/backends/cls_golden_apple.mm was a hand-built Apple-only harness that ran
// FUNSD crops through "Metal warp + MPSGraph" and through "cv::warpPerspective +
// ORT" and reported prob deltas and flip agreement. It found TWO real bugs:
//   * MpsClassifier defaulted to a 48x192 input when cls.onnx is [B,3,80,160];
//   * MpsClassifier flipped on a bare argmax, while OrtPaddleCls requires
//     s180 > s0 && s180 > 0.9 (costing 0.03pt and disagreeing on 2.3% of crops).
//
// It could only ever test Apple, because it was written against Metal types. The
// generalization is to diff at the STAGE seam instead of inside the vendor:
// IDetector / IClassifier / IRecognizer take an ImageView and return HOST types,
// so ANY backend can be diffed against the CPU reference with plain C++ — the
// same run works for nvidia and amd the day those targets build.
//
//   --stage det : identical page -> boxes.   Compared by IoU + count.
//   --stage cls : identical page AND identical REFERENCE BOXES -> which lines
//                 get flipped. This is the cls_golden case, preserved.
//   --stage rec : identical page AND identical REFERENCE BOXES -> (text, score).
//                 Compared by exact-string agreement + max |score delta|.
//
// Feeding the CANDIDATE the REFERENCE's boxes is what makes cls/rec a true
// per-stage golden: any disagreement is that stage's, not detection's leaking
// downstream.
//
// ============================================================================
// PER-STAGE TOLERANCES
// ============================================================================
// Stages differ in how much divergence is physically possible:
//
//   * Pure HOST post-process — DB box extraction, CTC greedy decode, the
//     argmax/threshold flip rule — CANNOT diverge per backend by construction:
//     there is exactly one implementation (detection::extract_boxes_from_bitmap,
//     recognition::ctc_greedy_decode, classification::should_flip_180) and every
//     backend calls it. That is the whole point of the dedup rebuild, and it is
//     why this tool tests the DEVICE stages, where divergence is possible.
//     (Both of the historical cls bugs were exactly there: a wrong device input
//     shape, and a device stage that had grown its OWN copy of the decision
//     rule instead of calling the shared one.)
//   * DEVICE conv/resample stages get a tolerance: a Metal bilinear resample is
//     not OpenCV's, and an fp16 engine is not fp32 ORT. Measured on this M3 Max,
//     Apple-vs-CPU cls flip agreement is ~0.995 end-to-end (1.1e-5 max prob
//     delta on identical crops), so the default tolerances below are set just
//     under the measured values, not at 1.0.
//
// Override any of them with --assert-agreement / --assert-mean-iou.
//
// USAGE
//   turbo_golden --backend apple [--ref cpu] --stage cls|det|rec|layout|all
//                (layout is NOT in "all": it loads a fourth model — own entry)
//                --images <dir> [--count N] [--tier tiny] [--iou 0.5]
//                [--assert-agreement X] [--assert-mean-iou X] [--report-only]
//                [--out golden.json]

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "harness.h"

using namespace turbo_ocr;
using namespace turbo_ocr::harness;

namespace {

// Default tolerances (see the header comment for why they are not 1.0).
struct Tolerance {
  double agreement;  // fraction of items that must agree exactly
  double mean_iou;   // geometry stages only
  const char *why;
};
// MEASURED BASELINES, apple-vs-cpu, FUNSD pages 0-9, tiny tier, M3 Max:
//   det  agreement 0.9114, mean IoU 0.8647 (SHIPPED models/det_tiny export)
//   cls  flip agreement 0.9881
//   rec  exact-string agreement 0.7223, max |score delta| 0.488
// The tolerances below sit just UNDER those, so they are TRIPWIRES for a
// backend that has started to diverge — not proofs of correctness. The accuracy
// gate of record is still end-to-end F1 (turbo_bench --assert-f1): two backends
// can disagree on 28% of individual lines and land within 0.07pt of each other
// on F1, which is exactly what cpu (85.79%) and apple (85.72%) do.
//
// HISTORY of the det baseline: 0.9625/0.9283 was measured through the Phase-0
// prototype exports (~/.apple_ocr_ml/exports/det_tiny992 — a policy-fit
// 992-canvas), which the harness's default model table silently pinned for
// every tier. The SHIPPED export (models/det_tiny, a 1280x1280 square canvas)
// STRETCHES non-square pages, so its boxes sit further from the CPU
// reference: 0.9114/0.8647 on the same 10 pages, F1 85.21% vs 85.72%. That
// gap is a real cost of the square canvas, not measurement noise — the fix is
// policy-fit multi-canvas det exports (the apple mirror of intel's
// snap_det_canvas_grid + per-canvas static compile), tracked as a follow-up.
Tolerance tolerance_for(const std::string &stage) {
  if (stage == "det") return {0.90, 0.85,
      "device resize/conv then SHARED DB post-process; measured 0.9114 / IoU "
      "0.8647 through the shipped 1280-square det export (see baseline note)"};
  if (stage == "cls") return {0.98, 0.0,
      "SHARED flip rule (should_flip_180) over device conv output; measured 0.9881"};
  if (stage == "rec") return {0.65, 0.0,
      "device warp+conv then SHARED ctc_greedy_decode; bilinear resampling differs "
      "per device, so per-character divergence is expected — measured 0.7223"};
  if (stage == "layout") return {0.98, 0.95,
      "PP-DocLayoutV3 fwd differs per EP (fp16 CoreML vs fp32 ORT) then SHARED "
      "postfilter; measured apple-vs-cpu 0.9955 / IoU 0.9802 / max score delta "
      "0.0136 (the one miss: a class flip on a 0.349-score borderline region). "
      "The hard guards (finite scores, non-empty on structured pages) are what "
      "catch the failure that motivated this stage — ORT 1.24 CoreML NaN'd "
      "every score and layout silently returned ZERO regions while the whole "
      "suite stayed green"};
  return {0.90, 0.85, "default"};
}

// One backend, opened with its stages exposed (not wrapped in a pipeline — this
// tool drives ONE stage at a time on purpose).
struct Opened {
  std::unique_ptr<backend::Backend> b;
  backend::StageSet stages;
  std::unique_ptr<backend::DeviceQueue> queue;
  std::shared_ptr<backend::IDeviceAllocator> alloc;
  ModelPaths models;
  backend::BackendCaps caps;
  bool ok = false;
};

Opened open(const std::string &name, const Args &args, const std::string &tier,
            bool want_layout) {
  Opened o;
  o.b = open_backend(name);
  if (!o.b) return o;
  o.caps = o.b->caps();
  o.models = resolve_models(args, o.caps.name, tier);
  // The layout model has ONE shipped artefact for every backend (each arm
  // differs only in which EP runs it), so the golden defaults it here rather
  // than in default_models — turning it on there would make every harness
  // load layout on every run.
  if (want_layout && o.models.layout.empty())
    o.models.layout = "models/layout/layout.onnx";
  o.stages = o.b->load_stages(to_config(o.models));
  if (!o.stages.available.detector || !o.stages.available.recognizer) {
    std::fprintf(stderr, "[%s] required stages missing (det=%d rec=%d)\n", name.c_str(),
                 o.stages.available.detector, o.stages.available.recognizer);
    return o;
  }
  o.queue = o.b->make_queue();
  o.alloc = o.b->allocator();
  o.ok = true;
  std::printf("[%s] device=%s  det=%s rec=%s cls=%s (cls stage %s)\n", o.caps.name.c_str(),
              backend::device_kind_name(o.caps.device), o.models.det.c_str(),
              o.models.rec.c_str(), o.models.cls.empty() ? "(off)" : o.models.cls.c_str(),
              o.stages.available.classifier ? "loaded" : "ABSENT");
  return o;
}

struct StageReport {
  std::string stage;
  long items = 0;
  long agree = 0;
  double iou_sum = 0;
  long iou_n = 0;
  double max_abs_delta = 0; // score/confidence delta where the stage has one
  double sum_abs_delta = 0;
  long delta_n = 0;
  long ref_count = 0, cand_count = 0;
  std::vector<std::string> examples;

  [[nodiscard]] double agreement() const {
    return items ? static_cast<double>(agree) / static_cast<double>(items) : 0.0;
  }
  [[nodiscard]] double mean_iou() const { return iou_n ? iou_sum / iou_n : 0.0; }
  [[nodiscard]] double mean_abs_delta() const {
    return delta_n ? sum_abs_delta / delta_n : 0.0;
  }
};

// A box is "flipped" by IClassifier by rotating its corner order; comparing the
// corner arrays therefore recovers exactly the flip decision, without needing
// the vendor's probabilities.
bool same_box(const Box &a, const Box &b) { return a == b; }

} // namespace

int main(int argc, char **argv) {
  Args args(argc, argv);
  auto bad = args.unknown({"backend", "ref", "stage", "images", "count", "tier", "iou",
                           "assert-agreement", "assert-mean-iou", "report-only", "out",
                           "det", "rec", "keys", "cls", "layout", "any-images", "help"});
  if (!bad.empty()) {
    for (const auto &f : bad) std::fprintf(stderr, "unknown flag --%s\n", f.c_str());
    return 2;
  }

  const std::string cand_name = args.get("backend");
  const std::string ref_name = args.get("ref", "cpu");
  const std::string stage_sel = args.get("stage", "all");
  const std::string images =
      args.get("images", args.positionals().empty() ? std::string() : args.positionals()[0]);
  const int count = args.get_int("count", 10);
  const std::string tier = args.get("tier", "tiny");
  const double iou_thresh = args.get_double("iou", 0.5);
  const bool report_only = args.get_bool("report-only", false);

  std::printf("turbo_golden — per-stage golden diff (candidate vs CPU reference)\n");
  if (cand_name.empty() || images.empty()) {
    std::fprintf(stderr,
                 "usage: %s --backend <name> --stage det|cls|rec|layout|all --images <dir> "
                 "[--count N]\n", argv[0]);
    return 2;
  }
  {
    auto avail = backend::available_backends();
    bool has_ref = false, has_cand = false;
    for (auto n : avail) {
      if (n == ref_name) has_ref = true;
      if (n == cand_name) has_cand = true;
    }
    if (!has_ref || !has_cand) {
      std::printf("SKIP: need both '%s' (reference) and '%s' (candidate) in ONE binary; "
                  "this build has:", ref_name.c_str(), cand_name.c_str());
      for (auto n : avail) std::printf(" %.*s", static_cast<int>(n.size()), n.data());
      std::printf("\n      Rebuild with -DTURBO_BACKENDS=\"%s;%s\".\n", ref_name.c_str(),
                  cand_name.c_str());
      return 77; // ctest SKIP_RETURN_CODE
    }
  }
  if (cand_name == ref_name) {
    std::printf("SKIP: candidate == reference (%s); nothing to diff.\n", cand_name.c_str());
    return 77;
  }

  ImageSet set = load_images(images, count, args.get_bool("any-images", false));
  if (set.imgs.empty()) {
    std::fprintf(stderr, "no images loaded from %s\n", images.c_str());
    return 2;
  }
  std::printf("images: %zu (sha256=%.16s...)\n", set.imgs.size(), set.sha256.c_str());

  const bool do_det = stage_sel == "all" || stage_sel == "det";
  const bool do_cls = stage_sel == "all" || stage_sel == "cls";
  const bool do_rec = stage_sel == "all" || stage_sel == "rec";
  // NOT part of "all": layout loads a fourth model (and on CoreML compiles
  // it), which would tax every det/cls/rec invocation; it has its own ctest
  // entry. --stage layout runs it alone.
  const bool do_layout = stage_sel == "layout";

  Opened ref = open(ref_name, args, tier, do_layout);
  Opened cand = open(cand_name, args, tier, do_layout);
  if (!ref.ok || !cand.ok) return 1;
  const bool ref_has_layout =
      ref.stages.available.optional.get(capability::CapabilityId::Layout) &&
      ref.stages.layout != nullptr;
  const bool cand_has_layout =
      cand.stages.available.optional.get(capability::CapabilityId::Layout) &&
      cand.stages.layout != nullptr;
  if (do_layout && (!ref_has_layout || !cand_has_layout)) {
    std::fprintf(stderr,
                 "layout stage missing (ref=%d cand=%d) — is %s present?\n",
                 ref_has_layout, cand_has_layout, ref.models.layout.c_str());
    return 1;
  }

  StageReport rdet, rcls, rrec, rlay;
  rdet.stage = "det";
  rcls.stage = "cls";
  rrec.stage = "rec";
  rlay.stage = "layout";
  // The two ABSOLUTE layout guards (see tolerance_for): counted across the
  // whole set, asserted after the loop.
  long lay_nonfinite = 0;      // candidate regions with NaN/Inf scores
  long lay_blank_pages = 0;    // pages where ref found >=3 regions, cand found 0

  for (std::size_t i = 0; i < set.imgs.size(); ++i) {
    const cv::Mat &img = set.imgs[i];
    // IDENTICAL INPUT: the same decoded cv::Mat is uploaded into each backend's
    // own address space. Nothing else about the two paths is shared.
    Uploaded uref = upload_page(*ref.b, ref.alloc, *ref.queue, img);
    Uploaded ucand = upload_page(*cand.b, cand.alloc, *cand.queue, img);

    // --- layout: SAME page, compare regions (class, geometry, score) --------
    if (do_layout) {
      const float th = backend::kDefaultLayoutScoreThreshold;
      auto la = ref.stages.layout->run(uref.view, img.rows, img.cols, th, *ref.queue);
      auto lb = cand.stages.layout->run(ucand.view, img.rows, img.cols, th, *cand.queue);
      rlay.ref_count += static_cast<long>(la.size());
      rlay.cand_count += static_cast<long>(lb.size());
      for (const auto &r : lb)
        if (!std::isfinite(r.score)) ++lay_nonfinite;
      if (la.size() >= 3 && lb.empty()) ++lay_blank_pages;
      std::vector<char> used(lb.size(), 0);
      for (const auto &ra : la) {
        double best = 0;
        std::size_t bj = lb.size();
        for (std::size_t j = 0; j < lb.size(); ++j) {
          if (used[j]) continue;
          double v = box_iou(ra.box, lb[j].box);
          if (v > best) { best = v; bj = j; }
        }
        ++rlay.items;
        if (bj < lb.size() && best >= iou_thresh && lb[bj].class_id == ra.class_id) {
          used[bj] = 1;
          ++rlay.agree;
          rlay.iou_sum += best;
          ++rlay.iou_n;
          const double d = std::fabs(static_cast<double>(ra.score) -
                                     static_cast<double>(lb[bj].score));
          rlay.max_abs_delta = std::max(rlay.max_abs_delta, d);
          rlay.sum_abs_delta += d;
          ++rlay.delta_n;
        } else if (rlay.examples.size() < 10) {
          char b[192];
          std::snprintf(b, sizeof b,
                        "page %zu: ref region class=%d score=%.3f unmatched "
                        "(best IoU %.3f)", i, ra.class_id, ra.score, best);
          rlay.examples.emplace_back(b);
        }
      }
      std::printf("  page %zu: layout ref=%zu cand=%zu\n", i, la.size(), lb.size());
      continue; // layout-only mode: det/cls/rec are separate ctest entries
    }

    auto ref_boxes = ref.stages.detector->run(uref.view, img.rows, img.cols, *ref.queue);

    if (do_det) {
      auto cand_boxes =
          cand.stages.detector->run(ucand.view, img.rows, img.cols, *cand.queue);
      rdet.ref_count += static_cast<long>(ref_boxes.size());
      rdet.cand_count += static_cast<long>(cand_boxes.size());
      std::vector<char> used(cand_boxes.size(), 0);
      for (const auto &rb : ref_boxes) {
        double best = 0;
        std::size_t bj = cand_boxes.size();
        for (std::size_t j = 0; j < cand_boxes.size(); ++j) {
          if (used[j]) continue;
          double v = box_iou(rb, cand_boxes[j]);
          if (v > best) { best = v; bj = j; }
        }
        ++rdet.items;
        if (bj < cand_boxes.size() && best >= iou_thresh) {
          used[bj] = 1;
          ++rdet.agree;
          rdet.iou_sum += best;
          ++rdet.iou_n;
          // Per-corner max abs delta, in pixels — the "max abs delta" the golden
          // reports for a geometry stage.
          for (int k = 0; k < 4; ++k)
            for (int d = 0; d < 2; ++d) {
              double dd = std::fabs(static_cast<double>(rb.pts[k][d]) -
                                    static_cast<double>(cand_boxes[bj].pts[k][d]));
              rdet.max_abs_delta = std::max(rdet.max_abs_delta, dd);
              rdet.sum_abs_delta += dd;
              ++rdet.delta_n;
            }
        } else if (rdet.examples.size() < 10) {
          char b[160];
          std::snprintf(b, sizeof b, "page %zu: reference box unmatched (best IoU %.3f)",
                        i, best);
          rdet.examples.emplace_back(b);
        }
      }
    }

    // --- cls: SAME page, SAME boxes, compare the flip decisions ---------------
    if (do_cls && ref.stages.available.classifier && cand.stages.available.classifier) {
      auto a = ref_boxes, b = ref_boxes;
      ref.stages.classifier->run(uref.view, a, *ref.queue);
      cand.stages.classifier->run(ucand.view, b, *cand.queue);
      for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) {
        ++rcls.items;
        const bool ref_flipped = !same_box(a[k], ref_boxes[k]);
        const bool cand_flipped = !same_box(b[k], ref_boxes[k]);
        // For cls the "produced" counts are the FLIP counts — the quantity the
        // old cls_golden_apple reported as "flips: A=%ld C=%ld".
        rcls.ref_count += ref_flipped ? 1 : 0;
        rcls.cand_count += cand_flipped ? 1 : 0;
        if (ref_flipped == cand_flipped) {
          ++rcls.agree;
        } else if (rcls.examples.size() < 10) {
          char buf[160];
          std::snprintf(buf, sizeof buf, "page %zu box %zu: %s flipped=%d, %s flipped=%d",
                        i, k, ref.caps.name.c_str(), static_cast<int>(ref_flipped),
                        cand.caps.name.c_str(), static_cast<int>(cand_flipped));
          rcls.examples.emplace_back(buf);
        }
      }
    }

    // --- rec: SAME page, SAME boxes, compare (text, score) -------------------
    if (do_rec) {
      auto ra = ref.stages.recognizer->run(uref.view, ref_boxes, *ref.queue);
      auto rb = cand.stages.recognizer->run(ucand.view, ref_boxes, *cand.queue);
      rrec.ref_count += static_cast<long>(ra.size());
      rrec.cand_count += static_cast<long>(rb.size());
      for (std::size_t k = 0; k < ra.size() && k < rb.size(); ++k) {
        ++rrec.items;
        double d = std::fabs(static_cast<double>(ra[k].second) -
                             static_cast<double>(rb[k].second));
        rrec.max_abs_delta = std::max(rrec.max_abs_delta, d);
        rrec.sum_abs_delta += d;
        ++rrec.delta_n;
        if (ra[k].first == rb[k].first) {
          ++rrec.agree;
        } else if (rrec.examples.size() < 10) {
          rrec.examples.push_back("page " + std::to_string(i) + ": '" + ra[k].first +
                                  "' | '" + rb[k].first + "'");
        }
      }
    }
    std::printf("  page %zu: boxes=%zu\n", i, ref_boxes.size());
  }

  int rc = 0;
  auto report = [&](const StageReport &s, bool ran) {
    if (!ran) return;
    if (s.items == 0) {
      std::printf("\n[stage %s] SKIPPED — stage not available on both backends\n",
                  s.stage.c_str());
      return;
    }
    const Tolerance tol = tolerance_for(s.stage);
    const double want_agree = args.has("assert-agreement")
                                  ? args.get_double("assert-agreement", tol.agreement)
                                  : tol.agreement;
    const double want_iou = args.has("assert-mean-iou")
                                ? args.get_double("assert-mean-iou", tol.mean_iou)
                                : tol.mean_iou;
    std::printf("\n=== stage %s : %s vs %s ===\n", s.stage.c_str(), ref.caps.name.c_str(),
                cand.caps.name.c_str());
    std::printf("  items compared    : %ld  (%s: ref %ld / cand %ld)\n", s.items,
                s.stage == "cls" ? "flips" : "produced", s.ref_count, s.cand_count);
    std::printf("  agreement rate    : %.4f   (tolerance %.4f — %s)\n", s.agreement(),
                want_agree, tol.why);
    if (s.iou_n) std::printf("  mean IoU          : %.4f   (tolerance %.4f)\n",
                             s.mean_iou(), want_iou);
    if (s.delta_n)
      std::printf("  max abs delta     : %.6f   mean %.6f\n", s.max_abs_delta,
                  s.mean_abs_delta());
    for (const auto &e : s.examples) std::printf("    e.g. %s\n", e.c_str());
    if (s.agreement() < want_agree) {
      std::fprintf(stderr, "GOLDEN FAILED [%s]: agreement %.4f < %.4f\n", s.stage.c_str(),
                   s.agreement(), want_agree);
      if (!report_only) rc = 1;
    }
    if (s.iou_n && want_iou > 0 && s.mean_iou() < want_iou) {
      std::fprintf(stderr, "GOLDEN FAILED [%s]: mean IoU %.4f < %.4f\n", s.stage.c_str(),
                   s.mean_iou(), want_iou);
      if (!report_only) rc = 1;
    }
  };
  report(rdet, do_det);
  report(rcls, do_cls);
  report(rrec, do_rec);
  report(rlay, do_layout);
  // The two ABSOLUTE layout guards. These are not tolerances — they are the
  // failure mode that motivated the stage: a broken EP returning NaN scores
  // (every region silently dropped) or zero regions on pages the reference
  // finds structured. Both fail even under --report-only tolerances.
  if (do_layout) {
    if (lay_nonfinite > 0) {
      std::fprintf(stderr,
                   "GOLDEN FAILED [layout]: %ld candidate region(s) with "
                   "non-finite scores — the ORT-1.24 CoreML NaN signature\n",
                   lay_nonfinite);
      rc = 1;
    }
    if (lay_blank_pages > 0) {
      std::fprintf(stderr,
                   "GOLDEN FAILED [layout]: %ld page(s) where the reference "
                   "found >=3 regions and the candidate found NONE — a blank "
                   "layout that would look like a clean page downstream\n",
                   lay_blank_pages);
      rc = 1;
    }
  }

  if (!args.get("out").empty()) {
    std::ofstream f(args.get("out"));
    f << "{\"reference\":" << jstr(ref.caps.name) << ",\"candidate\":"
      << jstr(cand.caps.name) << ",\"images\":" << set.imgs.size()
      << ",\"images_sha256\":" << jstr(set.sha256) << ",\"host\":" << jstr(host_name())
      << ",\"stages\":{";
    bool first = true;
    for (const StageReport *s : {&rdet, &rcls, &rrec, &rlay}) {
      if (s->items == 0) continue;
      f << (first ? "" : ",") << jstr(s->stage) << ":{\"items\":" << s->items
        << ",\"agreement\":" << s->agreement() << ",\"mean_iou\":" << s->mean_iou()
        << ",\"max_abs_delta\":" << s->max_abs_delta
        << ",\"mean_abs_delta\":" << s->mean_abs_delta() << "}";
      first = false;
    }
    f << "}}\n";
    std::printf("wrote %s\n", args.get("out").c_str());
  }
  if (rc == 0) std::printf("\nGOLDEN OK (within per-stage tolerances)\n");
  return rc;
}
