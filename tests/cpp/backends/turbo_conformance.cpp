// turbo_conformance — the CROSS-BACKEND conformance test ("the keystone").
//
// ============================================================================
// WHAT IT IS FOR
// ============================================================================
// The same image set is pushed through EVERY backend linked into this binary and
// the outputs are compared against a reference backend (cpu by default). It
// asserts:
//
//   * TEXT      — the recognized strings agree, per matched line and per page;
//   * GEOMETRY  — the detected boxes agree, matched by IoU above a threshold.
//
// This is what catches a backend silently diverging while still "working". Every
// bug in the list below was found by hand, one backend at a time, because this
// test did not exist:
//
//   - AMD classified with ImageNet normalisation while everyone else used the
//     PaddleCls 1/127.5-1 rule;
//   - AMD's layout decoder read a 6-column tensor as if it were 5;
//   - AMD ran the DB post-process with max_expand=0, disabling unclip entirely;
//   - Apple forked box_thresh (and bin_thresh/unclip/min_side) away from the
//     shared detection config;
//   - Apple's angle classifier flipped on a bare argmax with no 0.9 threshold,
//     and loaded a 48x192 input when cls.onnx is 80x160.
//
// Every one of those produces text or boxes that differ from the CPU reference
// on real pages, and every one of them would have been a one-line failure here.
//
// ============================================================================
// A REAL DIFFERENCE IS A FINDING, NOT A HARNESS BUG
// ============================================================================
// Backends will never be bit-identical: a Metal bilinear resample is not
// OpenCV's, an fp16 TensorRT engine is not fp32 ORT. The test therefore reports
// AGREEMENT RATES and a diff table, and only fails when an explicit threshold is
// passed (--assert-text-agreement / --assert-mean-iou / --assert-box-match).
// Set those thresholds from a measured baseline, not from hope.
//
// On a machine with only ONE backend compiled in there is nothing to compare, so
// the test SKIPS cleanly (exit 77, ctest's SKIP_RETURN_CODE) with a message
// telling you how to build a multi-backend binary.
//
// USAGE
//   turbo_conformance --images <dir> [--count N] [--tier tiny]
//                     [--backends cpu,apple] [--ref cpu] [--iou 0.5]
//                     [--max-diffs 20] [--out conformance.json]
//                     [--assert-text-agreement 0.9] [--assert-mean-iou 0.8]
//                     [--assert-box-match 0.9]
//
// NOTE ON MODELS: each backend gets ITS OWN default model artefacts (CPU eats
// models/*.onnx, Apple eats the MPSGraph export dirs) — they are different
// encodings of the same weights. --det/--rec/--cls override for ALL backends,
// which is only meaningful when they share a format.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "harness.h"
#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

using namespace turbo_ocr;
using namespace turbo_ocr::harness;

namespace {

struct Item {
  std::string text;
  Box box;
};
struct BackendRun {
  std::string name;
  bool ok = false;
  ModelPaths models;
  backend::BackendCaps caps;
  std::vector<std::vector<Item>> pages;
};

BackendRun run_backend(const std::string &name, const ImageSet &set, const Args &args,
                       const std::string &tier) {
  BackendRun br;
  br.name = name;
  auto b = open_backend(name);
  if (!b) return br;
  br.caps = b->caps();
  br.models = resolve_models(args, br.caps.name, tier);
  auto cfg = to_config(br.models);
  auto stages = b->load_stages(cfg);
  if (!stages.available.detector || !stages.available.recognizer) {
    std::fprintf(stderr, "[%s] load_stages: det=%d rec=%d — skipping this backend\n",
                 name.c_str(), stages.available.detector, stages.available.recognizer);
    return br;
  }
  std::printf("[%s] device=%s det=%s rec=%s cls=%s\n", br.caps.name.c_str(),
              backend::device_kind_name(br.caps.device), br.models.det.c_str(),
              br.models.rec.c_str(),
              br.models.cls.empty() ? "(off)" : br.models.cls.c_str());
  auto q = b->make_queue();
  pipeline::UnifiedOcrPipeline pipe(*b, std::move(stages), std::move(q));
  pipe.warmup();
  br.pages.reserve(set.imgs.size());
  for (const auto &img : set.imgs) {
    auto res = pipe.run(img);
    std::vector<Item> items;
    items.reserve(res.size());
    for (auto &r : res) items.push_back(Item{r.text, r.box});
    br.pages.push_back(std::move(items));
  }
  br.ok = true;
  return br;
}

struct Diff {
  int page = 0;
  std::string ref_text, cand_text;
  double iou = 0;
  const char *kind = "";
};

struct Comparison {
  std::string ref, cand;
  long ref_items = 0, cand_items = 0;
  long matched = 0;         // boxes paired above the IoU threshold
  long text_equal = 0;      // matched pairs whose text is identical
  long pages_identical = 0; // pages whose full text multiset matches
  double iou_sum = 0;
  std::vector<Diff> diffs;

  [[nodiscard]] double box_match_rate() const {
    const long denom = std::max(ref_items, cand_items);
    return denom ? static_cast<double>(matched) / static_cast<double>(denom) : 1.0;
  }
  [[nodiscard]] double text_agreement() const {
    return matched ? static_cast<double>(text_equal) / static_cast<double>(matched) : 0.0;
  }
  [[nodiscard]] double mean_iou() const {
    return matched ? iou_sum / static_cast<double>(matched) : 0.0;
  }
};

Comparison compare(const BackendRun &ref, const BackendRun &cand, double iou_thresh,
                   int max_diffs) {
  Comparison c;
  c.ref = ref.name;
  c.cand = cand.name;
  const std::size_t n = std::min(ref.pages.size(), cand.pages.size());
  for (std::size_t p = 0; p < n; ++p) {
    const auto &R = ref.pages[p];
    const auto &C = cand.pages[p];
    c.ref_items += static_cast<long>(R.size());
    c.cand_items += static_cast<long>(C.size());

    // Greedy best-IoU matching: for every reference line take the unmatched
    // candidate line with the highest IoU. Document lines are well separated, so
    // greedy == optimal here and it needs no dependency.
    std::vector<char> used(C.size(), 0);
    for (std::size_t i = 0; i < R.size(); ++i) {
      double best = 0;
      std::size_t bj = C.size();
      for (std::size_t j = 0; j < C.size(); ++j) {
        if (used[j]) continue;
        double v = box_iou(R[i].box, C[j].box);
        if (v > best) { best = v; bj = j; }
      }
      if (bj < C.size() && best >= iou_thresh) {
        used[bj] = 1;
        ++c.matched;
        c.iou_sum += best;
        if (R[i].text == C[bj].text) {
          ++c.text_equal;
        } else if (static_cast<int>(c.diffs.size()) < max_diffs) {
          c.diffs.push_back(Diff{static_cast<int>(p), R[i].text, C[bj].text, best, "text"});
        }
      } else if (static_cast<int>(c.diffs.size()) < max_diffs) {
        c.diffs.push_back(Diff{static_cast<int>(p), R[i].text, "(no box match)", best,
                               "missing"});
      }
    }
    // Page-level: identical set of strings regardless of order/boxes.
    std::map<std::string, int> rb, cb;
    for (const auto &i : R) ++rb[i.text];
    for (const auto &i : C) ++cb[i.text];
    if (rb == cb) ++c.pages_identical;
  }
  return c;
}

void print_table(const std::vector<Comparison> &cmps, std::size_t pages) {
  std::printf("\n=== CONFORMANCE TABLE (%zu pages) ===\n", pages);
  std::printf("%-14s %-14s %8s %8s %9s %9s %9s %9s\n", "reference", "candidate",
              "ref#", "cand#", "box-match", "mean-IoU", "text-agr", "pages-eq");
  for (const auto &c : cmps)
    std::printf("%-14s %-14s %8ld %8ld %8.2f%% %9.4f %8.2f%% %6ld/%zu\n", c.ref.c_str(),
                c.cand.c_str(), c.ref_items, c.cand_items, c.box_match_rate() * 100,
                c.mean_iou(), c.text_agreement() * 100, c.pages_identical, pages);
  for (const auto &c : cmps) {
    if (c.diffs.empty()) continue;
    std::printf("\n--- first %zu disagreements: %s vs %s ---\n", c.diffs.size(),
                c.ref.c_str(), c.cand.c_str());
    std::printf("%5s %-8s %6s  %-38s | %s\n", "page", "kind", "IoU", c.ref.c_str(),
                c.cand.c_str());
    for (const auto &d : c.diffs)
      std::printf("%5d %-8s %6.3f  %-38.38s | %.38s\n", d.page, d.kind, d.iou,
                  d.ref_text.c_str(), d.cand_text.c_str());
  }
}

std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i <= s.size()) {
    auto j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  Args args(argc, argv);
  auto bad = args.unknown({"images", "count", "tier", "backends", "ref", "iou",
                           "max-diffs", "out", "assert-text-agreement",
                           "assert-mean-iou", "assert-box-match", "det", "rec", "keys",
                           "cls", "layout", "any-images", "help"});
  if (!bad.empty()) {
    for (const auto &f : bad) std::fprintf(stderr, "unknown flag --%s\n", f.c_str());
    return 2;
  }

  const std::string images =
      args.get("images", args.positionals().empty() ? std::string() : args.positionals()[0]);
  const int count = args.get_int("count", args.positionals().size() > 1
                                              ? std::atoi(args.positionals()[1].c_str())
                                              : 20);
  const std::string tier = args.get("tier", "tiny");
  const double iou = args.get_double("iou", 0.5);
  const int max_diffs = args.get_int("max-diffs", 20);

  // Which backends: --backends, else every one linked into this binary.
  std::vector<std::string> names;
  if (args.has("backends")) {
    names = split_csv(args.get("backends"));
  } else {
    for (auto n : backend::available_backends()) names.emplace_back(n);
  }

  std::printf("turbo_conformance — same images, every backend, one verdict\n");
  std::printf("backends in this binary:");
  for (auto n : backend::available_backends())
    std::printf(" %.*s", static_cast<int>(n.size()), n.data());
  std::printf("\n");

  if (names.size() < 2) {
    std::printf("\nSKIP: cross-backend conformance needs at least TWO backends in one\n"
                "      binary; this build has %zu. Rebuild with e.g.\n"
                "        cmake -B build -DTURBO_BACKENDS=\"cpu;nvidia\"\n"
                "      This is a SKIP, not a failure: a single-backend build is a\n"
                "      legitimate configuration.\n", names.size());
    return 77; // ctest SKIP_RETURN_CODE
  }
  if (images.empty()) {
    std::fprintf(stderr, "usage: %s --images <dir> [--count N] [--backends a,b]\n", argv[0]);
    return 2;
  }

  ImageSet set = load_images(images, count, args.get_bool("any-images", false));
  if (set.imgs.empty()) {
    std::fprintf(stderr, "no images loaded from %s\n", images.c_str());
    return 2;
  }
  std::printf("images: %zu (set sha256=%.16s...)  IoU threshold=%.2f\n", set.imgs.size(),
              set.sha256.c_str(), iou);

  // Reference: --ref, else cpu when present (it is the slowest but the one every
  // other backend is a port OF), else the first named backend.
  std::string ref_name = args.get("ref");
  if (ref_name.empty()) {
    ref_name = names[0];
    for (const auto &n : names) if (n == "cpu") ref_name = "cpu";
  }

  std::vector<BackendRun> runs;
  for (const auto &n : names) {
    auto r = run_backend(n, set, args, tier);
    if (r.ok) runs.push_back(std::move(r));
    else std::fprintf(stderr, "backend '%s' unusable here — excluded\n", n.c_str());
  }
  const BackendRun *ref = nullptr;
  for (const auto &r : runs) if (r.caps.name == ref_name || r.name == ref_name) ref = &r;
  if (!ref) {
    std::fprintf(stderr, "reference backend '%s' did not run\n", ref_name.c_str());
    return 1;
  }
  if (runs.size() < 2) {
    std::printf("\nSKIP: only one backend actually loaded its models here.\n");
    return 77;
  }

  std::vector<Comparison> cmps;
  for (const auto &r : runs) {
    if (&r == ref) continue;
    cmps.push_back(compare(*ref, r, iou, max_diffs));
  }
  print_table(cmps, set.imgs.size());

  if (!args.get("out").empty()) {
    std::ofstream f(args.get("out"));
    f << "{\"images\":" << set.imgs.size() << ",\"images_sha256\":" << jstr(set.sha256)
      << ",\"iou_threshold\":" << iou << ",\"reference\":" << jstr(ref->name)
      << ",\"host\":" << jstr(host_name()) << ",\"os\":" << jstr(os_string())
      << ",\"comparisons\":[";
    for (std::size_t i = 0; i < cmps.size(); ++i) {
      const auto &c = cmps[i];
      f << (i ? "," : "") << "{\"candidate\":" << jstr(c.cand)
        << ",\"ref_items\":" << c.ref_items << ",\"cand_items\":" << c.cand_items
        << ",\"matched\":" << c.matched << ",\"box_match_rate\":" << c.box_match_rate()
        << ",\"mean_iou\":" << c.mean_iou()
        << ",\"text_agreement\":" << c.text_agreement()
        << ",\"pages_identical\":" << c.pages_identical << "}";
    }
    f << "]}\n";
    std::printf("wrote %s\n", args.get("out").c_str());
  }

  int rc = 0;
  auto gate = [&](const char *flag, double got, const char *what) {
    if (!args.has(flag)) return;
    const double want = args.get_double(flag, 0);
    if (got < want) {
      std::fprintf(stderr, "GATE FAILED: %s %.4f < required %.4f\n", what, got, want);
      rc = 1;
    } else {
      std::printf("GATE OK: %s %.4f >= %.4f\n", what, got, want);
    }
  };
  for (const auto &c : cmps) {
    gate("assert-text-agreement", c.text_agreement(), "text agreement");
    gate("assert-mean-iou", c.mean_iou(), "mean IoU");
    gate("assert-box-match", c.box_match_rate(), "box match rate");
  }
  if (rc == 0)
    std::printf("\nVERDICT: reported above. Disagreement between backends is a FINDING "
                "about the backends, not a failure of this harness — read the diff "
                "table before assuming the candidate is wrong.\n");
  return rc;
}
