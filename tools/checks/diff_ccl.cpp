// Side-by-side comparator: runs CPU CCL=1 and GPU CCL=2 on the same image(s),
// reports box count diff + IoU stats + missing/extra boxes. Iterate the GPU
// pipeline without HTTP server / batch evaluation.
//
// Build: cmake --build build --target diff_ccl
// Run:   ./build/diff_ccl path/to/image.png [more.png ...]
#include "nvidia/stages/paddle_det.h"
#include "nvidia/engine/onnx_to_trt.h"
#include "nvidia/support/cuda_check.h"
#include "turbo_ocr/image/gpu_image.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>
#include <algorithm>
#include <cmath>

using turbo_ocr::Box;
using turbo_ocr::detection::PaddleDet;

static cv::Rect aabb_of(const Box &b) {
  int xmin = std::min({b[0][0], b[1][0], b[2][0], b[3][0]});
  int ymin = std::min({b[0][1], b[1][1], b[2][1], b[3][1]});
  int xmax = std::max({b[0][0], b[1][0], b[2][0], b[3][0]});
  int ymax = std::max({b[0][1], b[1][1], b[2][1], b[3][1]});
  return cv::Rect(xmin, ymin, std::max(1, xmax - xmin), std::max(1, ymax - ymin));
}

static float iou_aabb(const cv::Rect &a, const cv::Rect &b) {
  cv::Rect inter = a & b;
  float i = (float)(inter.width * inter.height);
  float u = (float)(a.area() + b.area()) - i;
  return u > 0 ? i / u : 0.0f;
}

struct DetResult {
  std::vector<Box> boxes;
  std::vector<cv::Rect> aabbs;
};

static DetResult run_with_mode(const std::string &engine_path,
                                turbo_ocr::GpuImage &gpu_img, int orig_h,
                                int orig_w, int mode, cudaStream_t stream) {
  setenv("GPU_CCL", std::to_string(mode).c_str(), 1);
  PaddleDet det;
  if (!det.load_model(engine_path)) {
    fprintf(stderr, "load fail mode=%d\n", mode);
    std::exit(1);
  }
  // Warmup: triggers TRT context init outside the timed region (we don't time
  // here; just keeps first run from being penalized).
  (void)det.run(gpu_img, orig_h, orig_w, stream);
  cudaStreamSynchronize(stream);

  DetResult r;
  r.boxes = det.run(gpu_img, orig_h, orig_w, stream);
  cudaStreamSynchronize(stream);
  r.aabbs.reserve(r.boxes.size());
  for (auto &b : r.boxes) r.aabbs.push_back(aabb_of(b));
  return r;
}

static void diff_pair(const DetResult &cpu, const DetResult &gpu) {
  // For each CPU box, find best-IoU GPU box.
  std::vector<int> matched_gpu(gpu.aabbs.size(), -1);
  std::vector<float> match_iou(cpu.aabbs.size(), 0.0f);
  std::vector<int> match_idx(cpu.aabbs.size(), -1);
  for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
    float best = 0; int bi = -1;
    for (size_t j = 0; j < gpu.aabbs.size(); ++j) {
      float v = iou_aabb(cpu.aabbs[i], gpu.aabbs[j]);
      if (v > best) { best = v; bi = (int)j; }
    }
    match_iou[i] = best;
    match_idx[i] = bi;
    if (bi >= 0 && best > 0.5f) matched_gpu[bi] = (int)i;
  }

  int matched = 0, missed = 0;
  float iou_sum = 0.0f, iou_min = 1.0f;
  for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
    if (match_iou[i] > 0.5f) {
      matched++;
      iou_sum += match_iou[i];
      iou_min = std::min(iou_min, match_iou[i]);
    } else {
      missed++;
    }
  }
  int extra = 0;
  for (auto m : matched_gpu) if (m < 0) extra++;

  printf("  CPU=%zu GPU=%zu  matched=%d missed=%d extra=%d  "
         "mean_iou=%.3f min_iou=%.3f\n",
         cpu.aabbs.size(), gpu.aabbs.size(), matched, missed, extra,
         matched > 0 ? iou_sum / matched : 0.0f,
         matched > 0 ? iou_min : 0.0f);

  // Show first 5 missed CPU boxes
  if (missed > 0) {
    int shown = 0;
    for (size_t i = 0; i < cpu.aabbs.size() && shown < 5; ++i) {
      if (match_iou[i] > 0.5f) continue;
      const auto &r = cpu.aabbs[i];
      printf("    MISSED CPU [%zu]: x=%d y=%d w=%d h=%d  best_iou=%.2f\n",
             i, r.x, r.y, r.width, r.height, match_iou[i]);
      shown++;
    }
  }
  // Show first 5 extra GPU boxes
  if (extra > 0) {
    int shown = 0;
    for (size_t j = 0; j < gpu.aabbs.size() && shown < 5; ++j) {
      if (matched_gpu[j] >= 0) continue;
      const auto &r = gpu.aabbs[j];
      printf("    EXTRA  GPU [%zu]: x=%d y=%d w=%d h=%d\n",
             j, r.x, r.y, r.width, r.height);
      shown++;
    }
  }
}

// Optional: crop missed/extra boxes to /tmp/diff_crops/ for visual inspection.
// Triggered by setting DIFF_CROPS=1 env var.
static void maybe_save_crops(const cv::Mat &img, const std::string &stem,
                              const DetResult &cpu, const DetResult &gpu) {
  const char *env = std::getenv("DIFF_CROPS");
  if (!env || std::atoi(env) == 0) return;
  std::string dir = "/tmp/diff_crops";
  std::system(("mkdir -p " + dir).c_str());

  // Build matched mask
  std::vector<int> matched_gpu(gpu.aabbs.size(), -1);
  std::vector<float> match_iou(cpu.aabbs.size(), 0.0f);
  for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
    float best = 0; int bi = -1;
    for (size_t j = 0; j < gpu.aabbs.size(); ++j) {
      float v = iou_aabb(cpu.aabbs[i], gpu.aabbs[j]);
      if (v > best) { best = v; bi = (int)j; }
    }
    match_iou[i] = best;
    if (bi >= 0 && best > 0.5f) matched_gpu[bi] = (int)i;
  }

  auto pad = [&](const cv::Rect &r, int p) {
    int x = std::max(0, r.x - p);
    int y = std::max(0, r.y - p);
    int x2 = std::min(img.cols - 1, r.x + r.width + p);
    int y2 = std::min(img.rows - 1, r.y + r.height + p);
    return cv::Rect(x, y, x2 - x, y2 - y);
  };

  for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
    if (match_iou[i] > 0.5f) continue;
    auto rect = pad(cpu.aabbs[i], 8);
    cv::Mat crop = img(rect).clone();
    char name[256];
    std::snprintf(name, sizeof(name), "%s/%s_MISSED_cpu_%zu_x%d_y%d_w%d_h%d.png",
                  dir.c_str(), stem.c_str(), i, cpu.aabbs[i].x, cpu.aabbs[i].y,
                  cpu.aabbs[i].width, cpu.aabbs[i].height);
    cv::imwrite(name, crop);
  }
  for (size_t j = 0; j < gpu.aabbs.size(); ++j) {
    if (matched_gpu[j] >= 0) continue;
    auto rect = pad(gpu.aabbs[j], 8);
    cv::Mat crop = img(rect).clone();
    char name[256];
    std::snprintf(name, sizeof(name), "%s/%s_EXTRA_gpu_%zu_x%d_y%d_w%d_h%d.png",
                  dir.c_str(), stem.c_str(), j, gpu.aabbs[j].x, gpu.aabbs[j].y,
                  gpu.aabbs[j].width, gpu.aabbs[j].height);
    cv::imwrite(name, crop);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <image.png> [more.png ...]\n", argv[0]);
    return 1;
  }

  try {
  auto det_path =
      turbo_ocr::engine::ensure_trt_engine("models/det.onnx", "det");
  if (det_path.empty()) {
    fprintf(stderr, "det engine fail\n");
    return 1;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Aggregate stats across all images
  int agg_cpu = 0, agg_gpu = 0, agg_matched = 0, agg_missed = 0, agg_extra = 0;
  double agg_iou_sum = 0.0;

  for (int a = 1; a < argc; ++a) {
    cv::Mat img = cv::imread(argv[a], cv::IMREAD_COLOR);
    if (img.empty()) {
      fprintf(stderr, "skip: %s\n", argv[a]);
      continue;
    }
    void *d_buf;
    size_t pitch;
    CUDA_CHECK(cudaMallocPitch(&d_buf, &pitch, img.cols * 3, img.rows));
    CUDA_CHECK(cudaMemcpy2D(d_buf, pitch, img.data, img.step, img.cols * 3,
                            img.rows, cudaMemcpyHostToDevice));
    turbo_ocr::GpuImage gpu_img{d_buf, pitch, img.rows, img.cols};

    printf("[%d/%d] %s (%dx%d)\n", a, argc - 1, argv[a], img.cols, img.rows);
    auto cpu = run_with_mode(det_path, gpu_img, img.rows, img.cols, 1, stream);
    auto gpu = run_with_mode(det_path, gpu_img, img.rows, img.cols, 2, stream);
    diff_pair(cpu, gpu);
    {
      std::string stem = argv[a];
      auto p = stem.find_last_of('/');
      if (p != std::string::npos) stem = stem.substr(p + 1);
      auto d = stem.find_last_of('.');
      if (d != std::string::npos) stem = stem.substr(0, d);
      maybe_save_crops(img, stem, cpu, gpu);
    }

    // Aggregate
    agg_cpu += (int)cpu.aabbs.size();
    agg_gpu += (int)gpu.aabbs.size();
    for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
      float best = 0;
      for (auto &g : gpu.aabbs) {
        float v = iou_aabb(cpu.aabbs[i], g);
        if (v > best) best = v;
      }
      if (best > 0.5f) {
        agg_matched++;
        agg_iou_sum += best;
      } else {
        agg_missed++;
      }
    }
    std::vector<int> mark(gpu.aabbs.size(), 0);
    for (size_t i = 0; i < cpu.aabbs.size(); ++i) {
      float best = 0; int bi = -1;
      for (size_t j = 0; j < gpu.aabbs.size(); ++j) {
        float v = iou_aabb(cpu.aabbs[i], gpu.aabbs[j]);
        if (v > best) { best = v; bi = (int)j; }
      }
      if (bi >= 0 && best > 0.5f) mark[bi] = 1;
    }
    for (auto m : mark) if (!m) agg_extra++;

    cudaFree(d_buf);
  }

  printf("\n=== AGGREGATE ===\n");
  printf("CPU=%d GPU=%d  matched=%d (%.1f%% of CPU)  missed=%d  extra=%d  "
         "mean_iou=%.3f\n",
         agg_cpu, agg_gpu, agg_matched,
         agg_cpu > 0 ? 100.0 * agg_matched / agg_cpu : 0.0,
         agg_missed, agg_extra,
         agg_matched > 0 ? agg_iou_sum / agg_matched : 0.0);

  cudaStreamDestroy(stream);
  return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
