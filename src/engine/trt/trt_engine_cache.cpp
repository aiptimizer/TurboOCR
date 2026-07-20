// Engine-cache location, cache-key derivation and orphan temp-file sweep for
// the ONNX->TRT builder.

#include "turbo_ocr/engine/trt/onnx_to_trt.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "turbo_ocr/common/env_utils.h"
#include "engine_internal.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace turbo_ocr::engine {

namespace detail {

int read_det_effective_max_side() {
  return detection::effective_det_max_side(detection::read_det_resize());
}

int read_det_opt_batch() {
  static const int v = env::env_int("DET_OPT_BATCH", 4, 1, 8);
  return v;
}

int read_trt_opt_level() {
  static const int v = env::env_int("TRT_OPT_LEVEL", 5, 0, 5);
  return v;
}

} // namespace detail

std::string get_engine_cache_dir() {
  // User override
  if (auto *env = std::getenv("TRT_ENGINE_CACHE"))
    return env;

  // Try ~/.cache/turbo-ocr/
  if (auto *home = std::getenv("HOME")) {
    auto dir = std::string(home) + "/.cache/turbo-ocr";
    fs::create_directories(dir);
    return dir;
  }

  // Fallback to /tmp
  auto dir = std::string("/tmp/turbo-ocr-engines");
  fs::create_directories(dir);
  return dir;
}

std::string get_cached_engine_path(const std::string &onnx_path,
                                   const std::string &type) {
  auto cache_dir = get_engine_cache_dir();

  // Build a cache key from: onnx file size + mtime + TRT version
  auto onnx_size = fs::file_size(onnx_path);
  auto onnx_mtime = fs::last_write_time(onnx_path).time_since_epoch().count();

  int trt_major = 0, trt_minor = 0, trt_patch = 0;
#ifdef NV_TENSORRT_MAJOR
  trt_major = NV_TENSORRT_MAJOR;
  trt_minor = NV_TENSORRT_MINOR;
  trt_patch = NV_TENSORRT_PATCH;
#endif

  // GPU compute capability (engines are GPU-architecture specific)
  int gpu_major = 0, gpu_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_minor, cudaDevAttrComputeCapabilityMinor, 0));

  // CUDA driver + runtime versions: a host driver upgrade can invalidate
  // engines cached under the previous driver (cryptic CUDA errors at
  // deserialize time). TRT version covers majors but not driver minor
  // patches, so include both integers (e.g. 13020 for CUDA 13.2) directly.
  int cuda_driver = 0, cuda_runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

  // Cache key includes: onnx identity, TRT version, GPU arch, and profile version.
  // Bump kProfileVersion when optimization profiles change for det/rec/cls.
  // Adding a NEW model type (e.g. "layout") does NOT require a bump because
  // the cache key includes `type` — new types live in their own hash space.
  // 2026-04-26: bumped because the det profile MAX now tracks DET_MAX_SIDE.
  // 2026-06-15: v5->v6 det/rec swap — v5 engines are not reusable; bump
  // invalidates them belt-and-suspenders over the path+size+mtime key.
  // The det profile MAX follows the per-model effective max-side (default
  // 1280); the `:dms` suffix below separates engines built at different
  // effective max-sides by hash, so a max-side change needs no bump.
  static constexpr int kProfileVersion = 20260615;

  auto key = "v" + std::to_string(kProfileVersion) + ":" + type + ":" +
      onnx_path + ":" + std::to_string(onnx_size) + ":" +
      std::to_string(onnx_mtime) + ":" + std::to_string(trt_major) + "." +
      std::to_string(trt_minor) + "." + std::to_string(trt_patch) + ":sm" +
      std::to_string(gpu_major) + "." + std::to_string(gpu_minor) +
      ":drv" + std::to_string(cuda_driver) +
      ":rt" + std::to_string(cuda_runtime);
  // For det, the MAX dim of the optimization profile is the effective det
  // max-side (per-model max_side_limit, overridden by DET_MAX_SIDE), so each
  // operator config gets its own engine (Triton/vLLM pattern). A different
  // effective max => a different profile => a different cache key.
  if (type == "det")
    key += ":dms" + std::to_string(detail::read_det_effective_max_side()) +
           ":dob" + std::to_string(detail::read_det_opt_batch());
  // rec gets extra static (batch,width) profiles + maxAuxStreams=0 ONLY when
  // CUDA graphs are opted into. Default (graphs off) keeps the original
  // single-profile cache key, so existing cached engines are reused as-is and
  // no rebuild is triggered on upgrade.
  if (type == "rec" && TrtEngine::graphs_enabled())
    key += ":gp12aux0";
  // TRT_OPT_LEVEL changes which kernels TensorRT picks, so the produced
  // engine differs. Operators that toggle the level get separate cached
  // engines instead of silently reusing a stale one.
  key += ":opt" + std::to_string(detail::read_trt_opt_level());
  auto hash = std::hash<std::string>{}(key);

  return cache_dir + "/" + type + "_" + std::to_string(hash) + ".trt";
}

void sweep_orphan_engine_temps(int min_age_seconds) {
  std::error_code ec;
  const auto cache_dir = get_engine_cache_dir();
  if (!fs::exists(cache_dir, ec) || ec) return;

  const auto now = fs::file_time_type::clock::now();
  int removed = 0;
  for (const auto &entry : fs::directory_iterator(cache_dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const auto name = entry.path().filename().string();
    if (name.find(".tmp.") == std::string::npos) continue;

    const auto mtime = fs::last_write_time(entry.path(), ec);
    if (ec) { ec.clear(); continue; }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - mtime).count();
    if (age < min_age_seconds) continue;

    fs::remove(entry.path(), ec);
    if (!ec) ++removed;
    ec.clear();
  }
  if (removed > 0)
    std::cout << "[TRT] Removed " << removed
              << " orphan engine temp file(s) from " << cache_dir << '\n';
}

} // namespace turbo_ocr::engine
