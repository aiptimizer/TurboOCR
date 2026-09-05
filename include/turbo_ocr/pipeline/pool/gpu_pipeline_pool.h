#pragma once

#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include "turbo_ocr/pipeline/pool/pipeline_pool.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"

// TimeoutError now lives in turbo_ocr/common/errors.h (included above) so the
// CPU build sees it too.

namespace turbo_ocr::pipeline {

// Immutable recipe for (re)building one GpuPipelineEntry. Captured once at
// pool construction so a watchdog can rebuild a wedged entry without threading
// every model path back through the call site. All fields are owned by value
// so the spec safely outlives the request that triggered the rebuild.
struct PipelineBuildSpec {
  std::string det_model;
  std::string rec_model;
  std::string rec_dict;
  std::string cls_model;
  std::string layout_model;
  std::string doc_ori_model;
  DetInferConfig det_cfg{turbo_ocr::detection::kDetResizeDefault,
                         turbo_ocr::detection::kDbDefaults};
};

// Load the optional CUA router + table/formula stages onto `pipeline` when a
// backend is configured via env (matches the server's env-config style):
//   FORMULA_BACKEND  formulanet | ppformulanet_s | vlm  (vlm needs no local files)
//   FORMULA_ONNX / FORMULA_TOKENIZER         file-based formula backends
//   TABLE_{CLS,CELL_*,SLANEXT_*}_TRT         table stage (self-skips if absent)
// No-op when nothing is configured, so the text-only path is untouched.
inline void maybe_load_router_models(OcrPipeline &pipeline) {
  auto env = [](const char *k) {
    const char *v = std::getenv(k);
    return std::string(v ? v : "");
  };
  const bool want_router = std::getenv("TURBO_ROUTING_CONFIG") != nullptr ||
                           std::getenv("FORMULA_BACKEND") != nullptr ||
                           !env("FORMULA_ONNX").empty() ||
                           std::getenv("TABLE_BACKEND") != nullptr ||
                           !env("TABLE_CLS_TRT").empty() ||
                           !env("TABLE_SLANEXT_ENCODER_ONNX").empty() ||
                           !env("VLLM_TABLE_BASE_URL").empty();
  if (!want_router) {
    // Text-only is a valid mode, but running it UNINTENTIONALLY is the classic
    // footgun: tables come back empty and formulas are dropped with no error, so
    // a full-document benchmark silently scores at the floor. Say so once, loudly.
    static std::once_flag warned_text_only;
    std::call_once(warned_text_only, [] {
      std::cerr << "[Pipeline] NOTE: table + formula stages are DISABLED — running "
                   "TEXT-ONLY. Tables/formulas will be empty. To enable full-document "
                   "parsing set FORMULA_BACKEND=ppformulanet_s (+ FORMULA_ONNX / "
                   "FORMULA_TOKENIZER) and TABLE_BACKEND=slanext (+ "
                   "TABLE_SLANEXT_ENCODER_ONNX).\n"
                << std::flush;
    });
    return;
  }
  // Auto-resolve the baked formula weights when only FORMULA_BACKEND is set
  // (parity with TABLE_BACKEND=slanext, which defaults its encoder path): use
  // models/formula/<engine> and let the recognizer find its own files (fast/ for
  // -S, the dir itself for plus-M). Gated on FORMULA_BACKEND being EXPLICITLY
  // set, so text-only stays the default; the per-request ?formulas=1 opt-in
  // still gates execution (loading a backend != running it).
  std::string formula_onnx = env("FORMULA_ONNX");
  std::string formula_tok  = env("FORMULA_TOKENIZER");
  if (const char *fb = std::getenv("FORMULA_BACKEND");
      fb && *fb && formula_onnx.empty() &&
      (std::string(fb) == "ppformulanet_s" ||
       std::string(fb) == "ppformulanet_plus_m")) {
    formula_onnx = std::string("models/formula/") + fb;
    if (formula_tok.empty()) formula_tok = formula_onnx + "/tokenizer.json";
  }
  // FORMULA_BACKEND=auto (composite): point at the -S bundle so the load's
  // existence check passes; AutoCjkFormula resolves the plus-M sibling itself.
  if (const char *fb = std::getenv("FORMULA_BACKEND");
      fb && std::string(fb) == "auto" && formula_onnx.empty()) {
    formula_onnx = "models/formula/ppformulanet_s";
    if (formula_tok.empty()) formula_tok = formula_onnx + "/tokenizer.json";
  }
  // Router (CuaRouter) + formula stage. A
  // CONFIGURED backend that fails to load (out-of-memory / bad model) ABORTS
  // boot — we never start a server that silently produces no formulas/tables.
  if (!pipeline.load_router_models(
          env("TABLE_CLS_TRT"), env("TABLE_CELL_WIRED_TRT"),
          env("TABLE_CELL_WIRELESS_TRT"), env("TABLE_SLANEXT_WIRED_TRT"),
          env("TABLE_SLANEXT_WIRELESS_TRT"), formula_onnx,
          formula_tok))
    throw turbo_ocr::ModelLoadError(
        "configured formula backend failed to load (out-of-memory or bad model); "
        "refusing to start with formulas silently disabled — lower "
        "PIPELINE_POOL_SIZE, fix FORMULA_ONNX, or free VRAM");
  // Pluggable table backend (TABLE_BACKEND -> slanext|vlm; inferred from env).
  if (!pipeline.load_table_backend())
    throw turbo_ocr::ModelLoadError(
        "configured table backend failed to load (out-of-memory or bad model); "
        "refusing to start with tables silently disabled — lower "
        "PIPELINE_POOL_SIZE, fix the table model, or free VRAM");
}

/// OcrPipeline + its dedicated CUDA stream + its own nvJPEG decoder, managed
/// as a single poolable unit. One per dispatcher worker thread.
struct GpuPipelineEntry {
  std::unique_ptr<OcrPipeline> pipeline;
  cudaStream_t stream = nullptr;
  // Lazily constructed on the worker thread so the nvJPEG handle binds to
  // the same primary context that owns `stream` and `pipeline`.
  std::unique_ptr<decode::NvJpegDecoder> nvjpeg;
  // Second decoder on the hybrid backend, created only when a bitstream the
  // hardware path reports as unsupported (progressive, arithmetic) arrives,
  // so those still decode on the GPU instead of the host codec.
  std::unique_ptr<decode::NvJpegDecoder> nvjpeg_hybrid;

  GpuPipelineEntry() = default;

  GpuPipelineEntry(std::unique_ptr<OcrPipeline> p, cudaStream_t s)
      : pipeline(std::move(p)), stream(s) {}

  ~GpuPipelineEntry() noexcept {
    nvjpeg_hybrid.reset();
    nvjpeg.reset();
    if (stream)
      cudaStreamDestroy(stream);
  }

  decode::NvJpegDecoder &get_nvjpeg() {
    if (!nvjpeg) nvjpeg = std::make_unique<decode::NvJpegDecoder>();
    return *nvjpeg;
  }
  decode::NvJpegDecoder &get_nvjpeg_hybrid() {
    if (!nvjpeg_hybrid)
      nvjpeg_hybrid = std::make_unique<decode::NvJpegDecoder>(
          decode::NvJpegDecoder::Backend::Hybrid);
    return *nvjpeg_hybrid;
  }

  // Rebuild a wedged entry in place: tear down the old pipeline, stream, and
  // nvJPEG handle, then construct a fresh OcrPipeline + stream from `spec` and
  // re-warm it. Must run on the worker thread that owns this entry (so the new
  // CUDA resources bind to that thread's primary context, like get_nvjpeg).
  // Throws on init/load failure, leaving the entry in a torn-down (pipeline ==
  // nullptr) state; the caller treats that as a still-dead slot. A wedged
  // stream may refuse cudaStreamDestroy with a sticky error — that's tolerated
  // here (the GPU context itself is poisoned by then, handled separately).
  void recycle(const PipelineBuildSpec &spec) {
    nvjpeg_hybrid.reset();
    nvjpeg.reset();
    pipeline.reset();
    if (stream) {
      cudaStreamDestroy(stream); // best effort: may fail on a poisoned context
      stream = nullptr;
    }

    auto fresh = std::make_unique<OcrPipeline>();
    if (!fresh->init(spec.det_model, spec.rec_model, spec.rec_dict,
                     spec.cls_model, spec.det_cfg))
      throw turbo_ocr::ModelLoadError("[Recycle] Failed to re-init GPU pipeline");
    if (!spec.layout_model.empty() && !fresh->load_layout_model(spec.layout_model))
      throw turbo_ocr::ModelLoadError("[Recycle] Failed to reload layout model");
    if (!spec.doc_ori_model.empty())
      (void)fresh->load_doc_ori_model(spec.doc_ori_model); // soft-disable on fail
    maybe_load_router_models(*fresh);

    cudaStream_t fresh_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&fresh_stream, cudaStreamNonBlocking));
    pipeline = std::move(fresh);
    stream = fresh_stream;
    pipeline->warmup_gpu(stream);
  }

  GpuPipelineEntry(GpuPipelineEntry &&o) noexcept
      : pipeline(std::move(o.pipeline)), stream(o.stream),
        nvjpeg(std::move(o.nvjpeg)) {
    o.stream = nullptr;
  }
  GpuPipelineEntry &operator=(GpuPipelineEntry &&o) noexcept {
    if (this != &o) {
      nvjpeg.reset();
      if (stream) cudaStreamDestroy(stream);
      pipeline = std::move(o.pipeline);
      stream = o.stream;
      nvjpeg = std::move(o.nvjpeg);
      o.stream = nullptr;
    }
    return *this;
  }
  GpuPipelineEntry(const GpuPipelineEntry &) = delete;
  GpuPipelineEntry &operator=(const GpuPipelineEntry &) = delete;
};

/// Convenience alias — a PipelinePool of GpuPipelineEntry.
using GpuPipelinePool = PipelinePool<GpuPipelineEntry>;

/// Factory: create, init, warmup GPU pipelines and return a pool.
/// `layout_model` is an optional TRT engine path — pass "" to disable the
/// layout stage entirely (zero added cost at runtime).
[[nodiscard]] inline std::unique_ptr<GpuPipelinePool> make_gpu_pipeline_pool(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "",
    const DetInferConfig &det_cfg = {turbo_ocr::detection::kDetResizeDefault,
                                     turbo_ocr::detection::kDbDefaults}) {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Pool] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    auto pipeline = std::make_unique<OcrPipeline>();
    if (!pipeline->init(det_model, rec_model, rec_dict, cls_model, det_cfg)) {
      std::cerr << std::format("[Pool] Failed to init GPU pipeline {} of {}", i, pool_size) << '\n';
      continue;
    }
    if (!layout_model.empty()) {
      if (!pipeline->load_layout_model(layout_model)) {
        // Fail hard: mixing layout-on / layout-off pipelines in the same
        // pool would make response shape non-deterministic depending on
        // which handle the request happened to acquire.
        throw turbo_ocr::ModelLoadError(std::format(
            "[Pool] Failed to load layout model for pipeline {} of {}",
            i, pool_size));
      }
    }
    maybe_load_router_models(*pipeline);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    entries.push_back(std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(
        std::format("[Pool] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Warming up {} pipelines...", entries.size()) << '\n';
  for (auto &e : entries) {
    e->pipeline->warmup_gpu(e->stream);
  }
  std::cout << "Pipeline warmup complete." << '\n';

  return std::make_unique<GpuPipelinePool>(std::move(entries));
}

} // namespace turbo_ocr::pipeline
