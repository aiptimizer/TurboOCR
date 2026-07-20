#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/bootstrap/stages_cpu.h"

#include <cstdlib>
#include <stdexcept>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/env_utils.h"

namespace turbo_ocr::server {

namespace {

// Layout (on by default): a missing layout.onnx soft-disables the stage —
// the model is optional tooling, unlike the explicitly-configured backends
// below.
bool load_layout(pipeline::CpuPipelinePool &pool, int pool_size,
                 const ServerConfig &cfg) {
  if (cfg.layout_disabled) {
    TOCR_LOG_INFO("Layout detection disabled");
    return false;
  }
  if (cfg.layout_onnx.empty()) return false;
  for (int i = 0; i < pool_size; ++i) {
    auto handle = pool.acquire();
    if (!handle->load_layout_model(cfg.layout_onnx)) {
      TOCR_LOG_WARN("Layout model not found; layout disabled");
      return false;
    }
  }
  TOCR_LOG_INFO("Layout detection enabled (CPU/ONNX Runtime)");
  return true;
}

// Formula + table (optional, env-gated, same knobs as the GPU server).
// FORMULA_ONNX + FORMULA_TOKENIZER enable PP-FormulaNet-S on ORT-CPU;
// TABLE_SLANEXT_ENCODER_ONNX enables the SLANeXt ORT-CPU encoder + host GRU
// decode. A configured-but-unloadable backend THROWS (fail loud, never serve
// a silently structure-less pipeline).
void load_structure_backends(pipeline::CpuPipelinePool &pool, int pool_size,
                             CpuStageAvailability &avail) {
  std::string formula_onnx = env::env_or("FORMULA_ONNX", "");
  std::string formula_tok = env::env_or("FORMULA_TOKENIZER", "");
  // Auto-resolve baked formula weights when only FORMULA_BACKEND is set
  // (parity with the GPU build): default to models/formula/ppformulanet_s
  // (CPU uses the fused inference_trt.onnx inside it via
  // CpuFormulaRecognizer).
  if (env::env_or("FORMULA_BACKEND", "") == "ppformulanet_s" &&
      formula_onnx.empty()) {
    formula_onnx = "models/formula/ppformulanet_s";
    if (formula_tok.empty())
      formula_tok = "models/formula/ppformulanet_s/tokenizer.json";
  }
  // Auto-resolve the baked SLANeXt encoder when only TABLE_BACKEND=slanext
  // is set; load_table_backend() reads TABLE_SLANEXT_ENCODER_ONNX from the
  // env.
  if (env::env_or("TABLE_BACKEND", "") == "slanext" &&
      env::env_or("TABLE_SLANEXT_ENCODER_ONNX", "").empty()) {
    setenv("TABLE_SLANEXT_ENCODER_ONNX",
           "models/table/slanext_encoder/SLANeXt_wired_encoder.onnx", 1);
  }
  const bool want_formula = !formula_onnx.empty() && !formula_tok.empty();
  const bool want_table = !env::env_or("TABLE_SLANEXT_ENCODER_ONNX", "").empty();
  if (!want_formula && !want_table) return;

  for (int i = 0; i < pool_size; ++i) {
    auto handle = pool.acquire();
    if (want_formula &&
        !handle->load_formula_model(formula_onnx, formula_tok))
      throw std::runtime_error(
          "CPU formula backend failed to load (model: " + formula_onnx +
          ") — refusing to start");
    if (want_table && !handle->load_table_backend())
      throw std::runtime_error(
          "CPU table backend failed to load — refusing to start");
    // Read availability from what actually loaded into the pipeline (single
    // source of truth for the tables=1/formulas=1 fail-loud gate), not env
    // intent. All pipelines load identically, so the last wins.
    avail.table = handle->has_table_backend();
    avail.formula = handle->has_formula_backend();
  }
  if (avail.formula)
    TOCR_LOG_INFO("Formula stage enabled (CPU/ONNX Runtime)");
  if (avail.table) TOCR_LOG_INFO("Table stage enabled (CPU/ONNX Runtime)");
}

// Doc-orientation (optional). Powers /ocr/pdf?autorotate=1; absent model ->
// autorotate requests are rejected downstream.
bool load_doc_ori(pipeline::CpuPipelinePool &pool, int pool_size,
                  const ServerConfig &cfg) {
  if (cfg.doc_ori_onnx.empty()) return false;
  for (int i = 0; i < pool_size; ++i) {
    auto handle = pool.acquire();
    if (!handle->load_doc_ori_model(cfg.doc_ori_onnx)) {
      TOCR_LOG_INFO("Doc-orientation model not found; autorotate disabled");
      return false;
    }
  }
  TOCR_LOG_INFO("Doc-orientation (autorotate) enabled (CPU/ONNX Runtime)");
  return true;
}

} // namespace

CpuStageAvailability load_cpu_stages(pipeline::CpuPipelinePool &pool,
                                     int pool_size, const ServerConfig &cfg) {
  CpuStageAvailability avail;
  avail.layout = load_layout(pool, pool_size, cfg);
  load_structure_backends(pool, pool_size, avail);
  avail.doc_ori = load_doc_ori(pool, pool_size, cfg);
  return avail;
}

InferFunc make_cpu_infer_func(pipeline::CpuPipelinePool &pool) {
  return [&pool](const cv::Mat &img,
                 const InferOptions &opts) -> InferResult {
    auto handle = pool.acquire();
    auto out = handle->run_with_layout(img, opts.want_layout,
                                       opts.want_reading_order,
                                       opts.want_tables, opts.want_formulas);
    return InferResult{
        .results          = std::move(out.results),
        .layout           = std::move(out.layout),
        .reading_order    = std::move(out.reading_order),
        .tables           = std::move(out.tables),
        .formulas         = std::move(out.formulas),
        .formula_degraded = out.formula_degraded,
        .formula_warning  = std::move(out.formula_warning),
        .table_degraded   = out.table_degraded,
        .table_warning    = std::move(out.table_warning),
        .text_degraded    = out.text_degraded,
        .text_warning     = std::move(out.text_warning),
    };
  };
}

OrientFunc make_cpu_orient_func(pipeline::CpuPipelinePool &pool,
                                bool doc_ori_available) {
  if (!doc_ori_available) return {};
  return [&pool](const cv::Mat &img) -> int {
    auto handle = pool.acquire();
    return handle->detect_orientation(img);
  };
}

} // namespace turbo_ocr::server
