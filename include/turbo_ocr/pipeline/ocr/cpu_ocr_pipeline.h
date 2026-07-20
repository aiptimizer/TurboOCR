#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "turbo_ocr/classification/cpu_doc_orientation.h"
#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/formula/ppformulanet/cpu_formula_recognizer.h"
#include "turbo_ocr/layout/cpu_paddle_layout.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pipeline/ocr/i_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/recognition/cpu_paddle_rec.h"
#include "turbo_ocr/table/slanext/cpu_slanext_table.h"

namespace turbo_ocr::pipeline {

class CpuOcrPipeline : public IOcrPipeline {
public:
  CpuOcrPipeline();
  ~CpuOcrPipeline() noexcept override = default;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict, const std::string &cls_model = "",
                          const DetInferConfig &det_cfg =
                              {turbo_ocr::detection::kDetResizeDefault,
                               turbo_ocr::detection::kDbDefaults}) override;

  [[nodiscard]] bool load_layout_model(const std::string &onnx_path);

  // Load the CUDA-free PP-FormulaNet-S recognizer (fused graph on ORT-CPU).
  // `model_path` is inference_trt.onnx (or its parent dir); `tokenizer_json`
  // is tokenizer.json. Enables per-formula-region LaTeX in run_with_layout.
  [[nodiscard]] bool load_formula_model(const std::string &model_path,
                                        const std::string &tokenizer_json);

  // Load the CUDA-free SLANeXt table backend (ORT-CPU encoder + host decode).
  // Reads TABLE_SLANEXT_* env knobs internally. Enables per-table-region HTML.
  [[nodiscard]] bool load_table_backend();

  // Opt-in availability read from what actually loaded — the same pointers
  // run_structure_stages gates on — so the server's tables=1/formulas=1
  // fail-loud check can't drift from the loaded backends.
  [[nodiscard]] bool has_table_backend() const { return table_ != nullptr; }
  [[nodiscard]] bool has_formula_backend() const { return formula_ != nullptr; }

  // Load the document-orientation model (ONNX). Optional; powers autorotate.
  [[nodiscard]] bool load_doc_ori_model(const std::string &onnx_path);
  [[nodiscard]] bool has_doc_ori() const noexcept { return use_doc_ori_; }
  // Page's detected clockwise rotation (0/90/180/270), or 0 if unavailable.
  [[nodiscard]] int detect_orientation(const cv::Mat &bgr);

  void warmup() override;

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override;

  /// Run OCR + optional layout detection.
  /// `want_reading_order` is opt-in and has no effect when layout is
  /// unavailable — the returned vector stays empty so the JSON serializer
  /// omits the key.
  [[nodiscard]] OcrPipelineResult run_with_layout(const cv::Mat &img,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false,
                                                   bool want_tables = false,
                                                   bool want_formulas = false);

private:
  // Recognize formula + table regions of `img` from its layout boxes, filling
  // out.formulas / out.tables. Strict opt-in: a stage runs only when its flag
  // is set AND its backend is loaded AND a matching layout region exists.
  void run_structure_stages(const cv::Mat &img, OcrPipelineResult &out,
                            bool want_tables, bool want_formulas);

  // Core det -> sort -> (cls) -> rec -> combine, shared by run() and
  // run_with_layout(). Reports the detected box count via `num_boxes` so the
  // caller can apply the no-silent-failure text_degraded guard (parity with the
  // GPU pipeline's detail::flag_text_degraded). Public run() discards the count.
  [[nodiscard]] std::vector<OCRResultItem> run_core_(const cv::Mat &img,
                                                     std::size_t &num_boxes);

  std::unique_ptr<detection::CpuPaddleDet> det_;
  std::unique_ptr<classification::CpuPaddleCls> cls_;
  std::unique_ptr<recognition::CpuPaddleRec> rec_;
  std::unique_ptr<layout::CpuPaddleLayout> layout_;
  std::unique_ptr<classification::CpuDocOrientation> doc_ori_;
  std::unique_ptr<formula::CpuFormulaRecognizer> formula_;
  std::unique_ptr<table::CpuSlanextTableRecognizer> table_;

  bool use_cls_ = false;
  bool use_doc_ori_ = false;
};

} // namespace turbo_ocr::pipeline
