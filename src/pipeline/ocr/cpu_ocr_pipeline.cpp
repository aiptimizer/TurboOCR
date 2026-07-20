#include "turbo_ocr/pipeline/ocr/cpu_ocr_pipeline.h"
#include "turbo_ocr/pipeline/reading_order_util.h"

#include <algorithm>
#include <format>
#include <iostream>
#include <ranges>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/log/stage_profiler.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/router/router_types.h"

// Shared no-silent-failure guard for the base text stage (single source of
// truth for the text_degraded flag + warning string; keeps the CPU and GPU
// pipelines byte-identical on the degraded response).
#include "ocr_pipeline_detail.h"

namespace turbo_ocr::pipeline {

using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;
using ::turbo_ocr::sorted_boxes;
using ::turbo_ocr::is_vertical_box;

namespace {
// PP-DocLayoutV3 class ids that route to the structure recognizers (see
// turbo_ocr/layout/layout_types.h kLayoutLabels). Formula = display_formula(5)
// + formula_number(11) + inline_formula(15); table = table(21).
constexpr bool is_formula_class(int c) noexcept { return c == 5 || c == 11 || c == 15; }
constexpr bool is_table_class(int c) noexcept { return c == 21; }
}  // namespace

CpuOcrPipeline::CpuOcrPipeline()
    : det_(std::make_unique<detection::CpuPaddleDet>()),
      rec_(std::make_unique<recognition::CpuPaddleRec>()) {}

bool CpuOcrPipeline::init(const std::string &det_model,
                           const std::string &rec_model,
                           const std::string &rec_dict,
                           const std::string &cls_model,
                           const DetInferConfig &det_cfg) {
  if (!det_->load_model(det_model, det_cfg.resize, det_cfg.db))
    return false;
  if (!rec_->load_model(rec_model))
    return false;
  if (!rec_->load_dict(rec_dict))
    return false;

  if (!cls_model.empty()) {
    cls_ = std::make_unique<classification::CpuPaddleCls>();
    if (!cls_->load_model(cls_model)) {
      std::cerr << std::format("[Pipeline] Failed to load CPU cls model: {}", cls_model) << '\n';
      return false;
    }
    use_cls_ = true;
    std::cout << "[Pipeline] Angle classifier enabled (CPU, "
              << (classification::cls_all_boxes_enabled() ? "all boxes"
                                                          : "vertical-only gate")
              << ")" << '\n';
  }

  return true;
}

void CpuOcrPipeline::warmup() {
  cv::Mat dummy(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(dummy, cv::Point(10, 30), cv::Point(90, 70),
                cv::Scalar(0, 0, 0), 2);
  auto results = run(dummy);
  (void)results;
}

std::vector<OCRResultItem> CpuOcrPipeline::run(const cv::Mat &img) {
  std::size_t num_boxes = 0;
  return run_core_(img, num_boxes);
}

std::vector<OCRResultItem> CpuOcrPipeline::run_core_(const cv::Mat &img,
                                                     std::size_t &num_boxes) {
  namespace prof = turbo_ocr::prof;

  // Detection
  std::vector<Box> boxes;
  {
    prof::Scope _s(prof::DET);
    boxes = det_->run(img);
  }
  num_boxes = boxes.size();

  // Sort boxes top-to-bottom, left-to-right
  {
    prof::Scope _s(prof::SORT);
    sorted_boxes(boxes);
  }

  // Optional angle classification -- default: only when a vertical-looking
  // box is present; CLS_ALL_BOXES=1 checks every crop (mixed-orientation scans
  // have upside-down horizontal lines geometry alone cannot detect).
  if (use_cls_ && cls_ &&
      (classification::cls_all_boxes_enabled() ||
       std::ranges::any_of(boxes, is_vertical_box))) {
    prof::Scope _s(prof::CLS);
    cls_->run(img, boxes);
  }

  // Recognition
  std::vector<std::pair<std::string, float>> rec_results;
  {
    prof::Scope _s(prof::REC);
    rec_results = rec_->run(img, boxes);
  }

  // Combine (filter by drop_score)
  prof::Scope _scombine(prof::COMBINE);
  constexpr float kDropScore = turbo_ocr::kDropScore;
  std::vector<OCRResultItem> final_results;
  final_results.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (i < rec_results.size()) {
      if (rec_results[i].second < kDropScore)
        continue;
      if (rec_results[i].first.empty())
        continue;
      final_results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
      });
    }
  }
  if (rec_results.size() < boxes.size())
    // Recognizer under-ran the detector: fail loud (parity with formula/table),
    // don't silently drop detected text lines.
    std::cerr << std::format(
        "[Pipeline] rec returned {} of {} detected boxes — {} line(s) dropped "
        "(recognizer under-run, not empty text)\n",
        rec_results.size(), boxes.size(), boxes.size() - rec_results.size());

  return final_results;
}

bool CpuOcrPipeline::load_layout_model(const std::string &onnx_path) {
  layout_ = std::make_unique<layout::CpuPaddleLayout>();
  if (!layout_->load_model(onnx_path)) {
    layout_.reset();
    return false;
  }
  return true;
}

bool CpuOcrPipeline::load_formula_model(const std::string &model_path,
                                        const std::string &tokenizer_json) {
  formula_ = std::make_unique<formula::CpuFormulaRecognizer>();
  if (!formula_->load(model_path, tokenizer_json)) {
    formula_.reset();
    return false;
  }
  std::cout << "[Pipeline] Formula stage enabled (CPU, PP-FormulaNet-S fused/ORT-CPU)\n";
  return true;
}

bool CpuOcrPipeline::load_table_backend() {
  table_ = std::make_unique<table::CpuSlanextTableRecognizer>();
  if (!table_->load()) {
    table_.reset();
    return false;
  }
  // Per-cell crop OCR for grid cells the page detector under-segmented.
  if (rec_) table_->set_cell_recognizer(rec_.get());
  return true;
}

void CpuOcrPipeline::run_structure_stages(const cv::Mat &img,
                                          OcrPipelineResult &out,
                                          bool want_tables, bool want_formulas) {
  if (out.layout.empty()) return;
  if (!(want_tables && table_) && !(want_formulas && formula_)) return;

  // Formula regions -> LaTeX (display_formula / formula_number / inline_formula).
  if (want_formulas && formula_ && formula_->ready()) {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::FORMULA);
    std::vector<Box> fboxes;
    std::vector<int> flids;
    fboxes.reserve(out.layout.size());
    flids.reserve(out.layout.size());
    for (std::size_t lid = 0; lid < out.layout.size(); ++lid) {
      if (!is_formula_class(out.layout[lid].class_id)) continue;
      flids.push_back(static_cast<int>(lid));
      fboxes.push_back(out.layout[lid].box);
    }
    if (!fboxes.empty()) {
      auto latexes = formula_->recognize_regions(img, fboxes);
      std::size_t degraded = 0;
      out.formulas.reserve(latexes.size());
      for (std::size_t i = 0; i < latexes.size() && i < flids.size(); ++i) {
        const int lid = flids[i];
        if (latexes[i].empty()) ++degraded;  // dispatched-as-formula but no LaTeX
        router::FormulaResult fr;
        fr.layout_id = lid;
        fr.latex = std::move(latexes[i]);
        fr.score = out.layout[lid].score;
        fr.box = out.layout[lid].box;
        out.formulas.push_back(std::move(fr));
      }
      if (degraded > 0) {
        out.formula_degraded = true;
        out.formula_warning =
            "formula stage degraded: " + std::to_string(degraded) + " of " +
            std::to_string(flids.size()) + " region(s) produced no LaTeX";
      }
    }
  }

  // Table regions -> HTML.
  if (want_tables && table_ && table_->ready()) {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::TABLE);
    std::vector<Box> tboxes;
    std::vector<int> tlids;
    tboxes.reserve(out.layout.size());
    tlids.reserve(out.layout.size());
    for (std::size_t lid = 0; lid < out.layout.size(); ++lid) {
      if (!is_table_class(out.layout[lid].class_id)) continue;
      tlids.push_back(static_cast<int>(lid));
      tboxes.push_back(out.layout[lid].box);
    }
    if (!tboxes.empty()) {
      out.tables = table_->run(img, tboxes, out.results);
      std::size_t degraded = 0;
      for (std::size_t i = 0; i < out.tables.size() && i < tlids.size(); ++i) {
        out.tables[i].layout_id = tlids[i];
        if (out.tables[i].html.empty()) ++degraded;
      }
      if (degraded > 0) {
        out.table_degraded = true;
        out.table_warning =
            "table stage degraded: " + std::to_string(degraded) + " of " +
            std::to_string(out.tables.size()) + " region(s) produced no HTML";
      }
    }
  }
}

bool CpuOcrPipeline::load_doc_ori_model(const std::string &onnx_path) {
  if (onnx_path.empty()) return false;
  doc_ori_ = std::make_unique<classification::CpuDocOrientation>();
  if (!doc_ori_->load_model(onnx_path)) {
    doc_ori_.reset();
    return false;
  }
  use_doc_ori_ = true;
  return true;
}

int CpuOcrPipeline::detect_orientation(const cv::Mat &bgr) {
  if (!use_doc_ori_) return 0;
  return doc_ori_->detect(bgr);
}

OcrPipelineResult CpuOcrPipeline::run_with_layout(const cv::Mat &img,
                                                    bool want_layout,
                                                    bool want_reading_order,
                                                    bool want_tables,
                                                    bool want_formulas) {
  OcrPipelineResult out;
  std::size_t num_boxes = 0;
  out.results = run_core_(img, num_boxes);
  // No-silent-failure parity with the GPU pipeline: detection found text
  // regions but recognition produced nothing usable -> flag text_degraded so a
  // recognition failure on a text page is never a byte-identical clean empty
  // 200. A genuinely text-free page (num_boxes == 0) is not degraded.
  detail::flag_text_degraded(out, num_boxes);
  if (want_layout && layout_) {
    turbo_ocr::prof::Scope _s(turbo_ocr::prof::LAYOUT);
    out.layout = layout_->run(img);
  }

  // Formula + table recognition over the detected layout regions (CUDA-free).
  // Strict opt-in — layout alone never triggers them (parity with the GPU path).
  run_structure_stages(img, out, want_tables, want_formulas);

  // Reading-order over layout regions, with synthetic XY-cut entries
  // for orphan results (results whose centroid falls outside every
  // layout box). assign_layout_ids() resolves layout_id by centroid
  // containment so the helper can synthesise correctly. Idempotent —
  // serialization re-runs it but the second call is a no-op.
  maybe_assign_reading_order(want_reading_order, out.results, out.layout,
                             out.reading_order);

  return out;
}

} // namespace turbo_ocr::pipeline
