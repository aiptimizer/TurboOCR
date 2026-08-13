// Page preparation for the PDF job: text-layer open + quality gate, per-page
// mode resolution (prepopulate_pages), and the shared per-page serializer.

#include "turbo_ocr/pipeline/job/pdf_job.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/classification/doc_orientation_common.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/analysis/layout/order/reading_order.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include "pdf_job_internal.h"

namespace turbo_ocr::pipeline {

namespace {

void fill_from_text_layer_pt(PdfPageResult &pg,
                                    const pdf::PdfPageText &text) {
  pg.width  = static_cast<int>(std::round(text.page_width_pt));
  pg.height = static_cast<int>(std::round(text.page_height_pt));
  pg.effective_dpi = 72;
  // One OCRResultItem per extracted line, order and boxes preserved. The
  // extractor (pdf_text_layer.cpp) already groups the char stream into visual
  // lines via PDFium's own flow breaks, so no cross-line re-merging is done
  // here — merging by y-overlap would glue adjacent table cells / columns that
  // happen to share a row and would reorder multi-column reading order.
  pg.results.reserve(text.lines.size());
  for (const auto &line : text.lines) {
    OCRResultItem item;
    item.source = "pdf";
    item.confidence = 1.0f;
    item.text = line.text;
    int ix0 = static_cast<int>(std::round(line.x0_pt));
    int iy0 = static_cast<int>(std::round(line.y0_pt));
    int ix1 = static_cast<int>(std::round(line.x1_pt));
    int iy1 = static_cast<int>(std::round(line.y1_pt));
    item.box[0] = {ix0, iy0};
    item.box[1] = {ix1, iy0};
    item.box[2] = {ix1, iy1};
    item.box[3] = {ix0, iy1};
    pg.results.push_back(std::move(item));
  }
}

std::string_view text_layer_quality_for(const pdf::PdfPageText &text) {
  if (text.char_count == 0)                         return "absent";
  if (text.rotation_deg != 0)                       return "rejected";
  if (text.char_count < 10)                         return "absent";
  if (text.fffd_count * 20 > text.char_count)       return "rejected";
  if (text.nonprint_count * 10 > text.char_count)   return "rejected";
  if (text.lines.empty())                           return "absent";
  return "trusted";
}

} // namespace

// Open the PDF and pre-extract per-page text only when the mode needs it.
// mode=ocr skips this. On open failure, downgrade to ocr and clear the doc.
void open_pdf_for_text_layer(
    const uint8_t *pdf_data, size_t pdf_len, pdf::PdfMode &mode,
    std::unique_ptr<pdf::PdfDocument> &pdf_doc,
    std::vector<pdf::PdfPageText> &page_text_cache) {
  if (mode == pdf::PdfMode::Ocr) return;
  pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len);
  if (!pdf_doc->ok()) {
    TOCR_LOG_WARN("Failed to open PDF for text-layer lookup; falling back to mode=ocr",
                  "route", "/ocr/pdf");
    mode = pdf::PdfMode::Ocr;
    pdf_doc.reset();
    return;
  }
  int np = pdf_doc->page_count();
  page_text_cache.reserve(static_cast<size_t>(std::max(0, np)));
  for (int p = 0; p < np; ++p)
    page_text_cache.push_back(pdf_doc->extract_page(p));
}

// Decide per-page resolved_mode + whether each page needs rendering, from the
// text-layer quality. Only called for non-ocr modes. AutoVerified is GPU-only
// (CPU aliases it to Auto before calling). `want_page_image` forces a render
// for text-layer pages so the encoder has pixels.
void prepopulate_pages(pdf::PdfMode mode, bool layout_or_want_layout,
                              const std::vector<pdf::PdfPageText> &page_text_cache,
                              std::vector<PdfPageResult> &page_results,
                              std::vector<uint8_t> &need_render,
                              bool *any_need_render,
                              bool want_page_image) {
  int np = static_cast<int>(page_text_cache.size());
  page_results.resize(static_cast<size_t>(np));
  need_render.assign(static_cast<size_t>(np), 0);

  for (int p = 0; p < np; ++p) {
    const auto &text = page_text_cache[static_cast<size_t>(p)];
    auto &pg = page_results[static_cast<size_t>(p)];
    pg.text_layer_quality = text_layer_quality_for(text);
    bool has_good_layer = (pg.text_layer_quality == "trusted");

    switch (mode) {
      case pdf::PdfMode::Geometric:
        pg.resolved_mode = pdf::PdfMode::Geometric;
        if (has_good_layer) {
          fill_from_text_layer_pt(pg, text);
        } else {
          pg.width = static_cast<int>(std::round(text.page_width_pt));
          pg.height = static_cast<int>(std::round(text.page_height_pt));
          pg.effective_dpi = 72;
        }
        if (layout_or_want_layout) {
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      case pdf::PdfMode::Auto:
        if (has_good_layer) {
          pg.resolved_mode = pdf::PdfMode::Geometric;
          fill_from_text_layer_pt(pg, text);
          if (layout_or_want_layout) {
            need_render[static_cast<size_t>(p)] = 1;
            if (any_need_render) *any_need_render = true;
          }
        } else {
          pg.resolved_mode = pdf::PdfMode::Ocr;
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      // No AutoVerified case: run_pdf_job resolves it to Auto before this runs,
      // so a page can never be tagged resolved_mode=auto_verified by a build
      // that has no verification step. It used to be tagged exactly that.
      default: break;
    }

    // A page image was requested but this is a text-layer page that would
    // otherwise skip rasterization — force a render so the encoder has pixels.
    if (want_page_image && !need_render[static_cast<size_t>(p)]) {
      need_render[static_cast<size_t>(p)] = 1;
      if (any_need_render) *any_need_render = true;
    }
  }
}

// ── Shared per-page serializer (H7) ──────────────────────────────────────
//
// Produces the JSON body for one page's results (without any envelope). Both
// transports call this so the per-page result shape cannot drift: HTTP embeds
// it inside its {pages:[...]} object; gRPC stores it in OCRPageResult's
// json_response bytes. Mirrors the prior `emit_results_json` / `results_to_json`
// branch both sites used, so output is byte-identical.
[[nodiscard]] std::string
serialize_page_results(PdfPageResult &pg, bool want_blocks) {
  // Structure or degradation present -> full pipeline emitter so tables/formulas + the
  // *_degraded signals surface (they were dropped on /ocr/pdf). Otherwise the output is
  // byte-identical to the prior text-only branches.
  if (!pg.tables.empty() || !pg.formulas.empty() || pg.formula_degraded ||
      pg.table_degraded || pg.text_degraded) {
    OcrPipelineResult out;
    move_pipeline_fields(out, std::move(pg));
    return turbo_ocr::emit_pipeline_result_json(out, want_blocks);
  }
  if (!pg.reading_order.empty())
    return emit_results_json(pg.results, pg.layout, pg.reading_order,
                             want_blocks);
  return results_to_json(pg.results, pg.layout);
}

} // namespace turbo_ocr::pipeline
