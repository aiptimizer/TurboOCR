// PdfPageSink store layer: the helpers page tasks use to write finished pages
// into their slots under the sink lock. Compiled into both servers.

#include "turbo_ocr/pipeline/pdf/pdf_job.h"

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

#include "turbo_ocr/classification/doc_orientation_common.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif
#include "pdf_job_internal.h"
#include "turbo_ocr/pipeline/reading_order_util.h"

namespace turbo_ocr::pipeline {

namespace detail {

std::vector<uint8_t>
maybe_encode_page(const PdfPageSink &sink, const cv::Mat &img) {
  if (sink.image_mode != PdfImageMode::Inline) return {};
  return pdf::encode_page_image(img, sink.encode_opts);
}

pdf::PdfMode page_mode_of(PdfPageSink &sink, int page_idx) {
  std::lock_guard<std::mutex> lock(sink.results_mutex);
  return page_idx < static_cast<int>(sink.page_results.size())
      ? sink.page_results[page_idx].resolved_mode
      : pdf::PdfMode::Ocr;
}

// Store one OCR'd page: tag sources, apply auto_verified text-layer
// replacement, write the slot under the sink lock.
void store_ocr_page(PdfPageSink &sink, int page_idx,
                           OcrPipelineResult out, int width, int height,
                           std::vector<uint8_t> encoded_image,
                    int orientation_deg) {
  for (auto &it : out.results) it.source = "ocr";

  const pdf::PdfMode page_mode = page_mode_of(sink, page_idx);
  if (page_mode == pdf::PdfMode::AutoVerified &&
      page_idx < static_cast<int>(sink.page_text_cache.size()) && sink.pdf_doc)
    pdf::verify_results_with_text_layer(out.results, *sink.pdf_doc,
                                        page_idx, sink.dpi);

  std::lock_guard<std::mutex> lock(sink.results_mutex);
  auto &slot = sink.page_results[page_idx];
  move_pipeline_fields(slot, std::move(out));
  slot.width         = width;
  slot.height        = height;
  slot.effective_dpi = sink.dpi;
  slot.encoded_image = std::move(encoded_image);
  slot.orientation_deg = orientation_deg;
  if (page_mode == pdf::PdfMode::Ocr)
    slot.resolved_mode = pdf::PdfMode::Ocr;
}

// Rescale text-layer boxes from PDF points (DPI 72) to the render's pixel
// space. Shared by every geometric path (layout-only, +structure, CPU) so the
// scale/rounding can't drift between them.
void rescale_boxes_pt_to_px(std::vector<OCRResultItem> &results,
                             int dpi) {
  const float pt_to_px = static_cast<float>(dpi) / 72.0f;
  for (auto &item : results)
    for (int k = 0; k < 4; ++k) {
      item.box[k][0] = static_cast<int>(std::round(item.box[k][0] * pt_to_px));
      item.box[k][1] = static_cast<int>(std::round(item.box[k][1] * pt_to_px));
    }
}

// Geometric page: text came from the PDF layer in pt-space. Store layout, then
// rescale the stored text boxes to pixel space; compute reading order over the
// (now pixel-space) text + layout when requested, for parity with OCR pages.
void store_geometric_page(PdfPageSink &sink, int page_idx,
                                 std::vector<layout::LayoutBox> layout,
                                 int width, int height, bool want_reading_order,
                                 std::vector<uint8_t> encoded_image,
                          int orientation_deg) {
  std::lock_guard<std::mutex> lock(sink.results_mutex);
  auto &slot = sink.page_results[page_idx];
  rescale_boxes_pt_to_px(slot.results, sink.dpi);
  slot.layout        = std::move(layout);
  slot.width         = width;
  slot.height        = height;
  slot.effective_dpi = sink.dpi;
  slot.encoded_image = std::move(encoded_image);
  slot.orientation_deg = orientation_deg;
  maybe_assign_reading_order(want_reading_order, slot.results, slot.layout,
                             slot.reading_order);
}

// Render a finished page's Markdown via the sink hook. The slot is MOVED out
// under the mutex, rendered lock-free (the exporter crops figures from `img`,
// too heavy to hold the sink lock), and moved back. Safe because exactly one
// worker owns each page index and emit_page_ready has not run yet; a
// concurrent page_results resize only relocates the temporarily-empty slot.
void maybe_render_page_markdown(PdfPageSink &sink, int page_idx,
                                       const cv::Mat &img) {
  if (!sink.render_page_markdown) return;
  PdfPageResult local;
  {
    std::lock_guard<std::mutex> lk(sink.results_mutex);
    local = std::move(sink.page_results[static_cast<size_t>(page_idx)]);
  }
  local.markdown = sink.render_page_markdown(local, img);
  {
    std::lock_guard<std::mutex> lk(sink.results_mutex);
    sink.page_results[static_cast<size_t>(page_idx)] = std::move(local);
  }
}

} // namespace detail

} // namespace turbo_ocr::pipeline
