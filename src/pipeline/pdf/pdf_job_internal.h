#pragma once

// Internal seams of the pdf_job_* TUs: the page-prep helpers both run_pdf_job
// overloads call, the shared PdfPageSink page tasks write into, and the store
// helpers. Nothing here is part of the public pdf_job.h surface.

#include <atomic>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pdf/pdf_job.h"

namespace turbo_ocr::pipeline {

// One move of the full per-page pipeline field set (results / layout /
// reading_order / tables / formulas + the three degraded/warning pairs).
// Shared by every store path and the serializer's inverse copy so the field
// set cannot silently drift when a new pipeline output is added.
template <typename Dst, typename Src>
inline void move_pipeline_fields(Dst &dst, Src &&src) {
  dst.results = std::move(src.results);
  dst.layout = std::move(src.layout);
  dst.reading_order = std::move(src.reading_order);
  dst.tables = std::move(src.tables);
  dst.formulas = std::move(src.formulas);
  dst.formula_degraded = src.formula_degraded;
  dst.formula_warning = std::move(src.formula_warning);
  dst.table_degraded = src.table_degraded;
  dst.table_warning = std::move(src.table_warning);
  dst.text_degraded = src.text_degraded;
  dst.text_warning = std::move(src.text_warning);
}

// Open the PDF and pre-extract per-page text only when the mode needs it.
// mode=ocr skips this. On open failure, downgrade to ocr and clear the doc.
void open_pdf_for_text_layer(
    const uint8_t *pdf_data, size_t pdf_len, pdf::PdfMode &mode,
    std::unique_ptr<pdf::PdfDocument> &pdf_doc,
    std::vector<pdf::PdfPageText> &page_text_cache);

// Decide per-page resolved_mode + whether each page needs rendering, from the
// text-layer quality. Only called for non-ocr modes. AutoVerified is GPU-only
// (CPU aliases it to Auto before calling). `want_page_image` forces a render
// for text-layer pages so the encoder has pixels.
void prepopulate_pages(pdf::PdfMode mode, bool layout_or_want_layout,
                              const std::vector<pdf::PdfPageText> &page_text_cache,
                              std::vector<PdfPageResult> &page_results,
                              std::vector<uint8_t> &need_render,
                              bool *any_need_render,
                              bool want_page_image = false);

namespace detail {

// Shared state every page task writes into. Held by shared_ptr and captured BY
// VALUE into every page task, so a task abandoned on a deadline timeout (its
// future get()-with-timeout abandons it) safely outlives run_pdf_job: the sink
// stays alive until the last task that holds it finishes. It therefore OWNS the
// mutex / results / text cache / document outright (rather than referencing
// run_pdf_job locals that would die at return).
struct PdfPageSink {
  std::mutex results_mutex;
  std::vector<PdfPageResult> page_results;
  std::unique_ptr<pdf::PdfDocument> pdf_doc;  // null when no text layer was opened
  std::vector<pdf::PdfPageText> page_text_cache;
  int dpi = 0;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts{};
  bool autorotate = false;
  // Strict opt-in: run table / formula recognition on layout regions only when
  // the request asked (?tables=1 / ?formulas=1). Default off.
  bool want_tables = false;
  bool want_formulas = false;
  bool want_text = true;
  // Rendered pages whose PPM could not be read back (a server-side fault).
  std::atomic<int> decode_failures{0};
  // Pages whose OCR/inference threw — counted by the true page count (the whole
  // chunk on the batched path), so any > 0 fails the job rather than returning a
  // silently-empty page in a 200.
  std::atomic<int> page_failures{0};
  // Streaming hooks (/ocr/stream). on_page_ready fires exactly once per page,
  // AFTER the page's slot is fully stored, from whichever thread completed it
  // (dispatcher worker for rendered pages, run_pdf_job itself for render-skipped
  // geometric pages) — the callee must be thread-safe. It receives the page
  // moved out of the slot. on_page_failed mirrors the page_failures increment.
  // Both null on the non-streaming routes.
  std::function<void(int page_idx, PdfPageResult &&page)> on_page_ready;
  std::function<void(int page_idx)> on_page_failed;
  // Markdown hook (see PdfJobOptions::render_page_markdown). Applied by the
  // page worker between store and emit_page_ready.
  std::function<std::string(PdfPageResult &page, const cv::Mat &img)>
      render_page_markdown;
  // Move a finished page out of its slot (under the results mutex) and hand it
  // to on_page_ready. Call ONLY after the slot is final.
  void emit_page_ready(int page_idx) {
    PdfPageResult moved;
    {
      std::lock_guard<std::mutex> lk(results_mutex);
      if (page_idx < static_cast<int>(page_results.size()))
        moved = std::move(page_results[static_cast<size_t>(page_idx)]);
    }
    on_page_ready(page_idx, std::move(moved));
  }
};

[[nodiscard]] std::vector<uint8_t>
maybe_encode_page(const PdfPageSink &sink, const cv::Mat &img);

[[nodiscard]] pdf::PdfMode page_mode_of(PdfPageSink &sink, int page_idx);

// Store one OCR'd page: tag sources, apply auto_verified text-layer
// replacement, write the slot under the sink lock.
void store_ocr_page(PdfPageSink &sink, int page_idx, OcrPipelineResult out,
                    int width, int height,
                    std::vector<uint8_t> encoded_image = {},
                    int orientation_deg = 0);

// Rescale text-layer boxes from PDF points (DPI 72) to the render's pixel
// space. Shared by every geometric path (layout-only, +structure, CPU) so the
// scale/rounding can't drift between them.
void rescale_boxes_pt_to_px(std::vector<OCRResultItem> &results, int dpi);

// Geometric page: text came from the PDF layer in pt-space. Store layout, then
// rescale the stored text boxes to pixel space; compute reading order over the
// (now pixel-space) text + layout when requested, for parity with OCR pages.
void store_geometric_page(PdfPageSink &sink, int page_idx,
                          std::vector<layout::LayoutBox> layout, int width,
                          int height, bool want_reading_order,
                          std::vector<uint8_t> encoded_image = {},
                          int orientation_deg = 0);

// Render a finished page's Markdown via the sink hook (see the definition for
// the move-out/render/move-back locking contract).
void maybe_render_page_markdown(PdfPageSink &sink, int page_idx,
                                const cv::Mat &img);

} // namespace detail

} // namespace turbo_ocr::pipeline
