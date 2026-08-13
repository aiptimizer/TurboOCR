#pragma once

// Internal seams of the pdf_job_* TUs: the page-prep helpers both run_pdf_job
// overloads call, the shared PdfPageSink page tasks write into, and the store
// helpers. Nothing here is part of the public pdf_job.h surface.

#include <atomic>
#include <cstddef>    // size_t
#include <cstdint>    // uint8_t
#include <functional> // std::function
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/job/pdf_job.h"

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

// NOTE (removed): a PdfPageSink lived here — a mutex + slot vector + streaming
// hooks + nine store helpers (pdf_job_sink.cpp), for the PARALLEL page path the
// CUDA dispatcher used to drive. That dispatcher is gone, nothing constructed a
// sink any more, and the whole layer had zero callers: 193 lines of source and
// 53 of header describing a schedule this tree no longer has. Its one helper
// with a real caller, rescale_boxes_pt_to_px, now sits in pdf_job.cpp beside it.
//
// The sink is also where the /ocr/stream hooks used to fire, via an
// emit_page_ready() nothing called — which is why the streaming routes answered
// with a meta event, an end event, and no pages in between for as long as they
// existed on this path. They now fire from PdfStreamRenderState (below), which
// is the state the sequential path actually uses.

// The mutable page state a streamed-render driver reads and accumulates into,
// as ONE bundle of references to run_pdf_job's locals.
//
// WHY a struct of references rather than more parameters: the CPU driver used
// to take these four alongside thirteen fields that are LITERALLY PdfJobOptions
// members, as 22 positional arguments — seven of them bare bools in a row, where
// transposing two (want_tables/want_formulas, want_line_styles/want_movable_
// regions) compiles clean and silently changes what the response contains. The
// options now travel as the PdfJobOptions the caller already built, and what is
// left is exactly this: the accumulator. References (not values) because the
// driver must write through to the caller's locals, which is what the old
// `std::vector<PdfPageResult> &` / `int &` parameters did — same aliasing, same
// lifetimes, just named.
struct PdfStreamRenderState {
  // Per-page output slots, grown by the driver as pages arrive.
  std::vector<PdfPageResult> &page_results;
  // Which pages still need rasterising (empty for mode==Ocr, where every page
  // is rendered). Read-only input, bundled here because it is indexed in the
  // same per-page step as the slots above.
  const std::vector<uint8_t> &need_render;
  // Rendered pages whose PPM could not be read back (a server-side fault).
  int &decode_failures;
  // Pages whose OCR/inference threw; any > 0 fails the job rather than
  // returning a silently-empty page inside a 200.
  int &page_failures;
  // Which pages have already been handed to PdfJobOptions::on_page_ready.
  //
  // The hook's contract is EXACTLY ONCE PER PAGE, and two different code paths
  // finish a page: this render callback, and run_pdf_job for the pages that are
  // final without ever being rasterised (a text-layer page in mode=auto) or that
  // only appear when the slot vector is grown to the true page count at the end.
  // A shared ledger is what lets both emit without either double-sending or
  // dropping a page — and a dropped page here is invisible in the response,
  // because the non-streaming routes read `page_results` directly and never
  // notice the hook at all. Which is how `/ocr/stream` came to answer with a
  // `meta` and an `end` and nothing in between.
  //
  // Indexed like page_results and grown with it. Not atomic, but no longer
  // single-threaded either: the page workers in run_streamed_render_cpu touch
  // it only under that function's state mutex, and run_pdf_job only after the
  // workers have joined.
  std::vector<uint8_t> &emitted;
};

// Hand ONE finished page to PdfJobOptions::on_page_ready, at most once — see
// PdfStreamRenderState::emitted for why the ledger exists.
void emit_page(PdfStreamRenderState &state, const PdfJobOptions &opts,
               int page_idx);


} // namespace detail

} // namespace turbo_ocr::pipeline
