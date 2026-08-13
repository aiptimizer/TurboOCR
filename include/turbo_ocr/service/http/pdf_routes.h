#pragma once

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/service/server/server_types.h"

namespace turbo_ocr::routes {

/// Register /ocr/stream — NDJSON streaming endpoint. One request shape for PDFs
/// AND single images (content-sniffed): the response streams one JSON line per
/// finished page ({"event":"page",...} with page_index, dims, dpi, mode,
/// results, layout, ...) AS EACH PAGE COMPLETES — out of order — bracketed by a
/// meta line and an end line. Built for streaming consumers (e.g. RAG ingest
/// that embeds page k while pages k+1..N still OCR).
///
/// Runs on the device-agnostic InferFunc, so every backend serves it. The
/// previous declaration here was register_ocr_stream_route_gpu(PipelineDispatcher&),
/// which had no definition anywhere in the tree after the CUDA HTTP layer was
/// deleted — the endpoint simply did not exist.
void register_ocr_stream_route(server::WorkPool &pool,
                               const server::InferFunc &infer,
                               render::PdfRenderer &pdf_renderer,
                               const server::ImageDecoder &decode,
                               pdf::PdfMode default_pdf_mode,
                               const capability::CapabilityMask &loaded,
                               int default_dpi = 100,
                               int max_pdf_pages = 2000,
                               server::OrientFunc orient_fn = {});

/// Register /ocr/pdf — sequential page OCR via InferFunc.
/// `orient_fn` (optional) detects a page's clockwise rotation for autorotate;
/// pass an empty function to disable (autorotate requests then 400).
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        const capability::CapabilityMask &loaded,
                        int max_pdf_pages = 2000,
                        server::OrientFunc orient_fn = {},
                        // Render DPI when the request omits ?dpi= (PDF_DEFAULT_DPI).
                        // Was the compile-time constant kCpuDefaultDpi, one of four
                        // independent literal 100s for the same setting.
                        int default_dpi = 100);

} // namespace turbo_ocr::routes
