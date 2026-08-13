// pdf_documents.cpp — the transport-free half of "what to do with a finished
// PDF job". See the header for why it is not in src/service/http/pdf/.

#include "turbo_ocr/pdf/pdf_documents.h"

#include <format>
#include <memory>
#include <mutex>

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/forms/field_detector.h"
#include "turbo_ocr/analysis/forms/field_model.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/serialization/serialization.h" // assign_layout_ids, append_escaped_string

namespace turbo_ocr::pdf::documents {

// /ocr/pdf?markdown=1 response. Default: one text/markdown document, pages
// prefixed with `<!-- page N -->` (invisible when rendered, splittable by
// chunkers). ?as_pages=1: JSON array of per-page markdown for programmatic
// consumers. The markdown body intentionally drops failed/garbage regions, so
// per-stage degradation is surfaced in the X-OCR-Degraded header (with page
// numbers) and per-page flags in the as_pages shape — never silently.
PdfMarkdownPayload
build_pdf_markdown(std::vector<PdfPageResult> &pages, bool as_pages) {
  std::string dt, dtab, df;
  for (size_t i = 0; i < pages.size(); ++i) {
    auto mark = [&](std::string &s) {
      if (!s.empty()) s += ",";
      s += std::to_string(i + 1);
    };
    if (pages[i].text_degraded) mark(dt);
    if (pages[i].table_degraded) mark(dtab);
    if (pages[i].formula_degraded) mark(df);
  }
  std::string degraded;
  auto add = [&](const char *stage, const std::string &plist) {
    if (plist.empty()) return;
    if (!degraded.empty()) degraded += "; ";
    degraded += stage;
    degraded += "(p";
    degraded += plist;
    degraded += ")";
  };
  add("text", dt);
  add("table", dtab);
  add("formula", df);

  if (as_pages) {
    std::string body = "{\"pages\":[";
    for (size_t i = 0; i < pages.size(); ++i) {
      if (i) body += ",";
      body += "{\"page_index\":" + std::to_string(i) + ",\"markdown\":\"";
      turbo_ocr::detail::append_escaped_string(body, pages[i].markdown);
      body += "\"";
      if (pages[i].text_degraded) body += ",\"text_degraded\":true";
      if (pages[i].table_degraded) body += ",\"table_degraded\":true";
      if (pages[i].formula_degraded) body += ",\"formula_degraded\":true";
      body += "}";
    }
    body += "]}";
    return PdfMarkdownPayload{std::move(body), std::move(degraded), true};
  }
  {
    size_t total = 0;
    for (const auto &pg : pages) total += pg.markdown.size() + 24;
    std::string body;
    body.reserve(total);
    for (size_t i = 0; i < pages.size(); ++i) {
      body += "<!-- page ";
      body += std::to_string(i + 1);
      body += " -->\n\n";
      body += pages[i].markdown;
      if (i + 1 < pages.size()) body += "\n\n";
    }
    return PdfMarkdownPayload{std::move(body), std::move(degraded), false};
  }
}

// ?output=pdf — hand back the source document carrying an invisible text layer.
// The page results already know the raster each box was detected in, so the
// writer needs nothing from the request but the original bytes.
SearchablePdfPayload
build_searchable_pdf(std::vector<PdfPageResult> &pages,
                     const uint8_t *pdf_data, size_t pdf_len,
                     const SearchablePdfOptions &sopts) {
  const float min_confidence = sopts.min_confidence;
  const bool editable = sopts.editable;
  const bool movable = sopts.movable;
  const bool mark_regions = sopts.mark_regions;
  std::vector<pdf::SearchablePage> in;
  in.reserve(pages.size());
  for (size_t i = 0; i < pages.size(); ++i) {
    pdf::SearchablePage sp;
    sp.page_index = static_cast<int>(i);
    sp.raster_w = pages[i].width;
    sp.raster_h = pages[i].height;
    sp.orientation_deg = pages[i].orientation_deg;
    sp.results = &pages[i].results;
    sp.layout = &pages[i].layout;
    sp.mark_regions = mark_regions;
    if (editable) {
      sp.styles = &pages[i].line_styles;
      sp.font_match = pages[i].font_match;
    }
    if (movable) {
      sp.regions = &pages[i].region_images;
      sp.rules = &pages[i].rule_shapes;
      sp.blocks = &pages[i].block_shapes;
    }
    in.push_back(sp);
  }

  const auto mode =
      editable ? pdf::TextLayerMode::Visible : pdf::TextLayerMode::Invisible;
  pdf::SearchableStats stats;
  std::string err;
  std::string body = pdf::write_searchable_pdf(
      pdf_data, pdf_len, in, min_confidence, &stats, err, mode, movable);
  if (body.empty()) {
    TOCR_LOG_ERROR_RL("output=pdf write failed", "error", err);
    return SearchablePdfPayload{{}, err, 0, 0};
  }
  if (editable) {
    TOCR_LOG_INFO("editable pdf written", "pages", stats.pages, "pages_failed",
                  stats.pages_failed, "words",
                  stats.words, "visible", stats.visible, "left_as_scan",
                  stats.uncovered, "movable", stats.movable, "rules", stats.rules,
                  "blocks", stats.blocks, "regions",
                  stats.regions, "dropped", stats.dropped, "bytes", body.size());
  } else {
    TOCR_LOG_INFO("searchable pdf written", "pages", stats.pages,
                  "pages_failed", stats.pages_failed, "words",
                  stats.words, "regions", stats.regions, "dropped",
                  stats.dropped, "bytes", body.size());
  }

  return SearchablePdfPayload{std::move(body), {}, stats.pages_failed,
                             stats.dropped};
}

// Per-page Markdown renderer shared by the GPU and CPU /ocr/pdf handlers (set as
// PdfJobOptions::render_page_markdown). Moves the finished page's fields into an
// OcrPipelineResult, renders self-contained Markdown (figures embedded as data:
// URIs), and moves them back so the JSON envelope is unaffected. Runs on the
// pipeline worker while the page bitmap is still alive.
std::function<std::string(PdfPageResult &, const cv::Mat &)>
make_pdf_page_markdown_renderer() {
  return [](PdfPageResult &pg, const cv::Mat &img) -> std::string {
    turbo_ocr::assign_layout_ids(pg.results, pg.layout);
    pipeline::OcrPipelineResult res;
    res.results = std::move(pg.results);
    res.layout = std::move(pg.layout);
    res.reading_order = std::move(pg.reading_order);
    res.tables = std::move(pg.tables);
    res.formulas = std::move(pg.formulas);
    std::string md = markdown::render_markdown_with_assets(
        res, img, /*base_dir=*/".", /*embed_images=*/true);
    pg.results = std::move(res.results);
    pg.layout = std::move(res.layout);
    pg.reading_order = std::move(res.reading_order);
    pg.tables = std::move(res.tables);
    pg.formulas = std::move(res.formulas);
    return md;
  };
}

// ?fields=1 — per-page fillable-field detector, shared by the GPU and CPU
// /ocr/pdf handlers (set as PdfJobOptions::detect_page_fields). Runs on the
// pipeline worker while the page bitmap is still alive, because detectors 1
// and 2 read the RASTER: the printed rules a form draws its blanks with are
// invisible to the recogniser (it returns "Unterschrift:" and nothing for the
// rule beside it), so no amount of OCR geometry substitutes for the page.
//
// `pg.tables` is passed through as-is: a request without ?tables=1 leaves it
// empty and the empty-cell detector contributes nothing, which is why fields=1
// never forces the table stage to run.
//
// FFDetr joins as a fifth detector when its weights are present.
//
// THE MODEL IS A FUNCTION-LOCAL STATIC, and it has to be: this factory runs
// per REQUEST (pdf_route.cpp calls it inside the handler, once per ?fields=1
// submit), so anything owned by the returned closure would mean re-reading a
// 77 MB graph and rebuilding an ORT session on every request. A magic static
// is initialised exactly once, thread-safely, on the first request that asks
// for fields — which also keeps a server that never uses fields=1 from paying
// for the model at boot.
//
// A failed load is cached the same way, deliberately: no file, or a file ORT
// will not open, is a permanent condition, and retrying it per request would
// turn an absent optional model into a recurring cost.
std::function<std::vector<forms::FormField>(PdfPageResult &, const cv::Mat &)>
make_pdf_page_field_detector() {
  // One session, serialised. ORT's Run() is thread-safe but this instance's
  // staging buffers are not, and a pool would be the wrong shape of fix here:
  // fields=1 is a form-PREPARATION request, not the OCR hot path, so its
  // arrival rate is low and a 12 MB CHW buffer per extra worker buys nothing.
  // If that stops being true, give UnifiedPipelinePool a FieldModel per worker
  // the way it already does for the layout stage.
  struct Shared {
    std::unique_ptr<forms::FieldModel> model;
    std::mutex lock; // non-copyable, hence the work happens in the ctor
    Shared() {
      const std::string path = turbo_ocr::env::env_or(
          "FIELD_MODEL_ONNX", "models/forms/ffdetr.onnx");
      if (path != "none") model = forms::FieldModel::load(path);
      if (!model)
        TOCR_LOG_INFO("fields=1: no field model, geometry detectors only",
                      "path", std::string_view(path));
    }
  };
  static Shared shared;

  return [](PdfPageResult &pg, const cv::Mat &img) {
    std::vector<forms::FormField> from_model;
    if (shared.model) {
      const std::lock_guard<std::mutex> guard(shared.lock);
      from_model = shared.model->run(img);
    }
    return forms::detect_form_fields(img, pg.results, pg.tables,
                                     std::move(from_model));
  };
}

} // namespace turbo_ocr::pdf::documents
