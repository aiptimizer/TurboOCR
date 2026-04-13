#include "turbo_ocr/routes/pdf_routes.h"

#include <format>
#include <future>

#include "turbo_ocr/common/logger.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif
#include <mutex>

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/server/server_types.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::results_to_json;

static int max_pdf_pages() {
  static int val = [] {
    if (auto *env = std::getenv("MAX_PDF_PAGES"))
      return std::max(1, std::atoi(env));
    return 2000;
  }();
  return val;
}

namespace turbo_ocr::routes {

/// Helper: extract PDF bytes from a Drogon request (raw, base64 JSON, multipart).
/// Returns true on success, fills pdf_ptr/pdf_len and may fill decoded_buf.
/// On failure, calls cb with 400 and returns false.
static bool extract_pdf_bytes(const drogon::HttpRequestPtr &req,
                              std::string &decoded_buf,
                              const char *&pdf_ptr, size_t &pdf_len,
                              const std::function<void(const drogon::HttpResponsePtr &)> &cb) {
  auto ct = req->getHeader("Content-Type");
  if (ct.find("multipart/form-data") != std::string::npos) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
      cb(server::error_response(drogon::k400BadRequest, "INVALID_MULTIPART", "Failed to parse multipart body"));
      return false;
    }
    for (auto &file : parser.getFiles()) {
      auto name = file.getItemName();
      if (name == "file" || name == "pdf") {
        decoded_buf.assign(file.fileData(), file.fileLength());
        break;
      }
    }
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_FILE",
          "Multipart request must contain a 'file' or 'pdf' form field"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else if (ct.find("application/json") != std::string::npos) {
    auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf")) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_PDF",
          R"(JSON body must contain {"pdf": "<base64>"})"));
      return false;
    }
    auto b64 = (*json)["pdf"].asString();
    decoded_buf = turbo_ocr::base64_decode(b64);
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64 PDF"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else {
    if (req->body().empty()) {
      cb(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return false;
    }
    pdf_ptr = req->body().data();
    pdf_len = req->body().size();
  }
  return true;
}

#ifndef USE_CPU_ONLY
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available) {

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &dispatcher, &pdf_renderer, default_pdf_mode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    // Extract PDF bytes (lightweight, on event loop)
    auto pdf_buf = std::make_shared<std::string>();
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, *pdf_buf, pdf_ptr, pdf_len, callback))
      return;

    bool layout_enabled = false;
    if (auto err = server::parse_layout_query(req, layout_available, &layout_enabled); !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
      return;
    }

    int dpi = 100;
    auto dpi_str = req->getParameter("dpi");
    if (!dpi_str.empty()) dpi = std::atoi(dpi_str.c_str());
    if (dpi < 50 || dpi > 600) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI", "DPI must be between 50 and 600"));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty())
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);

    // For raw body case, pdf_ptr points into req->body() — copy into pdf_buf
    if (pdf_buf->empty())
      pdf_buf->assign(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, req, &dispatcher, &pdf_renderer,
         layout_enabled, dpi, req_mode](server::DrogonCallback &cb) {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      // Check page count limit
      {
        pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
        if (check_doc.ok()) {
          int np = check_doc.page_count();
          int limit = max_pdf_pages();
          if (np > limit) {
            cb(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
                std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                            np, limit)));
            return;
          }
        }
      }

      // Open PDF for text-layer modes
      std::unique_ptr<pdf::PdfDocument> pdf_doc;
      std::vector<pdf::PdfPageText> page_text_cache;
      pdf::PdfMode mode = req_mode;
      if (mode != pdf::PdfMode::Ocr) {
        pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len_local);
        if (!pdf_doc->ok()) {
          LOG_WARN("Failed to open PDF for text-layer lookup; falling back to mode=ocr", "route", "/ocr/pdf");
          mode = pdf::PdfMode::Ocr;
          pdf_doc.reset();
        } else {
          int np = pdf_doc->page_count();
          page_text_cache.reserve(static_cast<size_t>(std::max(0, np)));
          for (int p = 0; p < np; ++p)
            page_text_cache.push_back(pdf_doc->extract_page(p));
        }
      }

      // Shared state
      std::mutex results_mutex;
      struct PdfPageResult {
        std::vector<OCRResultItem> results;
        std::vector<layout::LayoutBox> layout;
        int width = 0, height = 0, effective_dpi = 0;
        pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
        std::string_view text_layer_quality = "absent";
      };
      std::vector<PdfPageResult> page_results;

      auto fill_from_text_layer_pt =
          [](PdfPageResult &pg, const pdf::PdfPageText &text) {
        pg.width  = static_cast<int>(std::round(text.page_width_pt));
        pg.height = static_cast<int>(std::round(text.page_height_pt));
        pg.effective_dpi = 72;
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
      };

      auto text_layer_quality_for =
          [](const pdf::PdfPageText &text) -> std::string_view {
        if (text.char_count == 0)         return "absent";
        if (text.rotation_deg != 0)       return "rejected";
        if (text.char_count < 10)         return "absent";
        if (text.fffd_count * 20 > text.char_count)     return "rejected";
        if (text.nonprint_count * 10 > text.char_count) return "rejected";
        if (text.lines.empty())           return "absent";
        return "trusted";
      };

      // Pre-populate pages that don't need rendering
      std::vector<uint8_t> need_render;
      bool any_need_render = (mode == pdf::PdfMode::Ocr);

      if (mode != pdf::PdfMode::Ocr) {
        int np = pdf_doc ? pdf_doc->page_count() : 0;
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
              if (has_good_layer) fill_from_text_layer_pt(pg, text);
              else {
                pg.width = static_cast<int>(std::round(text.page_width_pt));
                pg.height = static_cast<int>(std::round(text.page_height_pt));
                pg.effective_dpi = 72;
              }
              if (layout_enabled) {
                need_render[static_cast<size_t>(p)] = 1;
                any_need_render = true;
              }
              break;
            case pdf::PdfMode::Auto:
              if (has_good_layer) {
                pg.resolved_mode = pdf::PdfMode::Geometric;
                fill_from_text_layer_pt(pg, text);
                if (layout_enabled) {
                  need_render[static_cast<size_t>(p)] = 1;
                  any_need_render = true;
                }
              } else {
                pg.resolved_mode = pdf::PdfMode::Ocr;
                need_render[static_cast<size_t>(p)] = 1;
                any_need_render = true;
              }
              break;
            case pdf::PdfMode::AutoVerified:
              pg.resolved_mode = pdf::PdfMode::AutoVerified;
              need_render[static_cast<size_t>(p)] = 1;
              any_need_render = true;
              break;
            default: break;
          }
        }
      }

      // Streamed render + OCR
      std::mutex futures_mutex;
      std::vector<std::future<void>> page_futures;
      render::PdfRenderer::StreamHandle stream_handle;
      int num_pages = 0;

      if (any_need_render) {
        try {
          stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, dpi,
              [&](int page_idx, std::string ppm_path) {
                {
                  std::lock_guard<std::mutex> rlock(results_mutex);
                  if (page_idx >= static_cast<int>(page_results.size())) {
                    page_results.resize(page_idx + 1);
                    if (mode != pdf::PdfMode::Ocr &&
                        page_idx >= static_cast<int>(need_render.size()))
                      need_render.resize(page_idx + 1, 1);
                  }
                  if (mode != pdf::PdfMode::Ocr &&
                      page_idx < static_cast<int>(need_render.size()) &&
                      !need_render[page_idx])
                    return;
                }

                std::future<void> fut;
                try {
                fut = dispatcher.submit(
                    [&, page_idx, path = std::move(ppm_path)](auto &e) {
                  cv::Mat img = render::PdfRenderer::decode_ppm(path);
                  if (img.empty()) {
                    LOG_ERROR("Failed to decode PPM for page", "route", "/ocr/pdf", "page", page_idx);
                    return;
                  }
                  int pw = img.cols, ph = img.rows;

                  pdf::PdfMode page_mode;
                  {
                    std::lock_guard<std::mutex> rlock(results_mutex);
                    page_mode = (page_idx < static_cast<int>(page_results.size()))
                        ? page_results[page_idx].resolved_mode
                        : pdf::PdfMode::Ocr;
                  }

                  std::vector<OCRResultItem> rec_results;
                  std::vector<layout::LayoutBox> layout_snapshot;

                  if (page_mode == pdf::PdfMode::Geometric) {
                    auto lo = e.pipeline->run_layout_only(img, e.stream);
                    layout_snapshot = std::move(lo.layout);
                  } else {
                    auto pipeline_out = e.pipeline->run_with_layout(
                        img, e.stream, layout_enabled);
                    rec_results = std::move(pipeline_out.results);
                    layout_snapshot = std::move(pipeline_out.layout);
                    for (auto &it : rec_results) it.source = "ocr";
                  }

                  if (page_mode == pdf::PdfMode::AutoVerified &&
                      page_idx < static_cast<int>(page_text_cache.size()) && pdf_doc) {
                    for (auto &item : rec_results) {
                      const float px_to_pt = 72.0f / static_cast<float>(dpi);
                      auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
                      float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
                      float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
                      std::string native =
                          pdf_doc->text_in_rect_pt(page_idx, x0, y0, x1, y1);
                      auto verdict = pdf::passes_sanity_check(
                          native, x1 - x0, y1 - y0);
                      if (verdict.accept) {
                        item.text = std::move(native);
                        item.source = "pdf";
                        item.confidence = 1.0f;
                      }
                    }
                  }

                  std::lock_guard<std::mutex> rlock(results_mutex);
                  auto &slot = page_results[page_idx];
                  if (page_mode == pdf::PdfMode::Geometric) {
                    const float pt_to_px = static_cast<float>(dpi) / 72.0f;
                    for (auto &item : slot.results) {
                      for (int k = 0; k < 4; ++k) {
                        item.box[k][0] = static_cast<int>(
                            std::round(item.box[k][0] * pt_to_px));
                        item.box[k][1] = static_cast<int>(
                            std::round(item.box[k][1] * pt_to_px));
                      }
                    }
                  } else {
                    slot.results = std::move(rec_results);
                  }
                  slot.layout        = std::move(layout_snapshot);
                  slot.width         = pw;
                  slot.height        = ph;
                  slot.effective_dpi = dpi;
                  if (page_mode == pdf::PdfMode::Ocr)
                    slot.resolved_mode = pdf::PdfMode::Ocr;
                });

                } catch (const turbo_ocr::PoolExhaustedError &) {
                  LOG_WARN("GPU queue full, skipping page", "route", "/ocr/pdf", "page", page_idx);
                  return;
                }
                std::lock_guard lock(futures_mutex);
                page_futures.push_back(std::move(fut));
              });
          num_pages = stream_handle.num_pages;
        } catch (const std::exception &e) {
          for (auto &f : page_futures) { try { f.get(); } catch (...) {} }
          LOG_ERROR("PDF render failed", "route", "/ocr/pdf", "error", std::string_view(e.what()));
          cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED", "PDF render failed"));
          return;
        }
      } else {
        num_pages = pdf_doc ? pdf_doc->page_count() : 0;
      }

      {
        std::lock_guard<std::mutex> rlock(results_mutex);
        if (static_cast<int>(page_results.size()) < num_pages)
          page_results.resize(num_pages);
      }

      for (auto &f : page_futures) {
        try { f.get(); } catch (const std::exception &e) {
          LOG_ERROR("PDF page error", "route", "/ocr/pdf", "error", std::string_view(e.what()));
        }
      }

      if (num_pages == 0) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF", "PDF contains no pages"));
        return;
      }

      // Build JSON response
      size_t n_pages = static_cast<size_t>(num_pages);
      std::string json_str;
      json_str.reserve(n_pages * 1024);
      json_str += "{\"pages\":[";
      for (size_t i = 0; i < n_pages; ++i) {
        if (i > 0) json_str += ',';
        auto &pg = page_results[i];
        int page_dpi = pg.effective_dpi > 0 ? pg.effective_dpi : dpi;
        json_str += "{\"page\":";
        json_str += std::to_string(i + 1);
        json_str += ",\"page_index\":";
        json_str += std::to_string(i);
        json_str += ",\"dpi\":";
        json_str += std::to_string(page_dpi);
        json_str += ",\"width\":";
        json_str += std::to_string(pg.width);
        json_str += ",\"height\":";
        json_str += std::to_string(pg.height);
        json_str += ',';
        auto page_json = results_to_json(pg.results, pg.layout);
        json_str.append(page_json.data() + 1, page_json.size() - 2);
        json_str += ",\"mode\":\"";
        json_str += pdf::mode_name(pg.resolved_mode);
        json_str += "\",\"text_layer_quality\":\"";
        json_str += pg.text_layer_quality;
        json_str += "\"}";
      }
      json_str += "]}";

      cb(server::json_response(std::move(json_str)));
    });
  }, {drogon::Post});
}
#endif // !USE_CPU_ONLY

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available) {

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
      return;
    }

    int dpi = 100;
    auto dpi_str = req->getParameter("dpi");
    if (!dpi_str.empty()) dpi = std::atoi(dpi_str.c_str());
    if (dpi < 50 || dpi > 600) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI", "DPI must be between 50 and 600"));
      return;
    }

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout, dpi](server::DrogonCallback &cb) {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      // Check page count limit
      {
        pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
        if (check_doc.ok()) {
          int np = check_doc.page_count();
          int limit = max_pdf_pages();
          if (np > limit) {
            cb(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
                std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                            np, limit)));
            return;
          }
        }
      }

      struct PageResult {
        std::vector<OCRResultItem> results;
        std::vector<layout::LayoutBox> layout;
        int width = 0, height = 0;
      };
      std::vector<PageResult> page_results;

      try {
        auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, dpi,
            [&](int page_idx, std::string ppm_path) {
              cv::Mat img = render::PdfRenderer::decode_ppm(ppm_path);
              if (img.empty()) return;

              auto inf = infer(img, want_layout);

              if (page_idx >= static_cast<int>(page_results.size()))
                page_results.resize(page_idx + 1);
              auto &pg = page_results[page_idx];
              pg.results = std::move(inf.results);
              pg.layout = std::move(inf.layout);
              pg.width = img.cols;
              pg.height = img.rows;
              for (auto &item : pg.results) item.source = "ocr";
            });

        if (static_cast<int>(page_results.size()) < stream_handle.num_pages)
          page_results.resize(stream_handle.num_pages);
      } catch (const std::exception &e) {
        cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED",
            std::format("PDF render failed: {}", e.what())));
        return;
      }

      if (page_results.empty()) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF", "PDF contains no pages"));
        return;
      }

      size_t n_pages = page_results.size();
      std::string json_str;
      json_str.reserve(n_pages * 1024);
      json_str += "{\"pages\":[";
      for (size_t i = 0; i < n_pages; ++i) {
        if (i > 0) json_str += ',';
        auto &pg = page_results[i];
        json_str += "{\"page\":";
        json_str += std::to_string(i + 1);
        json_str += ",\"page_index\":";
        json_str += std::to_string(i);
        json_str += ",\"dpi\":";
        json_str += std::to_string(dpi);
        json_str += ",\"width\":";
        json_str += std::to_string(pg.width);
        json_str += ",\"height\":";
        json_str += std::to_string(pg.height);
        json_str += ',';
        auto page_json = results_to_json(pg.results, pg.layout);
        json_str.append(page_json.data() + 1, page_json.size() - 2);
        json_str += ",\"mode\":\"ocr\"}";
      }
      json_str += "]}";

      cb(server::json_response(std::move(json_str)));
    });
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
