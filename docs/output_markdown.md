# Faithful Markdown export

Native, in-process Markdown serializer for a parsed page — the faithful
counterpart of PP-StructureV3 `save_to_markdown` / MinerU / Marker / docling.
Markdown is the primary export format (the HTML page-view already exists
separately).

Module (self-contained, **already added**):

- `include/turbo_ocr/output/markdown_export.h`
- `src/output/markdown_export.cpp`

It consumes the CUDA-free result type `turbo_ocr::pipeline::OcrPipelineResult`
(text `results[]`, `layout[]`, `reading_order[]`, `tables[]`, `formulas[]`) and
emits one Markdown string. It is the same data the JSON route already serializes
via `emit_pipeline_result_json`.

## Conventions (matches PP-StructureV3 / MinerU / Marker / docling)

| Element                 | Markdown emitted                                              |
|-------------------------|--------------------------------------------------------------|
| `doc_title`             | `# …`                                                         |
| `paragraph_title`       | `## …`                                                        |
| `figure_title`          | `### …`                                                       |
| `text` / body classes   | paragraph (lines joined, single-spaced)                      |
| `abstract`              | `**Abstract** …`                                             |
| `algorithm`             | fenced ```` ``` ```` code block                              |
| `reference[_content]`   | `### References` once, then `- …`                            |
| `display_formula`       | `$$\n<latex>\n$$` ( + `\tag{N}` from a folded `formula_number`) |
| `inline_formula`        | `$<latex>$`                                                  |
| `table`                 | the **bare `<table>…</table>` HTML** embedded verbatim       |
| `image` / `chart`       | `![caption](assets/blockN.png)` (+ a crop to write/embed)   |
| `header` / `footer` / `number` / `seal` | **dropped by default** (still in JSON)       |

Faithfulness fixes baked in (these were the failures of the old
`tools/omnidoc_to_md.py`):

1. **Table wrapper stripped** — recognizers emit
   `<html><body><table>…</table></body></html>`; the document wrapper is
   invalid inside Markdown and makes most viewers show raw text or drop the
   table. `strip_table_wrapper()` keeps only `<table>…</table>` (colspan /
   rowspan preserved).
2. **LaTeX render-safety** — `latex_is_render_safe()` rejects the structural
   classes that throw in KaTeX/MathJax (unbalanced `{}` / `\begin\end` /
   `\left\right`, a `\left`/`\big`-family token not followed by a delimiter —
   the real `\left\mathrm…` breaker — a dangling backslash, an undefined
   `\<punct>` escape like `\?`, an accent escape with no argument, and an
   empty / doubled sub-superscript). A flagged formula falls back to inline
   code / a fenced block instead of a `$…$` / `$$…$$` that breaks the renderer.
3. **Orphan filter** — text regions with fewer than `min_text_codepoints`
   (default 2) Unicode codepoints are dropped (kills lone-glyph OCR noise such
   as a stray `制`). Titles / formulas / tables / images are never gated.
4. **Reading order** — blocks are emitted in reading order. `reading_order[]`
   ranks *text* results; structure-only regions (tables, images, standalone
   formulas with no OCR line) are threaded **inline** at the rank of the
   nearest text result, not dumped at the end. Header/body/footer class buckets
   are honored.

## API

```cpp
#include "turbo_ocr/output/markdown_export.h"
using namespace turbo_ocr::output;

// Pure string render (no OpenCV). Image blocks emit ![](assets/blockN.png);
// each image region is appended to `assets` for the caller to materialize.
MarkdownOptions opts;                       // see fields below
std::vector<MarkdownAsset> assets;
std::string md = render_markdown(res, opts, &assets);

// One-shot convenience that also handles the crops:
//   embed_images=false -> writes PNG crops under base_dir, links stay relative
//   embed_images=true  -> inlines crops as base64 data: URIs, nothing on disk
std::string md = render_markdown_with_assets(res, page /*cv::Mat*/, base_dir,
                                             /*embed_images=*/true, opts);
```

`MarkdownOptions`: `ignore_labels` (default `{header, header_image, footer,
footer_image, number, seal}`), `assets_dir` (link prefix, default `"assets"`),
`fold_formula_numbers` (default true), `min_text_codepoints` (default 2),
`safe_formula_fallback` (default true).

`render_markdown` should be called after `turbo_ocr::assign_layout_ids(
res.results, res.layout)` (the JSON routes already do this); it degrades
gracefully via centroid containment when ids are unset.

---

## How it is wired

This module is **already wired and shipped** as the GPU route `POST /ocr/markdown`
(`src/routes/image_routes.cpp`, `register_ocr_markdown_route_gpu`). The CPU server
does not expose it yet — `/ocr/markdown` only appears in `/capabilities` on the
GPU build. The sections below document the wiring as built.

### 1. CMake

`markdown_export.cpp` is CUDA-free and needs only OpenCV (already linked
`PUBLIC` by `turbo_ocr_common`). It is compiled into the `turbo_ocr_common
STATIC` source list in `CMakeLists.txt`:

```cmake
    src/output/markdown_export.cpp
```

That makes it available to both servers (`turboocr-server` and
`turboocr-cpu-server` link `turbo_ocr_common` transitively) and to the test exe.

### 2. Route — `POST /ocr/markdown` (GPU build)

A dedicated route is used rather than `?format=markdown` on `/ocr/raw` because
the Markdown handler must (a) force layout + reading order on, (b) always decode
to a `cv::Mat` (the nvJPEG GPU-direct fast path keeps no host bitmap, but the
page pixels are needed to crop figures). The handler lives in
`src/routes/image_routes.cpp` next to `register_ocr_raw_route_gpu`:

```cpp
// --- /ocr/markdown: faithful Markdown export ---
void register_ocr_markdown_route_gpu(server::WorkPool &pool,
                                     pipeline::PipelineDispatcher &dispatcher,
                                     const server::ImageDecoder &decode,
                                     bool layout_available) {
  drogon::app().registerHandler(
      "/ocr/markdown",
      [&pool, &dispatcher, &decode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "EMPTY_BODY", "Empty body"));
          return;
        }
        if (!layout_available) {
          callback(server::error_response(drogon::k400BadRequest,
              "LAYOUT_DISABLED",
              "/ocr/markdown requires the layout model (do not start with "
              "DISABLE_LAYOUT=1)"));
          return;
        }
        // HTTP is always self-contained base64 data: URIs. The file-ref mode
        // (embed_images=false) writes asset PNGs to the SERVER's filesystem —
        // unreachable for an HTTP client — so ?embed=0 is rejected with a loud
        // 400 INVALID_PARAMETER instead of being silently overridden. File-ref
        // markdown remains available to library/CLI consumers of
        // render_markdown_with_assets.
        if (auto p = req->getParameter("embed"); p == "0" || p == "false") {
          callback(server::error_response(
              drogon::k400BadRequest, "INVALID_PARAMETER",
              "embed=0 (file-ref markdown) is not supported over HTTP; "
              "assets are always embedded as data: URIs"));
          return;
        }
        const bool embed = true;

        server::submit_work(pool, std::move(callback),
            [req, &dispatcher, &decode, embed](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/markdown", [&] {
            const auto *data =
                reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();
            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest,
                  "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            const int kMaxImageDim = decode::max_image_dim();
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
              cb(server::error_response(drogon::k400BadRequest,
                  "DIMENSIONS_TOO_LARGE", "Image dimensions exceed maximum"));
              return;
            }

            pipeline::OcrPipelineResult out;
            try {
              out = dispatcher.submit_for_default([img](auto &e) {
                return e.pipeline->run_with_layout(
                    img, e.stream, /*want_layout=*/true,
                    /*want_reading_order=*/true, /*routing=*/{},
                    /*defer_external=*/true);
              });
            } catch (const turbo_ocr::TimeoutError &) {
              cb(timeout_response());
              return;
            }
            pipeline::finalize_deferred(out);

            turbo_ocr::assign_layout_ids(out.results, out.layout);
            std::string md = turbo_ocr::output::render_markdown_with_assets(
                out, img, /*base_dir=*/".", /*embed_images=*/embed);

            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(std::move(md));
            resp->setContentTypeString("text/markdown; charset=utf-8");
            cb(resp);
          });
        });
      }, {drogon::Post});
}
```

It is registered in `register_image_routes(...)` (same file) alongside the other
`register_ocr_*_route_gpu(...)` calls:

```cpp
  register_ocr_markdown_route_gpu(pool, dispatcher, decode, layout_available);
```

The CPU server does **not** yet expose `/ocr/markdown`. To add it, mirror the
same handler in `common_routes.cpp` against the `InferFunc`/`InferResult` path —
wrap the `InferResult` fields into an `OcrPipelineResult` exactly like
`server::emit_infer_result_json` does, then call `render_markdown_with_assets`.

### Usage

```bash
# self-contained markdown (images inline as data URIs)
curl --data-binary @page.png http://localhost:PORT/ocr/markdown > page.md

# file-ref image links (write the crops yourself from the JSON bboxes)
curl --data-binary @page.png 'http://localhost:PORT/ocr/markdown?embed=0'
```

For the offline `result/omnibench_layout_table/<doc>/page.md` exporter, call
`render_markdown_with_assets(out, page, "<doc-dir>", /*embed=*/false)` so each
figure crop lands in `<doc-dir>/assets/blockN.png`.

---

## Verification (rendered, not asserted)

Generated Markdown was rendered with `markdown-it` + KaTeX (`throwOnError`) over
the 125-doc OmniDocBench subset:

- **0** raw `<html>/<body>` wrapper leaks (all tables embed as bare `<table>`).
- **0** structural LaTeX breakers (unbalanced groups / `\left\right` /
  delimiters / dangling backslash / double scripts) — all fall back to code.
- KaTeX hard errors went **71 → 34** with **zero** valid formulas demoted
  (1524 valid spans unchanged). The residual 34 are upstream OCR
  hallucinations (undefined commands such as `\cccledast`, missing macro args);
  under the universal non-throwing renderer config they show as isolated red
  spans — the page still renders fully.
- Orphan lone-glyph fragments (e.g. `制`) dropped.

Rendered samples (open in a browser) for docs with both a table and formulas:
`result/omnibench_layout_table/book_zh_HGT51022016_extracted_page_12/page.new.rendered.html`
(+ `book_zh_GB115041989_extracted_page_9`, the Number-Theory doc).
