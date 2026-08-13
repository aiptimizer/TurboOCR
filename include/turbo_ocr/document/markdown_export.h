#pragma once

// Faithful Markdown export for a parsed page (PP-StructureV3 save_to_markdown /
// MinerU / Marker / docling conventions): blocks in reading order, ATX
// headings for titles, paragraphs for text, `$$…$$` / `$…$` for formulas,
// raw `<table>…</table>` HTML embedded verbatim, and `![caption](path)` for
// figures/charts. Self-contained: depends only on the CUDA-free result types
// (OcrPipelineResult + layout/router types). The OpenCV crop helpers are
// declared here but defined in the .cpp, so a consumer that only needs the
// string render does not drag in OpenCV.

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/pipeline/pipeline_result.h"

namespace cv { class Mat; }

namespace turbo_ocr::markdown {

// Layout-class labels dropped from the Markdown body by default (page
// furniture). Mirrors MinerU's `discarded_blocks` / PaddleX `ignore_labels`:
// header / footer / page-number / seal. The blocks remain in the JSON
// envelope — only the Markdown view omits them.
[[nodiscard]] const std::unordered_set<std::string> &default_ignore_labels();

// One image/figure/chart region whose crop the caller must materialize. Filled
// by render_markdown into `assets_out` so the handler can write the PNGs (or
// build base64 data URIs) without the renderer needing the page bitmap.
struct MarkdownAsset {
  int            layout_index = -1; // index into res.layout
  int            block_id     = -1; // stable id used in the asset filename
  std::string    kind;              // "image" | "chart" | "figure" | ...
  turbo_ocr::Box box{};             // crop rect in original-page pixels
  std::string    rel_path;          // link target, e.g. "assets/block12.png"
};

struct MarkdownOptions {
  // Layout-class labels to omit from the body. Defaults to
  // default_ignore_labels(); pass an empty set to keep every region.
  std::unordered_set<std::string> ignore_labels = default_ignore_labels();
  // Relative directory prefix for image links. "" => bare "<id>.png".
  std::string assets_dir = "assets";
  // Fold a `formula_number` region into the display formula it annotates as a
  // trailing \tag{…} instead of emitting it as a stray paragraph.
  bool fold_formula_numbers = true;
  // Drop text-bearing regions whose content has fewer than this many Unicode
  // codepoints — kills lone-character OCR noise (e.g. a stray "制"). Titles,
  // formulas, tables and images are never length-gated.
  int  min_text_codepoints = 2;
  // When a formula's LaTeX is not render-safe (unbalanced braces / begin-end,
  // or a \left|\right not followed by a delimiter) emit it as inline code / a
  // fenced block rather than a $…$ / $$…$$ that throws in KaTeX/MathJax.
  bool safe_formula_fallback = true;
  // Drop runaway / mode-collapsed formula LaTeX from the Markdown body.
  // PP-FormulaNet-S degenerates on Chinese-text-in-formula crops into
  // repetitive 1000+ char garbage (a short unit repeated dozens of times) that
  // is still brace-balanced, so safe_formula_fallback lets it through. This
  // gate is Markdown-VIEW-only: the JSON / scorer path keeps every formula,
  // because dropping collapsed strings pipeline-wide also drops legitimate long
  // matrices (measured CDM loss).
  bool drop_collapsed_formulas = true;
  // Placeholder emitted in place of a dropped collapsed formula. Empty => emit
  // nothing (the region vanishes). Default: a short marker so the omission is
  // visible rather than silent.
  std::string collapsed_formula_note = "*(formula not recognized)*";
  // Re-order the BODY blocks column-major for emission when the page bboxes
  // describe a clear multi-column layout. PaddleX's XY-cut reading_order weaves
  // a two-column page by horizontal band (L,R,L,R…), so a faithful render
  // zig-zags between columns; this pass detects a clean column split (well-
  // separated x-clusters with a real vertical gutter, side-by-side vertical
  // overlap, both columns a fair share of blocks) and emits each column
  // top→bottom, left→right, with full-width blocks (titles / spanning tables /
  // figures) kept as section breaks at their vertical position. Strictly
  // conservative: single-column / ambiguous / complex pages fall back to the
  // reading_order sequence untouched. Markdown-VIEW-only — the JSON envelope's
  // reading_order is never modified.
  bool column_aware_order = true;
};

// Resolve the Markdown `src` for an image asset (a path or a `data:` URI).
// Empty (default) => asset.rel_path. Supply this to inline base64 data URIs.
using ImageSrcResolver = std::function<std::string(const MarkdownAsset &)>;

// Render one parsed page to faithful Markdown. Pure (no OpenCV). Image blocks
// emit `![caption](src)`; when `assets_out != nullptr` each image region is
// appended so the caller can write its crop. `res` should have been through
// turbo_ocr::assign_layout_ids() (the JSON routes already do this); when it
// hasn't, result→region mapping degrades gracefully via centroid containment.
[[nodiscard]] std::string
render_markdown(const pipeline::OcrPipelineResult &res,
                const MarkdownOptions &opts = {},
                std::vector<MarkdownAsset> *assets_out = nullptr,
                const ImageSrcResolver &resolver = {});

// ── OpenCV-backed helpers (defined in the .cpp; link OpenCV) ─────────────

// Write each asset's crop (original-page pixels) to <base_dir>/<rel_path> as
// PNG, creating parent directories. Degenerate boxes are skipped. Returns the
// number of crops written.
int write_asset_crops(const cv::Mat &page,
                      const std::vector<MarkdownAsset> &assets,
                      const std::string &base_dir);

// Encode one box-crop of `page` as a `data:image/png;base64,…` URI — the body
// of an ImageSrcResolver when embedding images inline.
[[nodiscard]] std::string
crop_to_png_data_uri(const cv::Mat &page, const turbo_ocr::Box &box);

// Convenience one-shot: render and either write crops under `base_dir`
// (embed_images == false, links stay relative) or inline them as data URIs
// (embed_images == true, nothing written to disk).
[[nodiscard]] std::string
render_markdown_with_assets(const pipeline::OcrPipelineResult &res,
                            const cv::Mat &page, const std::string &base_dir,
                            bool embed_images, const MarkdownOptions &opts = {});

} // namespace turbo_ocr::markdown
