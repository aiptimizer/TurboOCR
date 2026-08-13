#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"

#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

#include "turbo_ocr/analysis/table/cell_matcher.h"     // OcrLine, match_cells_to_ocr
#include "turbo_ocr/analysis/table/html_reconstruct.h" // reconstruct_html
#include "turbo_ocr/analysis/table/table_cells.h"      // build_table_cells

namespace turbo_ocr::table {

namespace {

// Compute (h_scale, w_scale) used to project loc_preds back to the original
// image. SLANeXt uses the non-SLANet branch.
std::pair<float, float> bbox_scales(int padded_w, int padded_h,
                                    int ori_w, int ori_h) {
    const float pw = static_cast<float>(padded_w);
    const float ph = static_cast<float>(padded_h);
    const float rw = pw / static_cast<float>(ori_w);
    const float rh = ph / static_cast<float>(ori_h);
    const float r = std::min(rw, rh);
    return {ph / r, pw / r};
}

} // namespace

StructureResult decode_structure(
    const float* structure_probs,
    const float* loc_preds,
    std::size_t t,
    std::size_t v,
    const CharDict& dict,
    int padded_w,
    int padded_h,
    int ori_w,
    int ori_h) {
    const std::size_t sos = dict.sos_idx();
    const std::size_t eos = dict.eos_idx();
    const auto [h_scale, w_scale] = bbox_scales(padded_w, padded_h, ori_w, ori_h);
    // PaddleX: `scales[0::2] = [h_scale]*4; scales[1::2] = [w_scale]*4`.
    // h_scale lands on x-coordinate indices — preserved literally for
    // byte-equal output, even though the naming looks swapped.
    const std::array<float, 8> scales = {
        h_scale, w_scale, h_scale, w_scale, h_scale, w_scale, h_scale, w_scale,
    };

    // Emit straight into the final wrapped sequence: the intermediate token
    // vector and its move-loop are pure overhead. `t` bounds the kept tokens,
    // so one reservation covers the wrapper tags plus every decoded token with
    // no reallocation.
    std::vector<std::string> html;
    html.reserve(t + 6);
    html.emplace_back("<html>");
    html.emplace_back("<body>");
    html.emplace_back("<table>");

    std::vector<StructureCell> cells;
    float score_sum = 0.0f;
    std::size_t score_n = 0;

    for (std::size_t step = 0; step < t; ++step) {
        const float* row = structure_probs + step * v;
        std::size_t best_i = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < v; ++i) {
            if (row[i] > best_v) {
                best_v = row[i];
                best_i = i;
            }
        }
        if (step > 0 && best_i == eos) break;
        if (best_i == sos || best_i == eos) continue;

        if (dict.is_td_token(best_i)) {
            const float* lrow = loc_preds + step * 8;
            std::array<int, 8> bbox{};
            for (std::size_t k = 0; k < 8; ++k) {
                bbox[k] = static_cast<int>(lrow[k] * scales[k]);
            }
            // A blank (all-zero) quad means "no detected cell box". Push it as a
            // POSITIONAL PLACEHOLDER anyway so `cells` stays index-aligned with the
            // <td> tokens: dropping the slot shifts every later cell's text by one
            // (silent content corruption on tables with empty cells). A zero-area
            // quad matches no OCR line and renders <td></td>.
            cells.push_back(StructureCell{bbox});
        }
        html.emplace_back(dict.token(best_i));
        score_sum += best_v;
        ++score_n;
    }

    html.emplace_back("</table>");
    html.emplace_back("</body>");
    html.emplace_back("</html>");

    const float structure_score =
        score_n > 0 ? score_sum / static_cast<float>(score_n) : 0.0f;
    return StructureResult{std::move(html), std::move(cells), structure_score};
}


router::TableResult slanext_postprocess_region(
    const StructureResult &sr,
    const std::vector<OCRResultItem> &page_ocr,
    const Box &region,
    const SlanextCellRecFn &cell_rec) {
  // Crop origin the region-local cell quads are shifted back by; must equal
  // the (page top-left-clamped) origin the encoders crop at.
  int rbx1 = INT_MAX, rby1 = INT_MAX, rax2 = INT_MIN, ray2 = INT_MIN;
  for (const auto &p : region.pts) {
    rbx1 = std::min(rbx1, p[0]); rby1 = std::min(rby1, p[1]);
    rax2 = std::max(rax2, p[0]); ray2 = std::max(ray2, p[1]);
  }
  const int rx = std::max(rbx1, 0), ry = std::max(rby1, 0);

  // Cells from the page text-OCR: geometry-match each structure-order cell
  // quad to OCR lines inside the region, then reconstruct_html substitutes.
  // The matcher reads only OcrLine.bbox, so each in-region string is copied
  // once into region_texts (its index pool); region_ocr carries geometry only.
  std::vector<OcrLine> region_ocr;
  std::vector<std::string> region_texts;
  region_ocr.reserve(page_ocr.size());
  region_texts.reserve(page_ocr.size());
  for (const auto &r : page_ocr) {
    int bx1 = INT_MAX, by1 = INT_MAX, bx2 = INT_MIN, by2 = INT_MIN, cx = 0, cy = 0;
    for (const auto &p : r.box.pts) {
      cx += p[0]; cy += p[1];
      bx1 = std::min(bx1, p[0]); by1 = std::min(by1, p[1]);
      bx2 = std::max(bx2, p[0]); by2 = std::max(by2, p[1]);
    }
    cx /= 4; cy /= 4;
    if (cx < rx || cx > rax2 || cy < ry || cy > ray2) continue;
    region_ocr.push_back(OcrLine{{static_cast<float>(bx1), static_cast<float>(by1),
                                  static_cast<float>(bx2), static_cast<float>(by2)},
                                 {}});
    region_texts.push_back(r.text);
  }

  // SLANeXt per-token cell quads are region-local -> shift to page coords.
  std::vector<std::array<int, 8>> quads;
  quads.reserve(sr.cells.size());
  for (const auto &c : sr.cells) {
    std::array<int, 8> q = c.bbox;
    for (int k = 0; k < 4; ++k) { q[2 * k] += rx; q[2 * k + 1] += ry; }
    quads.push_back(q);
  }
  auto matched = match_cells_to_ocr(quads, region_ocr);

  // Per-cell crop OCR: SLANeXt grid cells with no page-OCR line get recognized
  // directly from their quad crop. Recovers dense-grid cells the page text
  // detector under-segmented (the dominant local-table content loss). One
  // batched rec call per table over all empty cells; page-coordinate quads.
  if (cell_rec && !matched.empty()) {
    std::vector<Box> empty_boxes;
    std::vector<std::size_t> empty_ci;
    for (std::size_t ci = 0; ci < matched.size() && ci < quads.size(); ++ci) {
      if (!matched[ci].ocr_indices.empty()) continue;
      const auto &q = quads[ci];
      const int w = std::abs(q[2] - q[0]);
      const int h = std::abs(q[5] - q[1]);
      if (w < 4 || h < 4) continue;  // skip degenerate / spacer cells
      empty_boxes.push_back(
          Box{{{{q[0], q[1]}, {q[2], q[3]}, {q[4], q[5]}, {q[6], q[7]}}}});
      empty_ci.push_back(ci);
    }
    if (!empty_boxes.empty()) {
      auto rec = cell_rec(empty_boxes);
      for (std::size_t k = 0; k < rec.size() && k < empty_ci.size(); ++k) {
        const std::string &txt = rec[k].first;
        if (txt.empty() || rec[k].second < 0.5f) continue;  // drop empty / low-conf noise
        region_texts.push_back(txt);
        matched[empty_ci[k]].ocr_indices.push_back(region_texts.size() - 1);
      }
    }
  }

  router::TableResult tr;
  tr.layout_id = -1;  // stamped by the caller (input order)
  tr.html      = reconstruct_html(sr.structure, matched, region_texts);
  // Built from the same quads/matched/text pool the HTML is built from, and
  // AFTER the crop-OCR backfill, so a cell recovered by crop OCR carries its
  // text here too.
  tr.cells     = build_table_cells(sr.structure, quads, matched, region_texts);
  tr.score     = sr.structure_score;
  tr.box       = region;
  return tr;
}

} // namespace turbo_ocr::table
