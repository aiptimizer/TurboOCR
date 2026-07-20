#include "turbo_ocr/table/slanext/slanext_postprocess.h"

#include <array>
#include <limits>
#include <utility>

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
            cells.push_back(StructureCell{best_i, bbox});
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

} // namespace turbo_ocr::table
