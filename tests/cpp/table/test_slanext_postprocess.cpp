#include <catch_amalgamated.hpp>

#include <cmath>

#include "turbo_ocr/analysis/table/cell_matcher.h"
#include "turbo_ocr/analysis/table/html_reconstruct.h"
#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"
#include "turbo_ocr/analysis/table/table_types.h"

using turbo_ocr::table::CharDict;
using turbo_ocr::table::decode_structure;
using turbo_ocr::table::default_dict;
using turbo_ocr::table::match_cells_to_ocr;
using turbo_ocr::table::OcrLine;
using turbo_ocr::table::reconstruct_html;
using turbo_ocr::table::SLANEXT_VOCAB;

namespace {

std::vector<float> one_hot(std::size_t t, std::size_t v,
                           const std::vector<std::size_t>& ids) {
    REQUIRE(ids.size() == t);
    std::vector<float> out(t * v, 0.0f);
    for (std::size_t step = 0; step < t; ++step) {
        out[step * v + ids[step]] = 1.0f;
    }
    return out;
}

std::size_t find_token(const CharDict& d, std::string_view t) {
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.token(i) == t) return i;
    }
    return d.len();
}

std::size_t any_td(const CharDict& d) {
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.is_td_token(i)) return i;
    }
    return d.len();
}

} // namespace

TEST_CASE("decode walks until eos", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    REQUIRE(v == SLANEXT_VOCAB);
    const std::size_t header_idx = find_token(dict, "<thead>");
    REQUIRE(header_idx < v);
    std::vector<std::size_t> ids = {dict.sos_idx(), header_idx, dict.eos_idx(),
                                    header_idx};
    auto probs = one_hot(4, v, ids);
    std::vector<float> loc(4 * 8, 0.0f);
    auto r = decode_structure(probs.data(), loc.data(), 4, v, dict, 488, 488,
                              100, 100);
    REQUIRE(r.structure.size() == 7); // 6 wrapper + 1 thead
    REQUIRE(r.structure[0] == "<html>");
    REQUIRE(r.structure[3] == "<thead>");
    REQUIRE(r.cells.empty());
}

TEST_CASE("td token emits scaled bbox", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td_idx = any_td(dict);
    REQUIRE(td_idx < v);
    std::vector<std::size_t> ids = {dict.sos_idx(), td_idx, dict.eos_idx()};
    auto probs = one_hot(3, v, ids);
    std::vector<float> loc(3 * 8, 0.0f);
    const float quad[8] = {0.1f, 0.1f, 0.5f, 0.1f, 0.5f, 0.5f, 0.1f, 0.5f};
    for (int k = 0; k < 8; ++k) loc[8 + k] = quad[k];
    auto r = decode_structure(probs.data(), loc.data(), 3, v, dict, 488, 488,
                              244, 244);
    REQUIRE(r.cells.size() == 1);
    // padded=488, ori=244 → ratio=2.0 → h_scale=w_scale=244.
    const int expected[8] = {
        static_cast<int>(0.1f * 244.0f), static_cast<int>(0.1f * 244.0f),
        static_cast<int>(0.5f * 244.0f), static_cast<int>(0.1f * 244.0f),
        static_cast<int>(0.5f * 244.0f), static_cast<int>(0.5f * 244.0f),
        static_cast<int>(0.1f * 244.0f), static_cast<int>(0.5f * 244.0f),
    };
    for (int k = 0; k < 8; ++k) REQUIRE(r.cells[0].bbox[k] == expected[k]);
}

TEST_CASE("structure score is mean of kept tokens", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td_idx = any_td(dict);
    std::vector<float> probs(4 * v, 0.0f);
    probs[dict.sos_idx()] = 1.0f;
    probs[v + td_idx] = 0.6f;
    probs[2 * v + td_idx] = 0.8f;
    probs[3 * v + dict.eos_idx()] = 1.0f;
    std::vector<float> loc(4 * 8, 0.0f);
    auto r = decode_structure(probs.data(), loc.data(), 4, v, dict, 488, 488,
                              244, 244);
    REQUIRE(std::fabs(r.structure_score - 0.7f) < 1e-5f);
    // Both quads are all-zero (blank). Blank cells are kept as positional
    // placeholders (zero quad) so `cells` stays index-aligned with the <td>
    // tokens in `structure`; the structure tokens still count toward the score.
    REQUIRE(r.cells.size() == 2);
    for (const auto& c : r.cells)
        for (int k = 0; k < 8; ++k) REQUIRE(c.bbox[k] == 0);
}

// M1 regression: a blank (all-zero quad) cell in the MIDDLE of a row must keep
// its slot so the non-blank cells' OCR text lands in the correct <td>. Before
// the fix the blank cell was dropped from `cells`, so every later cell rendered
// the NEXT cell's text (shifted by one) and the last cell came out empty.
TEST_CASE("blank middle cell keeps text aligned (M1)", "[slanext_postprocess]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td_idx = find_token(dict, "<td></td>");
    REQUIRE(td_idx < v);  // merged cell -> 1 td slot each in reconstruct

    // sos, cell0 (quad A), cell1 (BLANK), cell2 (quad C), eos.
    std::vector<std::size_t> ids = {dict.sos_idx(), td_idx, td_idx, td_idx,
                                    dict.eos_idx()};
    auto probs = one_hot(5, v, ids);
    std::vector<float> loc(5 * 8, 0.0f);
    // padded=488, ori=244 -> scale 244. quad A near top-left, quad C bottom-right.
    const float qa[8] = {0.05f, 0.05f, 0.2f, 0.05f, 0.2f, 0.2f, 0.05f, 0.2f};
    const float qc[8] = {0.6f, 0.6f, 0.9f, 0.6f, 0.9f, 0.9f, 0.6f, 0.9f};
    for (int k = 0; k < 8; ++k) loc[1 * 8 + k] = qa[k];  // cell0
    // cell1 stays all-zero (blank). cell2:
    for (int k = 0; k < 8; ++k) loc[3 * 8 + k] = qc[k];

    auto r = decode_structure(probs.data(), loc.data(), 5, v, dict, 488, 488,
                              244, 244);
    // 6 wrapper tags + 3 <td></td>.
    REQUIRE(r.structure.size() == 9);
    // The blank cell is NOT dropped: 3 cells, index-aligned with the 3 td tokens.
    REQUIRE(r.cells.size() == 3);
    for (int k = 0; k < 8; ++k) REQUIRE(r.cells[1].bbox[k] == 0);  // blank placeholder

    std::vector<std::array<int, 8>> quads;
    for (const auto& c : r.cells) quads.push_back(c.bbox);
    // OCR lines: "AAA" inside quad A, "CCC" inside quad C, none over the blank.
    std::vector<OcrLine> ocr = {
        OcrLine{{16.f, 16.f, 44.f, 44.f}, "AAA"},
        OcrLine{{150.f, 150.f, 210.f, 210.f}, "CCC"},
    };
    std::vector<std::string> texts = {"AAA", "CCC"};
    auto matched = match_cells_to_ocr(quads, ocr);
    REQUIRE(matched.size() == 3);

    const std::string html = reconstruct_html(r.structure, matched, texts);
    // AAA in cell0, blank cell1 renders empty, CCC in cell2 (NOT shifted into 1).
    REQUIRE(html ==
            "<html><body><table>"
            "<td>AAA</td><td></td><td>CCC</td>"
            "</table></body></html>");
}
