#include <catch_amalgamated.hpp>

#include <cmath>
#include <vector>

#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"
#include "turbo_ocr/analysis/table/table_types.h"

using turbo_ocr::table::CharDict;
using turbo_ocr::table::decode_structure;
using turbo_ocr::table::default_dict;
using turbo_ocr::table::SLANEXT_VOCAB;

namespace {

std::size_t find_token(const CharDict& d, std::string_view t) {
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.token(i) == t) return i;
    }
    return d.len();
}

void write_probs(std::vector<float>& probs, std::size_t step,
                 std::size_t v, std::size_t id, float p) {
    probs[step * v + id] = p;
}

void write_quad(std::vector<float>& loc, std::size_t step,
                const std::array<float, 8> &q) {
    for (std::size_t k = 0; k < 8; ++k) loc[step * 8 + k] = q[k];
}

} // namespace

TEST_CASE("eos at step 0 still emits token (matches RapidAI walk)",
          "[slanext_postprocess_aabb]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    REQUIRE(v == SLANEXT_VOCAB);
    std::vector<float> probs(2 * v, 0.0f);
    // Python `if idx > 0 and char_idx == end_idx: break` — eos at step 0 is
    // NOT a terminator (it falls into the sos/eos skip later instead).
    write_probs(probs, 0, v, dict.eos_idx(), 1.0f);
    write_probs(probs, 1, v, find_token(dict, "<thead>"), 1.0f);
    std::vector<float> loc(2 * 8, 0.0f);
    auto r = decode_structure(probs.data(), loc.data(), 2, v, dict, 488,
                              488, 200, 200);
    REQUIRE(r.structure.size() == 7);
    REQUIRE(r.structure[3] == "<thead>");
}

TEST_CASE("walk terminates at first non-step-0 eos",
          "[slanext_postprocess_aabb]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td = find_token(dict, "<td></td>");
    REQUIRE(td < v);
    std::vector<float> probs(5 * v, 0.0f);
    write_probs(probs, 0, v, dict.sos_idx(), 1.0f);
    write_probs(probs, 1, v, td, 1.0f);
    write_probs(probs, 2, v, dict.eos_idx(), 1.0f);
    // These should be ignored — walk halts at step 2.
    write_probs(probs, 3, v, td, 1.0f);
    write_probs(probs, 4, v, td, 1.0f);
    std::vector<float> loc(5 * 8, 0.0f);
    write_quad(loc, 1, {0.1f, 0.1f, 0.5f, 0.1f, 0.5f, 0.5f, 0.1f, 0.5f});
    write_quad(loc, 3, {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
    auto r = decode_structure(probs.data(), loc.data(), 5, v, dict, 488,
                              488, 200, 200);
    REQUIRE(r.cells.size() == 1);
    REQUIRE(r.structure.size() == 7);
    REQUIRE(r.structure[3] == "<td></td>");
}

TEST_CASE("td token quad scales as 4-corner AABB-style quad",
          "[slanext_postprocess_aabb]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td = find_token(dict, "<td></td>");
    REQUIRE(td < v);
    std::vector<float> probs(3 * v, 0.0f);
    write_probs(probs, 0, v, dict.sos_idx(), 1.0f);
    write_probs(probs, 1, v, td, 1.0f);
    write_probs(probs, 2, v, dict.eos_idx(), 1.0f);
    std::vector<float> loc(3 * 8, 0.0f);
    // 4 corners (axis-aligned rectangle interpreted as 4-corner quad).
    write_quad(loc, 1, {0.10f, 0.20f, 0.60f, 0.20f, 0.60f, 0.80f, 0.10f, 0.80f});
    // Anisotropic original size — long-side scaling applies after.
    // ori = (300 wide, 600 tall) → ratio = min(488/600, 488/300) = 488/600.
    // per-axis scale_w = 300 * 488 / (300 * 488/600) = 600,
    //                scale_h = 600 * 488 / (600 * 488/600) = 600.
    auto r = decode_structure(probs.data(), loc.data(), 3, v, dict, 488,
                              488, 300, 600);
    REQUIRE(r.cells.size() == 1);
    const auto& q = r.cells[0].bbox;
    REQUIRE(q[0] == static_cast<int>(0.10f * 600.0f));
    REQUIRE(q[1] == static_cast<int>(0.20f * 600.0f));
    REQUIRE(q[2] == static_cast<int>(0.60f * 600.0f));
    REQUIRE(q[3] == static_cast<int>(0.20f * 600.0f));
    REQUIRE(q[4] == static_cast<int>(0.60f * 600.0f));
    REQUIRE(q[5] == static_cast<int>(0.80f * 600.0f));
    REQUIRE(q[6] == static_cast<int>(0.10f * 600.0f));
    REQUIRE(q[7] == static_cast<int>(0.80f * 600.0f));
}

TEST_CASE("vocab boundary tokens decode to slanet-plus indices",
          "[slanext_postprocess_aabb]") {
    auto dict = default_dict();
    REQUIRE(dict.len() == SLANEXT_VOCAB);
    REQUIRE(dict.sos_idx() == 0);
    REQUIRE(dict.eos_idx() == SLANEXT_VOCAB - 1);
    // Authoritative PaddleX dict order: build() drops the bare "<td>" and
    // appends "<td></td>"; <tbody>/</tbody> live at the tail, not the head.
    // <thead>=1 </thead>=2 <tr>=3 </tr>=4 </td>=5 <td=6 >=7  colspan="2"=8.
    REQUIRE(dict.token(1) == "<thead>");
    REQUIRE(dict.token(2) == "</thead>");
    REQUIRE(dict.token(3) == "<tr>");
    REQUIRE(dict.token(4) == "</tr>");
    REQUIRE(dict.token(5) == "</td>");
    REQUIRE(dict.token(6) == "<td");
    REQUIRE(dict.token(7) == ">");
    REQUIRE(dict.token(8) == " colspan=\"2\"");
    // <td></td> is the merged sentinel — last slot before eos.
    REQUIRE(dict.is_td_token(SLANEXT_VOCAB - 2));
    REQUIRE(dict.token(SLANEXT_VOCAB - 2) == "<td></td>");
    // The colspan block runs "2".."19" then jumps to "25" (no "20".."24").
    REQUIRE(find_token(dict, " colspan=\"25\"") < dict.len());
    REQUIRE(find_token(dict, " colspan=\"20\"") == dict.len());
}

TEST_CASE("blank quad keeps an index-aligned placeholder cell",
          "[slanext_postprocess_aabb]") {
    auto dict = default_dict();
    const std::size_t v = dict.len();
    const std::size_t td = find_token(dict, "<td></td>");
    std::vector<float> probs(3 * v, 0.0f);
    write_probs(probs, 0, v, dict.sos_idx(), 1.0f);
    write_probs(probs, 1, v, td, 1.0f);
    write_probs(probs, 2, v, dict.eos_idx(), 1.0f);
    std::vector<float> loc(3 * 8, 0.0f);  // all zeros — blank quad
    auto r = decode_structure(probs.data(), loc.data(), 3, v, dict, 488,
                              488, 244, 244);
    // The all-zero quad is KEPT as a positional placeholder (zero quad) so
    // `cells` stays index-aligned with the <td> tokens in `structure` — dropping
    // it would shift every later cell's text by one (M1). The zero-area quad
    // matches no OCR line and renders as an empty <td></td>.
    REQUIRE(r.cells.size() == 1);
    for (int k = 0; k < 8; ++k) REQUIRE(r.cells[0].bbox[k] == 0);
    REQUIRE(r.structure.size() == 7);
    REQUIRE(r.structure[3] == "<td></td>");
}
