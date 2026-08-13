#include <catch_amalgamated.hpp>

#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/table_types.h"

using turbo_ocr::table::CharDict;
using turbo_ocr::table::default_dict;
using turbo_ocr::table::SLANEXT_VOCAB;

TEST_CASE("build inserts sos eos and merges td", "[slanext_dict]") {
    auto d = CharDict::build({"<td>", "</td>"});
    REQUIRE(d.token(0) == "sos");
    REQUIRE(d.token(d.eos_idx()) == "eos");
    bool has_merge = false;
    bool has_bare = false;
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.token(i) == "<td></td>") has_merge = true;
        if (d.token(i) == "<td>") has_bare = true;
    }
    REQUIRE(has_merge);
    REQUIRE_FALSE(has_bare);
}

TEST_CASE("default dict size equals SLANEXT_VOCAB", "[slanext_dict]") {
    auto d = default_dict();
    REQUIRE(d.len() == SLANEXT_VOCAB);
}

TEST_CASE("is_td_token matches the PaddleX set", "[slanext_dict]") {
    auto d = default_dict();
    bool found_td = false;
    for (std::size_t i = 0; i < d.len(); ++i) {
        if (d.is_td_token(i)) { found_td = true; break; }
    }
    REQUIRE(found_td);
    REQUIRE_FALSE(d.is_td_token(d.sos_idx()));
    REQUIRE_FALSE(d.is_td_token(d.eos_idx()));
}
