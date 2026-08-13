#include <catch_amalgamated.hpp>

#include <string>
#include <string_view>

#include "turbo_ocr/analysis/formula/cjk_stats.h"

using turbo_ocr::formula::cjk_stats;
using turbo_ocr::formula::text_has_cjk;

namespace {

// Encode one codepoint as UTF-8 (test helper; independent of the decoder
// under test).
std::string utf8(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (cp >> 18));
    s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

} // namespace

TEST_CASE("cjk_stats basics", "[cjk_stats]") {
  SECTION("empty") {
    auto st = cjk_stats("");
    CHECK(st.cjk == 0);
    CHECK(st.total == 0);
    CHECK_FALSE(text_has_cjk(""));
  }
  SECTION("pure ASCII counts codepoints, no CJK") {
    auto st = cjk_stats("E = mc^2");
    CHECK(st.cjk == 0);
    CHECK(st.total == 8);
    CHECK_FALSE(text_has_cjk("E = mc^2"));
  }
  SECTION("pure Chinese") {
    // 数学公式 — 4 ideographs
    const std::string s = "数学公式";
    auto st = cjk_stats(s);
    CHECK(st.cjk == 4);
    CHECK(st.total == 4);
    CHECK(text_has_cjk(s));
  }
  SECTION("mixed LaTeX with mangled CJK (the -S garbage signature)") {
    const std::string s = "\\frac{抎汎}{x+1}";
    auto st = cjk_stats(s);
    CHECK(st.cjk == 2);
    CHECK(text_has_cjk(s));
  }
}

TEST_CASE("cjk_stats ideograph range boundaries", "[cjk_stats]") {
  // Inside: CJK Unified incl. Ext A [3400,9FFF], Compatibility [F900,FAFF],
  // Ext B+ / SIP [20000,2FFFF]. Outside: neighbors on both sides.
  const uint32_t inside[] = {0x3400, 0x4E00, 0x9FFF, 0xF900, 0xFAFF,
                             0x20000, 0x2FFFF};
  const uint32_t outside[] = {0x33FF, 0xA000, 0xF8FF, 0xFB00, 0x1FFFF,
                              0x30000};
  for (uint32_t cp : inside) {
    INFO("codepoint 0x" << std::hex << cp);
    CHECK(text_has_cjk(utf8(cp)));
    CHECK(cjk_stats(utf8(cp)).total == 1);
  }
  for (uint32_t cp : outside) {
    INFO("codepoint 0x" << std::hex << cp);
    CHECK_FALSE(text_has_cjk(utf8(cp)));
    CHECK(cjk_stats(utf8(cp)).total == 1);  // still a counted codepoint
  }
}

TEST_CASE("cjk_stats excludes kana/hangul (ideographs only)", "[cjk_stats]") {
  // Escalation targets Chinese formula content; Japanese kana and Korean
  // hangul must not trip it.
  CHECK_FALSE(text_has_cjk(utf8(0x3042)));  // hiragana A
  CHECK_FALSE(text_has_cjk(utf8(0x30A2)));  // katakana A
  CHECK_FALSE(text_has_cjk(utf8(0xAC00)));  // hangul GA (0xAC00 > 0x9FFF)
}

TEST_CASE("cjk_stats malformed UTF-8 never miscounts or overruns",
          "[cjk_stats]") {
  SECTION("stray continuation bytes are skipped") {
    auto st = cjk_stats("\x80\x80\x80");
    CHECK(st.total == 0);
    CHECK(st.cjk == 0);
  }
  SECTION("invalid lead byte is skipped") {
    auto st = cjk_stats("\xFFhello");
    CHECK(st.total == 5);
    CHECK(st.cjk == 0);
  }
  SECTION("truncated multibyte at end of buffer stops cleanly") {
    std::string s = "ab";
    s += static_cast<char>(0xE6);  // lead of a 3-byte seq, no continuation
    auto st = cjk_stats(s);
    CHECK(st.total == 2);
    CHECK(st.cjk == 0);
  }
  SECTION("lead byte followed by non-continuation resyncs") {
    std::string s;
    s += static_cast<char>(0xE6);  // 3-byte lead
    s += 'x';                      // not a continuation
    s += utf8(0x4E2D);             // 中
    auto st = cjk_stats(s);
    CHECK(st.cjk == 1);
    CHECK(text_has_cjk(s));
  }
}

TEST_CASE("cjk_stats measured routing scenarios", "[cjk_stats]") {
  // The pipeline hint is: escalate iff cjk >= 3 && cjk*100 >= total
  // (>=3 absolute AND >=1%). These mirror the two measured cases that set
  // the threshold (ocr_pipeline_run.cpp).
  SECTION("one stray CJK glyph in 1700 chars must stay below threshold") {
    std::string page(1699, 'a');
    page += utf8(0x4E2D);
    auto st = cjk_stats(page);
    CHECK(st.cjk == 1);
    CHECK(st.total == 1700);
    CHECK_FALSE((st.cjk >= 3 && st.cjk * 100 >= st.total));
  }
  SECTION("genuinely Chinese page (23% CJK) must clear threshold") {
    std::string page;
    for (int i = 0; i < 23; ++i) page += utf8(0x4E00 + i);
    page += std::string(77, 'a');
    auto st = cjk_stats(page);
    CHECK(st.cjk == 23);
    CHECK(st.total == 100);
    CHECK((st.cjk >= 3 && st.cjk * 100 >= st.total));
  }
  SECTION("two CJK chars in a short caption stay below (absolute floor)") {
    std::string s = utf8(0x4E00) + utf8(0x4E01) + "figure 3";
    auto st = cjk_stats(s);
    CHECK(st.cjk == 2);
    CHECK_FALSE((st.cjk >= 3 && st.cjk * 100 >= st.total));
  }
}
