#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/document/docassembly/parsing_block.h"
#include "turbo_ocr/document/docassembly/repetition_guard.h"
#include "turbo_ocr/document/docassembly/table_merge.h"
#include "turbo_ocr/document/docassembly/title_level.h"

using namespace turbo_ocr::docassembly;

namespace {
ParsingBlock title(const std::string& content, int y0, int y1, int x0 = 0,
                   int x1 = 400) {
  ParsingBlock b;
  b.label = "paragraph_title";
  b.content = content;
  b.bbox = {x0, y0, x1, y1};
  return b;
}
ParsingBlock tbl(const std::string& html, int x0 = 0, int x1 = 400) {
  ParsingBlock b;
  b.label = "table";
  b.content = html;
  b.bbox = {x0, 0, x1, 50};
  return b;
}
} // namespace

// ---------------------------------------------------------------------------
// repetition guard (truncate_repetitive_content)
// ---------------------------------------------------------------------------

TEST_CASE("repetition: below min_count is untouched", "[docassembly][rep]") {
  std::string s = "short content";
  REQUIRE(truncate_repetitive_content(s, 10, 10, 10, 3000) == s);
}

TEST_CASE("repetition: line-level runaway collapses to the repeated line",
          "[docassembly][rep]") {
  std::string line = "the same row over and over\n";
  std::string s;
  for (int i = 0; i < 50; ++i) s += line;  // >> min_count=50
  auto out = truncate_repetitive_content(s, 10, 10, 10, 50);
  REQUIRE(out == "the same row over and over");
}

TEST_CASE("repetition: char-level repeat collapses to the unit",
          "[docassembly][rep]") {
  // Length 80: above min_count(50) but <=100, so Priority-1 (len>100) is
  // skipped and the char-level (Priority-2) branch is the one exercised.
  std::string s;
  for (int i = 0; i < 40; ++i) s += "ab";
  auto out = truncate_repetitive_content(s, 10, 10, 10, 50);
  REQUIRE(out == "ab");
}

TEST_CASE("repetition: phrase-suffix runaway keeps the real prefix",
          "[docassembly][rep]") {
  // Models the CDM mode-collapse: a real prefix then a long repeated tail
  // that dominates >50% of the string. Unit has no trailing whitespace so
  // strip() can't disturb the tiling — the prefix is recovered cleanly.
  std::string prefix = "Real formula prefix here. ";
  std::string tail;
  for (int i = 0; i < 400; ++i) tail += "QW";
  std::string s = prefix + tail;
  auto out = truncate_repetitive_content(s, 10, 10, 10, 50);
  REQUIRE(out == prefix);
}

// ---------------------------------------------------------------------------
// title leveling (get_symbol_and_level + assign_title_levels)
// ---------------------------------------------------------------------------

TEST_CASE("title: numbering symbol + level extraction", "[docassembly][title]") {
  REQUIRE(get_symbol_and_level("1 Introduction").symbol == "NUM_LIST");
  REQUIRE(get_symbol_and_level("1 Introduction").level == 1);
  REQUIRE(get_symbol_and_level("1.1 Background").level == 2);
  REQUIRE(get_symbol_and_level("1.2.3 Deep").level == 3);
  REQUIRE(get_symbol_and_level("(1) Bracketed").symbol == "NUM_LIST_BRACKET");
  REQUIRE(get_symbol_and_level("(1) Bracketed").level == 4);
  REQUIRE(get_symbol_and_level("1) Half bracket").symbol == "NUM_LIST_BRACKET");
  REQUIRE(get_symbol_and_level("IV. Roman").symbol == "ROMAN");
  REQUIRE(get_symbol_and_level("IV. Roman").level == 1);
  REQUIRE(get_symbol_and_level("A. Letter").symbol == "LETTER");
  REQUIRE(get_symbol_and_level("A. Letter").level == 2);
  REQUIRE(get_symbol_and_level("Plain Heading").level == -1);
}

TEST_CASE("title: numeric depth maps to heading level via vote",
          "[docassembly][title]") {
  std::vector<std::vector<ParsingBlock>> pages = {{
      title("1 Introduction", 0, 40),     // tall → level 1
      title("1.1 Motivation", 50, 75),    // shorter → level 2
      title("1.2 Scope", 80, 105),
      title("2 Methods", 120, 160),
  }};
  assign_title_levels(pages);
  REQUIRE(pages[0][0].title_level == 1);
  REQUIRE(pages[0][1].title_level == 2);
  REQUIRE(pages[0][2].title_level == 2);
  REQUIRE(pages[0][3].title_level == 1);
}

TEST_CASE("title: special keyword forces level 1", "[docassembly][title]") {
  std::vector<std::vector<ParsingBlock>> pages = {{
      title("1 Intro", 0, 40),
      title("References", 200, 224),  // small height, but keyword → 1
  }};
  assign_title_levels(pages);
  REQUIRE(pages[0][1].title_level == 1);
}

TEST_CASE("title: non-heading blocks are left untouched", "[docassembly][title]") {
  ParsingBlock text;
  text.label = "text";
  text.content = "1 this looks numbered but is body text";
  std::vector<std::vector<ParsingBlock>> pages = {{text}};
  assign_title_levels(pages);
  REQUIRE(pages[0][0].title_level == -1);
}

// ---------------------------------------------------------------------------
// cross-page table merge
// ---------------------------------------------------------------------------

TEST_CASE("table: full_to_half normalizes fullwidth chars",
          "[docassembly][table]") {
  // （１） in fullwidth → (1)
  REQUIRE(full_to_half("\xef\xbc\x88\xef\xbc\x91\xef\xbc\x89") == "(1)");
}

TEST_CASE("table: adjacent tables with matching columns merge across pages",
          "[docassembly][table]") {
  std::string head =
      "<html><body><table><tr><td>A</td><td>B</td></tr>";
  std::string prev = head + "<tr><td>1</td><td>2</td></tr></table></body></html>";
  std::string curr =
      "<html><body><table><tr><td>A</td><td>B</td></tr>"
      "<tr><td>3</td><td>4</td></tr></table></body></html>";
  std::vector<std::vector<ParsingBlock>> pages = {{tbl(prev)}, {tbl(curr)}};
  merge_tables_across_pages(pages);

  const std::string& merged = pages[0][0].content;
  // host table now carries both data rows
  REQUIRE(merged.find("<td>1</td><td>2</td>") != std::string::npos);
  REQUIRE(merged.find("<td>3</td><td>4</td>") != std::string::npos);
  // second table emptied and regrouped under the host
  REQUIRE(pages[1][0].content.empty());
  REQUIRE(pages[1][0].global_group_id == pages[0][0].global_block_id);
}

TEST_CASE("table: mismatched column counts do not merge",
          "[docassembly][table]") {
  std::string prev =
      "<html><body><table><tr><td>A</td><td>B</td></tr>"
      "<tr><td>1</td><td>2</td></tr></table></body></html>";
  std::string curr =
      "<html><body><table><tr><td>X</td><td>Y</td><td>Z</td></tr>"
      "<tr><td>7</td><td>8</td><td>9</td></tr></table></body></html>";
  std::vector<std::vector<ParsingBlock>> pages = {{tbl(prev)}, {tbl(curr)}};
  merge_tables_across_pages(pages);
  REQUIRE(pages[1][0].content == curr);                       // untouched
  REQUIRE(pages[1][0].global_group_id == pages[1][0].global_block_id);
}

TEST_CASE("table: non-furniture block between tables blocks the merge",
          "[docassembly][table]") {
  std::string prev =
      "<html><body><table><tr><td>A</td><td>B</td></tr>"
      "<tr><td>1</td><td>2</td></tr></table></body></html>";
  std::string curr =
      "<html><body><table><tr><td>A</td><td>B</td></tr>"
      "<tr><td>3</td><td>4</td></tr></table></body></html>";
  ParsingBlock para;
  para.label = "text";
  para.content = "an intervening paragraph";
  para.bbox = {0, 0, 400, 30};
  // paragraph appears BEFORE the table on the current page → not skippable
  std::vector<std::vector<ParsingBlock>> pages = {{tbl(prev)}, {para, tbl(curr)}};
  merge_tables_across_pages(pages);
  REQUIRE(pages[1][1].content == curr);  // not merged
}
