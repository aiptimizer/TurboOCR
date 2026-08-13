// Unit tests for the layout containment hierarchy (LayoutBox::parent_id).
//
// PP-DocLayoutV3 emits regions the model intends as NESTED — a figure_title
// inside an image, a formula_number inside a display_formula, a
// paragraph_title inside a content block. postfilter_layout_boxes records that
// structure as parent_id instead of flattening it away.
//
// The two properties that are easy to get wrong and expensive to debug later:
//   * parent_id can never form a cycle, including for the near-duplicate pairs
//     NMS does not catch (two boxes each >=90% inside the other);
//   * parent_id can never dangle — in kKeepOuter/kKeepInner a survivor whose
//     containment parent was DROPPED must be reparented to the nearest
//     surviving ancestor, or to -1.

#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/analysis/layout/layout_postfilter.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::layout::containment_parents;
using turbo_ocr::layout::layout_box_area;
using turbo_ocr::layout::LayoutBox;
using turbo_ocr::layout::MergeMode;
using turbo_ocr::layout::postfilter_layout_boxes;

namespace {

// Class ids from kLayoutLabels (pinned by static_assert in layout_types.h).
constexpr int kContent = 4;
constexpr int kDisplayFormula = 5;
constexpr int kFigureTitle = 7;
constexpr int kFormulaNumber = 11;
constexpr int kImage = 14;
constexpr int kTable = 21;
constexpr int kText = 22;

// A synthetic page big enough that the oversized-"image" filter (>82% of a
// portrait page) only fires when a test wants it to.
constexpr int kPageW = 800;
constexpr int kPageH = 1000;

LayoutBox make_layout(int x0, int y0, int x1, int y1, int class_id,
                      float score) {
  LayoutBox lb;
  lb.class_id = class_id;
  lb.score = score;
  lb.box = Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
  return lb;
}

// Index of the first box with the given class, or -1. Boxes come back in
// NMS score order, so tests give each region a distinct score and look it up
// by class rather than hard-coding positions.
int index_of(const std::vector<LayoutBox> &boxes, int class_id) {
  for (size_t i = 0; i < boxes.size(); ++i)
    if (boxes[i].class_id == class_id) return static_cast<int>(i);
  return -1;
}

} // namespace

TEST_CASE("hierarchy: a flat page leaves every region top-level",
          "[layout][hierarchy]") {
  std::vector<LayoutBox> in = {
      make_layout(50, 50, 750, 150, kText, 0.99f),
      make_layout(50, 200, 750, 400, kText, 0.98f),
      make_layout(50, 450, 750, 700, kText, 0.97f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);

  REQUIRE(out.size() == 3);
  for (const auto &lb : out) CHECK(lb.parent_id == -1);
}

TEST_CASE("hierarchy: parent is the SMALLEST container, not any container",
          "[layout][hierarchy]") {
  // caption inside figure inside content block. The caption must get the
  // figure, not the block that also contains it.
  std::vector<LayoutBox> in = {
      make_layout(50, 50, 750, 600, kContent, 0.99f),
      make_layout(100, 100, 400, 300, kImage, 0.95f),
      make_layout(120, 250, 380, 290, kFigureTitle, 0.90f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);
  REQUIRE(out.size() == 3);

  const int content = index_of(out, kContent);
  const int image = index_of(out, kImage);
  const int caption = index_of(out, kFigureTitle);
  REQUIRE(content >= 0);
  REQUIRE(image >= 0);
  REQUIRE(caption >= 0);

  CHECK(out[content].parent_id == -1);
  CHECK(out[image].parent_id == content);
  CHECK(out[caption].parent_id == image);
}

TEST_CASE("hierarchy: formula_number is parented to its display_formula",
          "[layout][hierarchy]") {
  std::vector<LayoutBox> in = {
      make_layout(100, 400, 600, 470, kDisplayFormula, 0.95f),
      make_layout(560, 420, 600, 450, kFormulaNumber, 0.90f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);
  REQUIRE(out.size() == 2);

  const int formula = index_of(out, kDisplayFormula);
  const int number = index_of(out, kFormulaNumber);
  REQUIRE(formula >= 0);
  REQUIRE(number >= 0);

  CHECK(out[formula].parent_id == -1);
  CHECK(out[number].parent_id == formula);
}

TEST_CASE("hierarchy: mutually-contained boxes cannot point at each other",
          "[layout][hierarchy]") {
  // Two near-duplicates NMS does not catch: cross-class NMS only suppresses at
  // IoU >= 0.98 and these sit at 0.95, yet each is >=90% inside the other. A
  // naive "any container" rule would make them each other's parent.
  std::vector<LayoutBox> in = {
      make_layout(100, 100, 700, 500, kText, 0.99f),   // area 240000
      make_layout(100, 100, 700, 480, kTable, 0.95f),  // area 228000
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);
  REQUIRE(out.size() == 2);

  const int larger = index_of(out, kText);
  const int smaller = index_of(out, kTable);
  REQUIRE(larger >= 0);
  REQUIRE(smaller >= 0);

  // Larger area wins the parent slot; the loser does not point back.
  CHECK(out[larger].parent_id == -1);
  CHECK(out[smaller].parent_id == larger);
}

TEST_CASE("hierarchy: equal-area mutual containment breaks the tie by score",
          "[layout][hierarchy]") {
  // Same area, each >=90% inside the other. The rank order is (area DESC,
  // index ASC) and the vector is NMS-sorted by score, so the HIGHER-scoring
  // box becomes the parent — deterministically, not by iteration accident.
  std::vector<LayoutBox> in = {
      make_layout(100, 5, 700, 95, kTable, 0.80f),  // area 54000, lower score
      make_layout(100, 0, 700, 90, kText, 0.99f),   // area 54000, higher score
  };
  REQUIRE(layout_box_area(in[0]) == layout_box_area(in[1]));

  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);
  REQUIRE(out.size() == 2);
  REQUIRE(out[0].class_id == kText);  // NMS sorted the higher score first

  CHECK(out[0].parent_id == -1);
  CHECK(out[1].parent_id == 0);
}

TEST_CASE("hierarchy: kKeepOuter reparents to the nearest SURVIVING ancestor",
          "[layout][hierarchy]") {
  // content > text > display_formula. kKeepOuter drops the text block (it is
  // nested) but keeps the formula (formulas are exempt from the nesting drop),
  // so the formula's immediate parent is gone and it must inherit the content
  // block rather than reference a deleted box.
  std::vector<LayoutBox> in = {
      make_layout(50, 50, 750, 900, kContent, 0.99f),
      make_layout(100, 100, 700, 500, kText, 0.95f),
      make_layout(200, 200, 600, 300, kDisplayFormula, 0.90f),
  };

  // Sanity: with nothing dropped the formula's parent IS the text block.
  auto all = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);
  REQUIRE(all.size() == 3);
  CHECK(all[index_of(all, kDisplayFormula)].parent_id == index_of(all, kText));

  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepOuter);
  REQUIRE(out.size() == 2);
  const int content = index_of(out, kContent);
  const int formula = index_of(out, kDisplayFormula);
  REQUIRE(content >= 0);
  REQUIRE(formula >= 0);
  CHECK(index_of(out, kText) == -1);  // the text block was dropped

  CHECK(out[content].parent_id == -1);
  CHECK(out[formula].parent_id == content);
}

TEST_CASE("hierarchy: kKeepOuter falls back to -1 when the whole chain goes",
          "[layout][hierarchy]") {
  // Mutually-contained text/table pair (both dropped by kKeepOuter, since each
  // is nested in the other) wrapping a formula (kept). Walking up from the
  // formula must run off the top and yield -1, not a dangling index.
  std::vector<LayoutBox> in = {
      make_layout(100, 100, 700, 500, kText, 0.99f),
      make_layout(100, 100, 700, 480, kTable, 0.95f),
      make_layout(200, 200, 600, 300, kDisplayFormula, 0.90f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepOuter);

  REQUIRE(out.size() == 1);
  CHECK(out[0].class_id == kDisplayFormula);
  CHECK(out[0].parent_id == -1);
}

TEST_CASE("hierarchy: kKeepInner keeps child links and drops dead parents",
          "[layout][hierarchy]") {
  std::vector<LayoutBox> in = {
      make_layout(50, 50, 750, 600, kContent, 0.99f),
      make_layout(100, 100, 400, 300, kImage, 0.95f),
      make_layout(120, 250, 380, 290, kFigureTitle, 0.90f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepInner);

  // kKeepInner drops pure containers: the content block contains others and is
  // itself contained by nothing, so it goes.
  REQUIRE(out.size() == 2);
  const int image = index_of(out, kImage);
  const int caption = index_of(out, kFigureTitle);
  REQUIRE(image >= 0);
  REQUIRE(caption >= 0);
  CHECK(index_of(out, kContent) == -1);

  CHECK(out[image].parent_id == -1);       // its parent was dropped
  CHECK(out[caption].parent_id == image);  // and this link still resolves
}

TEST_CASE("hierarchy: a dropped oversized image is never a parent",
          "[layout][hierarchy]") {
  // The full-page "image" detection is a detector artefact and step 2 removes
  // it. The hierarchy runs afterwards, so the text block must not be parented
  // to a box that is not in the response.
  std::vector<LayoutBox> in = {
      make_layout(0, 0, kPageW, kPageH, kImage, 0.99f),
      make_layout(100, 100, 700, 900, kText, 0.95f),
  };
  auto out = postfilter_layout_boxes(in, kPageH, kPageW, MergeMode::kKeepAll);

  REQUIRE(out.size() == 1);
  CHECK(out[0].class_id == kText);
  CHECK(out[0].parent_id == -1);
}

TEST_CASE("hierarchy: parent links always terminate (no cycles)",
          "[layout][hierarchy]") {
  // Property test over a deterministic pseudo-random pile of overlapping and
  // nested boxes: following parent_id from any box must reach -1 in at most n
  // steps, and every link must strictly ascend the (area DESC, index ASC) rank
  // order that makes that guarantee hold.
  std::vector<LayoutBox> boxes;
  uint32_t seed = 12345u;
  auto next = [&seed](int lo, int hi) {
    seed = seed * 1664525u + 1013904223u;
    return lo + static_cast<int>((seed >> 16) % static_cast<uint32_t>(hi - lo));
  };
  for (int i = 0; i < 200; ++i) {
    const int x0 = next(0, 700), y0 = next(0, 900);
    // Many boxes share edges/sizes on purpose so exact-area ties are common.
    const int w = next(1, 12) * 25, h = next(1, 12) * 25;
    boxes.push_back(make_layout(x0, y0, x0 + w, y0 + h, next(0, 25), 0.5f));
  }

  const auto parent = containment_parents(boxes);
  REQUIRE(parent.size() == boxes.size());

  for (size_t i = 0; i < boxes.size(); ++i) {
    int cur = static_cast<int>(i);
    size_t steps = 0;
    while (parent[static_cast<size_t>(cur)] >= 0) {
      const int p = parent[static_cast<size_t>(cur)];
      const long long ac = layout_box_area(boxes[static_cast<size_t>(cur)]);
      const long long ap = layout_box_area(boxes[static_cast<size_t>(p)]);
      // Strictly ascending rank: bigger area, or equal area and lower index.
      CHECK((ap > ac || (ap == ac && p < cur)));
      cur = p;
      ++steps;
      REQUIRE(steps <= boxes.size());  // a cycle would spin here
    }
  }
}

TEST_CASE("serialization: parent_id is emitted only when set",
          "[layout][hierarchy][serialization]") {
  std::vector<OCRResultItem> results;
  std::vector<LayoutBox> layout = {
      make_layout(50, 50, 750, 600, kContent, 0.99f),
      make_layout(100, 100, 400, 300, kImage, 0.95f),
  };
  layout[1].parent_id = 0;

  const std::string json = turbo_ocr::results_to_json(results, layout);

  // assign_layout_ids numbered the regions, and parent_id points at that id.
  CHECK(json.find("\"id\":0,\"class\":\"content\"") != std::string::npos);
  CHECK(json.find("\"id\":1,\"class\":\"image\"") != std::string::npos);
  CHECK(json.find("\"parent_id\":0") != std::string::npos);
  // Exactly one region carries the field — the top-level one must not.
  CHECK(json.find("\"parent_id\"") == json.rfind("\"parent_id\""));
}

TEST_CASE("serialization: a flat page emits no parent_id at all",
          "[layout][hierarchy][serialization]") {
  std::vector<OCRResultItem> results;
  std::vector<LayoutBox> layout = {
      make_layout(50, 50, 750, 150, kText, 0.99f),
      make_layout(50, 200, 750, 400, kText, 0.98f),
  };
  const std::string json = turbo_ocr::results_to_json(results, layout);

  CHECK(json.find("parent_id") == std::string::npos);
}
