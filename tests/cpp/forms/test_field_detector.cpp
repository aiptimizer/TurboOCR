// Geometry tests for the fillable-field detectors.
//
// The pages here are SYNTHESISED as rasters, not mocked, because the whole
// point of detectors 1 and 2 is that they read pixels the recogniser never
// reports. Mocking the mask would test nothing.

#include <catch_amalgamated.hpp>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/analysis/forms/field_detector.h"
#include "turbo_ocr/analysis/forms/field_serialization.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using namespace turbo_ocr::forms;

namespace {

Box mk_box(int x, int y, int w, int h) {
  return Box{{{{{x, y}}, {{x + w, y}}, {{x + w, y + h}}, {{x, y + h}}}}};
}

OCRResultItem mk_item(const std::string &text, int x, int y, int w, int h) {
  OCRResultItem it;
  it.text = text;
  it.confidence = 0.99f;
  it.box = mk_box(x, y, w, h);
  return it;
}

// A blank white page to draw a form on.
cv::Mat blank_page(int w = 800, int h = 1000) {
  return cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
}

void draw_rule(cv::Mat &page, int x, int y, int w, int thickness = 2) {
  cv::line(page, {x, y}, {x + w, y}, cv::Scalar(0, 0, 0), thickness);
}

void draw_box(cv::Mat &page, const cv::Rect &r, int thickness = 2) {
  cv::rectangle(page, r, cv::Scalar(0, 0, 0), thickness);
}

// Ink that is not a line, so the "whitespace above" test has something to
// reject. Text height 30 keeps the median where the tests assume it.
void draw_text_blob(cv::Mat &page, const cv::Rect &r) {
  cv::rectangle(page, r, cv::Scalar(0, 0, 0), cv::FILLED);
}

const FormField *find_by_label(const std::vector<FormField> &fields,
                               const std::string &label) {
  for (const auto &f : fields)
    if (f.label == label) return &f;
  return nullptr;
}

int count_type(const std::vector<FormField> &fields, FieldType t) {
  int n = 0;
  for (const auto &f : fields)
    if (f.type == t) ++n;
  return n;
}

} // namespace

TEST_CASE("median_text_height measures OCR boxes", "[forms]") {
  std::vector<OCRResultItem> text{
      mk_item("a", 0, 0, 50, 20),
      mk_item("b", 0, 40, 50, 30),
      mk_item("c", 0, 80, 50, 40),
  };
  CHECK(median_text_height(text, 1000) == Catch::Approx(30.0f));
}

TEST_CASE("median_text_height falls back to page scale with no text",
          "[forms]") {
  // A page of nothing but boxes still needs a scale unit, or every threshold
  // below collapses to zero and the detectors accept anything.
  const float h = median_text_height({}, 1100);
  CHECK(h > 8.0f);
  CHECK(h < 40.0f);
}

TEST_CASE("median_text_height ignores blank OCR entries", "[forms]") {
  std::vector<OCRResultItem> text{
      mk_item("   ", 0, 0, 50, 200), // whitespace-only run with a huge box
      mk_item("a", 0, 40, 50, 20),
      mk_item("b", 0, 80, 50, 20),
  };
  CHECK(median_text_height(text, 1000) == Catch::Approx(20.0f));
}

TEST_CASE("is_label_text needs a trailing colon", "[forms]") {
  CHECK(is_label_text("Name:"));
  CHECK(is_label_text("  Vorname:  "));
  CHECK(is_label_text("\xE5\xA7\x93\xE5\x90\x8D\xEF\xBC\x9A")); // CJK fullwidth
  CHECK_FALSE(is_label_text("Name"));
  CHECK_FALSE(is_label_text(":"));   // too short to be a label
  CHECK_FALSE(is_label_text(""));
}

TEST_CASE("is_rule_text spots a drawn blank", "[forms]") {
  CHECK(is_rule_text("_____"));
  CHECK(is_rule_text("-----"));
  CHECK(is_rule_text("....."));
  CHECK_FALSE(is_rule_text("__"));      // too short
  CHECK_FALSE(is_rule_text("__a__"));
  CHECK_FALSE(is_rule_text(""));
}

TEST_CASE("box_iou matches known overlaps", "[forms]") {
  CHECK(box_iou(mk_box(0, 0, 10, 10), mk_box(0, 0, 10, 10)) ==
        Catch::Approx(1.0f));
  CHECK(box_iou(mk_box(0, 0, 10, 10), mk_box(100, 100, 10, 10)) ==
        Catch::Approx(0.0f));
  // Half-overlap: intersection 50, union 150.
  CHECK(box_iou(mk_box(0, 0, 10, 10), mk_box(5, 0, 10, 10)) ==
        Catch::Approx(50.0f / 150.0f));
}

TEST_CASE("rect_to_box round-trips through box_to_rect", "[forms]") {
  const cv::Rect r(12, 34, 56, 78);
  CHECK(box_to_rect(rect_to_box(r)) == r);
}

TEST_CASE("group_into_lines bands runs by baseline, sorted left to right",
          "[forms]") {
  // Appended in the wrong order on purpose: detection order is not reading
  // order, and the grouping has to fix that itself.
  std::vector<OCRResultItem> text{
      mk_item("second", 300, 100, 80, 30),
      mk_item("below", 10, 300, 80, 30),
      mk_item("first", 10, 104, 80, 30),
  };
  const auto lines = group_into_lines(text, {});
  REQUIRE(lines.size() == 2);
  REQUIRE(lines[0].size() == 2);
  CHECK(text[lines[0][0]].text == "first");
  CHECK(text[lines[0][1]].text == "second");
  REQUIRE(lines[1].size() == 1);
  CHECK(text[lines[1][0]].text == "below");
}

TEST_CASE("find_label prefers the run to the left on the same line",
          "[forms]") {
  std::vector<OCRResultItem> text{
      mk_item("Above:", 100, 40, 90, 30),
      mk_item("Name:", 100, 100, 90, 30),
  };
  CHECK(find_label(cv::Rect(210, 100, 300, 30), text, 30.0f, {}) == "Name:");
}

TEST_CASE("find_label falls back to the run directly above", "[forms]") {
  std::vector<OCRResultItem> text{mk_item("Datum", 100, 100, 90, 30)};
  // Nothing to the left; the header sits one line up and its span overlaps.
  CHECK(find_label(cv::Rect(100, 140, 200, 30), text, 30.0f, {}) == "Datum");
}

TEST_CASE("find_label returns empty when nothing is near", "[forms]") {
  std::vector<OCRResultItem> text{mk_item("Far:", 10, 10, 90, 30)};
  CHECK(find_label(cv::Rect(600, 800, 100, 30), text, 30.0f, {}).empty());
}

TEST_CASE("find_label ignores a run above whose span does not overlap",
          "[forms]") {
  std::vector<OCRResultItem> text{mk_item("Elsewhere", 10, 100, 90, 30)};
  CHECK(find_label(cv::Rect(600, 140, 100, 30), text, 30.0f, {}).empty());
}

TEST_CASE("merge_fields collapses duplicates and records both sources",
          "[forms]") {
  std::vector<FormField> fields;
  FormField rule;
  rule.box = mk_box(200, 100, 300, 34);
  rule.confidence = 0.78f;
  rule.source = "rule";
  FormField gap;
  gap.box = mk_box(205, 100, 290, 34);
  gap.confidence = 0.55f;
  gap.source = "label_gap";
  gap.label = "Name:";
  fields.push_back(gap); // weaker one first: order must not decide the winner
  fields.push_back(rule);

  merge_fields(fields, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source == "rule+label_gap"); // strongest evidence first
  CHECK(fields[0].label == "Name:");           // label survives the merge
  CHECK(fields[0].confidence > 0.78f);         // agreement raises confidence
}

TEST_CASE("merge_fields keeps genuinely separate blanks", "[forms]") {
  std::vector<FormField> fields;
  FormField a;
  a.box = mk_box(200, 100, 300, 34);
  a.confidence = 0.78f;
  a.source = "rule";
  FormField b;
  b.box = mk_box(200, 300, 300, 34);
  b.confidence = 0.78f;
  b.source = "rule";
  fields = {a, b};
  merge_fields(fields, {});
  CHECK(fields.size() == 2);
}

TEST_CASE("merge_fields will not let a checkbox swallow a text field",
          "[forms]") {
  // The checkbox is contained in the text field but a fraction of its area.
  // Without the area-ratio guard the higher-confidence checkbox would absorb
  // the larger blank and the text field would vanish from the output.
  std::vector<FormField> fields;
  FormField cb;
  cb.box = mk_box(210, 105, 24, 24);
  cb.confidence = 0.90f;
  cb.source = "checkbox";
  cb.type = FieldType::Checkbox;
  FormField text_field;
  text_field.box = mk_box(200, 100, 400, 34);
  text_field.confidence = 0.88f;
  text_field.source = "box";
  fields = {cb, text_field};
  merge_fields(fields, {});
  CHECK(fields.size() == 2);
}

TEST_CASE("merge_fields is order independent", "[forms]") {
  FormField a;
  a.box = mk_box(200, 100, 300, 34);
  a.confidence = 0.78f;
  a.source = "rule";
  FormField b;
  b.box = mk_box(203, 101, 295, 33);
  b.confidence = 0.55f;
  b.source = "label_gap";

  std::vector<FormField> forward{a, b};
  std::vector<FormField> reverse{b, a};
  merge_fields(forward, {});
  merge_fields(reverse, {});
  REQUIRE(forward.size() == 1);
  REQUIRE(reverse.size() == 1);
  CHECK(forward[0].source == reverse[0].source);
  CHECK(forward[0].box == reverse[0].box);
}

TEST_CASE("find_rule_segments keeps long thin runs and drops short ones",
          "[forms]") {
  cv::Mat page = blank_page();
  draw_rule(page, 100, 500, 300);  // 10 text-heights long: a rule
  draw_rule(page, 100, 600, 40);   // ~1.3 text-heights: a dash, not a rule
  const cv::Mat bin = binarize_page(page);
  const LineMasks masks = extract_line_masks(bin, 45, 45);
  const auto rules = find_rule_segments(masks.horizontal, 30.0f, {});
  REQUIRE(rules.size() == 1);
  CHECK(rules[0].width >= 290);
  CHECK(std::abs(rules[0].y - 500) <= 3);
}

TEST_CASE("extract_line_masks skips a direction with a non-positive kernel",
          "[forms]") {
  cv::Mat page = blank_page(200, 200);
  draw_rule(page, 20, 100, 150);
  const cv::Mat bin = binarize_page(page);
  const LineMasks only_h = extract_line_masks(bin, 45, 0);
  CHECK_FALSE(only_h.horizontal.empty());
  CHECK(only_h.vertical.empty()); // the rule pass never pays for this one
}

TEST_CASE("find_rule_segments rejects a thick filled bar", "[forms]") {
  cv::Mat page = blank_page();
  // 20px tall at text height 30 — a printed bar, not something to write on.
  cv::rectangle(page, cv::Rect(100, 500, 300, 20), cv::Scalar(0, 0, 0),
                cv::FILLED);
  const cv::Mat bin = binarize_page(page);
  const LineMasks masks = extract_line_masks(bin, 45, 45);
  CHECK(find_rule_segments(masks.horizontal, 30.0f, {}).empty());
}

TEST_CASE("find_closed_boxes finds a drawn rectangle", "[forms]") {
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(100, 200, 400, 60));
  const cv::Mat bin = binarize_page(page);
  const LineMasks masks = extract_line_masks(bin, 12, 12);
  const auto boxes = find_closed_boxes(masks, 30.0f, {});
  REQUIRE(boxes.size() == 1);
  CHECK(std::abs(boxes[0].width - 400) < 12);
  CHECK(std::abs(boxes[0].height - 60) < 12);
}

TEST_CASE("find_closed_boxes rejects three sides of a rectangle", "[forms]") {
  // The four-edge coverage test is what separates a drawn box from an
  // accidental enclosure; an open bracket must not become a field.
  cv::Mat page = blank_page();
  draw_rule(page, 100, 200, 400);
  draw_rule(page, 100, 260, 400);
  cv::line(page, {100, 200}, {100, 260}, cv::Scalar(0, 0, 0), 2);
  const cv::Mat bin = binarize_page(page);
  const LineMasks masks = extract_line_masks(bin, 12, 12);
  CHECK(find_closed_boxes(masks, 30.0f, {}).empty());
}

TEST_CASE("detect_form_fields finds a rule as a writable band", "[forms]") {
  cv::Mat page = blank_page();
  draw_rule(page, 260, 300, 400);
  std::vector<OCRResultItem> text{mk_item("Unterschrift:", 100, 270, 150, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].type == FieldType::Text);
  CHECK(fields[0].label == "Unterschrift:");
  CHECK(fields[0].source.find("rule") != std::string::npos);
  const cv::Rect r = box_to_rect(fields[0].box);
  // The field stands ON the rule, not below it.
  CHECK(r.y + r.height <= 302);
  CHECK(r.y + r.height >= 290);
}

TEST_CASE("detect_form_fields ignores a rule underlining a heading",
          "[forms]") {
  // Ink directly above means the rule underlines something rather than
  // offering somewhere to write. This is the main false-positive guard.
  cv::Mat page = blank_page();
  draw_text_blob(page, cv::Rect(100, 270, 400, 28));
  draw_rule(page, 100, 302, 400);
  std::vector<OCRResultItem> text{mk_item("Section heading", 100, 270, 400, 28)};

  const auto fields = detect_form_fields(page, text, {});
  for (const auto &f : fields)
    CHECK(f.source.find("rule") == std::string::npos);
}

TEST_CASE("detect_form_fields classifies a small square box as a checkbox",
          "[forms]") {
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(300, 400, 28, 28));
  std::vector<OCRResultItem> text{mk_item("Ja", 200, 400, 60, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(count_type(fields, FieldType::Checkbox) == 1);
  const FormField *cb = find_by_label(fields, "Ja");
  REQUIRE(cb != nullptr);
  CHECK(cb->type == FieldType::Checkbox);
  CHECK(cb->source.find("checkbox") != std::string::npos);
}

TEST_CASE("detect_form_fields calls a wide empty box a text field",
          "[forms]") {
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(200, 400, 400, 40));
  std::vector<OCRResultItem> text{mk_item("Adresse:", 60, 405, 120, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(count_type(fields, FieldType::Text) >= 1);
  CHECK(count_type(fields, FieldType::Checkbox) == 0);
}

TEST_CASE("detect_form_fields does not propose a box that already has text",
          "[forms]") {
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(200, 400, 400, 40));
  // The OCR says this cell is occupied, so it is content and not a blank.
  std::vector<OCRResultItem> text{mk_item("already filled in", 210, 405, 300, 30)};

  const auto fields = detect_form_fields(page, text, {});
  for (const auto &f : fields) {
    const cv::Rect r = box_to_rect(f.box);
    CHECK_FALSE((r & cv::Rect(200, 400, 400, 40)).area() > 0.5 * r.area());
  }
}

TEST_CASE("detect_form_fields proposes the blank after a label", "[forms]") {
  cv::Mat page = blank_page();  // nothing drawn: label+gap is the only evidence
  std::vector<OCRResultItem> text{mk_item("Name:", 100, 300, 120, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source == "label_gap");
  CHECK(fields[0].label == "Name:");
  CHECK(box_to_rect(fields[0].box).x >= 220);
}

TEST_CASE("detect_form_fields reports one field for a label with a rule",
          "[forms]") {
  // Both detector 1 and detector 3 see this blank. Emitting it twice is the
  // most likely way the whole feature looks broken.
  cv::Mat page = blank_page();
  draw_rule(page, 240, 330, 400);
  std::vector<OCRResultItem> text{mk_item("Name:", 100, 300, 120, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].label == "Name:");
  CHECK(fields[0].source.find('+') != std::string::npos); // both agreed
  CHECK(fields[0].confidence > 0.78f);
}

TEST_CASE("detect_form_fields does not put a field above a bordered box",
          "[forms]") {
  // A box's TOP border is a long horizontal run with clean paper above it, so
  // the rule detector sees a write-on line and proposes a phantom field in the
  // gap above every bordered box on the page. The box detector already owns
  // that geometry.
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(200, 400, 400, 44));
  std::vector<OCRResultItem> text{mk_item("Firma:", 60, 405, 110, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].label == "Firma:");
  // The one field is the box itself, not a band floating above it.
  CHECK(box_to_rect(fields[0].box).y >= 395);
}

TEST_CASE("detect_form_fields ignores a gap that is followed by content",
          "[forms]") {
  // "Status: bereits ausgefuellt" — a wide gap after the colon, but the entry
  // is already filled and the whitespace is layout.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{
      mk_item("Status:", 100, 300, 110, 30),
      mk_item("bereits ausgefuellt", 400, 300, 260, 30),
  };
  CHECK(detect_form_fields(page, text, {}).empty());
}

TEST_CASE("detect_form_fields keeps two labelled blanks on one line",
          "[forms]") {
  // The gap-followed-by-content rule must not break the ordinary
  // two-fields-per-line form, where the run after the gap is another label.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{
      mk_item("Vorname:", 60, 300, 130, 30),
      mk_item("Nachname:", 400, 300, 150, 30),
  };
  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 2);
  CHECK(fields[0].label == "Vorname:");
  CHECK(fields[1].label == "Nachname:");
}

TEST_CASE("detect_form_fields folds a label gap into a checkbox inside it",
          "[forms]") {
  // "Ja: [ ]" — the gap detector proposes the blank after the colon, which
  // contains the checkbox. The drawn square is the field; the inferred blank
  // is the same answer with less evidence.
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(260, 300, 28, 28));
  std::vector<OCRResultItem> text{mk_item("Ja:", 100, 300, 60, 30)};

  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].type == FieldType::Checkbox);
  CHECK(fields[0].label == "Ja:");
}

TEST_CASE("detect_form_fields skips tables with no cell geometry", "[forms]") {
  // A backend that returns HTML only leaves `cells` empty; the detector must
  // contribute nothing rather than invent a grid.
  cv::Mat page = blank_page();
  turbo_ocr::router::TableResult t;
  t.html = "<table><tr><td></td></tr></table>";
  t.box = mk_box(100, 100, 400, 200);
  const auto fields = detect_form_fields(page, {}, {t});
  for (const auto &f : fields)
    CHECK(f.source.find("table_cell") == std::string::npos);
}

TEST_CASE("detect_form_fields proposes empty cells when geometry is present",
          "[forms]") {
  cv::Mat page = blank_page();
  turbo_ocr::router::TableResult t;
  t.box = mk_box(100, 100, 400, 200);
  turbo_ocr::router::TableCell filled;
  filled.box = mk_box(100, 100, 200, 50);
  filled.text = "Menge";
  turbo_ocr::router::TableCell empty;
  empty.box = mk_box(300, 100, 200, 50);
  turbo_ocr::router::TableCell degenerate; // model gave this slot no box
  t.cells = {filled, empty, degenerate};

  std::vector<OCRResultItem> text{mk_item("Menge", 110, 110, 100, 30)};
  const auto fields = detect_form_fields(page, text, {t});
  int cell_fields = 0;
  for (const auto &f : fields)
    if (f.source.find("table_cell") != std::string::npos) ++cell_fields;
  CHECK(cell_fields == 1); // only the empty cell with real geometry
}

TEST_CASE("detect_form_fields returns nothing for a blank page", "[forms]") {
  CHECK(detect_form_fields(blank_page(), {}, {}).empty());
}

TEST_CASE("detect_form_fields returns nothing for an empty image", "[forms]") {
  CHECK(detect_form_fields(cv::Mat(), {}, {}).empty());
}

TEST_CASE("detect_form_fields emits fields in reading order", "[forms]") {
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{
      mk_item("Third:", 100, 500, 120, 30),
      mk_item("First:", 100, 100, 120, 30),
      mk_item("Second:", 100, 300, 120, 30),
  };
  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 3);
  CHECK(fields[0].label == "First:");
  CHECK(fields[1].label == "Second:");
  CHECK(fields[2].label == "Third:");
}

TEST_CASE("append_fields_array emits the documented shape", "[forms]") {
  std::vector<FormField> fields;
  FormField f;
  f.type = FieldType::Checkbox;
  f.box = mk_box(10, 20, 30, 40);
  f.label = "Ja \"oder\" nein";
  f.confidence = 0.9f;
  f.source = "checkbox+label_gap";
  fields.push_back(f);

  std::string j;
  append_fields_array(j, fields);
  CHECK(j ==
        R"(,"fields":[{"type":"checkbox","bounding_box":[[10,20],[40,20],[40,60],[10,60]],)"
        R"("label":"Ja \"oder\" nein","confidence":0.9,"source":"checkbox+label_gap"}])");
}

TEST_CASE("append_fields_array emits an empty array", "[forms]") {
  std::string j;
  append_fields_array(j, {});
  CHECK(j == ",\"fields\":[]");
}

// ── The model as a fifth detector ─────────────────────────────────────────
//
// FieldModel itself needs the ONNX weights, which are an optional download and
// absent in CI. What must hold regardless is how a model proposal behaves once
// it reaches detect_form_fields — so these drive the merge with hand-built
// proposals, exactly the shape FieldModel::run returns.

namespace {

FormField mk_model_field(FieldType type, int x, int y, int w, int h,
                         float conf = 0.75f) {
  FormField f;
  f.type = type;
  f.box = mk_box(x, y, w, h);
  f.confidence = conf;
  f.source = "ffdetr";
  return f;
}

} // namespace

TEST_CASE("model fields with no geometry agreement are still emitted",
          "[forms]") {
  // The case the model exists for: a blank with nothing drawn round it, on a
  // page whose label carries no colon. No geometry detector can see this.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{mk_item("Vorname", 100, 300, 120, 30)};

  const auto fields = detect_form_fields(page, text, {},
                                         {mk_model_field(FieldType::Text,
                                                         260, 298, 300, 34)});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source == "ffdetr");
  CHECK(fields[0].type == FieldType::Text);
}

TEST_CASE("a model field agreeing with a drawn box merges into it", "[forms]") {
  // Both saw the same widget. The drawn box keeps its pixel-exact edges — a
  // detector regressing normalised coordinates cannot match them — and the
  // agreement is recorded rather than emitted as a second overlapping field.
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(300, 400, 200, 40));

  const auto fields = detect_form_fields(page, {}, {},
                                         {mk_model_field(FieldType::Text,
                                                         296, 396, 208, 48)});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source.find("box") != std::string::npos);
  CHECK(fields[0].source.find("ffdetr") != std::string::npos);
  // The survivor is the drawn box's own geometry — its enclosed interior,
  // one pixel inside the 1px border at (300,400) — and NOT the model's
  // (296,396). Asserting the difference is the point: this is the reason the
  // merge keeps geometry's edges rather than the higher-scoring proposal's.
  const auto r = turbo_ocr::aabb(fields[0].box);
  CHECK(r[0] >= 300);
  CHECK(r[0] <= 302);
  CHECK(r[1] >= 400);
  CHECK(r[1] <= 402);
}

TEST_CASE("a merge never downgrades a specific type to text", "[forms]") {
  // Only the model can say Signature; the geometry under a signature line is
  // an ordinary rule. If the survivor kept its own Text the class would be
  // silently lost, and the PDF would get a text field where a signature
  // widget belongs.
  cv::Mat page = blank_page();
  draw_rule(page, 300, 500, 260);

  const auto fields = detect_form_fields(page, {}, {},
                                         {mk_model_field(FieldType::Signature,
                                                         300, 470, 260, 34)});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].type == FieldType::Signature);
  CHECK(fields[0].source.find("ffdetr") != std::string::npos);
}

TEST_CASE("passing no model fields is exactly the geometry-only result",
          "[forms]") {
  // The optional model must be optional: a server without the weights has to
  // behave bit-for-bit as it did before the model existed.
  cv::Mat page = blank_page();
  draw_rule(page, 300, 500, 260);
  draw_box(page, cv::Rect(300, 600, 28, 28));
  std::vector<OCRResultItem> text{mk_item("Unterschrift:", 100, 470, 180, 30)};

  const auto geometry_only = detect_form_fields(page, text, {});
  // Explicitly typed: a bare `{}` in the fourth position is ambiguous between
  // FieldOptions and the proposals vector. That is a compile error rather than
  // a silent wrong overload, so it needs no API change — but a call site is
  // clearer for saying which one it means.
  const auto with_empty =
      detect_form_fields(page, text, {}, std::vector<FormField>{});
  REQUIRE(geometry_only.size() == with_empty.size());
  for (size_t i = 0; i < geometry_only.size(); ++i) {
    CHECK(geometry_only[i].box == with_empty[i].box);
    CHECK(geometry_only[i].type == with_empty[i].type);
    CHECK(geometry_only[i].source == with_empty[i].source);
    CHECK(geometry_only[i].confidence == with_empty[i].confidence);
  }
}

TEST_CASE("a signature field serialises as its own type", "[forms]") {
  std::vector<FormField> fields{mk_model_field(FieldType::Signature,
                                               10, 20, 30, 40, 0.8f)};
  std::string j;
  append_fields_array(j, fields);
  CHECK(j.find(R"("type":"signature")") != std::string::npos);
}

TEST_CASE("an uncorroborated label gap is dropped when the model ran the page",
          "[forms]") {
  // The model proposed most of what survived, so it has looked at this page
  // properly — and it did not propose anything at "Bemerkung:". Measured on
  // the CommonForms test split, proposals sourced only "label_gap" on such
  // pages were 64 false positives and zero true positives.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{mk_item("Bemerkung:", 60, 700, 150, 30)};

  std::vector<FormField> model;
  for (int i = 0; i < 4; ++i)
    model.push_back(mk_model_field(FieldType::Text, 300, 100 + i * 80, 300, 34));

  const auto fields = detect_form_fields(page, text, {}, model);
  CHECK(fields.size() == 4);
  for (const auto &f : fields)
    CHECK(f.source != "label_gap");
}

TEST_CASE("an uncorroborated label gap SURVIVES when the model said little",
          "[forms]") {
  // The mirror case, and the reason the rule is a majority test rather than a
  // flat "prefer the model". On a plain form with nothing drawn on it the
  // model is out of its distribution and proposes almost nothing; reading that
  // silence as a veto would delete the only answer there is. Measured on such
  // a page: 8 of 10 fields are label_gap alone.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{
      mk_item("Name:", 60, 200, 100, 30),
      mk_item("Vorname:", 60, 300, 140, 30),
      mk_item("Strasse:", 60, 400, 130, 30),
  };

  const auto fields = detect_form_fields(
      page, text, {}, {mk_model_field(FieldType::Text, 300, 600, 300, 34)});

  size_t inferred = 0;
  for (const auto &f : fields)
    if (f.source == "label_gap") ++inferred;
  CHECK(inferred == 3);
}

TEST_CASE("deferring to the model can be switched off", "[forms]") {
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{mk_item("Bemerkung:", 60, 700, 150, 30)};
  std::vector<FormField> model;
  for (int i = 0; i < 4; ++i)
    model.push_back(mk_model_field(FieldType::Text, 300, 100 + i * 80, 300, 34));

  FieldOptions opt;
  opt.defer_inference_to_model = false;
  const auto fields = detect_form_fields(page, text, {}, model, opt);
  CHECK(fields.size() == 5);
}

TEST_CASE("deferral never fires without model proposals", "[forms]") {
  // A server with no weights must behave exactly as it did before the model
  // existed, however few fields the geometry finds.
  cv::Mat page = blank_page();
  std::vector<OCRResultItem> text{mk_item("Name:", 60, 200, 100, 30)};
  const auto fields = detect_form_fields(page, text, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source == "label_gap");
}

TEST_CASE("one blank described twice on a short rule merges to one field",
          "[forms]") {
  // "Datum: ____" on the reference form: the rule band sits at y 651..674 and
  // the label gap at y 656..676, so the gap covers only 0.78 of the rule
  // against a 0.80 containment threshold and the two used to survive as
  // separate fields. Filling the form then printed both values on top of each
  // other. The gap runs to the page margin by construction, so drawn ink it
  // horizontally contains on its own baseline is the blank it was guessing at.
  std::vector<FormField> fields;
  FormField rule;
  rule.box = mk_box(195, 651, 130, 23);
  rule.confidence = 0.78f;
  rule.source = "rule";
  FormField gap;
  gap.box = mk_box(157, 656, 650, 20);
  gap.confidence = 0.55f;
  gap.source = "label_gap";
  gap.label = "Datum:";
  fields = {rule, gap};

  merge_fields(fields, {});
  REQUIRE(fields.size() == 1);
  CHECK(fields[0].source == "rule+label_gap");
  CHECK(fields[0].label == "Datum:");
}

TEST_CASE("a checkbox on the same baseline is still not absorbed", "[forms]") {
  // What the area-ratio guard was actually protecting, and what the type test
  // must not weaken: a checkbox sitting inside a wide DRAWN blank clears both
  // same-line overlap ratios trivially, and is not that blank.
  //
  // The wide field is sourced "box", not "label_gap", on purpose — a label_gap
  // folding into a checkbox is separate, deliberate behaviour (see "folds a
  // label gap into a checkbox inside it"), and would mask what is tested here.
  std::vector<FormField> fields;
  FormField wide;
  wide.box = mk_box(157, 656, 650, 20);
  wide.confidence = 0.88f;
  wide.source = "box";
  FormField cb;
  cb.box = mk_box(300, 654, 24, 24);
  cb.confidence = 0.90f;
  cb.source = "checkbox";
  cb.type = FieldType::Checkbox;
  fields = {wide, cb};

  merge_fields(fields, {});
  REQUIRE(fields.size() == 2);
  CHECK(count_type(fields, FieldType::Checkbox) == 1);
  CHECK(count_type(fields, FieldType::Text) == 1);
}

TEST_CASE("a rectangle enclosing several fields is dropped as a container",
          "[forms]") {
  // A requisition form's sample table is genuinely EMPTY, so its outer border
  // passes the emptiness test and the box detector returns it: one 676x337
  // proposal enclosing the 30 cell fields. Filling that form printed a single
  // value in enormous type across the whole table. A thing that holds three
  // places to write is not itself a place to write.
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(100, 100, 600, 300));  // the table's outer border
  for (int i = 0; i < 4; ++i)                    // four empty cells inside it
    draw_box(page, cv::Rect(140, 140 + i * 60, 200, 40));

  const auto fields = detect_form_fields(page, {}, {});
  for (const auto &f : fields) {
    const cv::Rect r = box_to_rect(f.box);
    CHECK(r.width < 500); // the 600-wide container must not survive
  }
  CHECK(fields.size() >= 4); // its children must
}

TEST_CASE("the container rule leaves an ordinary blank alone", "[forms]") {
  // Two fields on a page, neither inside the other: nothing to drop.
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(100, 100, 300, 40));
  draw_box(page, cv::Rect(100, 300, 300, 40));
  const auto fields = detect_form_fields(page, {}, {});
  CHECK(fields.size() == 2);
}

TEST_CASE("the container rule can be switched off", "[forms]") {
  cv::Mat page = blank_page();
  draw_box(page, cv::Rect(100, 100, 600, 300));
  for (int i = 0; i < 4; ++i)
    draw_box(page, cv::Rect(140, 140 + i * 60, 200, 40));

  FieldOptions opt;
  opt.container_min_children = 0;
  const auto fields = detect_form_fields(page, {}, {}, std::vector<FormField>{},
                                         opt);
  bool kept_wide = false;
  for (const auto &f : fields)
    if (box_to_rect(f.box).width >= 500) kept_wide = true;
  CHECK(kept_wide);
}

TEST_CASE("a row of equal checkboxes becomes one choice run", "[forms]") {
  // "Oats / Barley / Wheat / Corn" — four equal boxes across one line are one
  // control in the document's mind, and a caller that knows the form is
  // pick-one can turn the run into a radio group. This reports the run only.
  std::vector<FormField> fields;
  for (int i = 0; i < 4; ++i) {
    FormField f;
    f.type = FieldType::Checkbox;
    f.box = mk_box(200 + i * 120, 300, 24, 24);
    f.source = "checkbox";
    fields.push_back(f);
  }
  group_choice_runs(fields);
  for (const auto &f : fields) CHECK(f.group == 0);
}

TEST_CASE("a column of equal checkboxes is also one run", "[forms]") {
  std::vector<FormField> fields;
  for (int i = 0; i < 3; ++i) {
    FormField f;
    f.type = FieldType::Checkbox;
    f.box = mk_box(200, 300 + i * 60, 24, 24);
    f.source = "checkbox";
    fields.push_back(f);
  }
  group_choice_runs(fields);
  for (const auto &f : fields) CHECK(f.group == 0);
}

TEST_CASE("unrelated checkboxes are separate runs", "[forms]") {
  // Different size AND different axis: not alternatives to one another.
  std::vector<FormField> fields;
  FormField a;
  a.type = FieldType::Checkbox;
  a.box = mk_box(200, 300, 24, 24);
  a.source = "checkbox";
  FormField b;
  b.type = FieldType::Checkbox;
  b.box = mk_box(600, 900, 60, 60);
  b.source = "checkbox";
  fields = {a, b};
  group_choice_runs(fields);
  CHECK(fields[0].group == -1);
  CHECK(fields[1].group == -1);
}

TEST_CASE("text fields never join a choice run", "[forms]") {
  // A place to type is not an alternative to anything, however well aligned.
  std::vector<FormField> fields;
  for (int i = 0; i < 3; ++i) {
    FormField f;
    f.type = FieldType::Text;
    f.box = mk_box(200 + i * 120, 300, 24, 24);
    f.source = "box";
    fields.push_back(f);
  }
  group_choice_runs(fields);
  for (const auto &f : fields) CHECK(f.group == -1);
}

TEST_CASE("a choice run is serialised, a lone checkbox is not", "[forms]") {
  std::vector<FormField> grouped;
  FormField g;
  g.type = FieldType::Checkbox;
  g.box = mk_box(10, 20, 24, 24);
  g.confidence = 0.9f;
  g.source = "checkbox";
  g.group = 2;
  grouped.push_back(g);
  std::string j;
  append_fields_array(j, grouped);
  CHECK(j.find("\"group\":2") != std::string::npos);

  g.group = -1;
  std::string j2;
  append_fields_array(j2, {g});
  CHECK(j2.find("\"group\"") == std::string::npos);
}
