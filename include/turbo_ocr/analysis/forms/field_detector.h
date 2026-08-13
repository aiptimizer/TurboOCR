#pragma once

// Fillable-field proposal from page geometry — the "Prepare Form" capability.
//
// These four detectors read the raster the way a person does: a label, and
// then somewhere to write. They are HALF the answer. The other half is
// forms::FieldModel (FFDetr, see field_model.h), which is trained on real
// fillable PDFs and finds blanks that nothing was drawn around; it enters
// through the five-argument detect_form_fields below and is reconciled by the
// same merge step. Everything here still runs, and still runs alone when the
// model's weights are absent, because the two are right about different pages:
// morphology gives pixel-exact edges wherever the document actually drew the
// blank, which a detector regressing normalised coordinates does not.
//
// Four independent geometry detectors argue for rectangles:
//
//   rule        a long thin horizontal run with whitespace above it — the
//               printed underline in "Unterschrift: ______". OCR does NOT
//               read those as underscore characters (verified: it returns
//               only the label), so morphology is the only way to see them.
//   box         a closed rectangle from the horizontal ∩ vertical line masks,
//               with no OCR text inside. A small near-square one is a
//               CHECKBOX — a type no text-only heuristic can find.
//   label_gap   an OCR line ending in ':' followed by enough blank space on
//               the same baseline.
//   table_cell  an empty grid cell, when the table stage supplied cell
//               geometry. Self-disabling when it did not.
//
// SCALE: every threshold below is a MULTIPLE OF THE PAGE'S MEDIAN TEXT
// HEIGHT, never a pixel count, so one set of numbers holds at 100 and 300 dpi.

#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/types.h"
#include "turbo_ocr/analysis/forms/form_field.h"
#include "turbo_ocr/core/router_types.h"

namespace turbo_ocr::forms {

struct FieldOptions {
  // ── rule detector ──
  // Open kernel that keeps horizontal runs. Long enough that no glyph stroke
  // survives it; a printed write-on rule is many text-heights long.
  float rule_kernel = 1.5f;
  float min_rule_len = 2.5f;        // shorter runs are dashes/strikethroughs
  float max_rule_thickness = 0.35f; // thicker is a filled bar, not a rule
  float rule_field_height = 1.15f;  // the writable band standing on the rule
  float max_ink_above = 0.035f;     // ink fraction that still counts as blank

  // ── box detector ──
  // A checkbox side is about one text-height, so the box masks must use a
  // kernel SHORTER than that or the checkbox's own edges get erased. Letter
  // stems survive a kernel this short; the four-edge coverage test below is
  // what rejects them, not the kernel.
  float box_kernel = 0.4f;
  float min_edge_coverage = 0.65f;  // fraction of each side that must be line
  float min_box_side = 0.45f;
  float checkbox_max_side = 2.2f;   // above this a square box is a text field
  float checkbox_aspect = 1.6f;     // max of w/h and h/w to still be "square"

  // ── label + gap detector ──
  float min_gap = 1.8f;             // narrower blanks are just word spacing
  float max_field_width = 40.0f;    // cap so one field can't run off the page
  float same_line_tol = 0.6f;       // centre distance, in line heights
  float right_margin = 0.95f;       // where a trailing blank ends on the page
  // Require the gap to be followed by blank paper or another label, never by
  // content. "Status: bereits ausgefuellt" has a wide gap after the colon, but
  // the entry is filled; a field there would offer to overwrite it.
  bool require_blank_after_gap = true;

  // ── labelling ──
  float label_max_left_gap = 9.0f;  // how far left to look for a label
  float label_max_above_gap = 1.6f; // and how far up

  // A proposed blank must not sit on recognised text. Fraction of the
  // candidate's own area that may intersect an OCR box before it stops being
  // blank — small but non-zero, so a field can graze the label beside it.
  float max_text_overlap = 0.15f;

  // ── merge ──
  float merge_iou = 0.30f;
  float contain_frac = 0.80f;       // inter / smaller-area that counts as dup
  float contain_min_area_ratio = 0.25f; // below this they are separate fields
  // Two proposals on the SAME baseline that also overlap horizontally are one
  // blank described twice — the case IoU misses, because a label+gap runs to
  // the page margin while the rule under it stops where the ink stops. Both
  // fractions are measured against the SMALLER extent, so a narrow accurate
  // rule still recognises the wide guess as itself.
  float same_line_v_overlap = 0.60f;
  float same_line_h_overlap = 0.60f;

  // Fields smaller than this (in text heights) are dropped as noise.
  float min_field_width = 0.7f;
  float min_field_height = 0.5f;

  // ── choice-button runs ──
  // A row or column of checkboxes counts as one run when they are the same
  // size to within this fraction and their centres line up to within this
  // many of their own heights. Two is enough to be a run: "Ja / Nein" is the
  // commonest control on a printed form.
  float group_size_tol = 0.35f;
  float group_align_tol = 0.6f;
  int group_min_members = 2;

  // ── containers ──
  // A proposal that holds at least this many OTHER surviving proposals is the
  // rectangle drawn AROUND several fields, not a field. The case that forced
  // this: a requisition form whose sample table is genuinely empty, so its
  // outer border passes the emptiness test and comes back as one 676x337
  // proposal — 22% of the page — enclosing the 30 cell fields inside it.
  // Filling that form printed one value in 200pt type across the whole table.
  //
  // 3 rather than 2 because two nested proposals can be a labelling artefact,
  // whereas nothing that holds three separate places to write is itself one
  // place to write. 0 disables the check.
  int container_min_children = 3;

  // ── deferring to the model ──
  // Drop a label_gap proposal that NOTHING corroborated, on pages where the
  // model did the bulk of the work. See the comment on kModelMajority in
  // field_detector.cpp for the measurements; set false to emit every proposal
  // regardless. Has no effect when no model proposals were supplied.
  bool defer_inference_to_model = true;
};

// ── Public entry point ────────────────────────────────────────────────────
//
// `page` is the page raster (BGR or gray). `text` is that page's OCR (may be
// empty — the rule and box detectors still work, fields just come back
// unlabelled). `tables` may be empty, and cells-less tables are skipped, so
// this never depends on the table stage having run.
[[nodiscard]] std::vector<FormField>
detect_form_fields(const cv::Mat &page, const std::vector<OCRResultItem> &text,
                   const std::vector<router::TableResult> &tables,
                   const FieldOptions &opt = {});

// Same, plus proposals from forms::FieldModel (FFDetr) as a fifth source.
//
// The model enters as a PEER of the four geometry detectors, not as a
// replacement and not as a filter over them, because the two disagree in both
// directions and each is right about something the other cannot see:
//
//   only geometry   a bare "Name:" with whitespace after it. Nothing is drawn,
//                   so the model — trained on real drawn widgets — reads the
//                   page as having no field there. A human filling the form in
//                   would disagree.
//   only the model  a blank in a shaded panel with no rule under it, a
//                   checkbox drawn as a glyph rather than a rectangle, or a
//                   form whose labels do not end in a colon. No morphology
//                   sees these.
//   both            merge_fields folds them into one field sourced
//                   "ffdetr+box", and the geometry keeps the pixel-exact
//                   edges a normalised-coordinate regressor does not have.
//
// `model_fields` may be empty, which is exactly the three-argument behaviour.
[[nodiscard]] std::vector<FormField>
detect_form_fields(const cv::Mat &page, const std::vector<OCRResultItem> &text,
                   const std::vector<router::TableResult> &tables,
                   std::vector<FormField> model_fields,
                   const FieldOptions &opt = {});

// ── Pieces, exposed for testing ───────────────────────────────────────────

// Median height of the OCR boxes — the scale unit for every threshold. Falls
// back to a fraction of the page height when there is no text to measure.
[[nodiscard]] float median_text_height(const std::vector<OCRResultItem> &text,
                                       int page_height);

// Ink = 255. Otsu, with a guard for the pathological case where more than half
// the page comes back as ink (a near-blank page splits its own noise).
[[nodiscard]] cv::Mat binarize_page(const cv::Mat &page);

struct LineMasks {
  cv::Mat horizontal, vertical;
};

// Morphological open with a 1-D kernel in each direction. Kernel lengths are
// PIXELS here (the caller scales them by text height) because the rule pass
// and the box pass deliberately use different ones. A length <= 0 skips that
// direction, leaving its mask empty — the rule pass wants only the horizontal
// one and this is a whole full-page morphology per page not to do.
[[nodiscard]] LineMasks extract_line_masks(const cv::Mat &binary, int h_kernel,
                                           int v_kernel);

// Long thin horizontal runs from a horizontal mask, as page rects.
[[nodiscard]] std::vector<cv::Rect>
find_rule_segments(const cv::Mat &horizontal, float text_h,
                   const FieldOptions &opt);

// Closed rectangles: enclosed regions of (horizontal | vertical), each
// verified to have real line coverage on all four of its own edges — which is
// what separates a drawn box from the counter of an 'o'.
[[nodiscard]] std::vector<cv::Rect>
find_closed_boxes(const LineMasks &masks, float text_h,
                  const FieldOptions &opt);

// Band OCR items into visual lines (detection order is not reading order),
// each sorted left-to-right. Indices into `text`.
[[nodiscard]] std::vector<std::vector<int>>
group_into_lines(const std::vector<OCRResultItem> &text,
                 const FieldOptions &opt);

// "Name:" — a trailing colon is the clearest thing a printed form gives us.
[[nodiscard]] bool is_label_text(std::string_view text);

// A blank drawn with characters, on the occasions OCR does read it that way.
[[nodiscard]] bool is_rule_text(std::string_view text);

// Nearest plausible label for a rect.
//
// `label_follows` picks which side to read first, and it is not a preference —
// the two kinds of field are labelled from opposite directions. A blank to
// write in is named by what comes BEFORE it ("Name: ______"); a tick box is
// named by what it turns on, which is written AFTER it ("[ ] Oats"). Reading a
// checkbox leftwards takes the previous option's name, so a row of them comes
// out labelled one to the left all the way along: the box before "Oats" called
// "A. Grain", the box before "Barley" called "Oats".
//
// Falls back to the other side, then to the line above, so a checkbox with
// nothing to its right still gets a name.
[[nodiscard]] std::string find_label(const cv::Rect &field,
                                     const std::vector<OCRResultItem> &text,
                                     float text_h, const FieldOptions &opt,
                                     bool label_follows = false);

// Pulls field rectangles off any printed word they overlap.
//
// A detector's box is a little larger than the mark it found, and next to a
// tick box the very next thing on the line is usually its label — so the widget
// lands on the first letter of the word and the page reads as if the box were
// sitting on the text. Trimming is safe in a way that moving is not: the part
// being removed is the part that was never the field.
void trim_fields_off_text(std::vector<FormField> &fields,
                          const std::vector<OCRResultItem> &text,
                          const FieldOptions &opt = {});

// Names the fields that nothing adjacent explains, from the column they sit in.
//
// An empty cell in the middle of a grid has no label beside it and none
// directly above — its header is at the top of the column, several rows away.
// Left unnamed those fields come out as "field_25", which is what a form filler
// then shows the person filling it in. Walking up the column instead gives
// "Sample ID 3": the header, and which row of it this is.
//
// Deliberately a LAST resort, after the adjacent search has failed. Reaching
// this far for a label is right only when the alternative is no label at all.
void name_fields_from_columns(std::vector<FormField> &fields,
                              const std::vector<OCRResultItem> &text,
                              const FieldOptions &opt = {});

// Tag aligned, equally-sized runs of checkboxes with a shared `group` index.
// Reports the RUN, never exclusivity — see FormField::group for why.
void group_choice_runs(std::vector<FormField> &fields,
                       const FieldOptions &opt = {});

[[nodiscard]] float box_iou(const Box &a, const Box &b);
[[nodiscard]] Box rect_to_box(const cv::Rect &r);
[[nodiscard]] cv::Rect box_to_rect(const Box &b);

// Collapse overlapping proposals in place. Input order does not matter: the
// candidates are sorted by (confidence desc, y, x) first, so the same page
// always merges the same way. The survivor keeps its own geometry, records
// every source that argued for it, and gains confidence for the agreement.
void merge_fields(std::vector<FormField> &fields, const FieldOptions &opt);

} // namespace turbo_ocr::forms
