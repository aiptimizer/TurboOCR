// Orchestration: run the four detectors, label what they found, reconcile the
// overlaps.
//
// The detectors WILL propose the same field more than once — a label with a
// rule under it is seen by both detector 1 and detector 3 — so the merge step
// is not a tidy-up, it is the difference between working and looking broken.

#include "turbo_ocr/analysis/forms/field_detector.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>

#include <opencv2/imgproc.hpp>

#include "forms_internal.h"

namespace turbo_ocr::forms {

namespace {

// Base confidence per detector, ordered by how much geometry each one had to
// see. A closed rectangle is the document explicitly drawing a place to write;
// a rule is nearly as good; a colon-plus-whitespace is an inference.
constexpr float kBoxConfidence = 0.88f;
constexpr float kCheckboxConfidence = 0.90f;
constexpr float kRuleConfidence = 0.78f;
constexpr float kTableCellConfidence = 0.70f;

// A page where the model accounts for MORE THAN THIS SHARE of the surviving
// fields is one where the model is clearly working, and therefore one where
// its silence about a particular spot is evidence rather than absence.
//
// This is the whole basis of defer_inference_to_model, and it exists because
// the measurements pull in opposite directions on the two kinds of page:
//
//   CommonForms test, 40 pages, IoU 0.5 — real forms whose authors drew real
//   widgets. Proposals sourced ONLY "label_gap" were 64 false positives and
//   ZERO true positives. Dropping them took overall F1 0.786 -> 0.820 with
//   recall unchanged to three decimals (0.817, the same 620 true positives).
//
//   A plain scanned form (labels, no drawn widgets at all): 8 of its 10 fields
//   are label_gap and nothing else. Dropping them would leave 2. Here the
//   heuristic is not noise, it is the entire answer.
//
// What separates the two is not the document type but HOW MUCH THE MODEL SAID:
// it sourced 76 of 79 fields on the first, and 2 of 10 on the second. A model
// that has barely spoken is out of its distribution, and reading its silence
// as a veto would delete the only answer available. A model that produced
// most of the page has looked at that blank and declined it.
constexpr double kModelMajority = 0.5;

// Agreement between independent detectors is the strongest signal available,
// since they share no inputs beyond the page itself.
constexpr float kAgreementBonus = 0.07f;
constexpr float kLabelledBonus = 0.04f;
constexpr float kMaxConfidence = 0.99f;

// Is this horizontal run the top or bottom border of a rectangle the box
// detector already found? Requires the run to sit at the box's own y AND to
// span most of its width, so a rule that merely passes near a box is still a
// rule.
[[nodiscard]] bool is_box_edge(const cv::Rect &rule,
                               const std::vector<cv::Rect> &boxes,
                               int edge_tol) {
  for (const cv::Rect &b : boxes) {
    const int overlap =
        std::min(rule.x + rule.width, b.x + b.width) - std::max(rule.x, b.x);
    if (overlap < 0.6 * b.width) continue;
    if (std::abs(rule.y - b.y) <= edge_tol) return true;
    if (std::abs(rule.y - (b.y + b.height)) <= edge_tol) return true;
  }
  return false;
}

[[nodiscard]] bool has_source(const std::string &joined,
                              std::string_view token) {
  size_t pos = 0;
  while ((pos = joined.find(token, pos)) != std::string::npos) {
    const bool left_ok = pos == 0 || joined[pos - 1] == '+';
    const size_t end = pos + token.size();
    const bool right_ok = end == joined.size() || joined[end] == '+';
    if (left_ok && right_ok) return true;
    pos = end;
  }
  return false;
}

} // namespace

float median_text_height(const std::vector<OCRResultItem> &text,
                         int page_height) {
  std::vector<int> heights;
  heights.reserve(text.size());
  for (const auto &item : text) {
    if (detail::trim_copy(item.text).empty()) continue;
    const int h = detail::item_rect(item).height;
    if (h > 0) heights.push_back(h);
  }
  if (heights.empty()) {
    // No text to measure. 1/55 of the page is about a 12pt line on A4 — close
    // enough to keep the kernels sane on a page that is all boxes and rules.
    return std::max(8.0f, static_cast<float>(page_height) / 55.0f);
  }
  const size_t mid = heights.size() / 2;
  std::ranges::nth_element(heights, heights.begin() + mid);
  return std::max(4.0f, static_cast<float>(heights[mid]));
}

Box rect_to_box(const cv::Rect &r) {
  const int x0 = r.x, y0 = r.y, x1 = r.x + r.width, y1 = r.y + r.height;
  return Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
}

cv::Rect box_to_rect(const Box &b) {
  const auto r = aabb(b);
  return cv::Rect(r[0], r[1], std::max(0, r[2] - r[0]),
                  std::max(0, r[3] - r[1]));
}

float box_iou(const Box &a, const Box &b) {
  const cv::Rect ra = box_to_rect(a), rb = box_to_rect(b);
  const cv::Rect inter = ra & rb;
  const double ia = static_cast<double>(inter.width) * inter.height;
  if (ia <= 0.0) return 0.0f;
  const double ua = static_cast<double>(ra.width) * ra.height +
                    static_cast<double>(rb.width) * rb.height - ia;
  return ua > 0.0 ? static_cast<float>(ia / ua) : 0.0f;
}

void merge_fields(std::vector<FormField> &fields, const FieldOptions &opt) {
  // Deterministic: the same page must merge the same way whatever order the
  // detectors happened to append in.
  std::ranges::stable_sort(fields, [](const FormField &a, const FormField &b) {
    if (a.confidence != b.confidence) return a.confidence > b.confidence;
    const cv::Rect ra = box_to_rect(a.box), rb = box_to_rect(b.box);
    if (ra.y != rb.y) return ra.y < rb.y;
    return ra.x < rb.x;
  });

  std::vector<FormField> kept;
  kept.reserve(fields.size());
  for (auto &cand : fields) {
    const cv::Rect rc = box_to_rect(cand.box);
    const double ac = static_cast<double>(rc.width) * rc.height;

    // label_gap is the one purely INFERENTIAL detector: it argues from a colon
    // and some whitespace, never from anything the document drew. So whenever
    // it wraps a proposal that came from real ink, the ink wins outright and
    // the guess is folded into it — no area-ratio guard, because the guess
    // deliberately runs to the page margin and will always look much larger
    // than the rule it contains.
    const bool cand_is_inferred = cand.source == "label_gap";

    FormField *dup = nullptr;
    for (auto &k : kept) {
      if (box_iou(k.box, cand.box) >= opt.merge_iou) {
        dup = &k;
        break;
      }
      // Containment: a rule field sitting inside the wider label-gap field is
      // the same blank described twice. Guarded by an area ratio so a checkbox
      // can never swallow a text field ten times its size — at that ratio they
      // are genuinely two different places to write.
      const cv::Rect rk = box_to_rect(k.box);
      const cv::Rect inter = rk & rc;
      const double ai = static_cast<double>(inter.width) * inter.height;
      if (ai <= 0.0) continue;
      const double ak = static_cast<double>(rk.width) * rk.height;
      const double smaller = std::min(ak, ac), larger = std::max(ak, ac);
      if (smaller <= 0.0) continue;
      // Vertical containment alone is not enough for the inferential guess.
      // "Datum: ____" gave a rule band at y 651..674 and a label gap at
      // y 656..676 — the same blank, offset by five pixels, so the gap covers
      // only 0.78 of the rule against a 0.80 threshold and the two survived as
      // separate fields. Filling the form then printed both values on top of
      // each other, which is what makes a prepared form look broken.
      //
      // So also accept the same-baseline reading: the gap runs to the page
      // margin by construction, so if it horizontally CONTAINS drawn ink on
      // its own line, that ink is the blank it was guessing at. Restricted to
      // the inferential candidate on purpose — the same latitude given to
      // every pair costs 35 true positives (see the same-line branch below).
      const double v_ovl = std::min(rk.height, rc.height);
      const double h_ovl = std::min(rk.width, rc.width);
      const bool same_baseline =
          v_ovl > 0.0 && h_ovl > 0.0 &&
          inter.height / v_ovl >= opt.same_line_v_overlap &&
          inter.width / h_ovl >= opt.same_line_h_overlap;
      if (cand_is_inferred && ak > 0.0 &&
          (ai / ak >= opt.contain_frac || same_baseline)) {
        dup = &k;
        break;
      }
      if (ai / smaller >= opt.contain_frac &&
          smaller / larger >= opt.contain_min_area_ratio) {
        dup = &k;
        break;
      }
      // Same baseline and overlapping horizontally. This is the pair IoU
      // misses: "Unterschrift: ______" gives a label+gap field running to the
      // page margin and a rule field stopping where the ink stops, which
      // scored 0.295 IoU on the reference form — just under the threshold, and
      // so was emitted twice. Measuring each overlap against the SMALLER
      // extent asks the right question: is the narrow, accurate proposal
      // inside the wide guess?
      // Carries the same area-ratio guard as containment, and for the same
      // reason: a checkbox sitting inside a wide blank satisfies both overlap
      // tests trivially, and it is not that blank.
      //
      // Do NOT relax this guard to "same type merges anyway". Measured: it
      // fixes a visible duplicate on one form and costs 35 true positives
      // across 40 real ones (recall 0.819 -> 0.773, overall F1 0.801 ->
      // 0.772), because two genuinely distinct same-type fields on one line —
      // a wide one and a narrow one inside its span — look identical to this
      // test. The narrower rule for the inferential detector above is what
      // handles the duplicate without that cost.
      const double v_min = std::min(rk.height, rc.height);
      const double h_min = std::min(rk.width, rc.width);
      if (v_min > 0.0 && h_min > 0.0 &&
          smaller / larger >= opt.contain_min_area_ratio &&
          inter.height / v_min >= opt.same_line_v_overlap &&
          inter.width / h_min >= opt.same_line_h_overlap) {
        dup = &k;
        break;
      }
    }

    if (dup == nullptr) {
      kept.push_back(std::move(cand));
      continue;
    }
    // The survivor keeps its own geometry (it had the stronger evidence) but
    // records that another detector independently argued for the same blank.
    if (!has_source(dup->source, cand.source)) {
      dup->source += '+';
      dup->source += cand.source;
      dup->confidence = std::min(kMaxConfidence, dup->confidence + kAgreementBonus);
    }
    if (dup->label.empty()) dup->label = std::move(cand.label);
    // TYPE, unlike geometry, does NOT follow the higher-confidence proposal.
    // Text is every detector's fallback — what it emits when it found a blank
    // and had nothing to say about what kind. Checkbox and Signature are
    // positive identifications: the box detector says Checkbox only for a
    // drawn near-square, and only the model can say Signature at all. So a
    // specific type always beats the default, whichever side carried it, or a
    // model-detected signature merging into a geometry rule would silently
    // come out as a plain text field.
    if (dup->type == FieldType::Text && cand.type != FieldType::Text)
      dup->type = cand.type;
  }

  for (auto &f : kept) {
    if (!f.label.empty())
      f.confidence = std::min(kMaxConfidence, f.confidence + kLabelledBonus);
  }
  fields = std::move(kept);
}

void group_choice_runs(std::vector<FormField> &fields,
                       const FieldOptions &opt) {
  if (opt.group_min_members < 2) return;

  // Indices of the checkboxes only. Text and signature fields are never part
  // of a choice run — a run is a set of alternatives, and a place to type is
  // not an alternative to anything.
  std::vector<size_t> idx;
  for (size_t i = 0; i < fields.size(); ++i)
    if (fields[i].type == FieldType::Checkbox) idx.push_back(i);
  if (idx.size() < static_cast<size_t>(opt.group_min_members)) return;

  std::vector<cv::Rect> r(fields.size());
  for (size_t i : idx) r[i] = box_to_rect(fields[i].box);

  // Same control iff same size AND lined up on one axis. Both are needed:
  // size alone would join every checkbox on the page, and alignment alone
  // would join a checkbox to an unrelated one that happens to share a row.
  const auto similar_size = [&](size_t a, size_t b) {
    const double wa = r[a].width, wb = r[b].width;
    const double ha = r[a].height, hb = r[b].height;
    if (wa <= 0 || wb <= 0 || ha <= 0 || hb <= 0) return false;
    return std::abs(wa - wb) / std::max(wa, wb) <= opt.group_size_tol &&
           std::abs(ha - hb) / std::max(ha, hb) <= opt.group_size_tol;
  };
  // Alignment is tested on ONE axis at a time, and the two axes are clustered
  // in separate passes. Allowing "same row OR same column" inside one
  // transitive closure collapses an entire grid into a single run: a box
  // shares a row with its neighbour, that neighbour shares a column with the
  // box below, and the chain runs through the whole block. Measured on a
  // requisition form, that produced two runs of 12 and 20 boxes where the page
  // plainly has several short ones.
  const auto same_row = [&](size_t a, size_t b) {
    const double tol =
        opt.group_align_tol * std::max(1, std::min(r[a].height, r[b].height));
    return std::abs((r[a].y + r[a].height / 2.0) -
                    (r[b].y + r[b].height / 2.0)) <= tol;
  };
  const auto same_col = [&](size_t a, size_t b) {
    const double tol =
        opt.group_align_tol * std::max(1, std::min(r[a].width, r[b].width));
    return std::abs((r[a].x + r[a].width / 2.0) -
                    (r[b].x + r[b].width / 2.0)) <= tol;
  };

  // Transitive within an axis, so three boxes across a row group together even
  // though the outer two are only related through the middle one.
  const auto cluster = [&](const std::vector<size_t> &members,
                           const auto &alike) {
    std::map<size_t, size_t> parent;
    for (size_t i : members) parent[i] = i;
    const std::function<size_t(size_t)> find = [&](size_t x) -> size_t {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (size_t a = 0; a < members.size(); ++a)
      for (size_t b = a + 1; b < members.size(); ++b)
        if (similar_size(members[a], members[b]) &&
            alike(members[a], members[b]))
          parent[find(members[a])] = find(members[b]);

    std::map<size_t, std::vector<size_t>> runs;
    for (size_t i : members) runs[find(i)].push_back(i);
    return runs;
  };

  int next = 0;
  // Rows first: a printed form lays alternatives out left to right far more
  // often than top to bottom ("Ja  Nein"), so a box that belongs to both a row
  // and a column run is reported as part of its row.
  std::vector<size_t> leftover;
  for (auto &[root, members] : cluster(idx, same_row)) {
    if (members.size() < static_cast<size_t>(opt.group_min_members)) {
      leftover.insert(leftover.end(), members.begin(), members.end());
      continue;
    }
    for (size_t i : members) fields[i].group = next;
    ++next;
  }
  std::ranges::sort(leftover);
  for (auto &[root, members] : cluster(leftover, same_col)) {
    if (members.size() < static_cast<size_t>(opt.group_min_members)) continue;
    for (size_t i : members) fields[i].group = next;
    ++next;
  }
}

std::vector<FormField>
detect_form_fields(const cv::Mat &page, const std::vector<OCRResultItem> &text,
                   const std::vector<router::TableResult> &tables,
                   const FieldOptions &opt) {
  return detect_form_fields(page, text, tables, {}, opt);
}

std::vector<FormField>
detect_form_fields(const cv::Mat &page, const std::vector<OCRResultItem> &text,
                   const std::vector<router::TableResult> &tables,
                   std::vector<FormField> model_fields,
                   const FieldOptions &opt) {
  std::vector<FormField> out;
  if (page.empty()) return out;

  const float text_h = median_text_height(text, page.rows);
  const cv::Mat bin = binarize_page(page);
  if (bin.empty()) return out;
  const cv::Rect page_rect(0, 0, page.cols, page.rows);

  // ── Detector 2: closed boxes and checkboxes ─────────────────────────────
  // Runs FIRST because detector 1 needs to know which horizontal runs are
  // already spoken for. A short kernel: a checkbox side is about one text
  // height, and a kernel longer than that erases the checkbox's own edges.
  const int box_k =
      std::max(5, static_cast<int>(std::lround(opt.box_kernel * text_h)));
  const LineMasks box_masks = extract_line_masks(bin, box_k, box_k);
  const std::vector<cv::Rect> closed_boxes =
      find_closed_boxes(box_masks, text_h, opt);

  // ── Detector 1: ruling lines ────────────────────────────────────────────
  // A long kernel, so no glyph stroke can survive into the rule mask.
  {
    const int k = std::max(9, static_cast<int>(std::lround(opt.rule_kernel * text_h)));
    const LineMasks rule_masks = extract_line_masks(bin, k, /*v_kernel=*/0);
    const double band_h =
        std::max(2.0, static_cast<double>(opt.rule_field_height) * text_h);
    const int edge_tol =
        std::max(3, static_cast<int>(std::lround(0.2 * text_h)));
    for (const cv::Rect &rule : find_rule_segments(rule_masks.horizontal,
                                                   text_h, opt)) {
      // A box's own TOP border is a long horizontal run with clean paper above
      // it, so it looks exactly like a write-on line — and would put a phantom
      // field in the gap ABOVE every bordered box on the page (4 of them on
      // the box fixture). The box detector has already claimed that geometry.
      if (is_box_edge(rule, closed_boxes, edge_tol)) continue;

      // The writable band stands ON the rule, one text-height tall.
      cv::Rect band(rule.x, rule.y - static_cast<int>(std::lround(band_h)),
                    rule.width, static_cast<int>(std::lround(band_h)));
      band &= page_rect;
      if (band.width < opt.min_field_width * text_h ||
          band.height < opt.min_field_height * text_h)
        continue;
      // Whitespace above is what separates a write-on line from the underline
      // of a heading or the top border of a populated table row.
      if (detail::band_ink_fraction(bin, band) > opt.max_ink_above) continue;
      if (!detail::rect_is_empty(band, text, opt.max_text_overlap)) continue;

      FormField f;
      f.box = rect_to_box(band);
      f.confidence = kRuleConfidence;
      f.source = "rule";
      out.push_back(std::move(f));
    }
  }

  {
    for (const cv::Rect &r : closed_boxes) {
      if (!detail::rect_is_empty(r, text, opt.max_text_overlap)) continue; // a filled cell is content
      const double w = r.width, h = r.height;
      const double aspect = std::max(w / h, h / w);
      const bool square_ish = aspect <= opt.checkbox_aspect;
      const bool small = w <= opt.checkbox_max_side * text_h &&
                         h <= opt.checkbox_max_side * text_h;

      FormField f;
      f.box = rect_to_box(r);
      if (square_ish && small) {
        f.type = FieldType::Checkbox;
        f.confidence = kCheckboxConfidence;
        f.source = "checkbox";
      } else {
        if (w < opt.min_field_width * text_h || h < opt.min_field_height * text_h)
          continue;
        f.confidence = kBoxConfidence;
        f.source = "box";
      }
      out.push_back(std::move(f));
    }
  }

  // ── Detector 3: label then gap ──────────────────────────────────────────
  detail::collect_label_gap_fields(text, page.cols, text_h, opt, out);

  // ── Detector 4: empty table cells ───────────────────────────────────────
  // Self-disabling: a backend with no cell geometry (a remote VLM returns HTML
  // only) leaves `cells` empty, and this contributes nothing rather than
  // guessing a grid.
  for (const auto &t : tables) {
    for (const auto &cell : t.cells) {
      if (!detail::trim_copy(cell.text).empty()) continue;
      const cv::Rect r = box_to_rect(cell.box) & page_rect;
      if (r.width < opt.min_field_width * text_h ||
          r.height < opt.min_field_height * text_h)
        continue; // degenerate/zero box for a slot the model gave no geometry
      if (!detail::rect_is_empty(r, text, opt.max_text_overlap)) continue;
      FormField f;
      f.box = rect_to_box(r);
      f.confidence = kTableCellConfidence;
      f.source = "table_cell";
      out.push_back(std::move(f));
    }
  }

  // ── Detector 5: the model ───────────────────────────────────────────────
  // Appended raw. Deliberately NOT put through rect_is_empty like the four
  // geometry detectors: emptiness is the proxy those detectors use for "is
  // this a place to write", and applying it here would re-impose the exact
  // assumption the model exists to get past — a widget drawn over a shaded
  // panel, or one whose caption sits inside its own rectangle, is still a
  // widget. Its own confidence is what ranks it against the others.
  const bool had_model_fields = !model_fields.empty();
  out.insert(out.end(), std::make_move_iterator(model_fields.begin()),
             std::make_move_iterator(model_fields.end()));

  // Label BEFORE merging: detector 3 already knows its own label, the other
  // four do not, and merge_fields scores a labelled field higher. Doing it
  // here means that bonus is decided on the same basis for all of them.
  for (auto &f : out) {
    if (f.label.empty())
      f.label = find_label(box_to_rect(f.box), text, text_h, opt,
                           f.type == FieldType::Checkbox);
  }

  merge_fields(out, opt);

  // Drop CONTAINERS. merge_fields reconciles proposals that describe the same
  // blank; this removes the different mistake of a rectangle drawn around
  // several blanks — an empty table's outer border, a ruled panel enclosing a
  // column of fields. Neither IoU nor containment catches it, because the
  // container genuinely does not duplicate any single one of its children.
  //
  // Runs after the merge so the count is of SURVIVING fields: counting raw
  // proposals would let a wide blank that several detectors argued for look
  // like a container of its own duplicates.
  if (opt.container_min_children > 0 &&
      out.size() > static_cast<size_t>(opt.container_min_children)) {
    std::vector<cv::Rect> rects;
    rects.reserve(out.size());
    for (const auto &f : out) rects.push_back(box_to_rect(f.box));

    std::vector<bool> is_container(out.size(), false);
    for (size_t i = 0; i < out.size(); ++i) {
      int children = 0;
      for (size_t j = 0; j < out.size() && children < opt.container_min_children;
           ++j) {
        if (i == j) continue;
        const cv::Rect inter = rects[i] & rects[j];
        const double aj =
            static_cast<double>(rects[j].width) * rects[j].height;
        if (aj <= 0.0) continue;
        const double covered =
            static_cast<double>(inter.width) * inter.height / aj;
        // Measured against the CHILD's area: "is that whole field inside this
        // one", which is the question, and which a wide neighbour merely
        // overlapping cannot satisfy.
        if (covered >= opt.contain_frac) ++children;
      }
      is_container[i] = children >= opt.container_min_children;
    }
    size_t w = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      if (is_container[i]) continue;
      // The w != i guard is load-bearing: with no containers on the page every
      // element would otherwise be move-assigned ONTO ITSELF, which for the
      // std::string members is valid but leaves them unspecified — in practice
      // empty. Every field's `source` silently became "", which then defeated
      // the label_gap rules downstream.
      if (w != i) out[w] = std::move(out[i]);
      ++w;
    }
    out.resize(w);
  }

  // Defer to the model where it has earned it — see kModelMajority. Only
  // "label_gap" alone is dropped: it is the one detector that argues purely
  // from a colon and some whitespace, never from ink the document drew. A
  // solo "rule" or "box" survives, because a printed line IS evidence, and
  // measured it does catch fields the model misses (dropping those too cost
  // recall for almost no precision: 0.820 -> 0.821 F1).
  if (opt.defer_inference_to_model && had_model_fields && !out.empty()) {
    const auto backed = static_cast<double>(std::ranges::count_if(
        out, [](const FormField &f) { return has_source(f.source, "ffdetr"); }));
    if (backed > kModelMajority * static_cast<double>(out.size()))
      std::erase_if(out,
                    [](const FormField &f) { return f.source == "label_gap"; });
  }

  // Reading order, so a client can walk the fields the way the page reads.
  std::ranges::stable_sort(out, [](const FormField &a, const FormField &b) {
    const cv::Rect ra = box_to_rect(a.box), rb = box_to_rect(b.box);
    if (ra.y != rb.y) return ra.y < rb.y;
    return ra.x < rb.x;
  });

  // Last, so both act on the rectangles that actually ship.
  trim_fields_off_text(out, text, opt);
  name_fields_from_columns(out, text, opt);

  group_choice_runs(out, opt);
  return out;
}

} // namespace turbo_ocr::forms
