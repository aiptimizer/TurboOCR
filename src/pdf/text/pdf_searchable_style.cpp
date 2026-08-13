// pdf_searchable_style.cpp — phases 1-3: what the DOCUMENT decides.
//
// Decode every kept word to codepoints and CIDs, vote on one typeface across
// every page at once, then snap the per-line sizes. All three see the whole
// document and no page: resolving any of them per page is what lets a page of
// headings drift away from the body around it, which is the drift the vote
// exists to stop.
//
// See pdf_searchable_detail.h for the whole split.

#include "pdf_searchable_detail.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <fpdf_annot.h>

#include "turbo_ocr/base/log/logger.h"
#include "pdf_text_internal.h"

namespace turbo_ocr::pdf {
namespace searchable_detail {

// Pass 1 (lock-free): decode every kept word once and assign each distinct
// codepoint a CID. Doing it up front means the font, its ToUnicode and its
// CIDToGIDMap are built exactly once per document.
DocumentRuns collect_runs(const std::vector<SearchablePage> &pages,
                          float min_confidence) {
  DocumentRuns out;
  out.page_runs.resize(pages.size());
  std::vector<unsigned> cps;

  for (size_t p = 0; p < pages.size(); ++p) {
    if (!pages[p].results) continue;
    auto &runs = out.page_runs[p];
    runs.reserve(pages[p].results->size());
    const auto &items = *pages[p].results;
    for (size_t i = 0; i < items.size(); ++i) {
      const auto &item = items[i];
      if (!keep(item, min_confidence)) continue;
      if (!decode_utf8(item.text, cps)) {
        ++out.dropped;
        continue;
      }
      for (unsigned cp : cps) {
        auto [it, inserted] = out.cid_of.try_emplace(cp, 0);
        if (inserted) it->second = static_cast<unsigned>(out.cid_of.size());
      }
      runs.push_back(Run{&item, i, std::vector<uint32_t>(cps.begin(), cps.end())});
    }
  }
  return out;
}

// The document decides its own type ONCE, from every line on every page at
// once. Resolving per page would let a page of headings pick a different
// family from the body around it — the very drift the vote exists to stop.
DocumentStyle resolve_document_style(const std::vector<SearchablePage> &pages,
                                     TextLayerMode mode) {
  DocumentStyle out;
  out.style_base.assign(pages.size(), SIZE_MAX);
  if (mode == TextLayerMode::Visible) {
    for (size_t p = 0; p < pages.size(); ++p) {
      const auto *st = pages[p].styles;
      if (!st || !pages[p].results || st->size() != pages[p].results->size())
        continue;
      out.style_base[p] = out.styles.size();
      out.styles.insert(out.styles.end(), st->begin(), st->end());
    }
  }

  // The document's typeface, voted over the pages that were matched. Weighted
  // by how convincing each page's match was, so a page of two short labels
  // cannot outvote a page of prose. The measured features still decide bold and
  // italic per line — a document-wide match cannot see those at all.
  FontFamily voted = FontFamily::Sans;
  bool have_vote = false;
  if (mode == TextLayerMode::Visible) {
    std::map<int, float> weight;
    for (size_t p = 0; p < pages.size(); ++p) {
      if (out.style_base[p] == SIZE_MAX) continue;
      const PageFontMatch &m = pages[p].font_match;
      if (m.score <= 0.0f) continue;
      weight[static_cast<int>(m.family)] += m.score;
    }
    float best_w = 0.0f;
    for (const auto &[fam, w] : weight) {
      if (w <= best_w) continue;
      best_w = w;
      voted = static_cast<FontFamily>(fam);
      have_vote = true;
    }
  }

  out.fonts = out.styles.empty()
                  ? std::vector<FontChoice>{}
                  : resolve_document_fonts(out.styles,
                                           have_vote ? &voted : nullptr);
  return out;
}

// What point size each run wants, snapped across the document.
//
// Measured in em-per-raster-pixel, which is scale free, so one snapped set
// serves every page whatever its size; the per-page conversion to points
// happens where the page geometry is known. Done for the WHOLE document
// before anything is drawn — that is the point of it, since a size that is
// snapped per page would still step at every page boundary.
//
// Needs PDFium (natural_ink_height sets trial type), so unlike the two phases
// above it runs with the document open and the library lock held.
void snap_document_sizes(FPDF_DOCUMENT doc, FontCache &fonts,
                         const std::vector<SearchablePage> &pages,
                         const DocumentRuns &runs, TextLayerMode mode,
                         DocumentStyle &style) {
  std::vector<float> want_em_px(style.styles.size(), 0.0F);
  if (mode == TextLayerMode::Visible) {
    for (size_t p = 0; p < pages.size(); ++p) {
      if (style.style_base[p] == SIZE_MAX) continue;
      for (const Run &run : runs.page_runs[p]) {
        const size_t k = style.style_base[p] + run.index;
        if (k >= style.fonts.size()) continue;
        const LineStyle &st = (*pages[p].styles)[run.index];
        if (!st.measured || !st.flat_paper || !spellable_in_standard14(run.cps))
          continue;
        FPDF_FONT f = fonts.get(style.fonts[k].standard_name());
        if (!f) continue;
        const float nat_h = natural_ink_height(doc, f, run.cps);
        if (nat_h <= 0.0F) continue;
        if (st.ink_h <= 0) continue;
        want_em_px[k] = static_cast<float>(st.ink_h) * kTrialSize / nat_h;
      }
    }
  }
  style.em_px = snap_line_sizes(want_em_px);
}

} // namespace searchable_detail
} // namespace turbo_ocr::pdf
