#pragma once

// Identifies the typeface a scan was set in by SETTING IT AGAIN.
//
// Every candidate font renders the words the OCR read, and the rendering is
// compared, pixel against pixel, with the crop those words came from. The
// closest match wins. That is the whole idea, and it is a different idea from
// measuring features: it never has to name what makes Georgia Georgia, because
// it is looking at Georgia while it decides.
//
// Why the change. The first attempt measured hand-built features — stem-foot
// spread, stroke weight, slant — and mapped them to serif/sans/mono. Evaluated
// against 226 faces installed on a real machine, rendered at scan resolution
// with blur, sensor noise and JPEG, it read Times New Roman and Georgia as
// SANS, and scored 53% on serif faces. Its 100% on sans was hollow: sans is
// what it answers when it has no opinion. Blur closes the gap between a serif
// and its stem, and no threshold on that gap survives it. Comparing whole
// shapes does, because blur degrades the candidate render and the scan alike.
//
// The catalogue is deliberately small and deliberately licence-clean: the PDF
// standard-14, which every reader must provide and which need no embedding at
// all. Extra faces can be added, but only ones that may legally be embedded in
// and redistributed with the output.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/types.h"
#include "turbo_ocr/pdf/text/font_style.h"

namespace turbo_ocr::pdf {

// One typeface the matcher may choose.
struct FontCandidate {
  // What to call it in a report.
  std::string name;
  // PDF standard-14 name. Non-empty means the output needs no embedded font
  // file at all, because every conforming reader already has these.
  std::string standard14;
  // Path to a font file, used when standard14 is empty. Handles .ttf, .otf and
  // .ttc collections; `face` selects within a collection.
  //
  // Detecting against a font and EMBEDDING it are different permissions.
  // Anything may be used to recognise what a scan was set in — that is reading,
  // not copying — but only a font whose licence allows embedding and
  // redistribution may then be written into the output. Keep that distinction
  // when adding entries: `embeddable` is the flag that decides.
  std::string file;
  int face = 0;
  bool embeddable = false;

  FontFamily family = FontFamily::Sans;
  bool bold = false;
  bool italic = false;
};

// Reads a font file into the bytes PDFium wants, extracting a single face from
// a .ttc collection when needed. Returns empty on failure.
[[nodiscard]] std::vector<uint8_t> load_font_bytes(const std::string &path,
                                                   int face);

// A line of the document, as evidence. `crop` is the upright, ink-tight image
// of the line; `text` is what the recogniser read there.
struct FontSample {
  cv::Mat crop;
  std::string text;
};

struct FontMatch {
  // Index into the catalogue, or -1 when nothing could be matched — too few
  // usable lines, no renderable text, or every candidate scoring at chance.
  int index = -1;
  // Mean shape agreement of the winner, 0 to 1, BEFORE any family prior is
  // applied — so it measures the shapes alone and can be used as a weight
  // without counting the prior twice. Measured: the same words in the same face
  // score 0.84, a real but imperfect match 0.49 and up, and a face that cannot
  // render them at all about 0.20. A winner scoring under 0.35 is rejected
  // outright and reported as index -1.
  float score = 0.0F;
  // How far ahead of the runner-up, in the same units. A large score with a
  // tiny margin means several faces fit equally well, which is the honest
  // answer for a short, blurred sample.
  float margin = 0.0F;
};

// The shipped catalogue: standard-14 text faces, in all four styles each.
[[nodiscard]] const std::vector<FontCandidate> &standard_font_catalogue();

// What the measured-feature estimator thinks, offered to the matcher as a
// prior rather than as an answer.
//
// The two disagree in almost disjoint places, which is the whole reason to
// combine them. Measured on 90 verified text faces: shape matching alone gets
// Times New Roman, Georgia and every monospaced face right where the feature
// estimator misses them, and the feature estimator gets the slab serifs —
// Rockwell, Superclarendon, American Typewriter — right where shape matching
// misses them, because the catalogue holds no slab for those to match against.
// Only two faces defeat both.
struct FamilyPrior {
  FontFamily family = FontFamily::Sans;
  // 0 leaves the shape scores untouched. Higher lets the prior break a tie
  // without letting it overrule a clear shape win.
  float strength = 0.0F;
};

// Picks the closest candidate. Samples with empty crops or text the candidate
// cannot spell are skipped; if that leaves nothing, the result is index -1.
[[nodiscard]] FontMatch match_font(const std::vector<FontSample> &samples,
                                   const std::vector<FontCandidate> &catalogue,
                                   FamilyPrior prior = {});

// Runs the matcher over one page's recognised lines and reports the family it
// chose. Picks its own sample of lines — the longest and cleanest — because a
// three-word label carries far less evidence than a full line of prose, and
// comparing every line on a dense page would cost more than it adds.
[[nodiscard]] PageFontMatch match_page_family(
    const std::vector<OCRResultItem> &results, const cv::Mat &page,
    const std::vector<LineStyle> &styles);

// How closely `text` set in `candidate` resembles `crop`, from 0 to 1.
// Exposed for tests and for tuning; match_font is the normal entry point.
[[nodiscard]] float shape_agreement(const cv::Mat &crop, const std::string &text,
                                    const FontCandidate &candidate);

} // namespace turbo_ocr::pdf
