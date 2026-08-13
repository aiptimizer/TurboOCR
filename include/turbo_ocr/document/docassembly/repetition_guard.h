#pragma once

#include <string>

namespace turbo_ocr::docassembly {

// Truncate degenerate repetition in an autoregressive recognizer's output —
// the whitespace/phrase/line runaway documented in the CDM failure microscope
// (a decoder that locks on one token and emits it to the length cap). Direct
// port of PaddleX's truncate_repetitive_content (uilts.py).
//
// The pipeline calls this with min_count=5000 for table HTML and min_count=50
// for everything else, matching PaddleX.
//
// Operates on UTF-8 bytes rather than code points: the repetition detectors
// are byte-exact for any repeated unit (UTF-8 is self-synchronizing, so a
// repeated code-point run is a repeated byte run), and the only length
// thresholds (100, min_len) are coarse guards where byte vs code-point count
// makes no behavioral difference for real LaTeX/HTML.
std::string truncate_repetitive_content(const std::string& content,
                                        int line_threshold = 10,
                                        int char_threshold = 10,
                                        int min_len = 10,
                                        int min_count = 3000);

} // namespace turbo_ocr::docassembly
