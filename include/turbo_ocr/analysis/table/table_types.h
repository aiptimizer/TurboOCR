#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace turbo_ocr::table {

// SLANeXt vocab size after merge_no_span_structure=True expansion. Includes
// sos + eos.
inline constexpr std::size_t SLANEXT_VOCAB = 50;

// Output of TableCls — selects which wired/wireless variant of cell-det and
// SLANeXt to run for a given table region.
enum class TableVariant : std::uint8_t {
  Wired = 0,
  Wireless = 1,
};

// One <td>-family cell with its 8-coord quad in original-image pixels.
// Layout: [x1, y1, x2, y2, x3, y3, x4, y4].
struct StructureCell {
    // NOTE (removed): token_idx. Written by the decoder, read by nothing —
    // cell_matcher pairs cells to lines by geometry, not by token position.
    std::array<int, 8> bbox{};
};

// Full SLANeXt decode result for one sample.
struct StructureResult {
    // HTML token sequence wrapped with <html><body><table> / </table></body></html>.
    std::vector<std::string> structure;
    // Per <td>-family cell quad in original image space.
    std::vector<StructureCell> cells;
    // Mean of per-step argmax softmax scores across kept tokens.
    float structure_score = 0.0f;
};

} // namespace turbo_ocr::table
