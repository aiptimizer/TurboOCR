#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace turbo_ocr::table {

// Bundled default dict text — matches table_structure_dict_ch.txt from the
// PP-Structure SLANeXt release. Preserved verbatim so byte-equal output vs
// PaddleX is achievable.
extern const std::string_view DEFAULT_DICT_TEXT;

// SLANeXt character dictionary post merge_no_span_structure=True expansion.
// Index 0 is sos, last index is eos.
class CharDict {
public:
    // Build the dict from a list of base tokens.
    // Replicates TableLabelDecode.__init__:
    //   - if "<td></td>" not present, append it,
    //   - drop "<td>",
    //   - prepend "sos",
    //   - append "eos".
    static CharDict build(std::vector<std::string> base);

    // Build the dict from a \n-separated source string.
    static CharDict from_text(std::string_view text);

    // Load the dict from disk (one token per line).
    static CharDict from_path(const std::string& path);

    // Token at vocab index, or empty view if OOB.
    std::string_view token(std::size_t idx) const;

    std::size_t len() const noexcept { return chars_.size(); }
    bool empty() const noexcept { return chars_.size() <= 2; }

    // sos is always index 0.
    std::size_t sos_idx() const noexcept { return 0; }
    // eos is the last index.
    std::size_t eos_idx() const noexcept { return chars_.size() - 1; }

    // True if token at idx is one of "<td>", "<td", or "<td></td>" — the
    // td-bbox tokens that pull a per-step bbox out of loc_preds.
    bool is_td_token(std::size_t idx) const;

private:
    std::vector<std::string> chars_;
};

// Build the dict from the bundled DEFAULT_DICT_TEXT, or from
// $TURBO_OCR_TABLE_DICT_PATH if set.
CharDict default_dict();

} // namespace turbo_ocr::table
