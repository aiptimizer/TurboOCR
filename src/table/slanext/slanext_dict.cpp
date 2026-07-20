#include "turbo_ocr/table/slanext/slanext_dict.h"

#include "turbo_ocr/table/table_types.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace turbo_ocr::table {

namespace {

// Verbatim PaddleX table_structure_dict_ch.txt — 48 base tokens.
// Whitespace and quoting are byte-for-byte; alteration breaks SLANeXt
// vocab alignment.
constexpr std::string_view kDefaultDictText =
    "<thead>\n"
    "</thead>\n"
    "<tr>\n"
    "</tr>\n"
    "<td>\n"
    "</td>\n"
    "<td\n"
    ">\n"
    " colspan=\"2\"\n"
    " colspan=\"3\"\n"
    " colspan=\"4\"\n"
    " colspan=\"5\"\n"
    " colspan=\"6\"\n"
    " colspan=\"7\"\n"
    " colspan=\"8\"\n"
    " colspan=\"9\"\n"
    " colspan=\"10\"\n"
    " colspan=\"11\"\n"
    " colspan=\"12\"\n"
    " colspan=\"13\"\n"
    " colspan=\"14\"\n"
    " colspan=\"15\"\n"
    " colspan=\"16\"\n"
    " colspan=\"17\"\n"
    " colspan=\"18\"\n"
    " colspan=\"19\"\n"
    " colspan=\"25\"\n"
    " rowspan=\"2\"\n"
    " rowspan=\"3\"\n"
    " rowspan=\"4\"\n"
    " rowspan=\"5\"\n"
    " rowspan=\"6\"\n"
    " rowspan=\"7\"\n"
    " rowspan=\"8\"\n"
    " rowspan=\"9\"\n"
    " rowspan=\"10\"\n"
    " rowspan=\"11\"\n"
    " rowspan=\"12\"\n"
    " rowspan=\"13\"\n"
    " rowspan=\"14\"\n"
    " rowspan=\"15\"\n"
    " rowspan=\"16\"\n"
    " rowspan=\"17\"\n"
    " rowspan=\"18\"\n"
    " rowspan=\"19\"\n"
    " rowspan=\"20\"\n"
    "<tbody>\n"
    "</tbody>\n";

// Count of non-empty \n-separated lines — the base-token count from_text() sees.
constexpr std::size_t count_base_tokens(std::string_view t) noexcept {
    std::size_t n = 0, run = 0;
    for (char c : t) {
        if (c == '\n') { n += (run != 0); run = 0; }
        else ++run;
    }
    return n + (run != 0);
}

// True if any whole line of `text` equals `needle`.
constexpr bool has_token_line(std::string_view text, std::string_view needle) noexcept {
    for (std::size_t i = 0; i <= text.size();) {
        const std::size_t nl = text.find('\n', i);
        const std::size_t end = (nl == std::string_view::npos) ? text.size() : nl;
        if (text.substr(i, end - i) == needle) return true;
        if (nl == std::string_view::npos) break;
        i = nl + 1;
    }
    return false;
}

// Compile-time model of build(): merge_no_span_structure appends "<td></td>"
// when absent and drops "<td>", then sos/eos wrap the sequence.
constexpr std::size_t realized_vocab(std::string_view text) noexcept {
    std::size_t n = count_base_tokens(text);
    if (!has_token_line(text, "<td></td>")) ++n;
    if (has_token_line(text, "<td>")) --n;
    return n + 2;
}

// The bundled vocabulary is load-bearing: SLANeXt token IDs index directly into
// it, so a stray edit would silently corrupt every decoded table. Pin the shape
// here so mistakes fail the build, not TEDS.
static_assert(count_base_tokens(kDefaultDictText) == 48,
              "bundled SLANeXt dict must hold exactly 48 base tokens");
static_assert(realized_vocab(kDefaultDictText) == SLANEXT_VOCAB,
              "bundled dict is out of sync with SLANEXT_VOCAB");

} // namespace

const std::string_view DEFAULT_DICT_TEXT = kDefaultDictText;

CharDict CharDict::build(std::vector<std::string> base) {
    if (std::none_of(base.begin(), base.end(),
                     [](const std::string& s) { return s == "<td></td>"; })) {
        base.emplace_back("<td></td>");
    }
    base.erase(std::remove(base.begin(), base.end(), std::string("<td>")),
               base.end());

    CharDict d;
    d.chars_.reserve(base.size() + 2);
    d.chars_.emplace_back("sos");
    for (auto& t : base) d.chars_.emplace_back(std::move(t));
    d.chars_.emplace_back("eos");
    return d;
}

CharDict CharDict::from_text(std::string_view text) {
    std::vector<std::string> base;
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t nl = text.find('\n', i);
        std::string line(text.substr(i, (nl == std::string_view::npos
                                             ? text.size()
                                             : nl) - i));
        if (!line.empty()) base.push_back(std::move(line));
        if (nl == std::string_view::npos) break;
        i = nl + 1;
    }
    return build(std::move(base));
}

CharDict CharDict::from_path(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("CharDict: could not open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return from_text(ss.str());
}

std::string_view CharDict::token(std::size_t idx) const {
    if (idx >= chars_.size()) return {};
    return chars_[idx];
}

bool CharDict::is_td_token(std::size_t idx) const {
    if (idx >= chars_.size()) return false;
    // The three td-bbox tokens have distinct lengths, so a size switch dispatches
    // in O(1) and the common non-td token returns without any string compare.
    const std::string_view t = chars_[idx];
    switch (t.size()) {
        case 3: return t == "<td";
        case 4: return t == "<td>";
        case 9: return t == "<td></td>";
        default: return false;
    }
}

CharDict default_dict() {
    if (const char* override = std::getenv("TURBO_OCR_TABLE_DICT_PATH")) {
        if (*override) return CharDict::from_path(override);
    }
    return CharDict::from_text(DEFAULT_DICT_TEXT);
}

} // namespace turbo_ocr::table
