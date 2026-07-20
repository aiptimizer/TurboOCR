#include "turbo_ocr/common/string_utils.h"
#include "turbo_ocr/formula/ppformulanet/formula_tokenizer.h"

#include "turbo_ocr/formula/ppformulanet/latex_normalize.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>

#include "nlohmann/json.hpp"

namespace turbo_ocr::formula {

namespace {

// GPT-2/HF byte-level BPE reverse map: each codepoint in a token string encodes ONE
// original byte (the inverse of bytes_to_unicode); concatenating those bytes and reading
// them as UTF-8 yields the real text. Printable ASCII bytes map to themselves (so English
// is unchanged) and Ġ (U+0120) -> byte 0x20 (space); multi-byte UTF-8 (all Chinese,
// accented Latin) is reconstructed instead of left as byte-level mojibake (投 -> "æĬķ").
const std::array<int, 324>& byte_decoder_table() {
    static const std::array<int, 324> tbl = [] {
        std::array<int, 324> m{};
        m.fill(-1);
        auto printable = [](int b) {
            return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        };
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            int cp = printable(b) ? b : (256 + n++);
            if (cp >= 0 && cp < static_cast<int>(m.size())) m[cp] = b;
        }
        return m;
    }();
    return tbl;
}

// Replace malformed UTF-8 byte sequences with U+FFFD (HF tokenizer's errors="replace").
// A backend run on text it cannot model (e.g. -S on Chinese) can emit byte-level tokens
// whose reconstructed bytes are not valid UTF-8; left raw they would make nlohmann::json
// throw when serializing the response. Always returning valid UTF-8 keeps the server safe.
std::string utf8_sanitize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0, n = s.size();
    auto cont = [&](std::size_t k) {
        return k < n && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80;
    };
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len = (c < 0x80) ? 1 : ((c >> 5) == 0x6) ? 2 : ((c >> 4) == 0xE) ? 3
                                : ((c >> 3) == 0x1E) ? 4 : 0;
        bool ok = len >= 1;
        for (int k = 1; k < len && ok; ++k) ok = cont(i + k);
        if (ok) { out.append(s, i, len); i += static_cast<std::size_t>(len); }
        else { out.append("\xEF\xBF\xBD"); ++i; }  // U+FFFD
    }
    return out;
}

// Map a byte-level token string (UTF-8 of byte-level codepoints) back to real UTF-8 text.
std::string byte_level_to_utf8(const std::string& s) {
    const auto& inv = byte_decoder_table();
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6 && i + 1 < n) { cp = c & 0x1Fu; len = 2; }
        else if ((c >> 4) == 0xE && i + 2 < n) { cp = c & 0x0Fu; len = 3; }
        else if ((c >> 3) == 0x1E && i + 3 < n) { cp = c & 0x07u; len = 4; }
        else { out.push_back(s[i]); ++i; continue; }
        // Validate continuation bytes before shifting — a malformed
        // (non-byte-level) token would otherwise decode to a wrong byte.
        bool valid = true;
        for (int k = 1; k < len; ++k)
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0u) != 0x80u) { valid = false; break; }
        if (!valid) { out.push_back(s[i]); ++i; continue; }
        for (int k = 1; k < len; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        i += len;
        int b = (cp < inv.size()) ? inv[cp] : -1;
        if (b >= 0) out.push_back(static_cast<char>(b));
        else out.append(s, i - len, len);  // not a byte-level codepoint — pass through
    }
    return utf8_sanitize(out);  // reconstructed bytes may be invalid UTF-8 (mismatched model)
}


// Single-pass, in-place removal of the PP-FormulaNet literal frame markers. Exactly
// equivalent to three sequential replace_all() passes ([EOS] then [BOS] then [PAD]) —
// the markers are distinct, equal-length, and never rescanned — but allocates nothing:
// it compacts survivors left with a write cursor and shrinks. For this tokenizer the
// markers never occur (they live in no vocab token); the strip is kept for other vocabs.
void strip_frame_markers(std::string& s) {
    static constexpr std::array<std::string_view, 3> kMarkers{"[EOS]", "[BOS]", "[PAD]"};
    const std::size_t n = s.size();
    std::size_t w = 0, i = 0;
    while (i < n) {
        std::size_t skip = 0;
        if (s[i] == '[') {
            for (std::string_view m : kMarkers) {
                if (n - i >= m.size() && std::memcmp(s.data() + i, m.data(), m.size()) == 0) {
                    skip = m.size();
                    break;
                }
            }
        }
        if (skip) i += skip;
        else s[w++] = s[i++];
    }
    s.resize(w);
}

}  // namespace

std::string FormulaTokenizer::latex_post_process(const std::string& s) {
    // Delegates to the extracted normalization engine (latex_normalize.cpp) —
    // kept as a member so decode() call sites stay unchanged.
    return formula::latex_post_process(s);
}

std::optional<FormulaTokenizer> FormulaTokenizer::load(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) return std::nullopt;

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception&) {
        return std::nullopt;
    }

    FormulaTokenizer tok;

    // id_to_token_ is sized from ids read out of the file, so a single
    // corrupt/hostile id (e.g. 2^32-1) would force a multi-GB allocation.
    // Real BPE vocabularies are well under this cap.
    constexpr std::size_t kMaxTokenId = 1u << 20;

    try {
        // Build id -> token from model.vocab (token -> id). Resize as we go.
        if (!j.contains("model") || !j["model"].contains("vocab")) return std::nullopt;
        const auto& vocab = j["model"]["vocab"];
        if (!vocab.is_object()) return std::nullopt;

        std::size_t max_id = 0;
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            std::size_t id = it.value().get<std::size_t>();
            if (id > kMaxTokenId) return std::nullopt;
            if (id > max_id) max_id = id;
        }

        // Added/special tokens may extend the id space.
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto& at : j["added_tokens"]) {
                if (at.contains("id")) {
                    std::size_t id = at["id"].get<std::size_t>();
                    if (id > kMaxTokenId) return std::nullopt;
                    if (id > max_id) max_id = id;
                }
            }
        }

        tok.id_to_token_.assign(max_id + 1, std::string{});
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            std::size_t id = it.value().get<std::size_t>();
            tok.id_to_token_[id] = it.key();
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto& at : j["added_tokens"]) {
                if (!at.contains("id")) continue;
                std::size_t id = at["id"].get<std::size_t>();
                std::string content = at.value("content", std::string{});
                tok.id_to_token_[id] = content;
                bool is_special = at.value("special", false);
                if (is_special) tok.special_ids_.insert(static_cast<int64_t>(id));
                if (content == "</s>") tok.eos_id_ = static_cast<int64_t>(id);
            }
        }
    } catch (const std::exception&) {
        // Type-mismatch throws from nlohmann (non-numeric id, wrong shapes)
        // are a malformed tokenizer file, not a crash.
        return std::nullopt;
    }

    return tok;
}

std::string FormulaTokenizer::decode(std::span<const int64_t> ids, bool post_process) const {
    std::string raw;
    raw.reserve(ids.size() * 2);
    const std::size_t vocab = id_to_token_.size();
    for (int64_t id : ids) {
        if (id == eos_id_) break;
        if (special_ids_.count(id)) continue;
        if (id < 0 || static_cast<std::size_t>(id) >= vocab) continue;
        raw += id_to_token_[static_cast<std::size_t>(id)];
    }

    // Byte-level BPE -> real UTF-8 (Ġ -> space falls out of the map automatically), then
    // drop the literal frame markers and trim. Bit-identical to the prior
    // byte_level_to_utf8 + [EOS]/[BOS]/[PAD] replace_all + trim chain, but with one buffer
    // instead of five: the marker strip compacts in place and the trim returns a view.
    std::string text = byte_level_to_utf8(raw);
    strip_frame_markers(text);
    const std::string_view trimmed = trim_view(text);
    if (post_process) return latex_post_process(std::string(trimmed));
    return std::string(trimmed);
}

}  // namespace turbo_ocr::formula
