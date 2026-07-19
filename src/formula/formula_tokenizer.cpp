#include "turbo_ocr/formula/formula_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <regex>
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
        for (int k = 1; k < len; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        i += len;
        int b = (cp < inv.size()) ? inv[cp] : -1;
        if (b >= 0) out.push_back(static_cast<char>(b));
        else out.append(s, i - len, len);  // not a byte-level codepoint — pass through
    }
    return utf8_sanitize(out);  // reconstructed bytes may be invalid UTF-8 (mismatched model)
}

std::string replace_all(std::string s, std::string_view needle, std::string_view repl) {
    if (needle.empty()) return s;
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (true) {
        std::size_t hit = s.find(needle, i);
        if (hit == std::string::npos) {
            out.append(s, i, std::string::npos);
            return out;
        }
        out.append(s, i, hit - i);
        out.append(repl);
        i = hit + needle.size();
    }
}

std::string_view trim(std::string_view s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
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

// True if the [begin, end) UTF-8 span contains a CJK Unified Ideograph in
// U+4E00..U+9FFF (3-byte UTF-8 lead 0xE4 0xB8 .. 0xE9 0xBF).
bool contains_cjk(const char* begin, const char* end) {
    for (const char* p = begin; p + 2 < end;) {
        auto c0 = static_cast<unsigned char>(p[0]);
        if (c0 >= 0xE4 && c0 <= 0xE9) {
            auto c1 = static_cast<unsigned char>(p[1]);
            auto c2 = static_cast<unsigned char>(p[2]);
            // U+4E00..U+9FFF gates:
            //   0xE4 0xB8..0xBF *
            //   0xE5..0xE8 *      *
            //   0xE9 0x80..0xBF (whole range up to 0x9FFF -> 0xE9 0xBF 0xBF)
            bool ok = false;
            if (c0 == 0xE4) ok = (c1 >= 0xB8);
            else if (c0 >= 0xE5 && c0 <= 0xE8) ok = true;
            else if (c0 == 0xE9) ok = true;  // 0xE9 0x80 0x80 = U+9000, < U+9FFF+1
            (void)c2;
            if (ok) return true;
        }
        // advance by UTF-8 char width
        if (c0 < 0x80) p += 1;
        else if ((c0 & 0xE0) == 0xC0) p += 2;
        else if ((c0 & 0xF0) == 0xE0) p += 3;
        else if ((c0 & 0xF8) == 0xF0) p += 4;
        else p += 1;
    }
    return false;
}

// Port of `remove_chinese_text_wrapping` in tokenizer.rs:
//   regex: \\text\s*\{([^{}]*[一-鿿]+[^{}]*)\}  -> $1
// then drop every '"'. We hand-roll the scan because std::regex multi-byte
// Unicode character classes are unreliable.
std::string remove_chinese_text_wrapping(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    const char* data = s.data();
    std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        if (i + 5 < n && std::string_view(data + i, 5) == "\\text") {
            std::size_t j = i + 5;
            while (j < n && std::isspace(static_cast<unsigned char>(s[j]))) ++j;
            if (j < n && s[j] == '{') {
                std::size_t k = j + 1;
                std::size_t inner = k;
                bool ok = true;
                while (k < n && s[k] != '}' && s[k] != '{') ++k;
                if (k >= n || s[k] != '}') ok = false;
                if (ok && contains_cjk(data + inner, data + k)) {
                    out.append(data + inner, k - inner);
                    i = k + 1;
                    continue;
                }
            }
        }
        if (s[i] != '"') out.push_back(s[i]);
        ++i;
    }
    return out;
}

// `(\\(operatorname|mathrm|text|mathbf)\s?\*? \{.*?\})` — matches the wrapper
// macro and its (non-nested) argument; the inner content is space-stripped
// and substituted back in.
const std::regex& wrapper_regex() {
    static const std::regex r(
        R"((\\(?:operatorname|mathrm|text|mathbf)\s?\*? \{[^{}]*\}))",
        std::regex::ECMAScript);
    return r;
}

// Three whitespace-collapse passes. Char class `[\W_^\d]` in the Rust source
// is "anything that is not a letter" (\W excludes [a-zA-Z0-9_], then _ and \d
// add the latter back, ^ is literal inside the class). We use the same form;
// std::regex ECMAScript treats _ and ^ as literals here.
const std::regex& r_nn() {
    static const std::regex r(R"(([\W_^\d])\s+?([\W_^\d]))", std::regex::ECMAScript);
    return r;
}
const std::regex& r_nl() {
    static const std::regex r(R"(([\W_^\d])\s+?([a-zA-Z]))", std::regex::ECMAScript);
    return r;
}
const std::regex& r_ln() {
    static const std::regex r(R"(([a-zA-Z])\s+?([\W_^\d]))", std::regex::ECMAScript);
    return r;
}

// std::regex has no reliable lookbehind/lookahead; replicate PaddleX's
// `(?!\\ )` guard: if the captured `a` is a single backslash AND the very
// next source byte (i.e. the run of whitespace) starts with an ASCII space,
// keep the original substring instead of collapsing.
std::string collapse_unless_backslash_space(const std::regex& re, const std::string& src) {
    std::string out;
    out.reserve(src.size());
    auto begin = std::sregex_iterator(src.begin(), src.end(), re);
    auto end = std::sregex_iterator();
    std::size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        std::size_t start = static_cast<std::size_t>(m.position(0));
        std::size_t mlen = static_cast<std::size_t>(m.length(0));
        const std::string a = m[1].str();
        const std::string b = m[2].str();
        out.append(src, last, start - last);
        std::size_t after_a = start + a.size();
        bool guard = (a == "\\" && after_a < src.size() && src[after_a] == ' ');
        if (guard) {
            out.append(src, start, mlen);
        } else {
            out.append(a);
            out.append(b);
        }
        last = start + mlen;
    }
    out.append(src, last, std::string::npos);
    return out;
}

std::string normalize(const std::string& in) {
    // Stash wrapper-macro spans (operatorname/mathrm/text/mathbf) with their
    // inner whitespace already removed, swap them for a sentinel string, run
    // the collapse passes on the rest, then restore.
    std::vector<std::string> names;
    {
        auto it = std::sregex_iterator(in.begin(), in.end(), wrapper_regex());
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            std::string match = it->str(0);
            // Strip ASCII spaces inside the whole match (mirrors `.replace(' ', "")`).
            std::string stripped;
            stripped.reserve(match.size());
            for (char c : match) {
                if (c != ' ') stripped.push_back(c);
            }
            names.push_back(std::move(stripped));
        }
    }

    std::string s_out = in;
    if (!names.empty()) {
        std::string tmp;
        tmp.reserve(s_out.size());
        std::size_t last = 0, idx = 0;
        auto it = std::sregex_iterator(s_out.begin(), s_out.end(), wrapper_regex());
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            const auto& m = *it;
            std::size_t start = static_cast<std::size_t>(m.position(0));
            std::size_t mlen = static_cast<std::size_t>(m.length(0));
            tmp.append(s_out, last, start - last);
            tmp.append(names[idx++]);
            last = start + mlen;
        }
        tmp.append(s_out, last, std::string::npos);
        s_out = std::move(tmp);
    }

    while (true) {
        std::string p1 = collapse_unless_backslash_space(r_nn(), s_out);
        std::string p2 = collapse_unless_backslash_space(r_nl(), p1);
        std::string p3 = std::regex_replace(p2, r_ln(), "$1$2");
        if (p3 == s_out) break;
        s_out = std::move(p3);
    }
    return replace_all(s_out, "XXXXXXX", " ");
}

}  // namespace

std::string FormulaTokenizer::latex_post_process(const std::string& s) {
    return normalize(remove_chinese_text_wrapping(s));
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
    const std::string_view trimmed = trim(text);
    if (post_process) return latex_post_process(std::string(trimmed));
    return std::string(trimmed);
}

}  // namespace turbo_ocr::formula
