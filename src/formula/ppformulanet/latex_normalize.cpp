#include "turbo_ocr/formula/ppformulanet/latex_normalize.h"

#include <cctype>
#include <cstddef>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace turbo_ocr::formula {

namespace {

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
        if (i + 5 <= n && std::string_view(data + i, 5) == "\\text") {
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

std::string latex_post_process(const std::string& s) {
    return normalize(remove_chinese_text_wrapping(s));
}

}  // namespace turbo_ocr::formula
