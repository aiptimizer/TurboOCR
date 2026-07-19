#include "turbo_ocr/formula/latex_extract.h"

#include <regex>
#include <string_view>

namespace turbo_ocr::formula {

namespace {

std::string trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r'))
    s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r'))
    s.remove_suffix(1);
  return std::string(s);
}

} // namespace

std::string extract_latex(const std::string &msg) {
  static const std::regex re_fence(R"(```(?:latex|tex|math)?\s*\n?([\s\S]*?)```)",
                                   std::regex::ECMAScript);
  std::smatch m;
  if (std::regex_search(msg, m, re_fence) && m.size() >= 2) {
    std::string s = m[1].str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
      s.pop_back();
    return s;
  }
  static const std::regex re_disp(R"(\$\$([\s\S]*?)\$\$)");
  if (std::regex_search(msg, m, re_disp) && m.size() >= 2) return m[1].str();
  static const std::regex re_brk(R"(\\\[([\s\S]*?)\\\])");
  if (std::regex_search(msg, m, re_brk) && m.size() >= 2) return m[1].str();
  static const std::regex re_inline(R"(\$([^\$\n]+)\$)");
  if (std::regex_search(msg, m, re_inline) && m.size() >= 2) return m[1].str();
  std::string s = trim(msg);
  for (std::string_view pre :
       {"LaTeX:", "Latex:", "latex:", "Answer:", "answer:"}) {
    if (s.rfind(pre, 0) == 0) {
      s = trim(std::string_view(s).substr(pre.size()));
      break;
    }
  }
  return s;
}

} // namespace turbo_ocr::formula
