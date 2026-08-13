#include "turbo_ocr/document/docassembly/title_level.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

namespace turbo_ocr::docassembly {

namespace {

bool is_ws(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

std::size_t skip_ws(const std::string& s, std::size_t i) {
  while (i < s.size() && is_ws(s[i])) ++i;
  return i;
}

// Decode the UTF-8 code point at s[i]; sets `len` to its byte length.
std::uint32_t utf8_cp(const std::string& s, std::size_t i, int& len) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) { len = 1; return c; }
  if ((c >> 5) == 0x6 && i + 1 < s.size()) {
    len = 2;
    return ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
  }
  if ((c >> 4) == 0xE && i + 2 < s.size()) {
    len = 3;
    return ((c & 0x0Fu) << 12) |
           ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
  }
  if ((c >> 3) == 0x1E && i + 3 < s.size()) {
    len = 4;
    return ((c & 0x07u) << 18) |
           ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
           ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
           (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
  }
  len = 1;
  return c;
}

constexpr std::uint32_t kFwLParen = 0xFF08;  // （
constexpr std::uint32_t kFwRParen = 0xFF09;  // ）
constexpr std::uint32_t kFwPeriod = 0xFF0E;  // ．
constexpr std::uint32_t kDi = 0x7B2C;        // 第

bool is_chinese_num(std::uint32_t cp) {
  switch (cp) {
    case 0x4E00: case 0x4E8C: case 0x4E09: case 0x56DB: case 0x4E94:
    case 0x516D: case 0x4E03: case 0x516B: case 0x4E5D: case 0x5341:
      return true;
    default: return false;
  }
}

bool is_chinese_unit(std::uint32_t cp) {
  switch (cp) {
    case 0x7AE0: case 0x8282: case 0x7BC7: case 0x5377: case 0x90E8:  // 章节篇卷部
    case 0x6761: case 0x9898: case 0x8BB2: case 0x8BFE: case 0x56DE:  // 条题讲课回
      return true;
    default: return false;
  }
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Match \d+(?:\.\d+)* starting at p; returns end index or -1.
int match_number(const std::string& s, std::size_t p) {
  const std::size_t n = s.size();
  if (p >= n || !is_digit(s[p])) return -1;
  std::size_t q = p;
  while (q < n && is_digit(s[q])) ++q;
  while (q + 1 < n && s[q] == '.' && is_digit(s[q + 1])) {
    ++q;
    while (q < n && is_digit(s[q])) ++q;
  }
  return static_cast<int>(q);
}

} // namespace

SymbolLevel get_symbol_and_level(const std::string& content) {
  // mirror str(content).strip()
  std::size_t b = content.find_first_not_of(" \t\r\n\f\v");
  if (b == std::string::npos) return {};
  std::size_t e = content.find_last_not_of(" \t\r\n\f\v");
  const std::string txt = content.substr(b, e - b + 1);
  const std::size_t n = txt.size();

  // 1) NUM_LIST_WITH_BRACKET → ("NUM_LIST_BRACKET", 4)
  {
    std::size_t p = skip_ws(txt, 0);
    int len = 0;
    std::uint32_t cp = p < n ? utf8_cp(txt, p, len) : 0;
    std::size_t q = p;
    if (p < n && txt[p] == '(') q = p + 1;
    else if (cp == kFwLParen) q = p + len;
    int num_end = match_number(txt, q);
    if (num_end >= 0) {
      std::size_t r = static_cast<std::size_t>(num_end);
      if (r < n) {
        int l2 = 0;
        std::uint32_t c2 = utf8_cp(txt, r, l2);
        if (txt[r] == ')' || c2 == kFwRParen) return {"NUM_LIST_BRACKET", 4};
      }
    }
  }

  // 2) ROMAN → ("ROMAN", 1)
  {
    std::size_t p = skip_ws(txt, 0);
    std::size_t start = p;
    while (p < n && std::strchr("IVXivx", txt[p]) != nullptr) ++p;
    if (p > start) {
      if (p == n) return {"ROMAN", 1};
      int l2 = 0;
      std::uint32_t c2 = utf8_cp(txt, p, l2);
      if (txt[p] == '.' || txt[p] == ')' || is_ws(txt[p]) || c2 == kFwPeriod)
        return {"ROMAN", 1};
    }
  }

  // 3) CHINESE_NUM → ("CHINESE_NUM", 1)
  {
    std::size_t p = skip_ws(txt, 0);
    int len = 0;
    std::uint32_t cp = p < n ? utf8_cp(txt, p, len) : 0;
    if (cp == kDi) p += len;
    else if (p < n && txt[p] == '(') p += 1;
    else if (cp == kFwLParen) p += len;

    int count = 0;
    std::size_t q = p;
    while (count < 2 && q < n) {
      int l = 0;
      std::uint32_t c = utf8_cp(txt, q, l);
      if (!is_chinese_num(c)) break;
      q += l;
      ++count;
    }
    if (count > 0) {
      if (q == n) return {"CHINESE_NUM", 1};
      int l2 = 0;
      std::uint32_t c2 = utf8_cp(txt, q, l2);
      bool is_letter = (c2 >= 'A' && c2 <= 'Z') || (c2 >= 'a' && c2 <= 'z') ||
                       (c2 >= 0x4E00 && c2 <= 0x9FA5);
      if (is_chinese_unit(c2) || c2 == kFwRParen || txt[q] == ')' || !is_letter)
        return {"CHINESE_NUM", 1};
    }
  }

  // 4) LETTER → ("LETTER", 2)
  {
    std::size_t p = skip_ws(txt, 0);
    if (p < n && std::isalpha(static_cast<unsigned char>(txt[p]))) {
      std::size_t q = p + 1;
      if (q < n) {
        int l2 = 0;
        std::uint32_t c2 = utf8_cp(txt, q, l2);
        if (txt[q] == '.' || txt[q] == ')' || is_ws(txt[q]) || c2 == kFwPeriod)
          return {"LETTER", 2};
      }
    }
  }

  // 5) NUM_LIST → ("NUM_LIST", dots+1)
  {
    std::size_t p = skip_ws(txt, 0);
    int num_end = match_number(txt, p);
    if (num_end >= 0) {
      std::size_t r = static_cast<std::size_t>(num_end);
      bool blocked = false;
      if (r < n) {
        int l2 = 0;
        std::uint32_t c2 = utf8_cp(txt, r, l2);
        if (txt[r] == ')' || c2 == kFwRParen) blocked = true;  // negative lookahead
      }
      if (!blocked) {
        int dots = 0;
        for (std::size_t i = p; i < r; ++i)
          if (txt[i] == '.') ++dots;
        return {"NUM_LIST", dots + 1};
      }
    }
  }

  return {};
}

namespace {

const std::unordered_map<std::string, int>& special_keywords() {
  static const std::unordered_map<std::string, int> kw = {
      {"ABSTRACT", 1}, {"SUMMARY", 1}, {"RESUME", 1}, {"绪论", 1},
      {"引言", 1}, {"CONTENTS", 1}, {"REFERENCES", 1}, {"REFERENCE", 1},
      {"参考文献", 1}, {"APPENDIX", 1}, {"APPENDICES", 1},
      {"附录", 1}, {"ACKNOWLEDGMENTS", 1}, {"INTRODUCTION", 1},
      {"BACKGROUNDANDRELATEDWORK", 1}, {"BACKGROUND", 1}, {"RELATEDWORK", 1},
      {"THEORETICALMODELS", 1}, {"DATA", 1}, {"METHOD", 1}, {"METHODS", 1},
      {"METHODOLOGY", 1}, {"TOPICANALYSIS", 1}, {"RESULT", 1}, {"RESULTS", 1},
      {"DISCUSSION", 1}, {"CONCLUSIONS", 1}, {"CONCLUSION", 1}, {"LIMITATIONS", 1},
      {"研究背景", 1}, {"相关工作", 1},
      {"研究方法", 1}, {"实验结果", 1},
      {"讨论", 1}, {"结论", 1}, {"致谢", 1},
      {"目录", 1}};
  return kw;
}

// str(content).upper().strip().rstrip("：: ").replace(" ", "") — ASCII-upper
// only (matches Python for the Latin keys; non-ASCII bytes pass through, so
// the CJK keys still match by exact bytes).
std::string normalize_keyword(const std::string& content) {
  std::string s = content;
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  std::size_t b = s.find_first_not_of(" \t\r\n\f\v");
  if (b == std::string::npos) return {};
  std::size_t e = s.find_last_not_of(" \t\r\n\f\v");
  s = s.substr(b, e - b + 1);
  // rstrip "：: " (fullwidth colon ：is EF BC 9A)
  for (;;) {
    if (!s.empty() && (s.back() == ':' || s.back() == ' ')) { s.pop_back(); continue; }
    if (s.size() >= 3 && static_cast<unsigned char>(s[s.size() - 3]) == 0xEF &&
        static_cast<unsigned char>(s[s.size() - 2]) == 0xBC &&
        static_cast<unsigned char>(s[s.size() - 1]) == 0x9A) {
      s.erase(s.size() - 3);
      continue;
    }
    break;
  }
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    if (c != ' ') out.push_back(c);
  return out;
}

struct Entry {
  ParsingBlock* block = nullptr;
  std::string content;
  int height = 0;
  std::string symbol;
  int level = -1;
  int final_level = -1;
};

int get_title_height(const ParsingBlock& b) {
  int x1 = b.bbox[0], y1 = b.bbox[1], x2 = b.bbox[2], y2 = b.bbox[3];
  int h = y2 - y1, w = x2 - x1;
  if (h <= 0) h = 1;
  // content.strip().count("\n") + 1
  std::size_t s = b.content.find_first_not_of(" \t\r\n\f\v");
  int lines = 1;
  if (s != std::string::npos) {
    std::size_t e = b.content.find_last_not_of(" \t\r\n\f\v");
    for (std::size_t i = s; i <= e; ++i)
      if (b.content[i] == '\n') ++lines;
  }
  double aspect = static_cast<double>(w) / static_cast<double>(h);
  if (aspect >= 1.0) return static_cast<int>(h / lines);
  return static_cast<int>(w / lines);
}

// Deterministic 1-D k-means over heights; returns height→level (1=largest).
std::unordered_map<int, int> cluster_global_heights(const std::vector<Entry>& entries,
                                                    int k_clusters = 4) {
  std::vector<int> heights;
  heights.reserve(entries.size());
  for (const auto& e : entries) heights.push_back(e.height);
  std::set<int> uniq_set(heights.begin(), heights.end());
  std::unordered_map<int, int> mapping;
  if (uniq_set.empty()) return mapping;

  std::vector<int> uniq(uniq_set.begin(), uniq_set.end());  // ascending
  int k = std::min<int>(k_clusters, static_cast<int>(uniq.size()));

  // Init centers at evenly spaced quantiles of the unique values.
  std::vector<double> centers(k);
  if (k == 1) {
    centers[0] = uniq.front();
  } else {
    for (int j = 0; j < k; ++j) {
      int idx = static_cast<int>(std::lround(
          static_cast<double>(j) * (uniq.size() - 1) / (k - 1)));
      centers[j] = uniq[idx];
    }
  }

  for (int iter = 0; iter < 100; ++iter) {
    std::vector<double> sum(k, 0.0);
    std::vector<int> cnt(k, 0);
    for (int h : heights) {
      int best = 0;
      double bd = std::abs(h - centers[0]);
      for (int j = 1; j < k; ++j) {
        double d = std::abs(h - centers[j]);
        if (d < bd) { bd = d; best = j; }
      }
      sum[best] += h;
      cnt[best] += 1;
    }
    bool moved = false;
    for (int j = 0; j < k; ++j) {
      if (cnt[j] == 0) continue;
      double nc = sum[j] / cnt[j];
      if (std::abs(nc - centers[j]) > 1e-9) moved = true;
      centers[j] = nc;
    }
    if (!moved) break;
  }

  // Rank centers descending: larger font → higher level (1-based).
  std::vector<int> order(k);
  for (int j = 0; j < k; ++j) order[j] = j;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return centers[a] > centers[b]; });
  std::unordered_map<int, int> old2new;
  for (int rank = 0; rank < k; ++rank) old2new[order[rank]] = rank + 1;

  for (int h : uniq) {
    int best = 0;
    double bd = std::abs(h - centers[0]);
    for (int j = 1; j < k; ++j) {
      double d = std::abs(h - centers[j]);
      if (d < bd) { bd = d; best = j; }
    }
    mapping[h] = old2new[best];
  }
  return mapping;
}

} // namespace

void assign_title_levels(std::vector<std::vector<ParsingBlock>>& blocks_by_page) {
  std::vector<Entry> entries;
  for (std::size_t pi = 0; pi < blocks_by_page.size(); ++pi) {
    for (auto& block : blocks_by_page[pi]) {
      block.page_index = static_cast<int>(pi);
      if (block.label == "paragraph_title") {
        Entry e;
        e.block = &block;
        e.content = block.content;
        e.height = get_title_height(block);
        entries.push_back(std::move(e));
      }
    }
  }
  if (entries.empty()) return;

  // Per-entry semantic symbol/level.
  for (auto& e : entries) {
    SymbolLevel sl = get_symbol_and_level(e.content);
    e.symbol = sl.symbol;
    e.level = sl.level;
  }

  auto cluster_map = cluster_global_heights(entries);

  // Global ordering of distinct numbering styles (first-seen order).
  std::unordered_map<std::string, int> global_seq;
  {
    int counter = 1;
    for (const auto& e : entries) {
      if (e.level > 0 && global_seq.find(e.symbol) == global_seq.end())
        global_seq[e.symbol] = counter++;
    }
  }

  int first_num_level = 0;
  for (auto& e : entries) {
    enum { SEMANTIC, SPECIAL, CLUSTER } bucket;
    std::string norm;
    if (e.level > 0) {
      bucket = SEMANTIC;
    } else {
      norm = normalize_keyword(e.content);
      bucket = special_keywords().count(norm) ? SPECIAL : CLUSTER;
    }
    int cluster_level = cluster_map.count(e.height) ? cluster_map[e.height] : 1;
    int final_level;

    if (bucket == SEMANTIC) {
      int semantic_level = e.level;
      int relative_order_level;
      if (e.symbol == "NUM_LIST") {
        if (first_num_level != 0) {
          relative_order_level = global_seq[e.symbol] + (e.level - first_num_level);
        } else {
          first_num_level = e.level;
          relative_order_level = global_seq[e.symbol];
        }
      } else {
        relative_order_level = global_seq[e.symbol];
      }
      // Vote among the three signals; tie → relative order.
      int votes[3] = {semantic_level, relative_order_level, cluster_level};
      std::unordered_map<int, int> tally;
      for (int v : votes) ++tally[v];
      int mode = 0, mode_cnt = 0;
      // Preserve first-seen order on ties so behavior matches Counter.most_common.
      for (int v : votes) {
        if (tally[v] > mode_cnt) { mode_cnt = tally[v]; mode = v; }
      }
      final_level = (mode_cnt > 1) ? mode : relative_order_level;
    } else if (bucket == SPECIAL) {
      final_level = special_keywords().at(norm);
    } else {
      final_level = cluster_level;
    }

    // Clamp to valid Markdown heading depth: a shallower-numbered heading after a
    // deeper first one could drive this <=0 (empty prefix) and deep numbering
    // (1.2.3.4.5) past 6 ("#######" renders as literal text).
    final_level = std::clamp(final_level, 1, 6);
    e.final_level = final_level;
    e.block->title_level = final_level;
  }
}

} // namespace turbo_ocr::docassembly
