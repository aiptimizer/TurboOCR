#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace turbo_ocr::script_id {

// Seven script classes the multi-rec router can dispatch into. Order is
// stable: persisted in --rec-mode parses, server logs, and the script-id
// classifier's output mapping.
enum class ScriptId : uint8_t {
  Latin   = 0,
  Chinese = 1,
  Arabic  = 2,
  ESlav   = 3,
  Greek   = 4,
  Korean  = 5,
  Thai    = 6,
};

inline constexpr std::string_view to_string(ScriptId s) noexcept {
  switch (s) {
    case ScriptId::Latin:   return "latin";
    case ScriptId::Chinese: return "chinese";
    case ScriptId::Arabic:  return "arabic";
    case ScriptId::ESlav:   return "eslav";
    case ScriptId::Greek:   return "greek";
    case ScriptId::Korean:  return "korean";
    case ScriptId::Thai:    return "thai";
  }
  return "latin";
}

inline std::optional<ScriptId> parse_script(std::string_view s) noexcept {
  if (s == "latin")   return ScriptId::Latin;
  if (s == "chinese") return ScriptId::Chinese;
  if (s == "arabic")  return ScriptId::Arabic;
  if (s == "eslav")   return ScriptId::ESlav;
  if (s == "greek")   return ScriptId::Greek;
  if (s == "korean")  return ScriptId::Korean;
  if (s == "thai")    return ScriptId::Thai;
  return std::nullopt;
}

// Parse the --rec-mode spec into a set of loaded script engines.
//   ""              → {Latin}
//   "latin"         → {Latin}
//   "chinese"       → {Chinese}
//   "multilingual"  → all 7
//   "latin,chinese" → {Latin, Chinese}
// Returns nullopt if any token is unrecognized.
inline std::optional<std::set<ScriptId>>
parse_rec_mode(std::string_view spec) {
  std::set<ScriptId> out;
  if (spec.empty()) { out.insert(ScriptId::Latin); return out; }
  if (spec == "multilingual") {
    return std::set<ScriptId>{ScriptId::Latin, ScriptId::Chinese,
                               ScriptId::Arabic, ScriptId::ESlav,
                               ScriptId::Greek, ScriptId::Korean,
                               ScriptId::Thai};
  }
  std::size_t i = 0;
  while (i < spec.size()) {
    auto comma = spec.find(',', i);
    auto end = (comma == std::string_view::npos) ? spec.size() : comma;
    auto tok = spec.substr(i, end - i);
    while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.remove_prefix(1);
    while (!tok.empty() && (tok.back()  == ' ' || tok.back()  == '\t')) tok.remove_suffix(1);
    if (!tok.empty()) {
      auto parsed = parse_script(tok);
      if (!parsed) return std::nullopt;
      out.insert(*parsed);
    }
    i = (comma == std::string_view::npos) ? spec.size() : comma + 1;
  }
  if (out.empty()) out.insert(ScriptId::Latin);
  return out;
}

[[nodiscard]] inline bool rec_mode_needs_classifier(
    const std::set<ScriptId>& loaded) noexcept {
  return loaded.size() > 1;
}

} // namespace turbo_ocr::script_id
