#pragma once

// JSON shape for the /ocr/pdf?fields=1 "fields" array. Lives beside the other
// forms headers rather than in common/serialization/ so the field surface can
// evolve without touching the shared item serializers.

#include <string>
#include <vector>

#include "turbo_ocr/serialization/serialization_primitives.h"
#include "turbo_ocr/analysis/forms/form_field.h"

namespace turbo_ocr::forms {

// Appends `,"fields":[…]` — leading comma included, so the caller splices it
// into an already-open page object. Emitted even when empty: "we looked and
// found nothing" and "we never looked" are different answers, and a client
// that cannot tell them apart cannot tell a broken detector from a plain page.
inline void append_fields_array(std::string &j,
                                const std::vector<FormField> &fields) {
  j += ",\"fields\":[";
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto &f = fields[i];
    if (i) j += ',';
    j += "{\"type\":\"";
    j += field_type_name(f.type);
    j += "\",\"bounding_box\":";
    turbo_ocr::detail::append_box(j, f.box);
    j += ",\"label\":\"";
    turbo_ocr::detail::append_escaped_string(j, f.label);
    j += "\",\"confidence\":";
    turbo_ocr::detail::append_score(j, f.confidence);
    // INVARIANT: source is built only from the detectors' own string literals
    // joined by '+', never from user input or OCR text, so it needs no
    // escaping. If a detector name ever becomes data, route it through
    // turbo_ocr::detail::append_escaped_string.
    j += ",\"source\":\"";
    j += f.source;
    j += '"';
    // Only when the field is in one: an absent key means "not part of a run",
    // which is different from "run 0" and must not be confusable with it.
    if (f.group >= 0) {
      j += ",\"group\":";
      j += std::to_string(f.group);
    }
    j += '}';
  }
  j += ']';
}

} // namespace turbo_ocr::forms
