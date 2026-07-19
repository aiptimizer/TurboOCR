#pragma once

// The env parsers moved to common/env_utils.h (namespace turbo_ocr::env):
// they are used well beyond the server layer (decode, vlm, formula, table).
// This shim keeps the historical turbo_ocr::server:: spellings working.
#include "turbo_ocr/common/env_utils.h"

namespace turbo_ocr::server {
using env::env_bool_strict;
using env::env_choice_strict;
using env::env_enabled;
using env::env_float_strict;
using env::env_int;
using env::env_int_strict;
using env::env_or;
using env::env_present;
} // namespace turbo_ocr::server
