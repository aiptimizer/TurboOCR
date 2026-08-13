#!/usr/bin/env bash
# Architectural invariants that a compiler cannot state, checked mechanically.
#
# Every rule here is one this tree ALREADY broke. They were each found by hand,
# months apart, and each had been true for an unknown length of time — which is
# the argument for the file: a property nothing checks is a property that
# regresses.
#
#   tools/checks/architecture.sh          # all rules
#   tools/checks/architecture.sh layering # one rule
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

fail=0
report() { echo "FAIL: $1"; shift; printf '%s\n' "$@" | sed 's/^/  /'; fail=1; }
ok()     { echo "ok:   $1"; }

# ---------------------------------------------------------------------------
# LAYERING — vendor SDKs stay inside their backend.
#
# include/turbo_ocr/service/server/metrics.h included <cuda_runtime_api.h> and
# called cudaMemGetInfo from the /metrics handler, gated on the legacy
# USE_CPU_ONLY flag. A device-neutral service header, hard-wired to one vendor,
# and a build break waiting for the first non-NVIDIA GPU configure (that flag is
# off for those too). The fix was a device_memory() hook on the Backend seam;
# this is what stops the next one.
check_layering() {
  local hits arm_hits svc_hits
  hits=$(grep -rn --include=*.h --include=*.cpp --include=*.mm \
           -E '^\s*#\s*include\s*<(cuda[_./a-z]*|nvinfer[a-zA-Z_]*|nvjpeg)\.h>' \
           src include 2>/dev/null \
         | grep -v '^src/backends/nvidia/' \
         | grep -v '^tests/' || true)

  # The SDK check above only ever saw <angle-bracket> vendor SDK headers, so it
  # could not see the other half of the same rule: a file outside a vendor arm
  # including that arm's OWN header by its vendor-rooted path. That is how
  # src/image/page_image_encoder.cpp came to `#include
  # "nvidia/support/nvjpeg_encoder.h"` under `#ifndef USE_CPU_ONLY` and stay
  # green — a device-neutral TU hard-wired to one vendor behind a legacy build
  # flag, which is the EXACT shape of the bug this whole check was written for.
  # Fixed by a JpegEncodeHook the NVIDIA arm installs; this is what stops the
  # next one.
  arm_hits=$(grep -rn --include=*.h --include=*.cpp --include=*.mm --include=*.cu \
               -E '^\s*#\s*include\s*"(nvidia|amd|intel|apple|cpu)/' \
               src include 2>/dev/null \
             | grep -v '^src/backends/' || true)

  # And the reverse edge: a vendor arm reaching UP into the transport layer.
  # src/backends/nvidia/ used to include service/server/bootstrap/pool_sizing.h
  # for a policy that is arithmetic over two memory numbers; it now lives in
  # include/turbo_ocr/pipeline/. src/README.md's rule for service/ is "transport
  # only" — nothing below it may depend on it.
  svc_hits=$(grep -rn --include=*.h --include=*.cpp --include=*.mm --include=*.cu \
               -E '^\s*#\s*include\s*"turbo_ocr/service/' \
               src/backends 2>/dev/null || true)

  if [ -n "$hits" ]; then
    report "CUDA/TensorRT headers outside src/backends/nvidia/" "$hits"
  fi
  if [ -n "$arm_hits" ]; then
    report "vendor-arm headers included from outside src/backends/" "$arm_hits"
  fi
  if [ -n "$svc_hits" ]; then
    report "a vendor arm includes the transport layer (service/ is transport only)" "$svc_hits"
  fi
  if [ -z "$hits$arm_hits$svc_hits" ]; then
    ok "layering: vendor code stays inside its arm, and below the transport"
  fi
}

# ---------------------------------------------------------------------------
# DEAD HEADERS — a header nobody includes is dead weight that still gets read,
# maintained and believed.
#
# pipeline_pool.h survived as 155 lines of a template that was never
# instantiated, kept alive by three comments (two naming a file that had already
# been deleted). script_id_types.h outlived the feature it described and took
# two test files down with it — those were what stopped the GPU test binary from
# compiling at all.
check_dead_headers() {
  local dead=()
  while IFS= read -r h; do
    local base; base=$(basename "$h")
    grep -rq --include=*.h --include=*.cpp --include=*.cu --include=*.mm \
         "include.*${base}" src include tests 2>/dev/null || dead+=("$h")
  done < <(find include -name '*.h' 2>/dev/null)
  if [ ${#dead[@]} -gt 0 ]; then
    report "headers included by nothing" "${dead[@]}"
  else
    ok "dead headers: every public header has at least one includer"
  fi
}

# ---------------------------------------------------------------------------
# FILE SIZE — a ceiling, not a target.
#
# Advisory limits do not hold: this tree reached 1469 lines in one file and
# 1012 in another while the style guide already asked for smaller. The number is
# deliberately generous so it flags genuine outliers rather than becoming noise
# that gets ignored.
readonly MAX_LINES=900

# A RATCHET, not an amnesty. Files listed here are over the limit TODAY and are
# tracked for splitting; the check still fails for anything not on the list, so
# the count can only go down. Delete a line when the file drops under the limit
# — never add one to make a build pass.
readonly ALLOWED_OVERSIZE=(
)
check_file_size() {
  local big=() f n
  while read -r n f; do
    [ "$f" = "total" ] && continue
    [ "$n" -le $MAX_LINES ] && continue
    local allowed=0 a
    for a in "${ALLOWED_OVERSIZE[@]}"; do [ "$f" = "$a" ] && allowed=1; done
    [ $allowed -eq 1 ] || big+=("$f ($n lines)")
  done < <(find src include \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.mm' \) 2>/dev/null \
           | xargs wc -l 2>/dev/null)
  if [ ${#big[@]} -gt 0 ]; then
    report "files over ${MAX_LINES} lines (and not on the ratchet list)" "${big[@]}"
  else
    ok "file size: no new file over ${MAX_LINES} lines"
  fi
  # The ratchet must not rot either: a listed file that has since shrunk should
  # come OFF the list, or the limit silently stops applying to it.
  for a in "${ALLOWED_OVERSIZE[@]}"; do
    [ -f "$a" ] || { report "ratchet lists a file that no longer exists" "$a"; continue; }
    n=$(wc -l < "$a")
    [ "$n" -le $MAX_LINES ] && report "ratchet entry is now under the limit — remove it" "$a ($n lines)"
  done
}

# ---------------------------------------------------------------------------
# DEBT MARKERS must name a blocker: TODO(reason), not a bare TODO.
#
# A raw count of TODO/FIXME/HACK in this tree reads 24, and that number is
# nearly all noise: five are mkstemp templates ("/tmp/ocr_pdf_XXXXXX") caught by
# grepping for XXX, and most of the rest are PROSE CROSS-REFERENCES in
# explanatory comments ("layout is a TODO, see README") rather than work owed at
# that line. The markers that matter are the ones sitting on code, and the AMD
# and Intel backends already write those as TODO(on-hardware) — the blocker is
# in the marker, so a reader knows whether it is actionable without hunting.
#
# The rule enforces that form. It is the fix for how these rot: "ON-HARDWARE
# TODO: wire OrtCudaEngine's ctor knobs" named a task that could not exist
# because the class is constructed nowhere, and "UNVERIFIED: never compiled" sat
# above code that builds and serves. A marker that names its blocker can be
# checked against reality; a bare one cannot.
readonly ALLOWED_BARE_MARKERS=(
  "src/backends/nvidia/kernels_cuda/cuda_kernels.cpp"  # norm whitelist, task #9
)
check_debt_markers() {
  local bare=() line file
  while IFS= read -r line; do
    file=${line%%:*}
    local allowed=0 a
    for a in "${ALLOWED_BARE_MARKERS[@]}"; do [ "$file" = "$a" ] && allowed=1; done
    [ $allowed -eq 1 ] || bare+=("$line")
  done < <(grep -rn --include=*.cpp --include=*.h --include=*.cu --include=*.mm \
             -E '(^|[^A-Za-z(])(TODO|FIXME|HACK)([^(A-Za-z]|$)' src include 2>/dev/null \
           | grep -vE '(TODO|FIXME|HACK)\(' \
           | grep -vE '(TODOs|README)' \
           | grep -vE '(is a TODO|\.hip TODO|engine TODO|the .*TODO )' || true)
  if [ ${#bare[@]} -gt 0 ]; then
    report "debt markers with no named blocker — write TODO(reason)" "${bare[@]}"
  else
    ok "debt markers: every marker names its blocker"
  fi
}

# ---------------------------------------------------------------------------
# ENV READS go through turbo_ocr/base/env_utils.h, which RECORDS them.
#
# ServerConfig logs an "Effective server config" line that operators read as the
# truth about a running server. Roughly 80 reads across 39 files went straight to
# std::getenv behind it, so a knob like GPU_CCL or TURBO_DET_BATCH could be set,
# take effect, and appear nowhere — making "my override is not working"
# indistinguishable from "my override is working and something else is wrong".
#
# Reading through the helpers puts the knob in the startup inventory
# ("Environment overrides in effect"). Any new raw read is a failure, not a
# smaller number.
#
# The pattern matches getenv QUALIFIED OR NOT — `\bgetenv(`. It used to match
# only `std::getenv`, and two unqualified `getenv("MPS_...")` reads in the Apple
# rec builder slipped straight through while this very comment claimed the
# conversion was complete. That is the same shape as the *.mm blind spot noted
# below: a rule whose grep is narrower than the thing it governs is a rule with
# a hole. `\b` does not match inside `secure_getenv` (the `_` is a word char),
# so that stays exempt.
#
# The surviving reads each carry a `pre-commit-allow-getenv` marker (which this
# check honours) with a stated reason: logger.h x3 + host_ort_threads.cpp
# (bootstrap — the recorder allocates/locks, a hazard during static init or a
# noexcept function); HOME x2 (ambient identity, not a knob); and the two
# MPS_DEBUG/MPS_OUT dev-only debug knobs in the rec builder, read once at build.
check_env_reads() {
  local n
  n=$(grep -rnE --include=*.cpp --include=*.h --include=*.cu --include=*.mm '\bgetenv[[:space:]]*\(' src include 2>/dev/null \
      | grep -v 'base/env_utils.h' | grep -v 'pre-commit-allow-getenv' | wc -l)
  # *.mm IS counted. It was not, which hid 12 raw reads in the Apple backend --
  # every TURBO_APPLE_* knob was exempt from this rule by file extension alone.
  # check_layering scanned *.mm; this check, check_debt_markers and
  # check_file_size did not, so the same code was governed or ignored depending
  # on which rule was being applied. That is the identical blind spot that let an
  # Apple-only build break ship: .mm does not compile on Linux, and most of the
  # checks that would have flagged it were not reading it either.
  #
  # Zero, and it stays zero: a new raw read means a knob that will not appear in
  # the startup inventory, which is the whole failure this check exists for.
  # Raising it is never the fix — either convert the site or mark it
  # `pre-commit-allow-getenv` with the reason, as the survivors do.
  local allowed=0
  if [ "$n" -gt "$allowed" ]; then
    report "raw getenv sites went UP ($n > $allowed) — read through turbo_ocr::env instead" \
           "$(grep -rnE --include=*.cpp --include=*.h --include=*.mm '\bgetenv[[:space:]]*\(' src include | grep -v 'base/env_utils.h' | grep -v 'pre-commit-allow-getenv' | tail -5)"
  elif [ "$n" -lt "$allowed" ]; then
    report "raw std::getenv sites dropped to $n — lower ALLOWED in this check to $n" ""
  else
    ok "env reads: $n raw sites, none added"
  fi
}

# ---------------------------------------------------------------------------
# HTTP STATUS comes from the error code, never from the call site.
#
# error_codes.h has always mapped every wire code to its HTTP status AND its
# gRPC StatusCode — one row, both transports. It had zero HTTP callers: all 58
# route sites passed the status by hand next to the code, so "what status is
# IMAGE_DECODE_FAILED" was answered in 58 places and changing it meant editing
# all of them. Worse, several codes (BACKEND_UNAVAILABLE, PAGE_FAILED,
# ROUTING_*, ADHOC_*) had no row at all, so gRPC could not have agreed with HTTP
# even in principle.
#
# error_response() now takes NO status argument, which is what makes this rule
# mechanical: a status literal in the service layer means someone reintroduced
# the parameter or hand-built a response around it. Reads of a status
# (`statusCode() == ...`, e.g. the Retry-After middleware) are not emissions and
# do not count.
check_http_status() {
  local hits n
  hits=$(grep -rn --include=*.cpp --include=*.h -E 'drogon::k[45][0-9][0-9]' \
           src/service include/turbo_ocr/service 2>/dev/null \
         | grep -v 'server/error_codes.h' \
         | grep -v 'statusCode() *==' || true)
  n=$(printf '%s' "$hits" | grep -c . || true)
  if [ "$n" -gt 0 ]; then
    report "HTTP status literal in the service layer ($n) — pass the ErrorCode to error_response() and let error_codes.h answer" \
           "$hits"
  else
    ok "http status: derived from the error code, no literals"
  fi
}

# ---------------------------------------------------------------------------
# PRIVATE PATHS — nothing in the tree names one machine or one private project.
#
# Thirteen tracked files reached the v4 alpha carrying a separate private
# project's name (epAiland), a "turbo-private" server name, absolute
# /home/<user>/ dataset paths, one developer's home directory, and — in seven
# tools/ sources — an editor session's scratch directory as the argv DEFAULT.
# None of it existed at v3.5.0; all of it accumulated unnoticed over one release
# because nothing looked. Two of those scripts were also simply broken for
# everyone else, since the dataset path was hardcoded at module scope.
#
# Matching is on tracked files only (git grep), so a developer's own untracked
# scratch work is never flagged.
check_private_paths() {
  local hits n
  # TWO passes, because the exemptions differ.
  #
  # Pass 1 is never acceptable anywhere, in any form: a private project name, a
  # named SSH key, a tailnet address, an editor scratch dir. These take NO
  # portable-form exemption — writing "~/.ssh/id_ed25519_epailand" in a doc is
  # exactly as much of a leak as an absolute path, and an earlier draft of this
  # rule let that one through precisely because it contained "~/".
  local always svc
  always=$(git grep -nI -iE \
             'epailand|turbo-private|/private/tmp/claude-[0-9]+|\bid_ed25519|100\.(82|126)\.[0-9]+\.[0-9]+' \
             -- . 2>/dev/null | grep -vE '^tools/checks/architecture\.sh:' || true)

  # Pass 2 is home directories, where portable forms ARE the fix and /home/ocr
  # is the service account inside the container images. The allowlist is service
  # and CI accounts, not people; anything else under /home or /Users is a laptop.
  svc=$(git grep -nI -E '/home/[a-z][a-z0-9_-]*/|/Users/[a-z][a-z0-9_-]*/' -- . 2>/dev/null \
        | grep -vE '^tools/checks/architecture\.sh:' \
        | grep -vE '/home/(ocr|root|runner|ubuntu|app|node)/' \
        | grep -vE '\$HOME|\$\{HOME\}|<user>|<you>|example|placeholder' || true)

  hits=$(printf '%s\n%s' "$always" "$svc" | grep -c . >/dev/null 2>&1; printf '%s\n%s' "$always" "$svc" | grep . || true)
  n=$(printf '%s' "$hits" | grep -c . || true)
  if [ "$n" -gt 0 ]; then
    report "private path or identifier in a tracked file ($n) — use \$HOME, a repo-relative default, or an env var" \
           "$hits"
  else
    ok "private paths: no machine, user or private project named in tracked files"
  fi
}

case "${1:-all}" in
  layering)     check_layering ;;
  dead)         check_dead_headers ;;
  size)         check_file_size ;;
  markers)      check_debt_markers ;;
  env)          check_env_reads ;;
  status)       check_http_status ;;
  private)      check_private_paths ;;
  all)          check_layering; check_dead_headers; check_file_size; check_debt_markers; check_env_reads; check_http_status; check_private_paths ;;
  *) echo "usage: $0 [layering|dead|size|markers|env|status|private|all]"; exit 2 ;;
esac
exit $fail
