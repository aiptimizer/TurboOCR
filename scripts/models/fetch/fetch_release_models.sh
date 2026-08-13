#!/bin/bash
# Pre-bake every supported model bundle into /app/models/ at Docker build
# time, pulling from the TurboOCR GitHub Release and verifying SHA256.
#
# Output layout (matches what the server reads at startup):
#   /app/models/
#     det.onnx + rec.onnx + keys.txt            # PP-OCRv6 medium (flat default)
#     det_small.onnx + rec_small.onnx           # PP-OCRv6 small tier
#     det_tiny.onnx + rec_tiny.onnx + keys_tiny.txt   # PP-OCRv6 tiny tier
#     cls.onnx                                  # angle classifier (script-agnostic)
#     rec/greek/{rec.onnx, dict.txt}            # retained PP-OCRv5 scripts
#     rec/eslav/{rec.onnx, dict.txt}            # (v6 covers Latin/CJK only)
#     rec/arabic/{rec.onnx, dict.txt}
#     rec/korean/{rec.onnx, dict.txt}
#     rec/thai/{rec.onnx, dict.txt}
#     layout/layout.onnx                        # PP-DocLayoutV3 (?layout=1)
#     doc_ori.onnx                              # autorotate (variant release)

set -euo pipefail

# The whole cold-start self-heal depends on this release base being reachable.
# MODELS_RELEASE_URL is the primary (override to point at any HTTP mirror that
# hosts the same asset names + SHA256SUMS.txt). MODELS_RELEASE_URL_FALLBACK is
# tried per-asset when the primary download fails, so a mirror can stand in if
# the GitHub release vanishes or rate-limits. Empty fallback = primary only.
MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v3.0.0-ppocrv6}"
MODELS_RELEASE_URL_FALLBACK="${MODELS_RELEASE_URL_FALLBACK:-}"
OUT="${OUT:-models}"

mkdir -p "$OUT"
echo "[fetch_release_models] base=$MODELS_RELEASE_URL  out=$OUT"
[[ -n "$MODELS_RELEASE_URL_FALLBACK" ]] && \
  echo "[fetch_release_models] mirror=$MODELS_RELEASE_URL_FALLBACK"

# Download $1 (asset path) to $2, trying a primary base ($3, default
# MODELS_RELEASE_URL) then an optional mirror ($4, default
# MODELS_RELEASE_URL_FALLBACK). Hard-fails (exit 1) if neither serves it.
fetch_asset() {
  local asset=$1 dest=$2
  local base="${3:-$MODELS_RELEASE_URL}" mirror="${4:-$MODELS_RELEASE_URL_FALLBACK}"
  if wget --tries=3 --timeout=60 --retry-connrefused -nv \
      "${base}/${asset}" -O "$dest"; then
    return 0
  fi
  if [[ -n "$mirror" ]]; then
    echo "    primary failed for $asset, trying mirror" >&2
    if wget --tries=3 --timeout=60 --retry-connrefused -nv \
        "${mirror}/${asset}" -O "$dest"; then
      return 0
    fi
  fi
  echo "    ERROR: could not download $asset from primary or mirror" >&2
  return 1
}

SUMS_FILE="$OUT/SHA256SUMS.release.txt"
fetch_asset "SHA256SUMS.txt" "$SUMS_FILE"

fetch_verified() {
  local asset=$1 target=$2
  echo "  $asset -> $target"
  fetch_asset "$asset" "${target}.part"
  local expected
  expected=$(awk -v a="$asset" '$2 == a {print $1}' "$SUMS_FILE")
  [[ -z "$expected" ]] && { echo "    ERROR: no SHA entry for $asset" >&2; exit 1; }
  local actual
  actual=$(sha256sum "${target}.part" | awk '{print $1}')
  [[ "$actual" != "$expected" ]] && {
    echo "    ERROR: sha256 mismatch for $asset" >&2
    echo "      expected: $expected" >&2
    echo "      actual:   $actual" >&2
    rm -f "${target}.part"
    exit 1
  }
  mv "${target}.part" "$target"
}

# PP-OCRv6 medium (flat default) + small/tiny tiers as flat siblings.
# medium/small share keys.txt; tiny ships its own 6,904-char keys_tiny.txt.
fetch_verified "det.onnx"        "$OUT/det.onnx"
fetch_verified "rec.onnx"        "$OUT/rec.onnx"
fetch_verified "keys.txt"        "$OUT/keys.txt"
fetch_verified "det_small.onnx"  "$OUT/det_small.onnx"
fetch_verified "rec_small.onnx"  "$OUT/rec_small.onnx"
fetch_verified "det_tiny.onnx"   "$OUT/det_tiny.onnx"
fetch_verified "rec_tiny.onnx"   "$OUT/rec_tiny.onnx"
fetch_verified "keys_tiny.txt"   "$OUT/keys_tiny.txt"

# Angle classifier (script-agnostic, serves all paths; PP-OCRv6 ships none).
fetch_verified "cls.onnx"        "$OUT/cls.onnx"

# PP-LCNet_x1_0 textline-orientation variant (~6.8 MB) — OPTIONAL, selected via
# CLS_ONNX=x1_0 / CLS_MODEL=x1_0. Lives on its own release (the base v3.0.0
# models release is immutable), sha pinned here like doc_ori below (recipe:
# scripts/models/onnx/export_textline_ori_x1_0.py). Opt-in only: a failed fetch must not
# abort the build — the server refuses to boot loudly if a user selects x1_0
# without the file, so skipping stays visible, not silent.
CLS_X1_0_RELEASE_URL="${CLS_X1_0_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v3.4.0-textline-ori-x1-0}"
CLS_X1_0_RELEASE_URL_FALLBACK="${CLS_X1_0_RELEASE_URL_FALLBACK:-}"
CLS_X1_0_SHA256="e1e089ca7669e0ae28842a3b018152b6c320121fcd30e54664b4b5abbacda5a5"
echo "  cls_x1_0.onnx -> $OUT/cls_x1_0.onnx (optional)"
if fetch_asset "cls_x1_0.onnx" "$OUT/cls_x1_0.onnx.part" \
       "$CLS_X1_0_RELEASE_URL" "$CLS_X1_0_RELEASE_URL_FALLBACK" \
   && [[ "$(sha256sum "$OUT/cls_x1_0.onnx.part" | awk '{print $1}')" == "$CLS_X1_0_SHA256" ]]; then
  mv "$OUT/cls_x1_0.onnx.part" "$OUT/cls_x1_0.onnx"
else
  echo "    WARN: cls_x1_0.onnx unavailable or checksum mismatch — skipping." >&2
  echo "          CLS_ONNX=x1_0 will refuse to start; the default x0_25 classifier is unaffected." >&2
  rm -f "$OUT/cls_x1_0.onnx.part"
fi

# PP-DocLayoutV3 (~124 MB) — required for ?layout=1 endpoints.
mkdir -p "$OUT/layout"
fetch_verified "layout.onnx" "$OUT/layout/layout.onnx"

# FFDetr (~77 MB) — the learned half of ?fields=1. OPTIONAL: without it the
# four geometry detectors still run, they just cannot see a blank that nothing
# was drawn around, and cannot report a signature at all.
#
# Not carried in the models release yet. It is built from the Apache-2.0
# checkpoint on the Hub by tools/modelgen/export_ffdetr.py, which is deliberately a
# separate step: the export needs torch + rfdetr, ~3 GB of build-time
# dependencies that have no business in a runtime image.
FFDETR_RELEASE_URL="${FFDETR_RELEASE_URL:-}"
mkdir -p "$OUT/forms"
if [[ -f "$OUT/forms/ffdetr.onnx" ]]; then
  echo "  ffdetr.onnx -> already present, skipping"
elif [[ -n "$FFDETR_RELEASE_URL" ]] \
     && fetch_asset "ffdetr.onnx" "$OUT/forms/ffdetr.onnx.part" \
                    "$FFDETR_RELEASE_URL" ""; then
  mv "$OUT/forms/ffdetr.onnx.part" "$OUT/forms/ffdetr.onnx"
else
  rm -f "$OUT/forms/ffdetr.onnx.part"
  echo "  ffdetr.onnx -> not fetched (optional; ?fields=1 falls back to geometry)"
  echo "                build it with:  python3 tools/modelgen/export_ffdetr.py"
fi

# Retained PP-OCRv5 scripts (nested layout) with no PP-OCRv6 equivalent.
LANGS=(greek eslav arabic korean thai)

for lang in "${LANGS[@]}"; do
  mkdir -p "$OUT/rec/$lang"
  fetch_verified "rec-${lang}.onnx"  "$OUT/rec/${lang}/rec.onnx"
  fetch_verified "dict-${lang}.txt"  "$OUT/rec/${lang}/dict.txt"
done

# Table structure — SLANet-Plus (enable with TABLE_BACKEND=slanext). ~8 MB.
mkdir -p "$OUT/table/slanext_encoder"
fetch_verified "slanext_wired_encoder.onnx" "$OUT/table/slanext_encoder/SLANeXt_wired_encoder.onnx"
fetch_verified "slanext_wired_decoder.bin"  "$OUT/table/slanext_encoder/SLANeXt_wired_decoder.bin"
fetch_verified "slanext_dict_infer.txt"     "$OUT/table/slanext_encoder/SLANeXt_dict_infer.txt"

# Formula — PP-FormulaNet-S. GPU uses the fast/ split graphs (the only GPU path);
# the CPU server (turboocr-server) uses the fused inference_trt.onnx. Ship both
# + the shared tokenizer so either server variant can recognize formulas.
mkdir -p "$OUT/formula/ppformulanet_s/fast"
fetch_verified "ppformulanet_s_fast_encoder.onnx"      "$OUT/formula/ppformulanet_s/fast/encoder.onnx"
fetch_verified "ppformulanet_s_fast_prep.onnx"         "$OUT/formula/ppformulanet_s/fast/prep.onnx"
fetch_verified "ppformulanet_s_fast_step_batched.onnx" "$OUT/formula/ppformulanet_s/fast/step_batched.onnx"
fetch_verified "ppformulanet_s_trt.onnx"               "$OUT/formula/ppformulanet_s/inference_trt.onnx"
fetch_verified "ppformulanet_s_tokenizer.json"         "$OUT/formula/ppformulanet_s/tokenizer.json"

# Formula (Chinese swap) — PP-FormulaNet_plus-M (FORMULA_BACKEND=ppformulanet_plus_m).
# Split graphs live in the model dir (no fast/ subdir); decoder_step_384.onnx is the
# length-bucket for faster Chinese. Tokenizer is byte-identical to -S, so reuse it.
mkdir -p "$OUT/formula/ppformulanet_plus_m"
fetch_verified "ppformulanet_plus_m_encoder.onnx"          "$OUT/formula/ppformulanet_plus_m/encoder.onnx"
fetch_verified "ppformulanet_plus_m_prep.onnx"             "$OUT/formula/ppformulanet_plus_m/prep.onnx"
fetch_verified "ppformulanet_plus_m_decoder_step.onnx"     "$OUT/formula/ppformulanet_plus_m/decoder_step.onnx"
fetch_verified "ppformulanet_plus_m_decoder_step_384.onnx" "$OUT/formula/ppformulanet_plus_m/decoder_step_384.onnx"
cp "$OUT/formula/ppformulanet_s/tokenizer.json" "$OUT/formula/ppformulanet_plus_m/tokenizer.json"

# PP-LCNet_x1_0_doc_ori (~6.5 MB) — OPTIONAL autorotate (?autorotate=1). Lives on
# the pdf-page-images variant release, not the base bundle; sha pinned here so the
# asset can't change under us. Recipe: scripts/models/onnx/export_doc_ori.py. Because
# autorotate is opt-in and this asset comes from a SEPARATE (older) release, a
# failure here must NOT abort the whole fetch — every core model already succeeded
# above. Warn and continue so a vanished/renamed doc_ori release can't break the
# text/layout/table/formula cold start.
DOC_ORI_RELEASE_URL="${DOC_ORI_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v2.4.0-pdf-page-images}"
DOC_ORI_RELEASE_URL_FALLBACK="${DOC_ORI_RELEASE_URL_FALLBACK:-}"
DOC_ORI_SHA256="96e898f047a0e460ba0652e9afb8c874e53872821cfd7a3fec53a5ab62df92f0"
echo "  doc_ori.onnx -> $OUT/doc_ori.onnx (optional)"
if fetch_asset "doc_ori.onnx" "$OUT/doc_ori.onnx.part" \
       "$DOC_ORI_RELEASE_URL" "$DOC_ORI_RELEASE_URL_FALLBACK" \
   && [[ "$(sha256sum "$OUT/doc_ori.onnx.part" | awk '{print $1}')" == "$DOC_ORI_SHA256" ]]; then
  mv "$OUT/doc_ori.onnx.part" "$OUT/doc_ori.onnx"
else
  echo "    WARN: doc_ori.onnx unavailable or checksum mismatch — skipping." >&2
  echo "          autorotate (?autorotate=1) will be disabled; core pipeline unaffected." >&2
  rm -f "$OUT/doc_ori.onnx.part"
fi

rm -f "$SUMS_FILE"  # not shipped in image
echo ""
echo "[fetch_release_models] baked:"
# -printf is a GNU findutils extension that BSD find (macOS) does not have.
# Under `set -euo pipefail` this LAST, purely cosmetic line failed the whole
# script — so on macOS every model downloaded correctly and the build then died
# on the summary, with "find: -printf: unknown primary or operator" as the only
# clue. `wc -c` is POSIX and gives the same two columns everywhere.
find "$OUT" -type f | sort | while IFS= read -r _f; do
  printf "  %-40s  %s bytes\n" "$_f" "$(wc -c < "$_f" | tr -d ' ')"
done
