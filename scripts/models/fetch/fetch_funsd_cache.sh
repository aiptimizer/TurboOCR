#!/usr/bin/env bash
# Build the FUNSD evaluation cache the ctest accuracy gates consume
# (-DTURBO_FUNSD_CACHE=<dir>): the official FUNSD testing set's 50 pages,
# sorted by original filename and renamed funsd_000.png .. funsd_049.png.
#
# tests/benchmark/funsd_gt_words.json is indexed by THAT order — entry i is the
# word bag of the i-th sorted testing image — so the sort and the rename are
# load-bearing, not cosmetic. (Verified 2026-08-03: the reference cache's pages
# are pixel-identical to the sorted official images; only the PNG encoding
# differs, which cv::imread erases.)
#
#   scripts/models/fetch/fetch_funsd_cache.sh [dest-dir]   # default ./funsd_cache
set -euo pipefail

DEST="${1:-funsd_cache}"
URL="https://guillaumejaume.github.io/FUNSD/dataset.zip"
# Pinned; the dataset page has served this exact archive for years. If upstream
# re-rolls the zip, verify the 50 testing images are unchanged before bumping.
SHA256="c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"

if [ -d "$DEST" ] && [ "$(ls "$DEST"/funsd_*.png 2>/dev/null | wc -l)" -eq 50 ]; then
  echo "funsd cache already present at $DEST (50 pages)"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fsSL -o "$TMP/dataset.zip" "$URL"
# shasum on mac, sha256sum on most Linux images — accept either.
if command -v shasum >/dev/null 2>&1; then
  echo "$SHA256  $TMP/dataset.zip" | shasum -a 256 -c - >/dev/null
else
  echo "$SHA256  $TMP/dataset.zip" | sha256sum -c - >/dev/null
fi

unzip -q "$TMP/dataset.zip" 'dataset/testing_data/images/*' -d "$TMP"
mapfile -t IMGS < <(ls "$TMP"/dataset/testing_data/images/*.png | sort)
if [ "${#IMGS[@]}" -ne 50 ]; then
  echo "FATAL: expected 50 testing images, got ${#IMGS[@]} — dataset layout changed" >&2
  exit 1
fi

mkdir -p "$DEST"
i=0
for f in "${IMGS[@]}"; do
  cp "$f" "$(printf '%s/funsd_%03d.png' "$DEST" "$i")"
  i=$((i + 1))
done
echo "funsd cache built: $DEST (50 pages)"
