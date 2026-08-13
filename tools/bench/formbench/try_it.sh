#!/usr/bin/env bash
# One command to see ?fields=1 work on a document.
#
#   tools/bench/formbench/try_it.sh <some.pdf>
#
# Starts nothing — point FIELD_SERVER at a running turboocr-server, or use the
# default. Prints what was detected and where, so a wrong answer is visible
# rather than buried in JSON.
set -euo pipefail

PDF="${1:?usage: try_it.sh <file.pdf>}"
SERVER="${FIELD_SERVER:-http://127.0.0.1:8099}"

if ! curl -sf -m 3 "$SERVER/health" >/dev/null; then
  echo "No server at $SERVER. Start one with:" >&2
  echo "  FIELD_MODEL_ONNX=models/forms/ffdetr.onnx \\" >&2
  echo "    ./build-unified/turboocr-server --http-port 8099 --backend cpu" >&2
  exit 1
fi

echo "POST $SERVER/ocr/pdf?fields=1   <-- $PDF"
curl -s -X POST "$SERVER/ocr/pdf?fields=1" \
     --data-binary "@$PDF" -H "Content-Type: application/pdf" \
| python3 -c '
import json, sys, collections
doc = json.load(sys.stdin)
for pg in doc["pages"]:
    f = pg.get("fields", [])
    print(f"\npage {pg[\"page_index\"]}  raster {pg[\"width\"]}x{pg[\"height\"]}  -> {len(f)} field(s)")
    print("  by type   :", dict(collections.Counter(x["type"] for x in f)))
    print("  by source :", dict(collections.Counter(x["source"] for x in f).most_common(6)))
    runs = collections.Counter(x["group"] for x in f if x.get("group", -1) >= 0)
    if runs:
        print(f"  choice runs: {len(runs)} -> sizes {sorted(runs.values(), reverse=True)}")
    print("  first 12:")
    for x in f[:12]:
        b = x["bounding_box"]
        x0, y0 = min(p[0] for p in b), min(p[1] for p in b)
        x1, y1 = max(p[0] for p in b), max(p[1] for p in b)
        g = f"  group={x[\"group\"]}" if x.get("group", -1) >= 0 else ""
        lab = (x["label"] or "")[:24]
        print(f"    {x[\"type\"]:9s} {x[\"confidence\"]:.2f} "
              f"[{x0:5.0f},{y0:5.0f} {x1-x0:4.0f}x{y1-y0:3.0f}] "
              f"{x[\"source\"]:24s} {lab!r}{g}")
'
