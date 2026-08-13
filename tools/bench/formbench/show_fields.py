#!/usr/bin/env python3
"""Print the ?fields=1 response so a wrong answer is visible, not buried.

Reads the JSON on stdin. Used by try_it.sh; useful on its own too:

    curl -s -X POST 'localhost:8099/ocr/pdf?fields=1' --data-binary @form.pdf \\
         -H 'Content-Type: application/pdf' | tools/bench/formbench/show_fields.py
"""
import collections
import json
import sys


def main() -> int:
    doc = json.load(sys.stdin)
    for pg in doc.get("pages", []):
        fields = pg.get("fields", [])
        print(f"\npage {pg.get('page_index', 0)}  "
              f"raster {pg.get('width')}x{pg.get('height')}  "
              f"-> {len(fields)} field(s)")
        if not fields:
            print("  (none — the page has no blanks the detectors recognise)")
            continue

        by_type = collections.Counter(f["type"] for f in fields)
        by_src = collections.Counter(f["source"] for f in fields)
        print(f"  by type   : {dict(by_type)}")
        print(f"  by source : {dict(by_src.most_common(6))}")

        # 'ffdetr' in the source means the model argued for it; a source with
        # no 'ffdetr' came from page geometry alone.
        model = sum(1 for f in fields if "ffdetr" in f["source"])
        print(f"  model-backed: {model}/{len(fields)}"
              f"{'   (model not loaded?)' if model == 0 else ''}")

        runs = collections.Counter(f["group"] for f in fields
                                   if f.get("group", -1) >= 0)
        if runs:
            print(f"  choice runs : {len(runs)}, sizes "
                  f"{sorted(runs.values(), reverse=True)}")

        print("  fields (reading order):")
        for f in fields[:15]:
            b = f["bounding_box"]
            x0 = min(p[0] for p in b)
            y0 = min(p[1] for p in b)
            w = max(p[0] for p in b) - x0
            h = max(p[1] for p in b) - y0
            grp = f"  group={f['group']}" if f.get("group", -1) >= 0 else ""
            label = (f.get("label") or "")[:26]
            print(f"    {f['type']:9s} {f['confidence']:.2f} "
                  f"[{x0:5.0f},{y0:5.0f} {w:4.0f}x{h:3.0f}] "
                  f"{f['source']:24s} {label!r}{grp}")
        if len(fields) > 15:
            print(f"    ... and {len(fields) - 15} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
