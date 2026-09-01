#!/usr/bin/env python3
"""Generate a PEP 503 "simple" index for wheels hosted on a GitHub release.

    gen_simple_index.py --base-url \
        https://github.com/OWNER/REPO/releases/download/TAG/ \
        --out simple/  wheel1.whl [wheel2.whl ...]

Emits  <out>/index.html                    (project list)
       <out>/<normalized-name>/index.html  (one per distribution, links with
                                            #sha256= fragments and
                                            data-requires-python)

Published on GitHub Pages, this makes the release wheels resolve like PyPI:

    pip install turboocr-engine-cuda12 --extra-index-url https://<pages>/simple/

Why this exists: PyPI's per-file limit blocks the GPU wheels until their
size requests are approved (pypi/support#11962, #11963). Release assets +
this index are the standard bridge (flash-attention ships exclusively this
way). The index is REGENERATED whole from the full wheel list each run —
partial runs would silently drop distributions from the project list.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_dist_name(filename: str) -> str:
    """Distribution name from a wheel filename (the part before the version).

    Wheel filenames escape runs of [-_.] in the DISTRIBUTION part as '_', so
    splitting on '-' is safe: the first '-' ends the name.
    """
    return filename.split("-", 1)[0]


def requires_python(path: Path) -> str | None:
    """Requires-Python from the wheel's METADATA, or None."""
    with zipfile.ZipFile(path) as z:
        meta = next(
            (n for n in z.namelist()
             if n.endswith(".dist-info/METADATA") and n.count("/") == 1),
            None,
        )
        if meta is None:
            return None
        for line in z.read(meta).decode("utf-8", "replace").splitlines():
            if line.startswith("Requires-Python:"):
                return line.split(":", 1)[1].strip()
            if not line.strip():
                break  # headers end at the first blank line
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True,
                    help="URL prefix the wheel files are served from "
                         "(the release's download/ URL), with trailing slash")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory (the simple/ root)")
    ap.add_argument("wheels", nargs="+", type=Path)
    args = ap.parse_args()

    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    projects: dict[str, list[str]] = defaultdict(list)

    for w in args.wheels:
        if not w.name.endswith(".whl"):
            print(f"FATAL: not a wheel: {w}", file=sys.stderr)
            return 2
        digest = hashlib.sha256(w.read_bytes()).hexdigest()
        rp = requires_python(w)
        rp_attr = (f' data-requires-python="{html.escape(rp, quote=True)}"'
                   if rp else "")
        projects[normalize(wheel_dist_name(w.name))].append(
            f'    <a href="{base}{w.name}#sha256={digest}"{rp_attr}>'
            f"{html.escape(w.name)}</a><br/>"
        )

    for name, links in sorted(projects.items()):
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(
            "<!DOCTYPE html>\n<html>\n  <head>\n"
            f"    <title>Links for {name}</title>\n  </head>\n  <body>\n"
            f"    <h1>Links for {name}</h1>\n"
            + "\n".join(sorted(links))
            + "\n  </body>\n</html>\n"
        )

    (args.out / "index.html").write_text(
        "<!DOCTYPE html>\n<html>\n  <head>\n"
        "    <title>Simple index</title>\n  </head>\n  <body>\n"
        + "\n".join(f'    <a href="{n}/">{n}</a><br/>'
                    for n in sorted(projects))
        + "\n  </body>\n</html>\n"
    )
    print(f"wrote {args.out}/ with {len(projects)} project(s), "
          f"{sum(len(v) for v in projects.values())} file link(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
