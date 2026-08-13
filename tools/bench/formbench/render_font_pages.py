#!/usr/bin/env python3
"""Render one mock scanned page per installed font, for evaluating type detection.

Reads the manifest from font_ground_truth.py and, for every face, sets the same
handful of lines and saves them as a page raster — plus optional scan
degradation, because a clean render is not what the estimator meets in
production and measuring against one would flatter it.

    python3 tools/bench/formbench/font_ground_truth.py > /tmp/fonts.tsv
    python3 tools/bench/formbench/render_font_pages.py /tmp/fonts.tsv /tmp/fontpages

Writes /tmp/fontpages/*.png and /tmp/fontpages/manifest.tsv (png, label, family).
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Deliberately mixed: ascenders and descenders, capitals, digits, a line of
# form labels, and a line with the slash-and-capitals shape that broke the
# first serif detector.
LINES = [
    "Handgloves quick brown fox",
    "Invoice total due on receipt",
    "PLZ / Ort: Hamburg 20095",
    "The quality of mercy is not strained",
    "Name Vorname Strasse Telefon",
    "Reference 4711-A jumps over it",
]

PAGE_W = 900
MARGIN = 40
LINE_STEP = 62
FONT_PX = 30  # ~11pt at 150 dpi, where scanned business documents sit


def rotate_pt(x, y, deg, cx, cy, ncx, ncy):
    """PIL rotates anticlockwise about the centre, then re-centres on expand."""
    import math
    r = math.radians(deg)
    dx, dy = x - cx, y - cy
    return (ncx + dx * math.cos(r) + dy * math.sin(r),
            ncy - dx * math.sin(r) + dy * math.cos(r))


def render(path, index, out_png, degrade, skew=0.0):
    try:
        font = ImageFont.truetype(path, FONT_PX, index=index)
    except Exception:
        return False

    height = MARGIN * 2 + LINE_STEP * len(LINES)
    img = Image.new("RGB", (PAGE_W, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    boxes = []
    inked = 0
    for i, text in enumerate(LINES):
        y = MARGIN + i * LINE_STEP
        try:
            box = draw.textbbox((MARGIN, y), text, font=font)
        except Exception:
            return False
        # A face with no glyphs for the specimen draws nothing, or draws a row
        # of .notdef boxes. Either way it says nothing about type.
        if box[2] - box[0] < 40 or box[3] - box[1] < 8:
            return False
        draw.text((MARGIN, y), text, font=font, fill=(0, 0, 0))
        boxes.append(box)
        inked += 1
    if inked != len(LINES):
        return False

    # The line quadrilaterals, carried alongside the image. A detector hands
    # back rotated quads on a skewed page; finding lines by horizontal row
    # projection instead would merge them, so an evaluation that did that would
    # be measuring its own harness rather than the estimator.
    w0, h0 = img.width, img.height

    if skew:
        # A page that went through the feeder crooked. Detection boxes then
        # arrive as rotated quadrilaterals, and anything that normalises by an
        # axis-aligned height is working at the wrong scale — so an evaluation
        # with no skew in it cannot see that class of bug at all.
        img = img.rotate(skew, resample=Image.BICUBIC, expand=True,
                         fillcolor=(255, 255, 255))

    quads = []
    cx, cy = w0 / 2.0, h0 / 2.0
    ncx, ncy = img.width / 2.0, img.height / 2.0
    for (bx0, by0, bx1, by1) in boxes:
        corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        if skew:
            corners = [rotate_pt(px, py, skew, cx, cy, ncx, ncy) for px, py in corners]
        quads.append(corners)
    with open(out_png + ".boxes", "w") as fh:
        for q in quads:
            fh.write(" ".join(f"{int(round(px))} {int(round(py))}" for px, py in q) + "\n")

    if degrade:
        # What a scanner does: a little blur, a little noise, and JPEG. Without
        # these the evaluation is measuring vector renders, which is not the
        # input this code exists to handle.
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        import random

        rnd = random.Random(1234)
        px = img.load()
        for y in range(0, img.height, 3):
            for x in range(0, img.width, 3):
                n = rnd.randint(-14, 14)
                r, g, b = px[x, y]
                px[x, y] = (
                    max(0, min(255, r + n)),
                    max(0, min(255, g + n)),
                    max(0, min(255, b + n)),
                )
        img.save(out_png, quality=72)
        return True

    img.save(out_png)
    return True


def main():
    manifest_in = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fonts.tsv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fontpages"
    degrade = "--clean" not in sys.argv
    skew = 0.0
    for a in sys.argv:
        if a.startswith("--skew="):
            skew = float(a.split("=", 1)[1])
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for line in open(manifest_in):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        path, index, label, family = parts[0], int(parts[1]), parts[2], parts[3]
        # System-internal fallback faces, which no document is set in.
        if family.startswith("."):
            continue
        safe = "".join(c if c.isalnum() else "_" for c in family)
        out_png = os.path.join(out_dir, f"{safe}.png")
        if render(path, index, out_png, degrade, skew):
            rows.append((out_png, label, family))

    with open(os.path.join(out_dir, "manifest.tsv"), "w") as fh:
        for png, label, family in rows:
            fh.write(f"{png}\t{label}\t{family}\n")

    counts = {}
    for _, label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"rendered {len(rows)} faces "
          f"({'degraded' if degrade else 'clean'}, skew {skew}deg): {counts}")


if __name__ == "__main__":
    main()
