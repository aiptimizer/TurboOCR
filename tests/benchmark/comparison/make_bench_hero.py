#!/usr/bin/env python3
"""Generate the README benchmark hero (images/bench_hero_{light,dark}.svg).

One generator, two theme variants, picked at render time by the README's
<picture> element (GitHub swaps on prefers-color-scheme). Edit DATA / HERO
below and re-run; never hand-edit the SVGs.

Design notes (kept so future edits don't regress them):
- Horizontal bars, linear scale FROM ZERO. The competitor bars being a few
  pixels wide IS the honest reading — never log-scale a bar chart.
- TurboOCR bars wear the accent; competitors wear a deliberate neutral.
  Identity never rides on color alone: every row is direct-labeled.
- Text wears ink colors (primary/secondary), never the bar color.
- Both palettes were run through the dataviz validator against light/dark
  surfaces: separation + contrast pass in both modes.
"""

import os

# (label, sublabel, img/s, F1 %, is_turbo)
DATA = [
    ("TurboOCR tiny",        "default tier",        678, 85.4, True),
    ("TurboOCR small",       "",                    230, 90.3, True),
    ("TurboOCR medium",      "most accurate",        86, 91.9, True),
    ("PaddleOCR PP-OCRv5",   "Python",                6, 86.6, False),
    ("PaddleOCR-VL 1.6",     "VLM",                   5, 91.6, False),
    ("EasyOCR",              "",                      3, 59.8, False),
    ("RapidOCR",             "GPU",                   2, 69.1, False),
    ("Tesseract",            "",                      2, 62.3, False),
]

HERO = [
    ("650+", "img/s · FUNSD"),
    ("200+", "img/s · OmniDocBench"),
    ("20",   "pages/s · full parse"),
]

THEMES = {
    "light": dict(primary="#1a1a18", secondary="#52514e", muted="#767570",
                  accent="#2a78d6", neutral="#8f8e85", hairline="#d9d8d2"),
    "dark":  dict(primary="#ffffff", secondary="#c3c2b7", muted="#8a897f",
                  accent="#3987e5", neutral="#75746c", hairline="#3a3936"),
}

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
W, H = 1200, 470
BAR_X, BAR_MAX_W = 360, 660          # bars span 360..1020; labels to the right
ROW_Y0, ROW_H, BAR_H = 132, 38, 22


def svg(theme: dict) -> str:
    top = DATA[0][2]
    px = BAR_MAX_W / top
    p = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}" role="img" '
        f'aria-label="Whole-page OCR throughput on one RTX 5090: TurboOCR '
        f'678 images per second on FUNSD vs 2 to 6 for other engines">')

    # -- header: title left, hero stats right --------------------------------
    p.append(f'<text x="30" y="52" font-size="19" font-weight="650" '
             f'fill="{theme["primary"]}">Whole-page OCR throughput</text>')
    p.append(f'<text x="30" y="76" font-size="12.5" fill="{theme["secondary"]}">'
             f'FUNSD forms · one RTX 5090 · word-F1 beside each engine</text>')

    hero_x = [640, 830, 1040]
    for (num, cap), x in zip(HERO, hero_x):
        p.append(f'<text x="{x}" y="52" font-size="24" font-weight="700" '
                 f'fill="{theme["primary"]}">{num}</text>')
        p.append(f'<text x="{x}" y="72" font-size="11" '
                 f'fill="{theme["secondary"]}">{cap}</text>')

    # -- baseline hairline ---------------------------------------------------
    y_top, y_bot = ROW_Y0 - 10, ROW_Y0 + len(DATA) * ROW_H - 8
    p.append(f'<line x1="{BAR_X}" y1="{y_top}" x2="{BAR_X}" y2="{y_bot}" '
             f'stroke="{theme["hairline"]}" stroke-width="1"/>')

    # -- bars ----------------------------------------------------------------
    for i, (name, sub, rate, f1, ours) in enumerate(DATA):
        y = ROW_Y0 + i * ROW_H
        cy = y + BAR_H / 2
        w = max(rate * px, 3)
        rx = 4 if w >= 12 else 1.5
        fill = theme["accent"] if ours else theme["neutral"]
        name_fill = theme["primary"] if ours else theme["secondary"]

        label = name + (f' <tspan font-size="11" font-weight="400" '
                        f'fill="{theme["muted"]}">{sub}</tspan>' if sub else "")
        p.append(f'<text x="{BAR_X - 12}" y="{cy - 1}" text-anchor="end" '
                 f'font-size="13" font-weight="{600 if ours else 400}" '
                 f'fill="{name_fill}">{label}</text>')
        p.append(f'<text x="{BAR_X - 12}" y="{cy + 12}" text-anchor="end" '
                 f'font-size="10.5" fill="{theme["muted"]}">F1 {f1:.1f}%</text>')
        p.append(f'<rect x="{BAR_X}" y="{y}" width="{w:.1f}" height="{BAR_H}" '
                 f'rx="{rx}" fill="{fill}"/>')
        vx = BAR_X + w + 10
        p.append(f'<text x="{vx:.1f}" y="{cy + 4.5}" font-size="13" '
                 f'font-weight="{700 if ours else 500}" '
                 f'fill="{theme["primary"]}">{rate}'
                 f'<tspan font-size="10.5" font-weight="400" '
                 f'fill="{theme["muted"]}"> img/s</tspan></text>')

    # -- footer --------------------------------------------------------------
    p.append(f'<text x="30" y="{H - 12}" font-size="10.5" '
             f'fill="{theme["muted"]}">Same 50 FUNSD pages for every engine · '
             f'≥15 s timed windows with a dual-clock cross-check · '
             f'methodology: docs/benchmarks/comparison.md</text>')
    p.append('</svg>')
    return "\n".join(p) + "\n"


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    for mode, theme in THEMES.items():
        path = os.path.join(out_dir, f"bench_hero_{mode}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg(theme))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
