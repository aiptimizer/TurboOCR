#!/usr/bin/env python3
"""Label every installed font serif / sans / mono, from the font's own metadata.

Ground truth for the type estimator has to come from somewhere other than the
estimator, and it cannot come from me eyeballing 200 typefaces. It comes from
the fonts: OpenType requires a PANOSE classification in the OS/2 table, whose
second byte says which serif style the designer assigned and whose fourth says
whether the face is monospaced, and the `post` table carries an independent
isFixedPitch flag. Those are the designer's own answers.

Emits a TSV manifest: path, face-index, label, family-name. Faces PANOSE declines to
classify, and anything that is not Latin text (symbol, script, CJK, icon), are
dropped rather than guessed at — the estimator is not asked about them either.

    python3 tools/bench/formbench/font_ground_truth.py > /tmp/fonts.tsv
"""

import os
import struct
import sys

FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts",
]

# PANOSE bSerifStyle. 2-10 are the serif treatments, 11-13 the sans ones.
# 14 (flared) and 15 (rounded) are sans-serif designs too — Optima is flared,
# and no reader would call it a serif face.
SERIF_STYLES = set(range(2, 11))
SANS_STYLES = {11, 12, 13, 14, 15}


def _dir_at(data, base):
    """Table directory at `base`, as {tag: (offset, length)}."""
    if base + 12 > len(data):
        return None
    num = struct.unpack(">H", data[base + 4 : base + 6])[0]
    tables = {}
    for i in range(num):
        off = base + 12 + i * 16
        if off + 16 > len(data):
            break
        name, _, toff, tlen = struct.unpack(">4sIII", data[off : off + 16])
        tables[name] = (toff, tlen)
    return tables


def read_faces(path):
    """Every face in the file, as (index, tables). Handles .ttc collections.

    macOS keeps most of its text faces — Baskerville, Bodoni, Palatino,
    Athelas, American Typewriter — inside collections, so a reader that only
    understands single-face files sees almost none of the interesting type.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 12:
        return None, []
    tag = data[:4]
    if tag == b"ttcf":
        count = struct.unpack(">I", data[8:12])[0]
        out = []
        for i in range(min(count, 64)):
            off = 12 + i * 4
            if off + 4 > len(data):
                break
            base = struct.unpack(">I", data[off : off + 4])[0]
            tables = _dir_at(data, base)
            if tables:
                out.append((i, tables))
        return data, out
    if tag in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
        tables = _dir_at(data, 0)
        return (data, [(0, tables)]) if tables else (None, [])
    return None, []


# Every character the evaluation actually sets. A face is only admitted if it
# has a glyph for all of them.
REQUIRED = set("HandglovesquickbrownfxIitTPLZO/:0123456789-ABCDEFMNRSVW ")


def _cmap_lookup(data, base, chars):
    """Glyph ids for `chars` under the cmap subtable at `base`, or None."""
    fmt = struct.unpack(">H", data[base : base + 2])[0]
    out = {}
    if fmt == 4:
        seg2 = struct.unpack(">H", data[base + 6 : base + 8])[0]
        seg = seg2 // 2
        ends = base + 14
        starts = ends + seg2 + 2
        deltas = starts + seg2
        ranges = deltas + seg2
        for ch in chars:
            code = ord(ch)
            gid = 0
            for i in range(seg):
                end = struct.unpack(">H", data[ends + i * 2 : ends + i * 2 + 2])[0]
                if code > end:
                    continue
                start = struct.unpack(">H", data[starts + i * 2 : starts + i * 2 + 2])[0]
                if code < start:
                    break
                delta = struct.unpack(">h", data[deltas + i * 2 : deltas + i * 2 + 2])[0]
                ro = struct.unpack(">H", data[ranges + i * 2 : ranges + i * 2 + 2])[0]
                if ro == 0:
                    gid = (code + delta) & 0xFFFF
                else:
                    addr = ranges + i * 2 + ro + (code - start) * 2
                    if addr + 2 > len(data):
                        gid = 0
                    else:
                        g = struct.unpack(">H", data[addr : addr + 2])[0]
                        gid = 0 if g == 0 else (g + delta) & 0xFFFF
                break
            out[ch] = gid
        return out
    if fmt == 12:
        n = struct.unpack(">I", data[base + 12 : base + 16])[0]
        groups = base + 16
        for ch in chars:
            code = ord(ch)
            gid = 0
            for i in range(min(n, 20000)):
                g = groups + i * 12
                if g + 12 > len(data):
                    break
                lo, hi, first = struct.unpack(">III", data[g : g + 12])
                if lo <= code <= hi:
                    gid = first + (code - lo)
                    break
                if code < lo:
                    break
            out[ch] = gid
        return out
    return None


def has_basic_latin(data, tables):
    """The font has a real glyph for every character the specimen uses.

    The OS/2 Unicode-range bits are not enough on their own, and trusting them
    cost a tenth of the corpus: the Noto Serif faces for Myanmar, Ahom, Yezidi
    and the rest all claim Basic Latin, then set the specimen as a row of empty
    .notdef boxes. Those scored at chance against every candidate and were
    counted as detection failures when the real fault was that there was
    nothing there to detect. cmap is the only thing that actually answers it.
    """
    if b"cmap" not in tables:
        return False
    off, _ = tables[b"cmap"]
    if off + 4 > len(data):
        return False
    count = struct.unpack(">H", data[off + 2 : off + 4])[0]
    best = None
    for i in range(count):
        rec = off + 4 + i * 8
        if rec + 8 > len(data):
            break
        pid, eid, sub = struct.unpack(">HHI", data[rec : rec + 8])
        if (pid, eid) in ((3, 1), (3, 10), (0, 3), (0, 4), (0, 6)):
            got = _cmap_lookup(data, off + sub, REQUIRED)
            if got:
                best = got
                if (pid, eid) == (3, 1):
                    break
    if not best:
        return False
    return all(g != 0 for g in best.values())


def family_name(data, tables):
    """The name table's family entry, for readable output."""
    if b"name" not in tables:
        return "?"
    off, _ = tables[b"name"]
    if off + 6 > len(data):
        return "?"
    count, string_off = struct.unpack(">HH", data[off + 2 : off + 6])
    best = "?"
    for i in range(count):
        rec = off + 6 + i * 12
        if rec + 12 > len(data):
            break
        pid, eid, lid, nid, ln, no = struct.unpack(">HHHHHH", data[rec : rec + 12])
        if nid != 1:
            continue
        start = off + string_off + no
        raw = data[start : start + ln]
        try:
            text = raw.decode("utf-16-be" if pid == 3 else "latin-1", "ignore")
        except Exception:
            continue
        text = text.strip()
        if text:
            best = text
            if pid == 3:
                break
    return best


def classify_face(data, tables):
    if not tables or b"OS/2" not in tables:
        return None
    if not has_basic_latin(data, tables):
        return None
    off, ln = tables[b"OS/2"]
    if off + 42 > len(data) or ln < 42:
        return None
    family_class = struct.unpack(">h", data[off + 30 : off + 32])[0] >> 8
    panose = data[off + 32 : off + 42]
    if len(panose) < 4:
        return None
    kind, serif_style, _weight, proportion = panose[0], panose[1], panose[2], panose[3]

    # Only Latin text faces. Decorative, script, symbol and icon fonts are not
    # what a scanned business document is set in, and the estimator is never
    # asked to answer for them.
    if kind != 2 or family_class in (10, 12):
        return None

    fixed = False
    if b"post" in tables:
        poff, _ = tables[b"post"]
        if poff + 16 <= len(data):
            fixed = struct.unpack(">I", data[poff + 12 : poff + 16])[0] != 0
    # PANOSE proportion 9 is "Monospaced"; the post flag is the independent
    # second opinion. Either one is enough.
    if fixed or proportion == 9:
        return "mono", family_name(data, tables)

    if serif_style in SERIF_STYLES:
        return "serif", family_name(data, tables)
    if serif_style in SANS_STYLES:
        return "sans", family_name(data, tables)
    # 0 = "any" and 1 = "no fit" mean the designer declined to say. Fall back to
    # the OS/2 family class, which encodes 8 for sans-serif and 1-7 for the
    # serif groups; if that is also silent, drop the face.
    if family_class == 8:
        return "sans", family_name(data, tables)
    if 1 <= family_class <= 7:
        return "serif", family_name(data, tables)
    return None


def main():
    seen = set()
    rows = []
    for directory in FONT_DIRS:
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if not entry.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            path = os.path.join(directory, entry)
            try:
                data, faces = read_faces(path)
            except Exception:
                continue
            if not data:
                continue
            for index, tables in faces:
                try:
                    got = classify_face(data, tables)
                except Exception:
                    continue
                if not got:
                    continue
                label, family = got
                key = (family, label)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((path, index, label, family))

    for path, index, label, family in rows:
        print(f"{path}\t{index}\t{label}\t{family}")
    counts = {}
    for _, _, label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"# {len(rows)} faces: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
