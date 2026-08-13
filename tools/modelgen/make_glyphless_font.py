#!/usr/bin/env python3
"""Generate a minimal glyphless TrueType font for invisible OCR text layers.

One glyph, uniform advance 500/1000 em. The text is never drawn (render mode
3) and the characters are carried by the ToUnicode CMap supplied per document,
so the font only has to be a structurally valid CIDFontType2 donor.

The single glyph is a two-point contour spanning the em box: zero area, so it
paints nothing even if a viewer ignores the render mode, but a real bounding
box — PDFium discards rotated text whose glyphs measure empty, and viewers
size selection highlights from the glyph box.
"""
import struct, sys

UPEM, ADVANCE, ASC, DESC = 1000, 500, 800, -200
XMIN, YMIN, XMAX, YMAX = 0, DESC, ADVANCE, ASC


def head():
    return struct.pack(
        ">IIIIHHqqhhhhHHhhh",
        0x00010000, 0x00010000, 0, 0x5F0F3CF5, 3, UPEM,
        0, 0,            # created, modified
        XMIN, YMIN, XMAX, YMAX,
        0,               # macStyle
        8,               # lowestRecPPEM
        2, 0, 0)         # fontDirectionHint, indexToLocFormat, glyphDataFormat


def hhea():
    return struct.pack(
        ">IhhhHhhhhhhhhhhhH",
        0x00010000, ASC, DESC, 0, ADVANCE,
        XMIN, ADVANCE - XMAX, XMAX,  # min LSB, min RSB, xMaxExtent
        1, 0, 0,         # caret slope rise/run, offset
        0, 0, 0, 0,      # reserved
        0,               # metricDataFormat
        1)               # numberOfHMetrics


def maxp():
    # numGlyphs, then maxPoints, maxContours, maxCompositePoints/Contours, maxZones, ...
    return struct.pack(">IH", 0x00010000, 1) + struct.pack(
        ">13H", 2, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)


def hmtx():
    return struct.pack(">Hh", ADVANCE, 0)


def glyf():
    # one contour, two points: (XMIN, YMIN) -> (XMAX, YMAX). Deltas are int16.
    g = struct.pack(">hhhhh", 1, XMIN, YMIN, XMAX, YMAX)
    g += struct.pack(">H", 1)          # endPtsOfContours
    g += struct.pack(">H", 0)          # instructionLength
    g += bytes([0x01, 0x01])           # both points on-curve, int16 deltas
    g += struct.pack(">2h", XMIN, XMAX - XMIN)
    g += struct.pack(">2h", YMIN, YMAX - YMIN)
    return g


def loca():
    return struct.pack(">2H", 0, len(glyf()) // 2)


def cmap():
    sub = struct.pack(">7H", 4, 24, 0, 2, 2, 0, 0)
    sub += struct.pack(">H", 0xFFFF)          # endCode
    sub += struct.pack(">H", 0)               # reservedPad
    sub += struct.pack(">H", 0xFFFF)          # startCode
    sub += struct.pack(">h", 1)               # idDelta
    sub += struct.pack(">H", 0)               # idRangeOffset
    return struct.pack(">HHHHI", 0, 1, 3, 1, 12) + sub


def name():
    strings = {1: "TurboOCR Glyphless", 2: "Regular", 4: "TurboOCR Glyphless",
               6: "TurboOCRGlyphless"}
    records, blob = b"", b""
    for nid in sorted(strings):
        s = strings[nid].encode("utf-16-be")
        records += struct.pack(">HHHHHH", 3, 1, 0x409, nid, len(s), len(blob))
        blob += s
    return struct.pack(">HHH", 0, len(strings), 6 + 12 * len(strings)) + records + blob


def post():
    return struct.pack(">IIhhIIIII", 0x00030000, 0, -100, 50, 1, 0, 0, 0, 0)


def os2():
    return (struct.pack(">HhHHH", 4, ADVANCE, 400, 5, 0)
            + struct.pack(">8h", 650, 700, 0, 140, 650, 700, 0, 480)
            + struct.pack(">hh", 50, 300)
            + struct.pack(">h", 0) + b"\x02\x00\x06\x03\x00\x00\x00\x00\x00\x00"
            + struct.pack(">4I", 1, 0, 0, 0) + b"TOCR"
            + struct.pack(">HHH", 0x40, 0, 0xFFFF)
            + struct.pack(">hhh", ASC, DESC, 0)
            + struct.pack(">HH", ASC, -DESC)
            + struct.pack(">2I", 1, 0)
            + struct.pack(">hh", 500, 700)
            + struct.pack(">HHH", 0, 0, 0))


def checksum(data):
    data = data + b"\0" * (-len(data) % 4)
    return sum(struct.unpack(">%dI" % (len(data) // 4), data)) & 0xFFFFFFFF


def build():
    tables = {b"OS/2": os2(), b"cmap": cmap(), b"glyf": glyf(), b"head": head(),
              b"hhea": hhea(), b"hmtx": hmtx(), b"loca": loca(),
              b"maxp": maxp(), b"name": name(), b"post": post()}
    tags = sorted(tables)
    n = len(tags)
    entry_selector = max(n.bit_length() - 1, 0)
    search_range = (1 << entry_selector) * 16
    out = struct.pack(">IHHHH", 0x00010000, n, search_range, entry_selector,
                      n * 16 - search_range)
    offset = 12 + 16 * n
    directory, body = b"", b""
    for tag in tags:
        data = tables[tag]
        directory += struct.pack(">4sIII", tag, checksum(data), offset, len(data))
        pad = b"\0" * (-len(data) % 4)
        body += data + pad
        offset += len(data) + len(pad)
    font = out + directory + body
    adjustment = (0xB1B0AFBA - checksum(font)) & 0xFFFFFFFF
    head_off = font.index(b"head")
    head_start = struct.unpack(">I", font[head_off + 8:head_off + 12])[0]
    font = font[:head_start + 8] + struct.pack(">I", adjustment) + font[head_start + 12:]
    return font


if __name__ == "__main__":
    font = build()
    open(sys.argv[1], "wb").write(font)
    print(f"{len(font)} bytes -> {sys.argv[1]}")
