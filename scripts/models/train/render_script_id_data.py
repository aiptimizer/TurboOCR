#!/usr/bin/env python3
"""Render synthetic 32x128 RGB text-line crops for the 7-class script_id classifier.

Class order (load-bearing — must match ScriptId enum in include/turbo_ocr/models/lang_cls/script_id.h):
    0=latin, 1=han, 2=cyrillic, 3=arabic, 4=greek, 5=hangul, 6=thai

Output:
    /tmp/script_id_data/<class_name>/*.png         (32x128 RGB)
    /tmp/script_id_data/index.json                 ({class_name: [filenames]})

Rendering: PIL with system Noto fonts + downloaded Noto Sans SC / KR. Per-crop
randomization: font weight, font size, x-offset, background brightness, foreground
brightness. The classifier reads stroke statistics — we want enough geometric
variety that no class can be solved by mean-pixel alone.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- per-class corpora ----------------------------------------------------

LATIN_WORDS = (
    "the quick brown fox jumps over a lazy dog and runs through the fields "
    "while reading example sentences from various books and journals about "
    "modern machine learning research published in conferences last year"
).split()

# Use a wider random sample of common Chinese characters.
HAN_POOL = (
    "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下"
    "得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小"
    "军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从年明因"
    "动力百立长心因总现化两本表只手部相民开第知山理"
)

CYRILLIC_WORDS = (
    "привет мир пример предложение этот текст является просто демонстрацией "
    "случайных слов на кириллице для генерации тренировочных изображений "
    "классификатора алфавитов многоязычная оптическая распознавание символов"
).split()

GREEK_WORDS = (
    "γειά σου κόσμε παράδειγμα πρότασης αυτή είναι μια απλή επίδειξη "
    "τυχαίων λέξεων στα ελληνικά για τη δημιουργία εικόνων εκπαίδευσης "
    "ταξινομητής αλφαβήτων πολυγλωσσική οπτική αναγνώριση χαρακτήρων"
).split()

# Arabic: render right-to-left text. PIL handles BiDi at render time for
# Arabic-script codepoints when the font's GSUB tables drive contextual shaping;
# the modern Pillow (>=10) does this via libraqm if present, otherwise it falls
# back to glyph-by-glyph which is still close enough for a script-shape
# classifier (we're not predicting characters).
ARABIC_WORDS = (
    "مرحبا بالعالم مثال جملة هذا النص هو مجرد عرض توضيحي لكلمات عشوائية "
    "باللغة العربية لإنشاء صور تدريب لمصنف الأبجدية متعدد اللغات الاعتراف "
    "البصري على الحروف نص فني تجريبي للتعلم الآلي العميق"
).split()

THAI_WORDS = (
    "สวัสดีชาวโลกตัวอย่างประโยคนี้เป็นเพียงการสาธิตคำสุ่มในภาษาไทย"
    "เพื่อสร้างภาพฝึกอบรมสำหรับตัวจำแนกประเภทอักษรหลายภาษาการรู้จำ"
    "ตัวอักษรด้วยแสง"
).split()

HANGUL_POOL = (
    "안녕하세요 세계 예제 문장입니다 이것은 단순히 한국어 무작위 단어를 "
    "보여주는 예시이며 다국어 알파벳 분류기를 위한 학습 이미지를 "
    "생성하기 위한 것입니다 문자 인식 광학 시스템 자동화 학습"
).split()


# --- fonts -----------------------------------------------------------------

SYSTEM = "/usr/share/fonts"

FONT_POOLS = {
    "latin": [
        f"{SYSTEM}/TTF/DejaVuSans.ttf",
        f"{SYSTEM}/TTF/DejaVuSans-Bold.ttf",
        f"{SYSTEM}/gnu-free/FreeSans.otf",
        f"{SYSTEM}/gnu-free/FreeSerif.otf",
        f"{SYSTEM}/noto/NotoSans-Regular.ttf",
    ],
    "cyrillic": [
        f"{SYSTEM}/TTF/DejaVuSans.ttf",
        f"{SYSTEM}/TTF/DejaVuSans-Bold.ttf",
        f"{SYSTEM}/gnu-free/FreeSans.otf",
        f"{SYSTEM}/gnu-free/FreeSerif.otf",
        f"{SYSTEM}/noto/NotoSans-Regular.ttf",
    ],
    "greek": [
        f"{SYSTEM}/TTF/DejaVuSans.ttf",
        f"{SYSTEM}/TTF/DejaVuSans-Bold.ttf",
        f"{SYSTEM}/gnu-free/FreeSans.otf",
        f"{SYSTEM}/gnu-free/FreeSerif.otf",
        f"{SYSTEM}/noto/NotoSans-Regular.ttf",
    ],
    "arabic": [
        f"{SYSTEM}/noto/NotoSansArabic-Regular.ttf",
        f"{SYSTEM}/noto/NotoSansArabic-Bold.ttf",
        f"{SYSTEM}/noto/NotoSansArabic-Medium.ttf",
    ],
    "thai": [
        f"{SYSTEM}/noto/NotoSansThai-Regular.ttf",
        f"{SYSTEM}/noto/NotoSansThai-Bold.ttf",
        f"{SYSTEM}/noto/NotoSansThaiLooped-Regular.ttf",
    ],
    "han": [
        "/tmp/script_id_fonts/NotoSansSC-Regular.otf",
    ],
    "hangul": [
        "/tmp/script_id_fonts/NotoSansKR-Regular.otf",
    ],
}

CORPUS = {
    "latin": LATIN_WORDS,
    "cyrillic": CYRILLIC_WORDS,
    "greek": GREEK_WORDS,
    "arabic": ARABIC_WORDS,
    "thai": THAI_WORDS,
    "han": list(HAN_POOL),
    "hangul": HANGUL_POOL,
}


CLASS_ORDER = ["latin", "han", "cyrillic", "arabic", "greek", "hangul", "thai"]

H, W = 32, 128


def gen_text(klass: str, rng: random.Random) -> str:
    pool = CORPUS[klass]
    if klass in ("han",):
        # treat as character pool — pick a random span
        n = rng.randint(4, 9)
        return "".join(rng.choices(pool, k=n))
    if klass == "thai":
        # thai is unsegmented; pick contiguous slice of the joined string
        joined = "".join(pool)
        if len(joined) <= 12:
            return joined
        n = rng.randint(8, 18)
        start = rng.randint(0, len(joined) - n)
        return joined[start : start + n]
    # word-based pools
    n = rng.randint(2, 5)
    return " ".join(rng.sample(pool, k=min(n, len(pool))))


def render_one(klass: str, rng: random.Random) -> Image.Image:
    text = gen_text(klass, rng)
    font_path = rng.choice(FONT_POOLS[klass])
    size = rng.randint(18, 28)
    font = ImageFont.truetype(font_path, size)
    bg = rng.randint(220, 255)
    fg = rng.randint(0, 80)
    img = Image.new("RGB", (W, H), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    # measure to position roughly centered
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font.getlength(text), size
    x = rng.randint(-4, max(2, W - int(tw) - 2))
    y = rng.randint(-2, max(2, H - int(th) - 2))
    d.text((x, y), text, font=font, fill=(fg, fg, fg))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/script_id_data")
    ap.add_argument("--per-class", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    index = {k: [] for k in CLASS_ORDER}
    rng = random.Random(args.seed)

    for klass in CLASS_ORDER:
        cls_dir = out_root / klass
        cls_dir.mkdir(exist_ok=True)
        # purge prior contents
        for old in cls_dir.glob("*.png"):
            old.unlink()
        for i in range(args.per_class):
            img = render_one(klass, rng)
            fname = f"{i:06d}.png"
            img.save(cls_dir / fname, optimize=False, compress_level=1)
            index[klass].append(fname)
        print(f"[render] {klass}: {len(index[klass])}", flush=True)

    with open(out_root / "index.json", "w") as f:
        json.dump(index, f)
    print(f"[render] wrote index -> {out_root/'index.json'}", flush=True)


if __name__ == "__main__":
    main()
