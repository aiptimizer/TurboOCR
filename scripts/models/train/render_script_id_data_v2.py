#!/usr/bin/env python3
"""v2 multi-font renderer for the 7-class script_id training set.

Trainer class order (load-bearing — must match render output dirs):
    0=latin, 1=han, 2=cyrillic, 3=arabic, 4=greek, 5=hangul, 6=thai
Final ONNX will be permuted to canonical C++ enum order by the trainer.

Output:
    /tmp/script_id_data_v2/<class>/*.png      (32x128 RGB, ≥14k per class)
    /tmp/script_id_data_v2/index.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SYS = "/usr/share/fonts"
DL = "/tmp/script_id_fonts"

FONT_POOLS = {
    "latin": [
        f"{SYS}/TTF/DejaVuSans.ttf",
        f"{SYS}/TTF/DejaVuSans-Bold.ttf",
        f"{SYS}/gnu-free/FreeSans.otf",
        f"{SYS}/gnu-free/FreeSerif.otf",
        f"{SYS}/noto/NotoSans-Regular.ttf",
        f"{SYS}/liberation/LiberationSans-Regular.ttf",
        f"{SYS}/liberation/LiberationSerif-Regular.ttf",
        f"{SYS}/liberation/LiberationSans-Bold.ttf",
    ],
    "cyrillic": [
        f"{SYS}/TTF/DejaVuSans.ttf",
        f"{SYS}/TTF/DejaVuSans-Bold.ttf",
        f"{SYS}/gnu-free/FreeSans.otf",
        f"{SYS}/gnu-free/FreeSerif.otf",
        f"{SYS}/noto/NotoSans-Regular.ttf",
        f"{SYS}/liberation/LiberationSans-Regular.ttf",
        f"{SYS}/liberation/LiberationSerif-Regular.ttf",
    ],
    "greek": [
        f"{SYS}/TTF/DejaVuSans.ttf",
        f"{SYS}/TTF/DejaVuSans-Bold.ttf",
        f"{SYS}/gnu-free/FreeSans.otf",
        f"{SYS}/gnu-free/FreeSerif.otf",
        f"{SYS}/noto/NotoSans-Regular.ttf",
        f"{SYS}/liberation/LiberationSerif-Regular.ttf",
    ],
    "arabic": [
        f"{SYS}/noto/NotoSansArabic-Regular.ttf",
        f"{SYS}/noto/NotoSansArabic-Bold.ttf",
        f"{SYS}/noto/NotoSansArabic-Medium.ttf",
        f"{DL}/Amiri-Regular.ttf",
        f"{DL}/Amiri-Bold.ttf",
    ],
    "thai": [
        f"{SYS}/noto/NotoSansThai-Regular.ttf",
        f"{SYS}/noto/NotoSansThai-Bold.ttf",
        f"{SYS}/noto/NotoSansThaiLooped-Regular.ttf",
        f"{SYS}/noto/NotoSerifThai-Regular.ttf",
        f"{SYS}/noto/NotoSerifThai-Bold.ttf",
    ],
    "han": [
        f"{DL}/NotoSansSC-Regular.otf",
        f"{DL}/NotoSerifSC-Regular.otf",
    ],
    "hangul": [
        f"{DL}/NotoSansKR-Regular.otf",
        f"{DL}/NotoSerifKR-Regular.otf",
    ],
}


LATIN_WORDS = (
    "the quick brown fox jumps over a lazy dog and runs through the fields "
    "while reading example sentences from various books and journals about "
    "modern machine learning research published in conferences last year "
    "mathematics physics chemistry biology engineering computer science "
    "artificial intelligence neural network optimization regression"
).split()

HAN_POOL = (
    "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下"
    "得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小"
    "军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从年明因"
    "动力百立长心因总现化两本表只手部相民开第知山理科研究教育数学"
)

CYRILLIC_WORDS = (
    "привет мир пример предложение этот текст является просто демонстрацией "
    "случайных слов на кириллице для генерации тренировочных изображений "
    "классификатора алфавитов многоязычная оптическая распознавание символов "
    "наука техника искусство литература история философия математика физика"
).split()

GREEK_WORDS = (
    "γειά σου κόσμε παράδειγμα πρότασης αυτή είναι μια απλή επίδειξη "
    "τυχαίων λέξεων στα ελληνικά για τη δημιουργία εικόνων εκπαίδευσης "
    "ταξινομητής αλφαβήτων πολυγλωσσική οπτική αναγνώριση χαρακτήρων "
    "μαθηματικά φυσική χημεία βιολογία ιστορία φιλοσοφία"
).split()

ARABIC_WORDS = (
    "مرحبا بالعالم مثال جملة هذا النص هو مجرد عرض توضيحي لكلمات عشوائية "
    "باللغة العربية لإنشاء صور تدريب لمصنف الأبجدية متعدد اللغات الاعتراف "
    "البصري على الحروف نص فني تجريبي للتعلم الآلي العميق الذكاء"
).split()

THAI_WORDS = (
    "สวัสดีชาวโลกตัวอย่างประโยคนี้เป็นเพียงการสาธิตคำสุ่มในภาษาไทย"
    "เพื่อสร้างภาพฝึกอบรมสำหรับตัวจำแนกประเภทอักษรหลายภาษาการรู้จำ"
    "ตัวอักษรด้วยแสงคณิตศาสตร์ฟิสิกส์เคมีชีววิทยาประวัติศาสตร์"
).split()

HANGUL_POOL = (
    "안녕하세요 세계 예제 문장입니다 이것은 단순히 한국어 무작위 단어를 "
    "보여주는 예시이며 다국어 알파벳 분류기를 위한 학습 이미지를 "
    "생성하기 위한 것입니다 문자 인식 광학 시스템 자동화 학습 수학 과학 "
    "기술 공학 인공지능 신경망 컴퓨터 프로그래밍"
).split()

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
    if klass == "han":
        return "".join(rng.choices(pool, k=rng.randint(4, 9)))
    if klass == "thai":
        joined = "".join(pool)
        n = rng.randint(8, 18)
        if len(joined) <= n:
            return joined
        s = rng.randint(0, len(joined) - n)
        return joined[s : s + n]
    n = rng.randint(2, 5)
    return " ".join(rng.sample(pool, k=min(n, len(pool))))


def render_one(klass: str, rng: random.Random) -> Image.Image:
    text = gen_text(klass, rng)
    font_path = rng.choice(FONT_POOLS[klass])
    size = rng.randint(16, 24)
    font = ImageFont.truetype(font_path, size)
    bg_v = rng.randint(240, 255)
    bg = (bg_v, bg_v, bg_v)
    # Mostly dark fg, occasionally mid-gray
    if rng.random() < 0.85:
        fg_v = rng.randint(0, 50)
    else:
        fg_v = rng.randint(50, 100)
    fg = (fg_v, fg_v, fg_v)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font.getlength(text), size
    x = rng.randint(-4, max(2, W - int(tw) - 2))
    y = rng.randint(-2, max(2, H - int(th) - 2))
    d.text((x, y), text, font=font, fill=fg)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/script_id_data_v2")
    ap.add_argument("--per-class", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index = {}
    for klass in CLASS_ORDER:
        d = out_root / klass
        d.mkdir(exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        for i in range(args.per_class):
            img = render_one(klass, rng)
            img.save(d / f"{i:06d}.png", optimize=False, compress_level=1)
        index[klass] = args.per_class
        print(f"[render] {klass}: {args.per_class}", flush=True)
    (out_root / "index.json").write_text(json.dumps(index))
    print(f"[render] done -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
