"""Multi-language OCR smoke regression.

For each supported language bundle:
  1. Render a short multi-line phrase in that script using system fonts
  2. Download the language bundle if missing
  3. Boot the server with OCR_LANG=<lang>
  4. POST the image to /ocr/raw and score character recall
  5. Teardown and move on

Goals:
  - Confirm the configurable-language path is wired for every advertised bundle
  - Detect upstream model regressions (sudden accuracy drop on a language)

Non-goals:
  - Scene-text robustness (images are clean black-on-white rendered text)
  - Absolute accuracy claims (results are from a 3-line synthetic sample)

Usage (from repo root):
    python -m venv .venv && .venv/bin/pip install Pillow requests numpy
    LD_LIBRARY_PATH=$HOME/TensorRT-10.15.1.29/lib:$LD_LIBRARY_PATH \\
        .venv/bin/python tests/language_smoketest.py

Exit code: 0 on pass, 1 if any language drops below its threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
SERVER_BIN = REPO / "build" / "paddle_highspeed_cpp"
DOWNLOAD_SCRIPT = REPO / "scripts" / "download_models.sh"
PORT = int(os.environ.get("SMOKE_PORT", "8799"))

# Per-language test cases. Fonts must be installed locally; rendered text is
# what the OCR has to read back.
CASES = [
    # (lang, font_path, lines, min_char_recall)
    ("greek",  "/usr/share/fonts/TTF/DejaVuSans.ttf", [
        "Αθήνα Ελλάδα Θεσσαλονίκη",
        "Ωραίος καιρός σήμερα",
        "Καλημέρα κόσμε",
    ], 0.95),
    ("eslav",  "/usr/share/fonts/noto/NotoSans-Regular.ttf", [
        "Добро пожаловать",
        "Москва Санкт-Петербург",
        "Привет мир",
    ], 0.95),
    ("arabic", "/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf", [
        "مرحبا بالعالم",
        "القاهرة دبي الرياض",
        "اللغة العربية",
    ], 0.90),
    ("korean", "/tmp/fonts/NotoSansKR.otf", [
        "안녕하세요 한국",
        "서울 부산 제주도",
        "오늘 날씨 좋다",
    ], 0.95),
    ("thai",   "/usr/share/fonts/noto/NotoSansThai-Regular.ttf", [
        "สวัสดีชาวโลก",
        "กรุงเทพ เชียงใหม่",
        "ภาษาไทย",
    ], 0.95),
]


def render(lang: str, font_path: str, lines: list[str], out_dir: Path) -> Path:
    font = ImageFont.truetype(font_path, 36)
    width, height = 900, 60 + 70 * len(lines)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * 70), line, font=font, fill="black")
    path = out_dir / f"{lang}.png"
    img.save(path)
    return path


def _char_bag(text: str) -> Counter[str]:
    return Counter(c for c in text if not c.isspace() and c not in ".,:;!?\"'()[]{}—–-")


def recall(pred: str, gt: str) -> float:
    pb, gb = _char_bag(pred), _char_bag(gt)
    if not gb:
        return 1.0
    return sum((pb & gb).values()) / sum(gb.values())


def ensure_bundle(lang: str) -> None:
    if (REPO / "models" / "rec" / lang / "rec.onnx").exists():
        return
    subprocess.run(["bash", str(DOWNLOAD_SCRIPT), "--lang", lang],
                   cwd=REPO, check=True)


def boot_server(lang: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "OCR_LANG": lang,
        "PORT": str(PORT),
        "PIPELINE_POOL_SIZE": "1",
        "DISABLE_LAYOUT": "1",
    })
    proc = subprocess.Popen(
        [str(SERVER_BIN)],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{PORT}/health", timeout=1).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before becoming ready (lang={lang})")
        time.sleep(1)
    raise TimeoutError(f"server did not become ready within 120s (lang={lang})")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def ocr_image(image_path: Path) -> str:
    response = requests.post(
        f"http://localhost:{PORT}/ocr/raw",
        data=image_path.read_bytes(),
        headers={"Content-Type": "image/png"},
        timeout=30,
    )
    response.raise_for_status()
    return " ".join(item["text"] for item in response.json().get("results", []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated language subset")
    parser.add_argument("--out", type=Path, default=Path("/tmp/ocr_lang_test"))
    args = parser.parse_args()

    if not SERVER_BIN.is_file():
        print(f"Server binary not found at {SERVER_BIN} — build first "
              "(cmake --build build).", file=sys.stderr)
        return 2

    selected = set(args.only.split(",")) if args.only else None
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    image_dir.mkdir(exist_ok=True)

    summary = []
    for lang, font_path, lines, threshold in CASES:
        if selected and lang not in selected:
            continue
        if not Path(font_path).exists():
            print(f"[{lang}] SKIP — font missing: {font_path}")
            summary.append((lang, "SKIP", 0.0, threshold))
            continue

        image = render(lang, font_path, lines, image_dir)
        ensure_bundle(lang)
        proc = boot_server(lang)
        try:
            pred = ocr_image(image)
        finally:
            stop_server(proc)

        gt = " ".join(lines)
        score = recall(pred, gt)
        status = "PASS" if score >= threshold else "FAIL"
        summary.append((lang, status, score, threshold))
        print(f"[{lang}] {status:<4} char_recall={score:6.1%} (min={threshold:.0%}) "
              f"gt={gt!r} pred={pred!r}")

    passes = sum(1 for _, s, *_ in summary if s == "PASS")
    skips = sum(1 for _, s, *_ in summary if s == "SKIP")
    fails = sum(1 for _, s, *_ in summary if s == "FAIL")
    print(f"\n{passes} pass · {fails} fail · {skips} skip "
          f"({len(summary)} languages checked)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
