"""Cross-language Docker smoke test.

For every (image, language) pair:
  1. Boot a fresh container with a fresh named volume
  2. Wait for nginx /health on the mapped port
  3. POST a rendered per-language test image to /ocr/raw
  4. Score character recall against the rendered ground truth
  5. Tear down the container + volume

Prove-of-correctness: matches the server's startup flow exactly (entrypoint
validates OCR_LANG, downloads the bundle if needed, TRT/ORT loads it), so
this is what a real deployment does on first boot.

Usage (from repo root, Docker images turbo-ocr:prod and turbo-ocr-cpu:prod
already built):
    python tests/docker_language_matrix.py
    python tests/docker_language_matrix.py --image gpu --only latin,chinese

Exit 0 when all checked cells pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build" / "docker_lang_matrix"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    # (lang, font, lines, min_char_recall)
    ("latin",   "/usr/share/fonts/TTF/DejaVuSans.ttf", [
        "Invoice INV-2024-001",
        "Total USD 1,234.56",
        "Acme Corporation",
    ], 0.90),
    ("chinese", "/tmp/ocr_zh_bench/NotoSansSC.otf", [
        "发票编号 INV-2024-001",
        "金额 合计 壹万贰仟元整",
        "会议通知请全体员工",
    ], 0.90),
    ("greek",   "/usr/share/fonts/TTF/DejaVuSans.ttf", [
        "Αθήνα Ελλάδα Θεσσαλονίκη",
        "Ωραίος καιρός σήμερα",
        "Καλημέρα κόσμε",
    ], 0.95),
    ("eslav",   "/usr/share/fonts/noto/NotoSans-Regular.ttf", [
        "Добро пожаловать",
        "Москва Санкт-Петербург",
        "Привет мир",
    ], 0.95),
    ("arabic",  "/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf", [
        "مرحبا بالعالم",
        "القاهرة دبي الرياض",
        "اللغة العربية",
    ], 0.90),
    ("korean",  "/tmp/fonts/NotoSansKR.otf", [
        "안녕하세요 한국",
        "서울 부산 제주도",
        "오늘 날씨 좋다",
    ], 0.95),
    ("thai",    "/usr/share/fonts/noto/NotoSansThai-Regular.ttf", [
        "สวัสดีชาวโลก",
        "กรุงเทพ เชียงใหม่",
        "ภาษาไทย",
    ], 0.95),
]


def render(lang: str, font_path: str, lines: list[str]) -> Path:
    font = ImageFont.truetype(font_path, 36)
    w, h = 900, 60 + 70 * len(lines)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 70), line, font=font, fill="black")
    path = OUT / f"{lang}.png"
    img.save(path)
    return path


def _bag(s: str) -> Counter[str]:
    return Counter(c for c in s if not c.isspace()
                   and c not in ".,:;!?\"'()[]{}—–-")


def recall(pred: str, gt: str) -> float:
    pb, gb = _bag(pred), _bag(gt)
    return sum((pb & gb).values()) / max(1, sum(gb.values()))


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def test_case(image: str, lang: str, image_path: Path, gt: str, min_recall: float,
              port: int) -> tuple[str, float, str]:
    name = f"turbo-lang-{image}-{lang}"
    volume = f"turbo-lang-{image}-{lang}-vol"
    _run(["docker", "rm", "-f", name])
    _run(["docker", "volume", "rm", volume])
    _run(["docker", "volume", "create", volume])

    cmd = ["docker", "run", "-d", "--name", name,
           "-p", f"{port}:8000",
           "-v", f"{volume}:/home/ocr/.cache/turbo-ocr",
           "-e", f"OCR_LANG={lang}",
           "-e", "PIPELINE_POOL_SIZE=1",
           "-e", "DISABLE_LAYOUT=1"]
    if image == "gpu":
        cmd.insert(3, "--gpus")
        cmd.insert(4, "all")
        tag = "turbo-ocr:prod"
    else:
        tag = "turbo-ocr-cpu:prod"
    cmd.append(tag)

    boot = _run(cmd)
    if boot.returncode != 0:
        return ("DOCKER_FAIL", 0.0, boot.stderr.strip()[:200])

    # Wait for /health. TRT engine build on GPU can take ~90s; CPU is fast.
    deadline = time.time() + 240
    while time.time() < deadline:
        if not _run(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip() == "true":
            # container exited — read logs
            logs = _run(["docker", "logs", "--tail", "30", name]).stdout
            _run(["docker", "rm", "-f", name])
            _run(["docker", "volume", "rm", volume])
            return ("EXITED", 0.0, logs.strip()[:200])
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=1)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        _run(["docker", "rm", "-f", name])
        _run(["docker", "volume", "rm", volume])
        return ("TIMEOUT", 0.0, "health check never succeeded")

    # OCR
    try:
        resp = requests.post(
            f"http://localhost:{port}/ocr/raw",
            data=image_path.read_bytes(),
            headers={"Content-Type": "image/png"},
            timeout=60)
        resp.raise_for_status()
        pred = " ".join(r["text"] for r in resp.json().get("results", []))
        score = recall(pred, gt)
        status = "PASS" if score >= min_recall else "FAIL"
        note = pred[:80]
    except Exception as e:
        status, score, note = ("ERROR", 0.0, str(e)[:150])

    _run(["docker", "rm", "-f", name])
    _run(["docker", "volume", "rm", volume])
    return (status, score, note)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", choices=["gpu", "cpu", "both"], default="both")
    parser.add_argument("--only", help="comma-separated language subset")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    images = ("gpu", "cpu") if args.image == "both" else (args.image,)

    rendered = {}
    for lang, font, lines, _thr in CASES:
        if only and lang not in only:
            continue
        if not Path(font).exists():
            print(f"[{lang}] font missing: {font}")
            continue
        rendered[lang] = (render(lang, font, lines), " ".join(lines),
                          next(thr for l, _, _, thr in CASES if l == lang))

    print(f"{'image':<5} {'lang':<8} {'status':<10} {'recall':<8} note")
    print("-" * 100)
    port = 18500
    fails = 0
    for image in images:
        for lang in [c[0] for c in CASES]:
            if lang not in rendered:
                continue
            image_path, gt, min_recall = rendered[lang]
            status, score, note = test_case(image, lang, image_path, gt,
                                            min_recall, port)
            port += 1
            mark = "PASS" if status == "PASS" else "FAIL"
            if status != "PASS":
                fails += 1
            print(f"{image:<5} {lang:<8} {status:<10} {score:<7.2%} {note}")

    print("-" * 100)
    print(f"{fails} fail(s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
