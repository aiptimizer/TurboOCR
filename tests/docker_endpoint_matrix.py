"""Endpoint x language x image matrix tester.

For every (image in {gpu, cpu}, lang in SUPPORTED_LANGS):
  1. Boot a fresh container with a named volume + OCR_LANG=<lang>
  2. Wait for /health, tolerating slow TRT engine builds on first GPU boot
  3. Hit every public endpoint:
       language-agnostic:   /health, /health/live, /health/ready, /metrics
       image endpoints:     /ocr/raw, /ocr, /ocr/batch, /ocr/pixels
       pdf endpoint:        /ocr/pdf
  4. For image endpoints, score char-recall against rendered ground truth
  5. Tear down container + volume

Usage:
    python tests/docker_endpoint_matrix.py                  # full matrix
    python tests/docker_endpoint_matrix.py --image gpu      # GPU only
    python tests/docker_endpoint_matrix.py --only latin     # one language

Exit 0 iff every cell passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build" / "docker_endpoint_matrix"
OUT.mkdir(parents=True, exist_ok=True)

PDF_FIXTURE = REPO / "tests" / "fixtures" / "pdf" / "simple_letter.pdf"

SUPPORTED_LANGS = ["latin", "chinese", "greek", "eslav", "arabic", "korean", "thai"]

# (lang, font, lines, min_char_recall)
CASES = {
    "latin":   ("/usr/share/fonts/TTF/DejaVuSans.ttf", [
        "Invoice INV-2024-001",
        "Total USD 1,234.56",
        "Acme Corporation",
    ], 0.90),
    "chinese": ("/tmp/ocr_zh_bench/NotoSansSC.otf", [
        "发票编号 INV-2024-001",
        "金额 合计 壹万贰仟元整",
        "会议通知请全体员工",
    ], 0.95),
    "greek":   ("/usr/share/fonts/TTF/DejaVuSans.ttf", [
        "Αθήνα Ελλάδα Θεσσαλονίκη",
        "Ωραίος καιρός σήμερα",
        "Καλημέρα κόσμε",
    ], 0.95),
    "eslav":   ("/usr/share/fonts/noto/NotoSans-Regular.ttf", [
        "Добро пожаловать",
        "Москва Санкт-Петербург",
        "Привет мир",
    ], 0.95),
    "arabic":  ("/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf", [
        "مرحبا بالعالم",
        "القاهرة دبي الرياض",
        "اللغة العربية",
    ], 0.85),
    "korean":  ("/tmp/fonts/NotoSansKR.otf", [
        "안녕하세요 한국",
        "서울 부산 제주도",
        "오늘 날씨 좋다",
    ], 0.95),
    "thai":    ("/usr/share/fonts/noto/NotoSansThai-Regular.ttf", [
        "สวัสดีชาวโลก",
        "กรุงเทพ เชียงใหม่",
        "ภาษาไทย",
    ], 0.95),
}


def render(lang: str, font_path: str, lines: list[str]) -> Path:
    font = ImageFont.truetype(font_path, 36)
    w, h = 900, 260
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 20 + i * 70), line, font=font, fill="black")
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


def _inspect_state(name: str) -> tuple[str, str]:
    """Return (running_str, exit_code_str). Tolerate transient empty output."""
    for _ in range(3):
        r = _run(["docker", "inspect", "--format",
                  "{{.State.Running}} {{.State.ExitCode}}", name])
        out = r.stdout.strip()
        if out:
            running, _, code = out.partition(" ")
            return running, code
        time.sleep(0.5)
    return "", ""


def boot(image: str, lang: str, port: int) -> tuple[str, str, str]:
    """Boot a container. Returns (container_name, volume_name, error-or-empty)."""
    name = f"turbo-epmat-{image}-{lang}"
    volume = f"turbo-epmat-{image}-{lang}-vol"
    _run(["docker", "rm", "-f", name])
    _run(["docker", "volume", "rm", volume])
    _run(["docker", "volume", "create", volume])

    cmd = ["docker", "run", "-d", "--name", name]
    if image == "gpu":
        cmd += ["--gpus", "all"]
        tag = "turbo-ocr:prod"
    else:
        tag = "turbo-ocr-cpu:prod"
    cmd += ["-p", f"{port}:8000",
            "-v", f"{volume}:/home/ocr/.cache/turbo-ocr",
            "-e", f"OCR_LANG={lang}",
            "-e", "PIPELINE_POOL_SIZE=1",
            "-e", "DISABLE_LAYOUT=1",
            tag]
    boot = _run(cmd)
    if boot.returncode != 0:
        return name, volume, f"docker run: {boot.stderr.strip()[:200]}"
    return name, volume, ""


def wait_ready(name: str, port: int, timeout: int = 300) -> str:
    """Poll /health until 200, return '' on success or error string."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        running, exit_code = _inspect_state(name)
        if running == "false" and exit_code not in ("", "0"):
            logs = _run(["docker", "logs", "--tail", "30", name]).stdout
            return f"exited code={exit_code}: {logs.strip()[-250:]}"
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.ok:
                return ""
        except requests.RequestException:
            pass
        time.sleep(2)
    logs = _run(["docker", "logs", "--tail", "20", name]).stdout
    return f"timeout after {timeout}s: {logs.strip()[-200:]}"


def teardown(name: str, volume: str) -> None:
    _run(["docker", "rm", "-f", name])
    _run(["docker", "volume", "rm", volume])


# ------------------------- per-endpoint probes --------------------------- #

def probe_health(port: int, ep: str) -> tuple[str, str]:
    try:
        r = requests.get(f"http://localhost:{port}{ep}", timeout=5)
        if r.status_code == 200:
            return ("PASS", f"200 body={r.text.strip()[:40]!r}")
        return ("FAIL", f"{r.status_code} body={r.text.strip()[:80]!r}")
    except Exception as e:
        return ("ERROR", f"{type(e).__name__}: {e}")


def probe_metrics(port: int) -> tuple[str, str]:
    try:
        r = requests.get(f"http://localhost:{port}/metrics", timeout=5)
        if r.status_code == 200 and ("# HELP" in r.text or "# TYPE" in r.text):
            return ("PASS", f"200 bytes={len(r.content)}")
        return ("FAIL", f"{r.status_code} no prom body ({len(r.content)}B)")
    except Exception as e:
        return ("ERROR", f"{type(e).__name__}: {e}")


def _text_from_results(body: dict) -> str:
    results = body.get("results") or body.get("texts") or []
    if isinstance(results, list):
        parts = []
        for it in results:
            if isinstance(it, dict) and "text" in it:
                parts.append(it["text"])
            elif isinstance(it, str):
                parts.append(it)
            elif isinstance(it, list):
                for sub in it:
                    if isinstance(sub, dict) and "text" in sub:
                        parts.append(sub["text"])
        return " ".join(parts)
    return ""


def probe_ocr_raw(port: int, img_bytes: bytes, gt: str,
                  thr: float) -> tuple[str, float, str]:
    try:
        r = requests.post(f"http://localhost:{port}/ocr/raw",
                          data=img_bytes,
                          headers={"Content-Type": "image/png"}, timeout=60)
        if not r.ok:
            return ("FAIL", 0.0, f"{r.status_code} {r.text[:80]}")
        pred = _text_from_results(r.json())
        score = recall(pred, gt)
        return ("PASS" if score >= thr else "FAIL", score, pred[:80])
    except Exception as e:
        return ("ERROR", 0.0, f"{type(e).__name__}: {e}")


def probe_ocr_json(port: int, img_bytes: bytes, gt: str,
                   thr: float) -> tuple[str, float, str]:
    payload = {"image": base64.b64encode(img_bytes).decode("ascii")}
    try:
        r = requests.post(f"http://localhost:{port}/ocr",
                          json=payload, timeout=60)
        if not r.ok:
            return ("FAIL", 0.0, f"{r.status_code} {r.text[:80]}")
        pred = _text_from_results(r.json())
        score = recall(pred, gt)
        return ("PASS" if score >= thr else "FAIL", score, pred[:80])
    except Exception as e:
        return ("ERROR", 0.0, f"{type(e).__name__}: {e}")


def probe_ocr_batch(port: int, img_bytes: bytes, gt: str,
                    thr: float) -> tuple[str, float, str]:
    payload = {"images": [base64.b64encode(img_bytes).decode("ascii")]}
    try:
        r = requests.post(f"http://localhost:{port}/ocr/batch",
                          json=payload, timeout=60)
        if not r.ok:
            return ("FAIL", 0.0, f"{r.status_code} {r.text[:80]}")
        body = r.json()
        batch = (body.get("batch_results") or body.get("results")
                 or body.get("batch") or [])
        if batch and isinstance(batch, list) and isinstance(batch[0], dict):
            pred = _text_from_results(batch[0])
        else:
            pred = _text_from_results(body)
        score = recall(pred, gt)
        return ("PASS" if score >= thr else "FAIL", score, pred[:80])
    except Exception as e:
        return ("ERROR", 0.0, f"{type(e).__name__}: {e}")


def probe_ocr_pixels(port: int, img_bytes: bytes, gt: str,
                     thr: float) -> tuple[str, float, str]:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    # endpoint expects BGR interleaved
    rgb = img.tobytes()
    bgr = bytearray(rgb)
    bgr[0::3], bgr[2::3] = rgb[2::3], rgb[0::3]
    try:
        r = requests.post(f"http://localhost:{port}/ocr/pixels",
                          data=bytes(bgr),
                          headers={"Content-Type": "application/octet-stream",
                                   "X-Width": str(w),
                                   "X-Height": str(h)},
                          timeout=60)
        if not r.ok:
            return ("FAIL", 0.0, f"{r.status_code} {r.text[:80]}")
        pred = _text_from_results(r.json())
        score = recall(pred, gt)
        return ("PASS" if score >= thr else "FAIL", score, pred[:80])
    except Exception as e:
        return ("ERROR", 0.0, f"{type(e).__name__}: {e}")


def probe_ocr_pdf(port: int) -> tuple[str, str]:
    if not PDF_FIXTURE.exists():
        return ("SKIP", f"no fixture at {PDF_FIXTURE}")
    try:
        r = requests.post(f"http://localhost:{port}/ocr/pdf",
                          data=PDF_FIXTURE.read_bytes(),
                          headers={"Content-Type": "application/pdf"},
                          timeout=120)
        if not r.ok:
            return ("FAIL", f"{r.status_code} {r.text[:100]}")
        body = r.json()
        pages = body.get("pages") or body.get("results") or []
        return ("PASS", f"200 pages={len(pages) if isinstance(pages, list) else '?'}")
    except Exception as e:
        return ("ERROR", f"{type(e).__name__}: {e}")


# ------------------------------ driver ----------------------------------- #

def run_cell(image: str, lang: str, port: int, rendered: dict,
             pdf_done_for: set[str]) -> list[tuple[str, str, str, str, str]]:
    """Returns rows: (image, lang, endpoint, status, note)."""
    rows: list[tuple[str, str, str, str, str]] = []
    name, volume, err = boot(image, lang, port)
    if err:
        rows.append((image, lang, "boot", "ERROR", err))
        teardown(name, volume)
        return rows

    ready_err = wait_ready(name, port)
    if ready_err:
        rows.append((image, lang, "boot", "ERROR", ready_err))
        teardown(name, volume)
        return rows

    # Language-agnostic endpoints: run once per image only.
    key = f"{image}-agnostic"
    is_first_for_image = key not in pdf_done_for
    if is_first_for_image:
        for ep in ["/health", "/health/live", "/health/ready"]:
            s, n = probe_health(port, ep)
            rows.append((image, "-", ep, s, n))
        s, n = probe_metrics(port)
        rows.append((image, "-", "/metrics", s, n))
        pdf_done_for.add(key)

    img_path, gt, thr = rendered[lang]
    img_bytes = img_path.read_bytes()

    s, sc, n = probe_ocr_raw(port, img_bytes, gt, thr)
    rows.append((image, lang, "/ocr/raw", s, f"{sc:.2%} {n}"))

    s, sc, n = probe_ocr_json(port, img_bytes, gt, thr)
    rows.append((image, lang, "/ocr", s, f"{sc:.2%} {n}"))

    s, sc, n = probe_ocr_batch(port, img_bytes, gt, thr)
    rows.append((image, lang, "/ocr/batch", s, f"{sc:.2%} {n}"))

    s, sc, n = probe_ocr_pixels(port, img_bytes, gt, thr)
    rows.append((image, lang, "/ocr/pixels", s, f"{sc:.2%} {n}"))

    # PDF: language-agnostic for English letter fixture. Run once per image.
    pdf_key = f"{image}-pdf"
    if pdf_key not in pdf_done_for:
        s, n = probe_ocr_pdf(port)
        rows.append((image, "-", "/ocr/pdf", s, n))
        pdf_done_for.add(pdf_key)

    teardown(name, volume)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", choices=["gpu", "cpu", "both"], default="both")
    parser.add_argument("--only", help="comma-separated language subset")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    images = ("gpu", "cpu") if args.image == "both" else (args.image,)

    # Pre-render images up front so we fail fast on missing fonts.
    rendered: dict[str, tuple[Path, str, float]] = {}
    for lang in SUPPORTED_LANGS:
        if only and lang not in only:
            continue
        font, lines, thr = CASES[lang]
        if not Path(font).exists():
            print(f"[{lang}] SKIP — font missing: {font}")
            continue
        rendered[lang] = (render(lang, font, lines), " ".join(lines), thr)

    langs = [l for l in SUPPORTED_LANGS if l in rendered]
    if not langs:
        print("No languages to test (missing fonts?)")
        return 1

    print(f"Testing {len(langs)} langs x {len(images)} images. "
          f"Fixture PDF: {PDF_FIXTURE.name if PDF_FIXTURE.exists() else 'MISSING'}")

    all_rows: list[tuple[str, str, str, str, str]] = []
    port = 18800
    pdf_done_for: set[str] = set()
    for image in images:
        for lang in langs:
            t0 = time.time()
            print(f"\n=== {image}/{lang} (port {port}) ===")
            rows = run_cell(image, lang, port, rendered, pdf_done_for)
            dt = time.time() - t0
            for r in rows:
                print(f"  [{r[3]:<5}] {r[2]:<14} {r[4]}")
            print(f"  ({dt:.0f}s)")
            all_rows.extend(rows)
            port += 1

    # Summary
    print("\n" + "=" * 90)
    print(f"{'image':<5} {'lang':<8} {'endpoint':<15} {'status':<6} note")
    print("-" * 90)
    fails = 0
    for image, lang, ep, status, note in all_rows:
        if status not in ("PASS", "SKIP"):
            fails += 1
        print(f"{image:<5} {lang:<8} {ep:<15} {status:<6} {note[:55]}")
    print("-" * 90)
    print(f"{len(all_rows)} cells, {fails} failure(s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
