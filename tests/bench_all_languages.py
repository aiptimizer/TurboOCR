"""Per-language throughput + latency benchmark against the GH-Releases-backed image.

For every supported OCR_LANG:
  1. Spin up a container with OCR_LANG=<lang>, PIPELINE_POOL_SIZE=3, DISABLE_LAYOUT=1
  2. Render 20 test images in that script (~3 lines each)
  3. Warm up (2 calls on image[0])
  4. Sequential accuracy + latency over 20 images
  5. Concurrent throughput: c=16, 200 requests
  6. Tear down, move on
  7. Print a consolidated table

Defaults: `turbo-ocr:prod` on GPU, base port 19000. CPU run available via --image cpu.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# Per-language corpus — 20 short phrases each, same layout as the smoketest.
# Fonts mirror tests/language_smoketest.py; the Korean Noto CJK OTF is
# grabbed into /tmp/fonts/ by that script — we reuse it if present.
CORPORA = {
    "latin": [
        "Invoice INV-{i:04d}-{y}", "Total USD {i},{y:03d}.00",
        "Quantity {i} units", "Vendor Acme Corporation",
        "Account {i}4{y}", "Shipping to warehouse {i}",
        "Receipt {i} dated today", "Contract expires {y}-{i:02d}-01",
        "Reference {i}-{y}", "Department OCR-{i}",
        "Approved by M. Smith", "Net payable within 30 days",
        "Order OR-{i}-{y} confirmed", "Customer ID CUST{y}-{i:04d}",
        "Freight EUR {i}.{y}", "Line item subtotal",
        "Discount {i}% applied", "Due before {y}-12-31",
        "Issued on the {i}th", "Terms net thirty days",
    ],
    "chinese": [
        "发票编号INV-{i:04d}-{y}", "金额合计人民币壹万元整",
        "数量{i}件", "供应商爱可美公司",
        "账户{i}4{y}", "收货仓库编号{i}",
        "收据{i}今日开具", "合同有效期{y}-{i:02d}-01",
        "参考号{i}-{y}", "文字识别部门OCR-{i}",
        "批准人张三", "三十日内付清",
        "订单OR-{i}-{y}已确认", "客户编号CUST{y}-{i:04d}",
        "运费人民币{i}.{y}元", "行项目小计",
        "折扣{i}%已生效", "截止日期{y}-12-31",
        "于{i}日签发", "付款期限三十日",
    ],
    "greek": [
        "Τιμολόγιο INV-{i:04d}-{y}", "Σύνολο EUR {i},{y:03d}",
        "Ποσότητα {i} μονάδες", "Προμηθευτής Acme",
        "Λογαριασμός {i}4{y}", "Αποθήκη {i}",
        "Απόδειξη {i} σήμερα", "Συμβόλαιο {y}-{i:02d}-01",
        "Αναφορά {i}-{y}", "Τμήμα OCR-{i}",
        "Εγκρίθηκε από κ. Σμίθ", "Πληρωμή σε 30 ημέρες",
        "Παραγγελία OR-{i}-{y}", "Πελάτης CUST{y}-{i:04d}",
        "Μεταφορικά EUR {i}.{y}", "Υπολογισμός γραμμής",
        "Έκπτωση {i}%", "Πριν τις {y}-12-31",
        "Εκδόθηκε στις {i}", "Όροι τριάντα ημέρες",
    ],
    "eslav": [
        "Счёт INV-{i:04d}-{y}", "Итого RUB {i},{y:03d}",
        "Количество {i} шт", "Поставщик Акмэ",
        "Счёт {i}4{y}", "Склад номер {i}",
        "Квитанция {i} сегодня", "Договор {y}-{i:02d}-01",
        "Ссылка {i}-{y}", "Отдел распознавания-{i}",
        "Утверждено Ивановым", "Оплата в течение 30 дней",
        "Заказ OR-{i}-{y}", "Клиент CUST{y}-{i:04d}",
        "Доставка RUB {i}.{y}", "Подытог строки",
        "Скидка {i}%", "До {y}-12-31",
        "Выдано {i} числа", "Срок тридцать дней",
    ],
    "arabic": [
        "فاتورة INV-{i:04d}-{y}", "المجموع دولار {i},{y:03d}",
        "الكمية {i} وحدات", "المورد شركة أكمي",
        "الحساب {i}4{y}", "المستودع رقم {i}",
        "إيصال {i} اليوم", "العقد {y}-{i:02d}-01",
        "المرجع {i}-{y}", "قسم التعرف-{i}",
        "معتمد من سميث", "الدفع خلال 30 يوما",
        "الطلب OR-{i}-{y}", "العميل CUST{y}-{i:04d}",
        "الشحن يورو {i}.{y}", "إجمالي السطر",
        "خصم {i}%", "قبل {y}-12-31",
        "صدر في {i}", "الشروط ثلاثون يوما",
    ],
    "korean": [
        "인보이스 INV-{i:04d}-{y}", "합계 원 {i},{y:03d}",
        "수량 {i} 개", "공급사 아크미",
        "계좌 {i}4{y}", "창고 번호 {i}",
        "영수증 {i} 오늘", "계약 {y}-{i:02d}-01",
        "참조 {i}-{y}", "인식 부서-{i}",
        "승인자 홍길동", "삼십일 이내 지불",
        "주문 OR-{i}-{y}", "고객 CUST{y}-{i:04d}",
        "운송 원 {i}.{y}", "라인 소계",
        "할인 {i}%", "마감 {y}-12-31",
        "{i}일에 발행", "조건 삼십일",
    ],
    "thai": [
        "ใบแจ้งหนี้ INV-{i:04d}-{y}", "ยอดรวม บาท {i},{y:03d}",
        "จำนวน {i} หน่วย", "ผู้ขาย แอคมี",
        "บัญชี {i}4{y}", "คลังสินค้าเลข {i}",
        "ใบเสร็จ {i} วันนี้", "สัญญา {y}-{i:02d}-01",
        "อ้างอิง {i}-{y}", "แผนก OCR-{i}",
        "อนุมัติโดย สมิธ", "ชำระภายใน 30 วัน",
        "ใบสั่ง OR-{i}-{y}", "ลูกค้า CUST{y}-{i:04d}",
        "ค่าขนส่ง ดอลล่าร์ {i}.{y}", "ยอดย่อยรายการ",
        "ส่วนลด {i}%", "ก่อน {y}-12-31",
        "ออกเมื่อวันที่ {i}", "เงื่อนไข สามสิบวัน",
    ],
}

FONTS = {
    "latin":   "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "chinese": "/tmp/ocr_zh_bench/NotoSansSC.otf",
    "greek":   "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "eslav":   "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "arabic":  "/usr/share/fonts/noto/NotoNaskhArabic-Regular.ttf",
    "korean":  "/tmp/fonts/NotoSansKR.otf",
    "thai":    "/usr/share/fonts/noto/NotoSansThai-Regular.ttf",
}

OUT = Path("/tmp/bench_all_languages")
N_IMAGES = 20
THROUGHPUT_REQS = 200
CONCURRENCY = 16


def render_corpus(lang: str, out_dir: Path) -> tuple[list[Path], list[str]]:
    font_path = FONTS[lang]
    if not Path(font_path).exists():
        raise RuntimeError(f"missing font for {lang}: {font_path}")
    font_big = ImageFont.truetype(font_path, 36)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths, gts = [], []
    template = CORPORA[lang]
    for i in range(N_IMAGES):
        line1 = template[i % len(template)].format(i=i + 1, y=2024 + (i % 4))
        line2 = template[(i + 5) % len(template)].format(i=i + 7, y=2024 + (i % 4))
        line3 = template[(i + 11) % len(template)].format(i=i + 13, y=2024 + (i % 4))
        img = Image.new("RGB", (900, 240), "white")
        d = ImageDraw.Draw(img)
        d.text((30, 30), line1, font=font_big, fill="black")
        d.text((30, 100), line2, font=font_big, fill="black")
        d.text((30, 170), line3, font=font_big, fill="black")
        p = out_dir / f"{lang}_{i:02d}.png"
        img.save(p)
        paths.append(p)
        gts.append(f"{line1} {line2} {line3}")
    return paths, gts


def char_bag(s: str) -> Counter[str]:
    return Counter(c for c in s if not c.isspace() and c not in ".,:;!?\"'()[]{}—–-")


def recall(pred: str, gt: str) -> float:
    pb, gb = char_bag(pred), char_bag(gt)
    return sum((pb & gb).values()) / max(1, sum(gb.values()))


def _docker(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def bench_language(lang: str, image_tag: str, image_kind: str, port: int) -> dict:
    name = f"turbo-bench-{image_kind}-{lang}"
    vol = f"{name}-vol"

    _docker(["rm", "-f", name], check=False)
    _docker(["volume", "rm", vol], check=False)
    _docker(["volume", "create", vol])

    cmd = ["run", "-d", "--name", name, "-p", f"{port}:8000",
           "-v", f"{vol}:/home/ocr/.cache/turbo-ocr",
           "-e", f"OCR_LANG={lang}",
           "-e", "PIPELINE_POOL_SIZE=3", "-e", "DISABLE_LAYOUT=1"]
    if image_kind == "gpu":
        cmd.insert(2, "--gpus")
        cmd.insert(3, "all")
    cmd.append(image_tag)
    _docker(cmd)

    # Wait for /health
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=1).ok:
                break
        except requests.RequestException:
            pass
        # Detect early exit
        inspect = _docker(["inspect", "-f", "{{.State.Running}}", name], check=False)
        if inspect.stdout.strip() == "false":
            logs = _docker(["logs", "--tail", "30", name], check=False).stdout
            _docker(["rm", "-f", name], check=False)
            _docker(["volume", "rm", vol], check=False)
            raise RuntimeError(f"{lang} container exited: {logs[-300:]}")
        time.sleep(2)
    else:
        _docker(["rm", "-f", name], check=False)
        raise TimeoutError(f"{lang} container never became ready")

    # Workload
    paths, gts = render_corpus(lang, OUT / image_kind / lang)
    urls = f"http://localhost:{port}/ocr/raw"
    payloads = [p.read_bytes() for p in paths]

    def one(i: int) -> str:
        r = requests.post(urls, data=payloads[i],
                          headers={"Content-Type": "image/png"}, timeout=30)
        r.raise_for_status()
        return " ".join(item["text"] for item in r.json().get("results", []))

    # Warmup
    one(0); one(0)

    # Sequential accuracy + latency
    accs, lats = [], []
    for i in range(N_IMAGES):
        t0 = time.perf_counter()
        pred = one(i)
        lats.append((time.perf_counter() - t0) * 1000)
        accs.append(recall(pred, gts[i]))

    # Concurrent throughput
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(lambda k: one(k % N_IMAGES), range(THROUGHPUT_REQS)))
    throughput = THROUGHPUT_REQS / (time.perf_counter() - t0)

    _docker(["rm", "-f", name], check=False)
    _docker(["volume", "rm", vol], check=False)

    return {
        "lang": lang,
        "image": image_kind,
        "recall_mean": float(np.mean(accs)),
        "throughput_img_s": throughput,
        "p50_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", choices=["gpu", "cpu", "both"], default="gpu")
    parser.add_argument("--only", help="comma-separated lang subset")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    images = (("gpu", "turbo-ocr:prod"),) if args.image == "gpu" \
        else (("cpu", "turbo-ocr-cpu:prod"),) if args.image == "cpu" \
        else (("gpu", "turbo-ocr:prod"), ("cpu", "turbo-ocr-cpu:prod"))

    results: list[dict] = []
    port = 19000
    for kind, tag in images:
        for lang in CORPORA:
            if only and lang not in only:
                continue
            print(f"\n=== {kind}/{lang} (port {port}, image {tag}) ===")
            try:
                r = bench_language(lang, tag, kind, port)
                results.append(r)
                print(f"  recall={r['recall_mean']:.1%}  tp={r['throughput_img_s']:.1f} img/s  "
                      f"p50={r['p50_ms']:.0f}ms  p95={r['p95_ms']:.0f}ms")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"lang": lang, "image": kind, "error": str(e)})
            port += 1

    # Final table
    print("\n" + "=" * 92)
    print(f"{'image':<5} {'lang':<8} {'recall':>9} {'throughput':>15} {'p50':>8} {'p95':>8}")
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['image']:<5} {r['lang']:<8}  ERROR: {r['error'][:60]}")
            continue
        print(f"{r['image']:<5} {r['lang']:<8} {r['recall_mean']:>8.1%} "
              f"{r['throughput_img_s']:>10.1f} img/s {r['p50_ms']:>5.0f} ms {r['p95_ms']:>5.0f} ms")
    print("=" * 92)
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
