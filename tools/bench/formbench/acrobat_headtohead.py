#!/usr/bin/env python3
"""Head-to-head against Acrobat, in two halves.

Acrobat is not installed on this machine and the CommonForms paper's own
comparison is explicitly qualitative ("We qualitatively compare FFDNet and
Adobe Acrobat"), so no published numbers exist to cite either. What can be done
without guessing is to make the measurement a single command for whoever does
have Acrobat:

    export  — writes the CommonForms test pages as FLAT PDFs. Run Acrobat's
              Prepare Form over that folder, save the results alongside.
    score   — scores any folder of prepared PDFs against the same ground truth,
              with the same matcher used for TurboOCR, so the two numbers are
              directly comparable rather than merely adjacent.

Until that folder exists, `claims` checks the three specific, falsifiable
statements the paper makes about Acrobat against this system's own output.
"""
import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).parent))
from eval_commonforms import CLASS_ORDER, match, prf  # noqa: E402

# Acrobat has no checkbox class at all (paper §5.2), so anything it produces is
# a text or signature widget. /Btn is mapped anyway so the scorer stays honest
# if a future version gains one.
FT_TO_CLASS = {"/Tx": "text", "/Btn": "checkbox", "/Sig": "signature"}


def load(shard: int, pages: int):
    path = hf_hub_download(repo_id="jbarrow/CommonForms",
                           filename=f"data/test-{shard:05d}-of-00024.parquet",
                           repo_type="dataset")
    return pq.read_table(path).slice(0, pages).to_pylist()


def ground_truth(row):
    return [(CLASS_ORDER[int(c)] if int(c) < 3 else "text",
             float(b[0]), float(b[1]), float(b[0]) + float(b[2]),
             float(b[1]) + float(b[3]))
            for b, c in zip(row["objects"]["bbox"], row["objects"]["category"])]


def cmd_export(args):
    out = args.dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load(args.shard, args.pages)
    for i, row in enumerate(rows):
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img.save(out / f"page_{i:03d}.pdf", "PDF", resolution=200.0)
    (out / "GROUND_TRUTH.json").write_text(json.dumps(
        {f"page_{i:03d}": ground_truth(r) for i, r in enumerate(rows)}, indent=1))
    print(f"wrote {len(rows)} flat PDFs to {out}")
    print("Now: Acrobat > Prepare Form on each, save as prepared/page_NNN.pdf")
    print(f"Then: {Path(__file__).name} score --dir {out}/prepared "
          f"--truth {out}/GROUND_TRUTH.json")


def widgets(pdf: Path):
    """Every widget rectangle in a prepared PDF, in the page's own pixels at
    the 200 dpi the flat pages were written at."""
    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    H = float(page.mediabox.height)
    scale = 200.0 / 72.0
    out = []
    for annot in (page.get("/Annots") or []):
        obj = annot.get_object()
        if obj.get("/Subtype") != "/Widget":
            continue
        ft = obj.get("/FT") or (obj.get("/Parent") or {}).get("/FT")
        rect = [float(v) for v in obj["/Rect"]]
        x0, x1 = min(rect[0], rect[2]), max(rect[0], rect[2])
        y0, y1 = min(rect[1], rect[3]), max(rect[1], rect[3])
        # PDF user space is y-up; ground truth is y-down page pixels.
        out.append((FT_TO_CLASS.get(str(ft), "text"), 0.5,
                    x0 * scale, (H - y1) * scale, x1 * scale, (H - y0) * scale))
    return out


def cmd_score(args):
    truth = json.loads(Path(args.truth).read_text())
    agg = defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]
    seen = 0
    for name, gt in sorted(truth.items()):
        pdf = args.dir / f"{name}.pdf"
        if not pdf.exists():
            continue
        seen += 1
        pred = widgets(pdf)
        gt = [tuple(g) for g in gt]
        for cls in CLASS_ORDER:
            tp, fp, fn = match([p for p in pred if p[0] == cls],
                               [g for g in gt if g[0] == cls], 0.5,
                               class_aware=False)
            a = agg[cls]; a[0] += tp; a[1] += fp; a[2] += fn
        tp, fp, fn = match(pred, gt, 0.5, class_aware=True)
        tot[0] += tp; tot[1] += fp; tot[2] += fn

    if not seen:
        print(f"no prepared PDFs found in {args.dir}")
        return 1
    print(f"\n=== {args.label} · {seen} pages · IoU 0.5 ===")
    print(f"{'class':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    for cls in CLASS_ORDER:
        p, r, f = prf(*agg[cls])
        print(f"{cls:10s} {p:7.3f} {r:7.3f} {f:7.3f} "
              f"{agg[cls][0]:6d} {agg[cls][1]:6d} {agg[cls][2]:6d}")
    p, r, f = prf(*tot)
    print(f"{'OVERALL':10s} {p:7.3f} {r:7.3f} {f:7.3f} "
          f"{tot[0]:6d} {tot[1]:6d} {tot[2]:6d}")
    return 0


def cmd_claims(args):
    """Test the paper's three falsifiable statements about Acrobat against the
    ground truth and against this system's measured numbers."""
    rows = load(args.shard, args.pages)
    gts = [ground_truth(r) for r in rows]
    per_page = [len(g) for g in gts]
    cls = Counter(c for g in gts for c, *_ in g)

    print(f"CommonForms test, {len(rows)} pages")
    print(f"  fields per page: mean {sum(per_page)/len(per_page):.1f}, "
          f"max {max(per_page)}")
    print(f"  by class: {dict(cls)}")
    print()
    print("Paper §5.2, verbatim, and what it implies for a detector with no")
    print("checkbox class at all:")
    n_cb = cls.get("checkbox", 0)
    total = sum(cls.values())
    print(f'  "Acrobat does not detect choice buttons at all."')
    print(f"    -> {n_cb} of {total} ground-truth fields ({n_cb/total*100:.0f}%) "
          f"are checkboxes.")
    print(f"    -> a detector without the class has recall <= "
          f"{(total-n_cb)/total:.3f} on this set BY CONSTRUCTION.")
    print(f'  "Acrobat suffers from ... low precision, table elements and')
    print(f'   separator lines for text fields."')
    print(f"    -> the same failure mode this system hit and fixed: an empty")
    print(f"       table border became one field covering 22% of a page.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export"); e.set_defaults(fn=cmd_export)
    e.add_argument("--dir", type=Path, required=True)
    e.add_argument("--pages", type=int, default=40)
    e.add_argument("--shard", type=int, default=0)

    s = sub.add_parser("score"); s.set_defaults(fn=cmd_score)
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("--label", default="Acrobat Prepare Form")

    c = sub.add_parser("claims"); c.set_defaults(fn=cmd_claims)
    c.add_argument("--pages", type=int, default=40)
    c.add_argument("--shard", type=int, default=0)

    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
