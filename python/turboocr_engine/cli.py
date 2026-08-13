"""``turboocr`` command-line interface.

    turboocr doctor                 # detect hardware, print the pip install for it
    turboocr ocr image.png          # OCR an image
    turboocr pdf doc.pdf --markdown # OCR a PDF to Markdown
    turboocr models                 # list available models
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def _parse_pages(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out or None


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="model/alias override (tiny|small|medium|arabic|...)")
    p.add_argument("--lang", default=None, help="language: en/ch/ja (+ --tier) or a script ko/ar/th/el/ru")
    p.add_argument("--tier", default=None, help="Latin/CJK accuracy tier: tiny|small|medium")
    p.add_argument("--backend", default="auto", help="auto|fast|cpu | vendor backends: nvidia(turbo/tensorrt)|intel|amd|apple(metal) | ORT EPs: cuda|openvino|coreml|... (default: auto)")
    p.add_argument("--models-dir", default=None, help="directory of ONNX models (else auto/download)")
    p.add_argument("--device", default=None, help="device hint for OpenVINO (CPU|GPU|NPU|AUTO)")
    p.add_argument("--cls", action="store_true", help="enable 180° angle classification")
    p.add_argument("--drop-score", type=float, default=0.5, help="min confidence to keep (default 0.5)")


def _build_ocr(args):
    from .pipeline import OCR

    ocr = OCR(
        args.model,
        args.backend,
        lang=getattr(args, "lang", None),
        tier=getattr(args, "tier", None),
        models_dir=args.models_dir,
        device=args.device,
        use_cls=args.cls,
        layout=getattr(args, "layout", False),
    )
    if args.verbose:
        print(f"[turboocr] {ocr.provider_summary} | model={ocr.model_name}", file=sys.stderr)
    return ocr


def _emit(page_or_doc, fmt: str, out) -> None:
    """Write a PageResult/DocumentResult in the requested format to `out`."""
    if fmt == "json":
        out.write(page_or_doc.to_json(indent=2) + "\n")
    elif fmt == "markdown":
        out.write(page_or_doc.to_markdown() + "\n")
    elif fmt == "tsv":
        pages = getattr(page_or_doc, "pages", [page_or_doc])
        for p in pages:
            out.write(p.to_tsv() + "\n")
    elif fmt == "hocr":
        pages = getattr(page_or_doc, "pages", [page_or_doc])
        for i, p in enumerate(pages, 1):
            out.write(p.to_hocr(page_id=i) + "\n")
    else:  # text
        out.write(page_or_doc.text + "\n")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import doctor

    doctor(plain=args.plain)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .catalog import catalog

    for e in catalog():
        det = e.det or "det.onnx"
        print(f"{e.name:8s}  family={e.family:7s}  rec={e.rec:24s}  det={det}")
    return 0


def _resolve_fmt(args) -> str:
    if getattr(args, "format", None):
        return args.format
    if getattr(args, "json", False):
        return "json"
    if getattr(args, "markdown", False):
        return "markdown"
    return "text"


def _open_out(path):
    import contextlib

    if not path:
        return contextlib.nullcontext(sys.stdout)
    return open(path, "w", encoding="utf-8")


def cmd_ocr(args: argparse.Namespace) -> int:
    import glob as _glob

    # Expand globs the shell didn't (Windows, quoted patterns).
    paths = []
    for pat in args.images:
        hits = _glob.glob(pat)
        paths.extend(hits if hits else [pat])
    if not paths:
        print("turboocr: no input images", file=sys.stderr)
        return 2

    ocr = _build_ocr(args)
    fmt = _resolve_fmt(args)
    multi = len(paths) > 1
    with _open_out(args.output) as out:
        for i, path in enumerate(paths):
            res = ocr.read(path, drop_score=args.drop_score, layout=args.layout or None)
            if multi and fmt in ("text", "markdown"):
                out.write(f"===== {path} =====\n")
            _emit(res, fmt, out)
            if args.overlay:
                # For a single input use the path as-is; for many, suffix each
                # so nothing is silently overwritten/ignored.
                dst = args.overlay
                if multi:
                    stem, ext = os.path.splitext(args.overlay)
                    dst = f"{stem}_{i}{ext or '.png'}"
                res.save_overlay(dst, show_text=False)
    return 0


def cmd_pdf(args: argparse.Namespace) -> int:
    ocr = _build_ocr(args)
    if args.searchable:
        if not args.output:
            print("turboocr: --searchable requires -o/--output <file.pdf>", file=sys.stderr)
            return 2
        # Stream page-by-page (constant memory) straight to the searchable PDF.
        ocr.pdf_to_searchable(
            args.file,
            args.output,
            dpi=args.dpi,
            pages=_parse_pages(args.pages),
            max_pages=args.max_pages,
            drop_score=args.drop_score,
            progress=True if args.verbose else None,
        )
        return 0
    doc = ocr.read_pdf(
        args.file,
        dpi=args.dpi,
        pages=_parse_pages(args.pages),
        max_pages=args.max_pages,
        drop_score=args.drop_score,
        progress=True if args.verbose else None,
    )
    fmt = _resolve_fmt(args)
    with _open_out(args.output) as out:
        if fmt == "text":
            for page in doc.pages:
                out.write(f"----- page {page.page} -----\n")
                out.write(page.text + "\n")
        else:
            _emit(doc, fmt, out)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    from ._version import __version__

    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turboocr", description="TurboOCR — fast multi-backend OCR")
    p.add_argument("-v", "--verbose", action="store_true", help="log backend/model to stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser(
        "doctor",
        help="detect your hardware and print the pip install command for it",
        description="Detect the accelerator on this machine and name the one wheel that "
        "serves it (turboocr-engine-cpu | -cuda12 | -cuda13 | -openvino | -rocm), "
        "with the exact install command.",
    )
    d.add_argument("--plain", action="store_true", help="plain text (no rich formatting)")
    d.set_defaults(func=cmd_doctor)

    m = sub.add_parser("models", help="list available models")
    m.set_defaults(func=cmd_models)

    o = sub.add_parser("ocr", help="OCR one or more images")
    o.add_argument("images", nargs="+", help="image path(s) or glob(s)")
    _add_common(o)
    o.add_argument("-f", "--format", choices=["text", "json", "markdown", "tsv", "hocr"], default=None)
    o.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
    o.add_argument("--layout", action="store_true", help="also detect layout regions")
    o.add_argument("--overlay", default=None, help="save a boxes-overlay image to this path (single input)")
    o.add_argument("--json", action="store_true", help="shorthand for --format json")
    o.set_defaults(func=cmd_ocr)

    pf = sub.add_parser("pdf", help="OCR a PDF")
    pf.add_argument("file")
    _add_common(pf)
    pf.add_argument("--dpi", type=int, default=150, help="render DPI (default 150)")
    pf.add_argument("--pages", default=None, help="1-based pages, e.g. 1,3,5-8")
    pf.add_argument("--max-pages", type=int, default=None, help="cap page count")
    pf.add_argument("--layout", action="store_true", help="also detect layout regions")
    pf.add_argument("--searchable", action="store_true", help="write a searchable PDF (needs -o out.pdf)")
    pf.add_argument("-f", "--format", choices=["text", "json", "markdown", "tsv", "hocr"], default=None)
    pf.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
    pf.add_argument("--json", action="store_true", help="shorthand for --format json")
    pf.add_argument("--markdown", action="store_true", help="shorthand for --format markdown")
    pf.set_defaults(func=cmd_pdf)

    v = sub.add_parser("version", help="print version")
    v.set_defaults(func=cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # user-facing: keep it terse
        print(f"turboocr: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
