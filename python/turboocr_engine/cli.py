"""``turboocr`` command-line interface.

    turboocr doctor                 # detect hardware, print the pip install for it
    turboocr ocr image.png          # OCR an image
    turboocr pdf doc.pdf --markdown # OCR a PDF to Markdown
    turboocr models                 # list available models
    turboocr info                   # build the engine, print its resolved config
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .options import DEFAULT_DPI, DROP_SCORE, OUTPUT_FORMATS, PDF_MODES


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
    p.add_argument("--replicas", type=int, default=1,
                    help="engine replicas — parallel pages/images (default 1)")
    p.add_argument("--cls", action="store_true", help="enable 180° angle classification")
    p.add_argument("--tables", action="store_true",
                   help="recognize tables (HTML) inside layout regions")
    p.add_argument("--formulas", action="store_true",
                   help="recognize formulas (LaTeX) inside layout regions")
    p.add_argument("--autorotate", action="store_true",
                   help="detect and correct 0/90/180/270 page rotation")
    p.add_argument("--drop-score", type=float, default=DROP_SCORE,
                   help=f"min confidence to keep (default {DROP_SCORE})")


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
        # reading_order rides the layout model, so requesting it loads it
        layout=(getattr(args, "layout", False)
                or getattr(args, "reading_order", False)),
        tables=getattr(args, "tables", False),
        formulas=getattr(args, "formulas", False),
        autorotate=getattr(args, "autorotate", False),
        replicas=max(1, getattr(args, "replicas", 1)),
    )
    if args.verbose:
        print(f"[turboocr] {ocr.provider_summary} | model={ocr.model_name}", file=sys.stderr)
    return ocr


def _emit(page_or_doc, fmt: str, out) -> None:
    """Write a PageResult/DocumentResult in the requested format to `out`."""
    is_doc = hasattr(page_or_doc, "pages")
    if fmt == "json":
        out.write(page_or_doc.to_json(indent=2) + "\n")
    elif fmt == "markdown":
        out.write(page_or_doc.to_markdown() + "\n")
    elif fmt == "tsv":
        # DocumentResult.to_tsv: ONE header + a page column. Looping per-page
        # to_tsv repeated the header and lost page identity.
        out.write(page_or_doc.to_tsv() + "\n")
    elif fmt == "hocr":
        # A COMPLETE hOCR document (html/head/ocr-system meta, real page
        # numbers) — bare sibling ocr_page divs are not parseable XML, so
        # hocr-tools rejected the old multi-page output.
        text = page_or_doc.to_hocr() if is_doc else page_or_doc.to_hocr(full=True)
        out.write(text + "\n")
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


def cmd_warmup(args: argparse.Namespace) -> int:
    import time

    import numpy as np

    t0 = time.perf_counter()
    ocr = _build_ocr(args)  # model download + the backend's load-time warmup
    t1 = time.perf_counter()
    # A realistic page, not a thumbnail: detector graphs are compiled per
    # canvas SIZE, so a tiny image would warm only the smallest canvas and
    # the first real document would still pay the big one.
    page = np.full((1400, 1000, 3), 255, np.uint8)
    for y in range(120, 1300, 44):
        page[y : y + 18, 80:920] = 16
    ocr.read(page, **_stage_kwargs(args))
    t2 = time.perf_counter()
    summary = f"{ocr.provider_summary} | model={ocr.model_name}"
    ocr.close()
    print(
        f"warmup complete: load {t1 - t0:.1f}s, first read {t2 - t1:.1f}s "
        f"({summary})"
    )
    return 0


def _stage_kwargs(args) -> dict:
    """The per-call stage request, derived from the parsed flags in ONE place.
    Both subcommands forward these (Load is not Run: the constructor only
    decides what is loadable) — cmd_pdf once parsed --layout/--tables/
    --formulas and then never passed them, so they did nothing."""
    return {
        "layout": args.layout or None,
        "tables": args.tables or None,
        "formulas": args.formulas or None,
        "reading_order": args.reading_order,
    }


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

    fmt = _resolve_fmt(args)
    if args.reading_order and fmt != "json":
        # Only to_dict() carries the indices; any other format would compute
        # them and silently drop them — refuse BEFORE the engine builds.
        print("turboocr: --reading-order is only representable in JSON — "
              "add --format json", file=sys.stderr)
        return 2
    ocr = _build_ocr(args)
    multi = len(paths) > 1
    # json/tsv/hocr ALWAYS emit the document shape ({"pages":[...]} / page
    # column / one hOCR doc), single input included: the shape used to
    # depend on how many files the GLOB matched, so `... | jq '.pages'`
    # worked until a directory happened to contain one file.
    accumulate = fmt in ("json", "tsv", "hocr")
    json_pages = []
    doc_pages = []
    skipped = 0
    with _open_out(args.output) as out:
        for i, path in enumerate(paths):
            try:
                res = ocr.read(path, drop_score=args.drop_score,
                               **_stage_kwargs(args))
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                # Best-effort mode: one corrupt file in a 500-scan glob must
                # not discard every already-OCR'd result. Noted on stderr;
                # the exit code still reports partial failure.
                skipped += 1
                print(f"turboocr: skipped {path}: {exc}", file=sys.stderr)
                continue
            if args.overlay:
                # Overlays need the raster — draw BEFORE the accumulators
                # strip it. For a single input use the path as-is; for many,
                # suffix each so nothing is silently overwritten.
                dst = args.overlay
                if multi:
                    stem, ext = os.path.splitext(args.overlay)
                    dst = f"{stem}_{i}{ext or '.png'}"
                res.save_overlay(dst, show_text=False, layout=bool(args.layout))
            if accumulate and fmt == "json":
                d = res.to_dict()
                d["source"] = path
                d["page"] = i + 1
                json_pages.append(d)
            elif accumulate:
                res.page = i + 1
                # Accumulating whole PageResults kept every ~6 MB raster
                # alive for the run; only the text/boxes are needed here.
                res.image = None
                doc_pages.append(res)
            elif multi:
                out.write(f"===== {path} =====\n")
                _emit(res, fmt, out)
            else:
                _emit(res, fmt, out)
        if accumulate and fmt == "json":
            import json as _json

            # ONE parseable document, same {"pages": [...]} envelope the pdf
            # subcommand emits (concatenated JSON objects are unparseable; a
            # bare array differed from the pdf shape for no reason).
            out.write(_json.dumps({"pages": json_pages}, ensure_ascii=False,
                                  indent=2) + "\n")
        elif accumulate:
            from .result import DocumentResult

            # tsv: one header + page column. hocr: ONE complete document —
            # per-image emission stacked N <html> documents (unparseable).
            _emit(DocumentResult(pages=doc_pages), fmt, out)
    return 1 if skipped else 0


def cmd_pdf(args: argparse.Namespace) -> int:
    # Argument contradictions fail BEFORE the engine builds — constructing a
    # model pipeline just to print a usage error wastes seconds and downloads.
    if args.searchable:
        if not args.output:
            print("turboocr: --searchable requires -o/--output <file.pdf>", file=sys.stderr)
            return 2
        if args.mode == "text":
            print("turboocr: --searchable always renders and OCRs (its output "
                  "embeds page images) — it cannot run with --mode text",
                  file=sys.stderr)
            return 2
    if args.reading_order and (args.searchable or _resolve_fmt(args) != "json"):
        print("turboocr: --reading-order is only representable in JSON — "
              "add --format json (and drop --searchable)", file=sys.stderr)
        return 2
    if args.mode == "text" and (args.layout or args.tables or args.formulas
                                or args.reading_order):
        # Same refusal read_pdf raises — but here BEFORE the engine builds
        # (constructing a model pipeline just to print a usage error wastes
        # seconds and downloads).
        print('turboocr: --mode text serves the embedded text layer only (no '
              'rendering, no models) — it cannot run --layout/--tables/'
              '--formulas/--reading-order. Use --mode auto or --mode ocr.',
              file=sys.stderr)
        return 2
    ocr = _build_ocr(args)
    if args.searchable:
        # Stream page-by-page (constant memory) straight to the searchable PDF.
        ocr.pdf_to_searchable(
            args.file,
            args.output,
            dpi=args.dpi,
            pages=_parse_pages(args.pages),
            max_pages=args.max_pages,
            drop_score=args.drop_score,
            password=args.password,
            progress=True if args.verbose else None,
        )
        return 0
    doc = ocr.read_pdf(
        args.file,
        dpi=args.dpi,
        pages=_parse_pages(args.pages),
        max_pages=args.max_pages,
        mode=args.mode,
        drop_score=args.drop_score,
        **_stage_kwargs(args),
        password=args.password,
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


def cmd_info(args: argparse.Namespace) -> int:
    import json

    ocr = _build_ocr(args)
    print(json.dumps(ocr.info(), indent=2))
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

    w = sub.add_parser(
        "warmup",
        help="pay the one-time model download + engine compilation now",
        description="Construct the requested engine and run one synthetic page "
        "through it, so model downloads and per-machine engine compilation "
        "(TensorRT engines, MIGraphX programs, CoreML specialization) happen "
        "here — at install or image-build time — instead of on the first real "
        "document. Loads every requested stage and exercises the det/cls/rec "
        "core path. Idempotent: costs seconds once the caches are warm.",
    )
    _add_common(w)
    w.add_argument("--layout", action="store_true",
                   help="also load and run the layout stage")
    w.set_defaults(func=cmd_warmup, reading_order=False)

    o = sub.add_parser("ocr", help="OCR one or more images")
    o.add_argument("images", nargs="+", help="image path(s) or glob(s)")
    _add_common(o)
    o.add_argument("-f", "--format", choices=list(OUTPUT_FORMATS), default=None)
    o.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
    o.add_argument("--layout", action="store_true", help="also detect layout regions")
    o.add_argument("--reading-order", action="store_true", dest="reading_order",
                   help="compute reading-order indices (JSON output only — the "
                        "other formats have no field for them)")
    o.add_argument("--overlay", default=None, help="save a boxes-overlay image to this path (single input)")
    o.add_argument("--json", action="store_true", help="shorthand for --format json")
    o.add_argument("--on-error", choices=["raise", "skip"], default="raise",
                   dest="on_error",
                   help="skip = note unreadable images on stderr and keep "
                        "going (exit 1 if any were skipped); raise = stop at "
                        "the first failure (default)")
    o.set_defaults(func=cmd_ocr)

    pf = sub.add_parser("pdf", help="OCR a PDF")
    pf.add_argument("file")
    _add_common(pf)
    pf.add_argument("--mode", choices=list(PDF_MODES), default="ocr",
                    help="ocr = render and OCR every page (default); auto = "
                         "use a page's embedded text layer when one is there "
                         "and passes the quality gate (much faster on "
                         "born-digital PDFs), OCR the rest; text = embedded "
                         "text layer only, never OCR")
    pf.add_argument("--password", default=None,
                    help="password for an encrypted PDF (user or owner)")
    pf.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                    help=f"render DPI (default {DEFAULT_DPI})")
    pf.add_argument("--pages", default=None, help="1-based pages, e.g. 1,3,5-8")
    pf.add_argument("--max-pages", type=int, default=None, help="cap page count")
    pf.add_argument("--layout", action="store_true", help="also detect layout regions")
    pf.add_argument("--reading-order", action="store_true", dest="reading_order",
                    help="compute reading-order indices (JSON output only — the "
                         "other formats have no field for them)")
    pf.add_argument("--searchable", action="store_true", help="write a searchable PDF (needs -o out.pdf)")
    pf.add_argument("-f", "--format", choices=list(OUTPUT_FORMATS), default=None)
    pf.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
    pf.add_argument("--json", action="store_true", help="shorthand for --format json")
    pf.add_argument("--markdown", action="store_true", help="shorthand for --format markdown")
    pf.set_defaults(func=cmd_pdf)

    i = sub.add_parser(
        "info",
        help="build the engine and print its resolved configuration (JSON)",
        description="Construct the engine with the given options and print "
        "OCR.info() — resolved model, backend, mode, capabilities — as JSON.",
    )
    _add_common(i)
    i.add_argument("--layout", action="store_true", help="also load the layout model")
    i.set_defaults(func=cmd_info)

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
