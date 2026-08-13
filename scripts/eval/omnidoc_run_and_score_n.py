#!/usr/bin/env python3
"""Run OCR -> markdown -> OmniDocBench scoring for an N-doc subset.

Generalized from scripts/eval/omnidoc_run_and_score.py (which is hardwired to the
guaranteed-text+formula+table 5-doc set). Here the subset is stratified and
heterogeneous: a given doc may have NO table or NO formula, so it legitimately
won't appear in that element's per-page file. The completeness check is
therefore element-aware: a doc is only "dropped" (a hard error) if its GT
actually contains instances of an element but the scorer produced no per-page
edit for it.

Workflow:
  1. (re)generate the N-doc subset GT via scripts/eval/omnidoc_subset_n.py.
  2. POST each subset image to the OCR server (parallel).
  3. Convert per-image JSON to OmniDocBench-style markdown.
  4. Render a per-experiment scorer config and invoke pdf_validation.py.
  5. Read *_metric_result.json + the per-page edit files and emit one compact
     summary at /tmp/omnidoc_runs/<experiment>_<timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("OMNIDOCBENCH", REPO.parent / "omnidocbench"))
SUBSET_N = 125
SUBSET_DIR = Path("/tmp/omnidoc_subset125")
SUBSET_GT = SUBSET_DIR / "OmniDocBench_subset125.json"
SUBSET_SCRIPT = REPO / "scripts" / "eval" / "omnidoc_subset_n.py"
RUN_TOOL = REPO / "tools" / "bench" / "omnidoc_run.py"
MD_TOOL = REPO / "tools" / "bench" / "omnidoc_to_md.py"
RUNS_DIR = Path("/tmp/omnidoc_runs")
BASE_CONFIG = BENCH / "configs" / "end2end_subset125.yaml"
IMAGES_DIR = BENCH / "data" / "images"
BENCH_PY = BENCH / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True,
                   help="Slug used for prediction dir + summary filename.")
    p.add_argument("--server-url", default="http://localhost:8080")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--n", type=int, default=SUBSET_N)
    p.add_argument("--ocr-timeout-sec", type=int, default=1800)
    p.add_argument("--score-timeout-sec", type=int, default=900)
    p.add_argument("--skip-ocr", action="store_true",
                   help="Reuse existing JSON predictions if present.")
    p.add_argument("--base-config", default=str(BASE_CONFIG),
                   help="Scorer config template (default: full text+formula+table).")
    return p.parse_args()


def _load_subset() -> list[dict]:
    with SUBSET_GT.open() as f:
        return json.load(f)


def _ensure_subset(n: int) -> list[dict]:
    if not SUBSET_GT.exists():
        subprocess.run(
            [sys.executable, str(SUBSET_SCRIPT),
             "--n", str(n), "--out-dir", str(SUBSET_DIR)],
            check=True)
    return _load_subset()


def _cats(doc: dict) -> set[str]:
    return {b.get("category_type") for b in doc.get("layout_dets", [])}


def _stage_images(filenames: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.iterdir():
        if f.is_file():
            f.unlink()
    for name in filenames:
        src = IMAGES_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"missing image: {src}")
        shutil.copy2(src, dest / name)


def _run(cmd: list[str], *, timeout: int, cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(cmd)}  (timeout={timeout}s)", flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        print(f"[abort] command exceeded {timeout}s", file=sys.stderr)
        raise
    print(f"[run] done in {time.time() - t0:.1f}s", flush=True)


def _render_config(experiment: str, pred_dir: Path,
                   base_config: Path = BASE_CONFIG) -> Path:
    base = base_config.read_text()
    rendered = base.replace(
        "data_path: /tmp/omnidoc_subset125/md",
        f"data_path: {pred_dir}",
    )
    if f"data_path: {pred_dir}" not in rendered:
        raise RuntimeError(
            "config render failed: did not find the prediction data_path token "
            f"to replace in {BASE_CONFIG}")
    cfg_path = BENCH / "configs" / f"end2end_subset125__{experiment}.yaml"
    cfg_path.write_text(rendered)
    return cfg_path


def _read_per_page(save_name: str, element: str) -> dict[str, float] | None:
    p = BENCH / "result" / f"{save_name}_{element}_per_page_edit.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _summarize(experiment: str, save_name: str, docs: list[dict],
               ocr_sec: float, md_sec: float, score_sec: float) -> dict:
    metric_path = BENCH / "result" / f"{save_name}_metric_result.json"
    if not metric_path.exists():
        raise FileNotFoundError(f"scorer did not produce {metric_path}")
    metric = json.loads(metric_path.read_text())

    def _all(section: str, key: str, *prefer: str):
        try:
            block = metric[section]["all"][key]
        except (KeyError, TypeError):
            return None
        if not isinstance(block, dict):
            return block
        for k in prefer:
            if k in block:
                return block[k]
        for k in ("ALL_page_avg", "all", "edit_sample_avg", "edit_whole"):
            if k in block:
                return block[k]
        return None

    summary_all = {
        "text_block_edit_dist": _all("text_block", "Edit_dist"),
        "display_formula_edit_dist": _all("display_formula", "Edit_dist"),
        "display_formula_cdm": _all("display_formula", "CDM"),
        "table_teds": _all("table", "TEDS"),
        "table_teds_structure_only": _all("table", "TEDS_structure_only"),
        "table_edit_dist": _all("table", "Edit_dist"),
        "reading_order_edit_dist": _all("reading_order", "Edit_dist"),
    }

    per_page = {
        "text_block": _read_per_page(save_name, "text_block"),
        "display_formula": _read_per_page(save_name, "display_formula"),
        "table": _read_per_page(save_name, "table"),
        "reading_order": _read_per_page(save_name, "reading_order"),
    }

    # GT element presence per doc, so a missing per-page entry is only an error
    # when the doc actually HAS that element in the ground truth.
    GT_ELEM = {
        "text_block": "text_block",
        "display_formula": "equation_isolated",
        "table": "table",
        "reading_order": None,  # reading_order applies whenever there is content
    }
    by_path = {d["page_info"]["image_path"]: d for d in docs}
    filenames = [d["page_info"]["image_path"] for d in docs]

    missing: dict[str, list[str]] = {}
    scored_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for elem, edits in per_page.items():
        gt_cat = GT_ELEM[elem]
        scored = 0
        expected = 0
        for fn in filenames:
            doc = by_path[fn]
            has_elem = True if gt_cat is None else (gt_cat in _cats(doc))
            if has_elem:
                expected += 1
            if edits is not None and fn in edits:
                scored += 1
            elif has_elem:
                # doc has this element in GT but scorer produced no edit for it
                missing.setdefault(elem, []).append(fn)
        scored_counts[elem] = scored
        expected_counts[elem] = expected

    # A doc that scored on NONE of the elements it should have scored on is a
    # genuine failure (e.g. its prediction markdown was empty / unmatched).
    failed_docs: list[str] = []
    for fn in filenames:
        doc = by_path[fn]
        any_scored = False
        any_expected = False
        for elem, edits in per_page.items():
            gt_cat = GT_ELEM[elem]
            has_elem = True if gt_cat is None else (gt_cat in _cats(doc))
            if has_elem:
                any_expected = True
                if edits is not None and fn in edits:
                    any_scored = True
        if any_expected and not any_scored:
            failed_docs.append(fn)

    n_docs = len(filenames)
    n_failed = len(failed_docs)
    n_scored = n_docs - n_failed

    per_doc = []
    for fn in filenames:
        rec = {"image": fn}
        for elem in ("text_block", "display_formula", "table", "reading_order"):
            edits = per_page[elem]
            rec[f"{elem}_edit"] = (edits.get(fn) if edits is not None else None)
        per_doc.append(rec)

    n_table = sum(1 for d in docs if "table" in _cats(d))
    n_formula = sum(1 for d in docs if "equation_isolated" in _cats(d))
    n_both = sum(1 for d in docs
                 if "table" in _cats(d) and "equation_isolated" in _cats(d))
    n_text = sum(1 for d in docs if "text_block" in _cats(d))

    return {
        "experiment": experiment,
        "save_name": save_name,
        "timestamp": int(time.time()),
        "n_docs": n_docs,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "failed_docs": failed_docs,
        "composition": {
            "table": n_table,
            "formula": n_formula,
            "both": n_both,
            "text_block": n_text,
        },
        "scored_counts": scored_counts,
        "expected_counts": expected_counts,
        "docs_with_gt_element_unscored": missing,
        "subset": filenames,
        "summary": summary_all,
        "per_doc": per_doc,
        "runtime_sec": {
            "ocr": round(ocr_sec, 2),
            "md": round(md_sec, 2),
            "score": round(score_sec, 2),
            "total": round(ocr_sec + md_sec + score_sec, 2),
        },
    }


def main() -> int:
    args = parse_args()
    docs = _ensure_subset(args.n)
    filenames = [d["page_info"]["image_path"] for d in docs]
    print(f"[orchestrator] subset of {len(filenames)} docs", flush=True)

    exp_dir = Path(f"/tmp/omnidoc_subset125_{args.experiment_name}")
    images_stage = exp_dir / "images"
    json_dir = exp_dir / "json"
    md_dir = exp_dir / f"md_{args.experiment_name}"

    _stage_images(filenames, images_stage)
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.skip_ocr and any(json_dir.glob("*.json")):
        print(f"[orchestrator] reusing JSON in {json_dir}", flush=True)
        ocr_sec = 0.0
    else:
        _run([
            sys.executable, str(RUN_TOOL),
            "--server", args.server_url,
            "--out-dir", str(json_dir),
            "--images-dir", str(images_stage),
            "--concurrency", str(args.concurrency),
        ], timeout=args.ocr_timeout_sec)
        ocr_sec = time.time() - t0

    produced = sorted(p.name for p in json_dir.glob("*.json")
                      if p.name != "_errors.txt")
    expected = sorted(Path(f).stem + ".json" for f in filenames)
    if produced != expected:
        only_missing = sorted(set(expected) - set(produced))
        only_extra = sorted(set(produced) - set(expected))
        raise RuntimeError(
            f"OCR stage incomplete: missing={only_missing} extra={only_extra}")

    t1 = time.time()
    _run([
        sys.executable, str(MD_TOOL),
        "--in-dir", str(json_dir),
        "--out-dir", str(md_dir),
    ], timeout=300)
    md_sec = time.time() - t1

    cfg_path = _render_config(args.experiment_name, md_dir,
                              Path(args.base_config))
    save_name = f"{md_dir.name}_quick_match"

    scorer_py = str(BENCH_PY) if BENCH_PY.exists() else sys.executable
    t2 = time.time()
    _run([scorer_py, str(BENCH / "pdf_validation.py"),
          "--config", str(cfg_path)],
         timeout=args.score_timeout_sec, cwd=BENCH)
    score_sec = time.time() - t2

    summary = _summarize(args.experiment_name, save_name, docs,
                         ocr_sec, md_sec, score_sec)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.experiment_name}_{summary['timestamp']}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    tmp.replace(out_path)

    print(f"\n[summary] {out_path}")
    print("=== metrics ===")
    print(json.dumps(summary["summary"], indent=2))
    print("=== scoring coverage ===")
    print(json.dumps({
        "n_docs": summary["n_docs"],
        "n_scored": summary["n_scored"],
        "n_failed": summary["n_failed"],
        "failed_docs": summary["failed_docs"],
        "composition": summary["composition"],
        "scored_counts": summary["scored_counts"],
        "expected_counts": summary["expected_counts"],
    }, indent=2))
    print("=== runtime ===")
    print(json.dumps(summary["runtime_sec"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
