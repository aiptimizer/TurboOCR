#!/usr/bin/env python3
"""Run OCR -> markdown -> OmniDocBench scoring for the 5-doc subset.

Workflow:
  1. POST each of the 5 subset images to the OCR server (parallel).
  2. Convert per-image JSON to OmniDocBench-style markdown.
  3. Render a per-experiment scorer config (so its save_name doesn't
     collide with concurrent / past runs) and invoke pdf_validation.py.
  4. Read the scorer's *_metric_result.json plus the 4 *_per_page_edit.json
     files and emit one compact summary at
     /tmp/omnidoc_runs/<experiment>_<timestamp>.json.

The whole chain is capped with explicit per-stage timeouts. Any of the 5
docs failing to score (missing from any per-page file) is a hard error.
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
BENCH = REPO.parent / "omnidocbench"
SUBSET_DIR = Path("/tmp/omnidoc_subset5")
SUBSET_GT = SUBSET_DIR / "OmniDocBench_subset5.json"
SUBSET_SCRIPT = REPO / "scripts" / "eval" / "omnidoc_subset.py"
RUN_TOOL = REPO / "tools" / "bench" / "omnidoc_run.py"
MD_TOOL = REPO / "tools" / "bench" / "omnidoc_to_md.py"
RUNS_DIR = Path("/tmp/omnidoc_runs")
BASE_CONFIG = BENCH / "configs" / "end2end_subset5.yaml"
IMAGES_DIR = BENCH / "data" / "images"
BENCH_PY = BENCH / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-name", required=True,
                   help="Slug used for prediction dir + summary filename.")
    p.add_argument("--server-url", default="http://localhost:8080")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--ocr-timeout-sec", type=int, default=180)
    p.add_argument("--score-timeout-sec", type=int, default=240)
    p.add_argument("--skip-ocr", action="store_true",
                   help="Reuse existing JSON predictions if present.")
    return p.parse_args()


def _load_subset_filenames() -> list[str]:
    with SUBSET_GT.open() as f:
        gt = json.load(f)
    return [d["page_info"]["image_path"] for d in gt]


def _ensure_subset() -> list[str]:
    if not SUBSET_GT.exists():
        subprocess.run([sys.executable, str(SUBSET_SCRIPT)], check=True)
    return _load_subset_filenames()


def _stage_images(filenames: list[str], dest: Path) -> None:
    """Copy the 5 subset images into a dedicated dir so omnidoc_run.py
    only sees those (it walks its --images-dir top-level)."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.iterdir():
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


def _render_config(experiment: str, pred_dir: Path) -> Path:
    base = BASE_CONFIG.read_text()
    rendered = base.replace(
        "data_path: /tmp/omnidoc_subset5/md",
        f"data_path: {pred_dir}",
    )
    cfg_path = BENCH / "configs" / f"end2end_subset5__{experiment}.yaml"
    cfg_path.write_text(rendered)
    return cfg_path


def _read_per_page(save_name: str, element: str) -> dict[str, float]:
    p = BENCH / "result" / f"{save_name}_{element}_per_page_edit.json"
    if not p.exists():
        raise FileNotFoundError(f"scorer did not produce {p}")
    return json.loads(p.read_text())


def _summarize(experiment: str, save_name: str, filenames: list[str],
               ocr_sec: float, md_sec: float, score_sec: float) -> dict:
    metric_path = BENCH / "result" / f"{save_name}_metric_result.json"
    if not metric_path.exists():
        raise FileNotFoundError(f"scorer did not produce {metric_path}")
    metric = json.loads(metric_path.read_text())

    def _all(section: str, key: str, *prefer: str):
        """Pull a scalar from metric[section]['all'][key]. Schema varies:
        Edit_dist blocks expose ALL_page_avg/edit_whole/edit_sample_avg;
        TEDS / CDM blocks expose a flat 'all'. Try keys in order; if the
        block itself is a scalar, return it directly."""
        try:
            block = metric[section]["all"][key]
        except KeyError:
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

    missing: dict[str, list[str]] = {}
    for elem, edits in per_page.items():
        for fn in filenames:
            if fn not in edits:
                missing.setdefault(elem, []).append(fn)
    if missing:
        raise RuntimeError(
            "scorer dropped docs (a doc with no matching block is fine ONLY if "
            "the GT has no instances of that element; but our subset is "
            "guaranteed text+formula+table). missing="
            + json.dumps(missing, ensure_ascii=False))

    per_doc = []
    for fn in filenames:
        per_doc.append({
            "image": fn,
            "text_block_edit": per_page["text_block"][fn],
            "display_formula_edit": per_page["display_formula"][fn],
            "table_edit": per_page["table"][fn],
            "reading_order_edit": per_page["reading_order"][fn],
        })

    return {
        "experiment": experiment,
        "save_name": save_name,
        "timestamp": int(time.time()),
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
    filenames = _ensure_subset()
    print(f"[orchestrator] subset of {len(filenames)} docs", flush=True)

    exp_dir = Path(f"/tmp/omnidoc_subset5_{args.experiment_name}")
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

    produced = sorted(p.name for p in json_dir.glob("*.json") if p.name != "_errors.txt")
    expected = sorted(Path(f).stem + ".json" for f in filenames)
    if produced != expected:
        raise RuntimeError(
            f"OCR stage produced {produced}, expected {expected}")

    t1 = time.time()
    _run([
        sys.executable, str(MD_TOOL),
        "--in-dir", str(json_dir),
        "--out-dir", str(md_dir),
    ], timeout=60)
    md_sec = time.time() - t1

    cfg_path = _render_config(args.experiment_name, md_dir)
    save_name = f"{md_dir.name}_quick_match"

    scorer_py = str(BENCH_PY) if BENCH_PY.exists() else sys.executable
    t2 = time.time()
    _run([scorer_py, str(BENCH / "pdf_validation.py"),
          "--config", str(cfg_path)],
         timeout=args.score_timeout_sec, cwd=BENCH)
    score_sec = time.time() - t2

    summary = _summarize(args.experiment_name, save_name, filenames,
                         ocr_sec, md_sec, score_sec)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{args.experiment_name}_{summary['timestamp']}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    tmp.replace(out_path)

    print(f"\n[summary] {out_path}")
    print(json.dumps(summary["summary"], indent=2))
    print(json.dumps(summary["runtime_sec"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
