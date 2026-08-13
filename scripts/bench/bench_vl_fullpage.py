#!/usr/bin/env python3
"""Benchmark PaddleOCR-VL doing the WHOLE document (standalone) — the head-to-head baseline for the
hybrid C++ pipeline.

The C++ server only ever sends table/formula CROPS to PaddleOCR-VL; it has no full-page VL mode. This
runs PaddleOCR-VL the way it is designed to be used: the official `PaddleOCRVL` pipeline
(`tools/bench/omnidoc_run_paddlevl.py` — local PP-DocLayoutV3 layout, then VL on EVERY region incl. text)
over the SAME stratified OmniDocBench subset as the hybrid runs, times it (pages/s), and scores the
markdown with the SAME OmniDocBench scorer + config, so the metrics are directly comparable.

  python3 scripts/bench/bench_vl_fullpage.py --subset 125 --server http://localhost:8077/v1

Needs the `compare-ocrs/.venv` (has `paddleocr` 3.4) for the VL pipeline, and the `omnidocbench/.venv`
for scoring (both auto-detected). Reuses scripts/eval/omnidoc_run_and_score_n.py for the subset + scorer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))
import omnidoc_run_and_score_n as acc  # subset gen + image staging + scorer config + summary

VL_RUNNER = REPO / "tools" / "bench" / "omnidoc_run_paddlevl.py"
VL_VENV = REPO.parent / "compare-ocrs" / ".venv" / "bin" / "python"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=int, default=125, help="OmniDocBench subset size (same as the hybrid runs)")
    p.add_argument("--server", default="http://localhost:8077/v1", help="vLLM PaddleOCR-VL OpenAI endpoint")
    p.add_argument("--pipeline-version", default="v1.5",
                   help="standalone PaddleOCRVL pipeline version (paddleocr 3.4 supports v1/v1.5, NOT v1.6)")
    p.add_argument("--vl-concurrency", type=int, default=16)
    p.add_argument("--doc-workers", type=int, default=2)
    p.add_argument("--experiment", default="vl_fullpage")
    p.add_argument("--out", default=str(REPO / "result" / "bench_vl_fullpage.json"))
    p.add_argument("--ocr-timeout-sec", type=int, default=3600)
    p.add_argument("--score-timeout-sec", type=int, default=1800)
    a = p.parse_args()

    # 1. the shared stratified subset (identical docs to the hybrid accuracy/speed runs)
    docs = acc._ensure_subset(a.subset)
    filenames = [d["page_info"]["image_path"] for d in docs]
    print(f"[vl] {len(filenames)}-doc subset (shared with the hybrid runs)", flush=True)

    work = Path(f"/tmp/omnidoc_subset{a.subset}_{a.experiment}")
    images_stage = work / "images"
    acc._stage_images(filenames, images_stage)
    vl_out = work / f"vl_{a.pipeline_version}"
    md_dir = vl_out / "md"  # the runner writes per-image markdown here

    # 2. run PaddleOCR-VL full-page (timed -> pages/s)
    vl_py = str(VL_VENV) if VL_VENV.is_file() else sys.executable
    env = {**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
    cmd = [vl_py, str(VL_RUNNER), "--images-dir", str(images_stage), "--out-dir", str(vl_out),
           "--server", a.server, "--pipeline-version", a.pipeline_version,
           "--vl-concurrency", str(a.vl_concurrency), "--doc-workers", str(a.doc_workers)]
    print(f"[vl] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    import subprocess
    subprocess.run(cmd, check=True, timeout=a.ocr_timeout_sec, env=env)
    ocr_sec = time.time() - t0
    n_md = len(list(md_dir.glob("*.md")))
    pages_per_s = round(n_md / ocr_sec, 3) if ocr_sec else 0.0
    print(f"[vl] {n_md} pages in {ocr_sec:.1f}s = {pages_per_s} pages/s", flush=True)

    # 3. score with the SAME OmniDocBench scorer/config as the hybrid runs. The scorer derives its
    #    save_name from the prediction dir basename — give it a unique, experiment-named dir.
    pred_dir = work / f"md_{a.experiment}"
    if pred_dir.is_symlink() or pred_dir.exists():
        pred_dir.unlink() if pred_dir.is_symlink() else None
    if not pred_dir.exists():
        try:
            pred_dir.symlink_to(md_dir)
        except OSError:
            import shutil
            shutil.copytree(md_dir, pred_dir)
    cfg = acc._render_config(a.experiment, pred_dir)
    save_name = f"{pred_dir.name}_quick_match"
    scorer_py = str(acc.BENCH_PY) if acc.BENCH_PY.exists() else sys.executable
    t1 = time.time()
    acc._run([scorer_py, str(acc.BENCH / "pdf_validation.py"), "--config", str(cfg)],
             timeout=a.score_timeout_sec, cwd=acc.BENCH)
    score_sec = time.time() - t1

    summary = acc._summarize(a.experiment, save_name, docs, ocr_sec, 0.0, score_sec)
    summary["pipeline"] = f"PaddleOCR-{a.pipeline_version} full-page (standalone, VL on every region)"
    summary["pages_per_s"] = pages_per_s
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    metrics = summary.get("summary", {})
    print("\n=== VL full-page result ===")
    print(f"pages/s = {pages_per_s}   scored {summary['n_scored']}/{summary['n_docs']} "
          f"(failed {summary['n_failed']})")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"[vl] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
