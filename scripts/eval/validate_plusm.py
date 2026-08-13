#!/usr/bin/env python3
"""PP-FormulaNet_plus-M accuracy gate harness.

Scores any backend's per-crop LaTeX predictions against ground truth on the
mixed EN+ZH formula set:
  - CDM-F1 on English crops (image-level Character Detection Matching, via the
    OmniDocBench scorer). Renders LaTeX -> needs the omnidocbench CDM deps, so
    run this under .venv-omnidoc.
  - CJK-char-F1 on Chinese (CJK-containing) crops. Render-free multiset F1 over
    the CJK characters -- the faithful proxy because the local TeX has no CJK
    fonts (CDM can't render Chinese -> would return 0 for every backend).
  - normalized edit similarity as a secondary signal for both buckets.

Usage (run with the omnidocbench venv so `cdm` imports):
  .venv-omnidoc/bin/python scripts/eval/validate_plusm.py \
      --manifest <gate_dir>/manifest.json \
      --preds <preds>           # JSONL {"id","latex"} | dir of <id>.txt | preds.json
      [--preds-key plusM]       # which key to read if --preds is a preds.json map
      [--out report.json] [--cdm-workers 10]

GATE: plus-M must beat PP-FormulaNet-S/plus-S on ZH CJK-char-F1 (it does:
~0.73 vs 0.00) and hold EN CDM (~0.99). Use --baseline to print a delta vs a
reference preds file (e.g. the paddle reference) for C++-prototype regression.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

OMNIDOCBENCH = os.environ.get(
    "OMNIDOCBENCH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "omnidocbench"))


def is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def clean_latex(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\$+", "", s)
    s = re.sub(r"\$+$", "", s)
    return s.strip()


def cjk_f1(pred: str, gt: str) -> float | None:
    from collections import Counter
    pc = Counter(c for c in pred if is_cjk(c))
    gc = Counter(c for c in gt if is_cjk(c))
    if sum(gc.values()) == 0:
        return None
    inter = sum((pc & gc).values())
    p = inter / max(sum(pc.values()), 1)
    r = inter / sum(gc.values())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def edit_sim(a: str, b: str) -> float:
    a = re.sub(r"\s+", "", a or "")
    b = re.sub(r"\s+", "", b or "")
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[lb] / max(la, lb, 1)


def load_preds(path: str, key: str | None) -> dict[str, str]:
    """Accept: JSONL {"id","latex"}; a dir of <id>.txt; or a JSON map
    {id: latex} / preds.json {"preds": {id: {backend: latex}}}."""
    if os.path.isdir(path):
        out = {}
        for fn in os.listdir(path):
            if fn.endswith(".txt"):
                out[fn[:-4]] = open(os.path.join(path, fn), encoding="utf-8").read().strip()
        return out
    txt = open(path, encoding="utf-8").read().strip()
    if path.endswith(".jsonl"):
        out = {}
        for line in txt.splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = d.get("latex", d.get("pred", ""))
        return out
    obj = json.loads(txt)
    if isinstance(obj, dict) and "preds" in obj and isinstance(obj["preds"], dict):
        inner = obj["preds"]
        sample = next(iter(inner.values()))
        if isinstance(sample, dict):
            k = key or next(iter(sample))
            return {cid: v.get(k, "") for cid, v in inner.items()}
        return inner
    if isinstance(obj, dict):
        return {k2: (v if isinstance(v, str) else v.get(key or "latex", "")) for k2, v in obj.items()}
    raise SystemExit(f"unrecognized preds format: {path}")


def _cdm_one(args):
    cid, pred, gt, tmp = args
    sys.path.insert(0, OMNIDOCBENCH)
    from src.metrics.cdm.cdm import cdm
    try:
        return cid, float(cdm(pred, gt, save_vis=False, tmp_dir=tmp))
    except Exception:
        return cid, 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest.json: [{id,bucket,cjk,gt}]")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--preds-key", default=None, help="backend key if --preds is a preds.json map")
    ap.add_argument("--baseline", default=None, help="optional reference preds to delta against")
    ap.add_argument("--baseline-key", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cdm-workers", type=int, default=10)
    ap.add_argument("--tmp", default=os.path.join(tempfile.gettempdir(), "plusm_validate"))
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding="utf-8"))
    gt = {m["id"]: m for m in man}
    preds = load_preds(args.preds, args.preds_key)
    os.makedirs(args.tmp, exist_ok=True)

    en = [m for m in man if m["bucket"] == "EN"]
    zh = [m for m in man if m["bucket"] == "ZH"]

    # CDM on EN (parallel)
    cdm_jobs = [(m["id"], clean_latex(preds.get(m["id"], "")), m["gt"],
                 f'{args.tmp}/{m["id"]}') for m in en]
    cdm = {}
    if cdm_jobs:
        with ProcessPoolExecutor(max_workers=args.cdm_workers) as ex:
            for cid, s in ex.map(_cdm_one, cdm_jobs):
                cdm[cid] = s

    rows = {}
    for m in man:
        cid = m["id"]
        p = clean_latex(preds.get(cid, ""))
        rows[cid] = {
            "bucket": m["bucket"],
            "cdm": cdm.get(cid),
            "cjk_f1": cjk_f1(p, m["gt"]),
            "edit_sim": edit_sim(p, m["gt"]),
            "pred_empty": not p,
        }

    en_cdm = [rows[m["id"]]["cdm"] for m in en if rows[m["id"]]["cdm"] is not None]
    zh_f1 = [rows[m["id"]]["cjk_f1"] for m in zh if rows[m["id"]]["cjk_f1"] is not None]
    en_es = [rows[m["id"]]["edit_sim"] for m in en]
    zh_es = [rows[m["id"]]["edit_sim"] for m in zh]
    mean = lambda v: round(statistics.mean(v), 4) if v else None

    summary = {
        "n_en": len(en), "n_zh": len(zh),
        "EN_CDM_F1": mean(en_cdm),
        "ZH_CJK_char_F1": mean(zh_f1),
        "EN_edit_sim": mean(en_es),
        "ZH_edit_sim": mean(zh_es),
        "n_empty": sum(1 for r in rows.values() if r["pred_empty"]),
    }

    print(f"=== plus-M accuracy gate ({args.preds}) ===")
    print(f"  EN  (n={len(en)}): CDM-F1 {summary['EN_CDM_F1']}   editSim {summary['EN_edit_sim']}")
    print(f"  ZH  (n={len(zh)}): CJK-char-F1 {summary['ZH_CJK_char_F1']}   editSim {summary['ZH_edit_sim']}")
    print(f"  empty preds: {summary['n_empty']}/{len(man)}")

    if args.baseline:
        base = load_preds(args.baseline, args.baseline_key)
        # token-free agreement: identical-after-whitespace-norm rate + mean editSim(pred, base)
        ident = sum(1 for m in man
                    if re.sub(r"\s+", "", clean_latex(preds.get(m["id"], "")))
                    == re.sub(r"\s+", "", clean_latex(base.get(m["id"], "")))) / len(man)
        es = statistics.mean(edit_sim(preds.get(m["id"], ""), base.get(m["id"], "")) for m in man)
        summary["vs_baseline_exact"] = round(ident, 4)
        summary["vs_baseline_edit_sim"] = round(es, 4)
        print(f"  vs baseline ({args.baseline}): exact {ident:.3f}  editSim {es:.3f}")

    if args.out:
        json.dump({"summary": summary, "per_sample": rows}, open(args.out, "w"),
                  ensure_ascii=False, indent=1)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
