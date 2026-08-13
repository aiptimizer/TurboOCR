#!/usr/bin/env python3
"""Speed + VRAM regression matrix for the TurboOCR server across pipeline combinations.

For each named config (text-only, +table, +formula, +both; local SLANeXt / PP-FormulaNet
vs external PaddleOCR-VL) this:
  1. launches the server with the config's env/flags,
  2. waits for /capabilities,
  3. measures the server's VRAM footprint (delta over the pre-boot baseline) idle and under load,
  4. benchmarks the full structured pipeline (POST images to /ocr/raw?layout=1) -> images/s + p50/p90,
  5. probes single-region /infer table+formula latency where the config routes them,
  6. tears the server down and waits for VRAM to free.

Results are written to JSON and printed as a markdown table. With --baseline it compares
against a saved run and exits non-zero on a regression (throughput drop or VRAM growth past
--tolerance), so it doubles as a CI/regression gate.

Re-run (after building the server):
  python3 scripts/bench/bench_speed_matrix.py                       # full matrix, fresh baseline
  python3 scripts/bench/bench_speed_matrix.py --baseline result/bench_speed_matrix.json   # regression check
  python3 scripts/bench/bench_speed_matrix.py --only both_vl text_only --n 40 --concurrency 8

VL configs need a PaddleOCR-VL endpoint (default http://localhost:8077); they are SKIPPED
(not failed) when it is unreachable. Local-formula needs models/formula/{encoder.onnx,tokenizer.json};
SLANeXt needs models/table/slanext_encoder/. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def nvidia_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def http_get(url: str, timeout: float = 3.0):
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return -1, b""


def vllm_up(url: str) -> bool:
    code, _ = http_get(url.rstrip("/") + "/v1/models", timeout=3)
    return code == 200


def write_routing(path: Path, table: str | None, formula: str | None, ext: dict):
    """Emit a TURBO_ROUTING_CONFIG json. table/formula in {None,'default','slanext','vl'}.
    The external (kind:openai) table and formula backends may point at DIFFERENT models on
    DIFFERENT hosts — any OpenAI-compatible vision endpoint, not only PaddleOCR-VL."""
    backends, routes = {}, {"text": "default"}
    if table == "vl":
        backends["vl_table"] = {"kind": "openai", "base_url": ext["table_url"], "model": ext["table_model"],
                                 "prompt": "Table Recognition:", "parser": "otsl",
                                 "max_tokens": 4096, "timeout_s": 60}
    if formula == "vl":
        backends["vl_formula"] = {"kind": "openai", "base_url": ext["formula_url"], "model": ext["formula_model"],
                                   "prompt": "Formula Recognition:", "parser": "latex",
                                   "max_tokens": 1024, "timeout_s": 60}
    if table == "slanext":
        backends["slanext_local"] = {"kind": "local", "engine": "slanext"}
    routes["table"] = {"vl": "vl_table", "slanext": "slanext_local"}.get(table, "default")
    routes["formula"] = {"vl": "vl_formula"}.get(formula, "default")
    path.write_text(json.dumps({"backends": backends, "routes": routes}, indent=2))


def make_configs(a) -> list[dict]:
    enc = f"{a.models_dir}/table/slanext_encoder"
    slanext_env = {
        "TABLE_SLANEXT_ENCODER_ONNX": f"{enc}/SLANeXt_wired_encoder.onnx",
        "TABLE_SLANEXT_WIRELESS_ENCODER_ONNX": f"{enc}/SLANeXt_wireless_encoder.onnx",
        "TABLE_SLANEXT_DICT": f"{enc}/SLANeXt_dict_infer.txt",
    }
    formula_env = {
        "FORMULA_BACKEND": "ppformulanet_s",
        "FORMULA_ONNX": f"{a.models_dir}/formula/ppformulanet_s/inference_trt.onnx",
        "FORMULA_TOKENIZER": f"{a.models_dir}/formula/ppformulanet_s/tokenizer.json",
    }
    # (name, desc, extra_env, flags, route_table, route_formula, needs_vllm)
    raw = [
        ("text_only",   "Pure OCR (det+rec+cls), no layout", {}, ["--disable-layout"], None, None, False),
        ("layout_only", "OCR + layout, no table/formula", {}, [], "default", "default", False),
        ("table_slanext", "OCR + layout + SLANeXt table (local)", slanext_env, [], "slanext", "default", False),
        ("table_vl",    "OCR + layout + VL table (external)", {}, [], "vl", "default", True),
        ("formula_local", "OCR + layout + PP-FormulaNet (local)", formula_env, [], "default", "default", False),
        ("formula_vl",  "OCR + layout + VL formula (external)", {}, [], "default", "vl", True),
        ("both_local",  "OCR + layout + SLANeXt table + local formula", {**slanext_env, **formula_env}, [], "slanext", "default", False),
        ("both_vl",     "OCR + layout + VL table + VL formula (external)", {}, [], "vl", "vl", True),
        ("hybrid_slanext_vlformula", "SLANeXt table (local) + VL formula", slanext_env, [], "slanext", "vl", True),
    ]
    cfgs = []
    for name, desc, env, flags, rt, rf, needs_vllm in raw:
        cfgs.append(dict(name=name, desc=desc, env=env, flags=flags,
                         route_table=rt, route_formula=rf, needs_vllm=needs_vllm))
    return cfgs


def launch_server(a, cfg, routing_path: Path, log_path: Path, pool, ext) -> subprocess.Popen:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = a.trt_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    env["TURBO_ALLOW_ADHOC_BACKENDS"] = "1"
    # The local formula backend (ppformulanet_s) runs fully in-process on
    # ONNX Runtime CUDA — no Python sidecar, no PPFNS_* sidecar env. We still put
    # the modelopt venv first on PATH so any external Python tooling resolves a
    # CUDA-13-capable interpreter.
    modelopt_bin = str(REPO / ".venv-modelopt" / "bin")
    if os.path.isdir(modelopt_bin):
        env["PATH"] = modelopt_bin + ":" + env.get("PATH", "")
    if pool is not None:
        env["PIPELINE_POOL_SIZE"] = str(pool)  # else the server auto-sizes by free VRAM (inconsistent)
    env.update(cfg["env"])
    # Use a routing file ONLY when a VL (external) backend is involved — a routing file
    # overrides env-synth entirely, so an all-local config must drive table/formula via env
    # (TABLE_BACKEND + FORMULA_BACKEND/FORMULA_ONNX) or the unrouted modality is disabled.
    needs_routing = "vl" in (cfg["route_table"], cfg["route_formula"])
    if needs_routing:
        write_routing(routing_path, cfg["route_table"], cfg["route_formula"], ext)
        env["TURBO_ROUTING_CONFIG"] = str(routing_path)
    elif cfg["route_table"] == "slanext":
        env["TABLE_BACKEND"] = "slanext"  # local SLANeXt via env-synth (FORMULA_BACKEND already in env)
    args = [str(REPO / "build" / "turboocr-server"),
            "--http-port", str(a.port), "--grpc-port", str(a.grpc_port)] + cfg["flags"]
    log = open(log_path, "w")
    return subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))


def wait_ready(a, proc, timeout=120) -> bool:
    base = f"http://localhost:{a.port}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        code, _ = http_get(base + "/capabilities", timeout=3)
        if code == 200:
            return True
        time.sleep(2)
    return False


def teardown(proc, baseline_mib: int, timeout=30):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if nvidia_used_mib() <= baseline_mib + 500:
            break
        time.sleep(1)


def gt_counts(subset_gt_path: Path) -> dict:
    """GT denominators for a subset GT json: how many tables / formulas / text blocks the GROUND
    TRUTH contains, so extracted counts can be reported as 'N / GT' (coverage) instead of bare N."""
    try:
        gt = json.loads(subset_gt_path.read_text())
    except Exception:
        return {}
    def n(cat):
        return sum(1 for d in gt for b in d.get("layout_dets", [])
                   if b.get("category_type") == cat)
    return {"docs": len(gt), "gt_tables": n("table"),
            "gt_formulas": n("equation_isolated"), "gt_text": n("text_block")}


def load_images(a):
    """Return (list[(path, bytes)], provenance, gt_denominators).

    Default uses the SAME stratified subset as the accuracy harness — generated by
    scripts/eval/omnidoc_subset_n.py into /tmp/omnidoc_subset<N> — so speed and accuracy run on
    IDENTICAL docs and the GT denominators apply. Falls back to --images-dir / repo fixtures."""
    files, prov, gt = [], "", {}
    if a.images_set == "subset":
        subset_dir = Path(a.subset_dir or f"/tmp/omnidoc_subset{a.n}")
        subset_gt = subset_dir / f"OmniDocBench_subset{a.n}.json"
        gen = REPO / "scripts" / "eval" / "omnidoc_subset_n.py"
        if not subset_gt.is_file() and gen.is_file() and Path(a.gt).is_file():
            try:
                subprocess.run([sys.executable, str(gen), "--n", str(a.n),
                                "--out-dir", str(subset_dir), "--gt", a.gt,
                                "--images-dir", a.images_dir],
                               check=True, capture_output=True, timeout=180)
            except Exception as e:
                print(f"[bench] subset generation failed: {e}", file=sys.stderr)
        if subset_gt.is_file():
            for d in json.loads(subset_gt.read_text()):
                ip = Path(a.images_dir) / d.get("page_info", {}).get("image_path", "")
                if ip.is_file():
                    files.append(ip)
            gt = gt_counts(subset_gt)
            prov = (f"OmniDocBench {a.n}-doc stratified subset via scripts/eval/omnidoc_subset_n.py "
                    f"(identical docs to the accuracy harness): {len(files)} images; ground truth = "
                    f"{gt.get('gt_tables','?')} tables, {gt.get('gt_formulas','?')} formulas, "
                    f"{gt.get('gt_text','?')} text blocks")
    if not files:
        d = Path(a.images_dir)
        if d.is_dir():
            files = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[: a.n]
            prov = f"first {len(files)} images (sorted) from {d}"
    if not files:
        fix = sorted((REPO / "tests/fixtures/images/png").glob("*.png"))
        files = [fix[i % len(fix)] for i in range(a.n)] if fix else []
        prov = f"repo fixtures cycled to n={a.n}"
    imgs = [(p, p.read_bytes()) for p in files]
    return imgs, prov, gt


def post_layout(base: str, img: bytes, ct: str, timeout: int):
    """Return (latency_s, n_tables_extracted, n_formulas_extracted). Counts REAL extracted
    elements from the parsed /ocr/raw?layout=1 JSON: `tables[]` entries with non-empty `html`
    and `formulas[]` entries with non-empty `latex` — NOT a substring count of the `"latex"`
    field key (which appears once per formula record AND would miss table-cell LaTeX)."""
    t = time.time()
    req = urllib.request.Request(base + "/ocr/raw?layout=1", img, {"Content-Type": ct})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(r.read())
        nt = sum(1 for x in d.get("tables", [])
                 if isinstance(x, dict) and str(x.get("html", "")).strip())
        nf = sum(1 for x in d.get("formulas", [])
                 if isinstance(x, dict) and str(x.get("latex", "")).strip())
        return time.time() - t, nt, nf
    except Exception:
        return time.time() - t, 0, 0


def benchmark(a, imgs) -> dict:
    base = f"http://localhost:{a.port}"
    # warmup
    if imgs:
        for _, b in imgs[: min(4, len(imgs))]:
            post_layout(base, b, "image/png", a.req_timeout)
    used_idle = nvidia_used_mib()
    lat, tabs, forms = [], 0, 0
    peak = [used_idle]

    def work(item):
        p, b = item
        ct = "image/jpeg" if str(p).lower().endswith((".jpg", ".jpeg")) else "image/png"
        dt, nt, nf = post_layout(base, b, ct, a.req_timeout)
        peak[0] = max(peak[0], nvidia_used_mib())
        return dt, nt, nf

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for dt, nt, nf in ex.map(work, imgs):
            lat.append(dt * 1000); tabs += nt; forms += nf
    wall = time.time() - t0
    lat.sort()

    def pct(p):
        return round(lat[min(len(lat) - 1, int(len(lat) * p))], 1) if lat else 0.0
    return dict(n=len(imgs), wall_s=round(wall, 2),
                images_per_s=round(len(imgs) / wall, 2) if wall else 0.0,
                p50_ms=pct(0.5), p90_ms=pct(0.9), p99_ms=pct(0.99),
                tables_found=tabs, formulas_found=forms,
                vram_idle_mib=used_idle, vram_peak_mib=peak[0])


def infer_latency(a, modality: str, backend) -> float | None:
    base = f"http://localhost:{a.port}"
    import base64
    table_img = (REPO / "tests/fixtures/images/png/table.png")
    if not table_img.exists():
        return None
    b64 = base64.b64encode(table_img.read_bytes()).decode()
    lat = []
    for _ in range(8):
        t = time.time()
        body = json.dumps({"image": b64, "modality": modality, "backend": backend}).encode()
        req = urllib.request.Request(base + "/infer", body, {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=a.req_timeout).read()
            lat.append((time.time() - t) * 1000)
        except Exception:
            return None
    lat.sort()
    return round(lat[len(lat) // 2], 1)


def run_matrix(a):
    cfgs = make_configs(a)
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only]
    imgs, prov, gt = load_images(a)
    if not imgs:
        print("ERROR: no benchmark images (set --images-dir / --gt)", file=sys.stderr)
        sys.exit(2)
    pools = a.pool_sizes if a.pool_sizes else [None]  # None -> server auto-sizes by free VRAM
    ext = {"table_url": a.table_vl_url or a.vllm_url, "table_model": a.table_vl_model,
           "formula_url": a.formula_vl_url or a.vllm_url, "formula_model": a.formula_vl_model}
    print(f"[bench] image set: {prov}")
    print(f"[bench] {len(imgs)} images, concurrency={a.concurrency}, pools={pools}, port={a.port}")
    have_ext = vllm_up(ext["table_url"]) or vllm_up(ext["formula_url"])
    print(f"[bench] external table={ext['table_url']}({ext['table_model']}) "
          f"formula={ext['formula_url']}({ext['formula_model']}): "
          f"{'UP' if have_ext else 'DOWN (VL configs skipped)'}")
    routing_path = Path("/tmp/bench_speed_routing.json")
    results = {}
    for cfg in cfgs:
        for pool in pools:
            key = cfg["name"] if pool is None else f"{cfg['name']}@pool{pool}"
            if cfg["needs_vllm"] and not have_ext:
                print(f"[bench] {key}: SKIP (no external endpoint)")
                results[key] = {"skipped": "no external endpoint", "desc": cfg["desc"], "pool_size": pool}
                continue
            baseline = nvidia_used_mib()
            log_path = Path(f"/tmp/bench_server_{key.replace('@', '_')}.log")
            print(f"[bench] {key}: launching ({cfg['desc']}) ...")
            proc = launch_server(a, cfg, routing_path, log_path, pool, ext)
            try:
                if not wait_ready(a, proc):
                    tail = log_path.read_text()[-500:] if log_path.exists() else ""
                    print(f"[bench] {key}: FAILED to start. log tail:\n{tail}")
                    results[key] = {"error": "server did not become ready", "desc": cfg["desc"], "pool_size": pool}
                    continue
                res = benchmark(a, imgs)
                res["vram_footprint_mib"] = res["vram_idle_mib"] - baseline
                res["pool_size"] = pool
                res["desc"] = cfg["desc"]
                if cfg["route_table"] in ("vl", "slanext"):
                    be = {"vl": "vl_table", "slanext": {"kind": "local", "engine": "slanext"}}[cfg["route_table"]]
                    res["infer_table_p50_ms"] = infer_latency(a, "table", be)
                if cfg["route_formula"] == "vl":
                    res["infer_formula_p50_ms"] = infer_latency(a, "formula", "vl_formula")
                results[key] = res
                print(f"[bench] {key}: {res['images_per_s']} images/s  p50={res['p50_ms']}ms  "
                      f"VRAM={res['vram_footprint_mib']}MiB  tables={res['tables_found']} formulas={res['formulas_found']}")
            finally:
                teardown(proc, baseline)
    return results, {"image_set": prov, "gt": gt, "images": [p.name for p, _ in imgs]}


def markdown_table(results: dict, gt: dict = None) -> str:
    gt = gt or {}
    gtt, gtf = gt.get("gt_tables"), gt.get("gt_formulas")
    tcol = f"tables (/{gtt} GT)" if gtt else "tables"
    fcol = f"formulas (/{gtf} GT)" if gtf else "formulas"
    cols = ["config", "pool", "images/s", "p50 ms", "p90 ms", "VRAM MiB", tcol, fcol, "infer tbl ms", "infer fml ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, r in results.items():
        pool = r.get("pool_size") if r.get("pool_size") is not None else "auto"
        if "skipped" in r or "error" in r:
            lines.append(f"| {name} | {pool} | _{r.get('skipped') or r.get('error')}_ | | | | | | | |")
            continue
        lines.append("| " + " | ".join(str(x) for x in [
            name, pool, r["images_per_s"], r["p50_ms"], r["p90_ms"], r["vram_footprint_mib"],
            r["tables_found"], r["formulas_found"],
            r.get("infer_table_p50_ms", ""), r.get("infer_formula_p50_ms", "")]) + " |")
    return "\n".join(lines)


def regression_check(results: dict, baseline_path: Path, tol: float) -> int:
    base = json.loads(baseline_path.read_text()).get("results", {})
    bad = 0
    for name, r in results.items():
        b = base.get(name)
        if not b or "images_per_s" not in r or "images_per_s" not in b:
            continue
        if b["images_per_s"] and r["images_per_s"] < b["images_per_s"] * (1 - tol):
            print(f"REGRESSION {name}: images/s {r['images_per_s']} < {b['images_per_s']} (-{tol:.0%})"); bad += 1
        if b.get("vram_footprint_mib", 0) and r.get("vram_footprint_mib", 0) > b["vram_footprint_mib"] * (1 + tol):
            print(f"REGRESSION {name}: VRAM {r['vram_footprint_mib']} > {b['vram_footprint_mib']} (+{tol:.0%})"); bad += 1
    print("REGRESSION CHECK:", "FAIL" if bad else "PASS")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--trt-lib", default=os.environ.get(
        "TENSORRT_LIB", os.path.expanduser("~/TensorRT-10.15.1.29/lib")))
    p.add_argument("--vllm-url", default="http://localhost:8077", help="default external endpoint for table+formula")
    p.add_argument("--pool-sizes", nargs="*", type=int,
                   help="explicit PIPELINE_POOL_SIZE values to sweep (default: server auto-sizes, one run/config)")
    p.add_argument("--table-vl-url", help="external TABLE endpoint base_url (default --vllm-url; can differ from formula)")
    p.add_argument("--table-vl-model", default="PaddleOCR-VL", help="external table model name")
    p.add_argument("--formula-vl-url", help="external FORMULA endpoint base_url (default --vllm-url; can differ from table)")
    p.add_argument("--formula-vl-model", default="PaddleOCR-VL", help="external formula model name")
    p.add_argument("--images-dir", default=str(REPO.parent / "omnidocbench" / "data" / "images"))
    p.add_argument("--gt", default=str(REPO.parent / "omnidocbench" / "data" / "OmniDocBench.json"),
                   help="OmniDocBench GT json (for the shared stratified subset + GT denominators)")
    p.add_argument("--images-set", choices=["subset", "dir"], default="subset",
                   help="subset = the SAME stratified OmniDocBench subset as the accuracy harness "
                        "(scripts/eval/omnidoc_subset_n.py, default); dir = first-N from --images-dir")
    p.add_argument("--subset-dir", default=None, help="subset cache dir (default /tmp/omnidoc_subset<N>)")
    p.add_argument("--n", type=int, default=125, help="subset size / images per config (deterministic)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--port", type=int, default=8833)
    p.add_argument("--grpc-port", type=int, default=50833)
    p.add_argument("--req-timeout", type=int, default=120)
    p.add_argument("--only", nargs="*", help="subset of config names to run")
    p.add_argument("--out", default=str(REPO / "result" / "bench_speed_matrix.json"))
    p.add_argument("--baseline", help="compare against this saved run; non-zero exit on regression")
    p.add_argument("--tolerance", type=float, default=0.15)
    a = p.parse_args()

    results, meta = run_matrix(a)
    payload = {"results": results,
               "config": {"n": a.n, "concurrency": a.concurrency, "images_dir": a.images_dir,
                          "images_set": a.images_set},
               "image_set": meta["image_set"], "gt": meta.get("gt", {}), "images": meta["images"]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    md = (f"**Image set:** {meta['image_set']}  \n"
          f"**Concurrency:** {a.concurrency}\n\n" + markdown_table(results, meta.get("gt", {})))
    print("\n" + md + "\n")
    (out.with_suffix(".md")).write_text(md + "\n")
    print(f"[bench] wrote {out} and {out.with_suffix('.md')}")
    if a.baseline:
        return 1 if regression_check(results, Path(a.baseline), a.tolerance) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
