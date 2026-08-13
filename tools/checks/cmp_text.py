#!/usr/bin/env python3
"""Compare per-box recognized text between two server env configs (e.g. scalar
vs batched rec) to quantify accuracy drift: #boxes differing + char error rate.
Usage: tools/checks/cmp_text.py 'LABEL_A:ENV..' 'LABEL_B:ENV..'
"""
import base64, glob, json, os, signal, subprocess, sys, time, urllib.request

# repo root = up out of tools/bench/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = f"{REPO}/build_cpu/turboocr-server"
FUNSD = os.path.join(os.path.dirname(REPO), "compare-ocrs", "funsd_cache")
PORT = 18090
N = int(os.environ.get("BENCH_N", "30"))


def collect(env_extra):
    env = dict(os.environ)
    env.update({"PORT": str(PORT), "GRPC_PORT": str(PORT + 1),
                "PIPELINE_POOL_SIZE": "2", "DET_ONNX": "models/det.onnx",
                "REC_ONNX": env_extra.get("REC_ONNX", "models/rec.onnx"),
                "CLS_ONNX": "models/cls.onnx", "REC_DICT": "models/keys.txt",
                "OCR_LANG": "", "LAYOUT_DISABLED": "1"})
    env.update(env_extra)
    imgs = sorted(glob.glob(f"{FUNSD}/*.png"))[:N]
    b64 = [base64.b64encode(open(p, "rb").read()).decode() for p in imgs]
    log = open("/tmp/cmp_srv.log", "w")
    sv = subprocess.Popen([BIN], cwd=REPO, env=env, stdout=log, stderr=log)
    try:
        for _ in range(60):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health/ready", timeout=3).read()
                break
            except Exception:
                pass
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/ocr/batch",
                                     data=json.dumps({"images": b64}).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=200).read())
    finally:
        sv.send_signal(signal.SIGTERM)
        try: sv.wait(timeout=15)
        except Exception: sv.kill()
    items = resp.get("batch_results", resp.get("results", resp))
    out = []
    for it in items:
        res = it.get("results", it) if isinstance(it, dict) else it
        out.append([x.get("text", "") for x in res] if isinstance(res, list) else [])
    return out


def lev(a, b):
    if a == b: return 0
    m, n = len(a), len(b)
    if not m: return n
    if not n: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i-1] != b[j-1]))
        prev = cur
    return prev[n]


def parse(arg):
    label, _, rest = arg.partition(":")
    env = dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)
    return label, env


def main():
    la, ea = parse(sys.argv[1])
    lb, eb = parse(sys.argv[2])
    A = collect(ea)
    B = collect(eb)
    tot_boxes = sum(len(p) for p in A)
    diff_boxes = 0
    tot_chars = 0
    err_chars = 0
    examples = []
    for pa, pb in zip(A, B):
        for i in range(max(len(pa), len(pb))):
            ta = pa[i] if i < len(pa) else ""
            tb = pb[i] if i < len(pb) else ""
            tot_chars += max(len(ta), 1)
            if ta != tb:
                diff_boxes += 1
                err_chars += lev(ta, tb)
                if len(examples) < 12:
                    examples.append((ta, tb))
    print(f"{la} vs {lb}: boxes={tot_boxes} differing_boxes={diff_boxes} "
          f"({100*diff_boxes/max(tot_boxes,1):.2f}%) char_err={err_chars} "
          f"CER={100*err_chars/max(tot_chars,1):.3f}%")
    for ta, tb in examples:
        print(f"   A:{ta!r}\n   B:{tb!r}")


if __name__ == "__main__":
    main()
