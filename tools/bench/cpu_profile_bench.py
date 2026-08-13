#!/usr/bin/env python3
"""Profile-driven CPU OCR benchmark. Starts turboocr-server with a given env,
warms up, resets per-stage profiling, runs timed /ocr/batch rounds, reads the
/profile breakdown, and emits img/s + per-stage ms/img + an accuracy signature
(box count + text hash) for regression gating.

Usage: tools/bench/cpu_profile_bench.py [label] [KEY=VAL ...]  # extra server env
"""
import base64, glob, hashlib, json, os, signal, subprocess, sys, time, urllib.request

# repo root = up out of tools/bench/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = f"{REPO}/build_cpu/turboocr-server"
FUNSD = os.path.join(os.path.dirname(REPO), "compare-ocrs", "funsd_cache")
PORT = int(os.environ.get("BENCH_PORT", "18080"))
GRPC = PORT + 1
N_IMAGES = int(os.environ.get("BENCH_N", "30"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))

label = sys.argv[1] if len(sys.argv) > 1 else "run"
extra_env = dict(kv.split("=", 1) for kv in sys.argv[2:] if "=" in kv)


def http(path, data=None, timeout=120):
    url = f"http://127.0.0.1:{PORT}{path}"
    if data is not None:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    imgs = sorted(glob.glob(f"{FUNSD}/*.png"))[:N_IMAGES]
    assert imgs, "no FUNSD images"
    b64 = [base64.b64encode(open(p, "rb").read()).decode() for p in imgs]

    env = dict(os.environ)
    env.update({
        "PORT": str(PORT), "GRPC_PORT": str(GRPC),
        "PIPELINE_POOL_SIZE": env.get("PIPELINE_POOL_SIZE", "4"),
        "DET_ONNX": env.get("DET_ONNX", "models/det.onnx"),
        "REC_ONNX": "models/rec.onnx", "CLS_ONNX": "models/cls.onnx",
        "REC_DICT": "models/keys.txt", "OCR_LANG": "",
        "PROFILE_STAGES": "1", "LAYOUT_DISABLED": "1",
    })
    env.update(extra_env)

    log = open(f"/tmp/bench_{label}.log", "w")
    sv = subprocess.Popen([BIN], cwd=REPO, env=env, stdout=log, stderr=log)
    try:
        for _ in range(60):
            time.sleep(0.5)
            try:
                if http("/health/ready", timeout=3):
                    break
            except Exception:
                pass
        else:
            raise RuntimeError("server not ready; see " + log.name)

        # warmup
        http("/ocr/batch", {"images": b64})
        http("/ocr/batch", {"images": b64})
        # reset profile
        http("/profile")

        wall = 0.0
        last = None
        per_round = []
        for _ in range(ROUNDS):
            t0 = time.time()
            resp = http("/ocr/batch", {"images": b64})
            dt = time.time() - t0
            wall += dt
            per_round.append(len(b64) / dt)
            last = resp

        total_imgs = len(b64) * ROUNDS
        ips = total_imgs / wall

        prof = json.loads(http("/profile").decode())

        # accuracy signature from last response
        r = json.loads(last.decode())
        items = r.get("batch_results", r.get("results", r))
        boxes = 0
        texts = []
        if isinstance(items, list):
            for it in items:
                res = it.get("results", it) if isinstance(it, dict) else it
                if isinstance(res, list):
                    boxes += len(res)
                    for x in res:
                        texts.append(x.get("text", "") if isinstance(x, dict) else "")
        sig = hashlib.sha1("\x1f".join(texts).encode()).hexdigest()[:12]

        per_img = {k: round(v["ms"] / total_imgs, 3) for k, v in prof.items()}
        out = {
            "label": label, "img_per_s": round(ips, 2),
            "round_ips": [round(x, 1) for x in per_round],
            "n_images": len(b64), "rounds": ROUNDS,
            "boxes_total": boxes, "text_sig": sig,
            "stage_ms_per_img": per_img,
            "extra_env": extra_env,
        }
        print(json.dumps(out, indent=2))
    finally:
        sv.send_signal(signal.SIGTERM)
        try:
            sv.wait(timeout=15)
        except Exception:
            sv.kill()
        log.close()


if __name__ == "__main__":
    main()
