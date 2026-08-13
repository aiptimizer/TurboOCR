#!/usr/bin/env python3
"""Run cpu_profile_bench.py across a list of configs and tabulate throughput,
key per-stage costs, and accuracy pass/fail against the baseline signature.

Configs are given as 'label KEY=VAL KEY=VAL ...' lines from a file or stdin.
The first config is treated as the reference for accuracy (its text_sig+boxes
become the gate). Any later config whose sig/boxes differ is flagged FAIL.
"""
import json, subprocess, sys, os

# repo root = up out of tools/bench/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = f"{REPO}/tools/bench/cpu_profile_bench.py"


def run_one(label, env_kvs):
    env = dict(os.environ)
    env["BENCH_N"] = env.get("BENCH_N", "30")
    env["BENCH_ROUNDS"] = env.get("BENCH_ROUNDS", "4")
    # PIPELINE_POOL_SIZE may be in env_kvs; bench reads it from process env too
    for kv in env_kvs:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
    cmd = ["python3", BENCH, label] + env_kvs
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=400)
    try:
        # bench prints a JSON object (indented); grab from first '{'
        out = p.stdout
        j = json.loads(out[out.index("{"):])
        return j
    except Exception:
        sys.stderr.write(f"[{label}] FAILED:\n{p.stdout[-500:]}\n{p.stderr[-500:]}\n")
        return None


def main():
    lines = [l.strip() for l in (open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin) if l.strip() and not l.startswith("#")]
    ref = None
    rows = []
    for line in lines:
        parts = line.split()
        label, kvs = parts[0], parts[1:]
        j = run_one(label, kvs)
        if not j:
            rows.append((label, None)); continue
        if ref is None:
            ref = (j["text_sig"], j["boxes_total"])
        rows.append((label, j))

    print(f"\n{'label':18s} {'img/s':>7s} {'Δ%':>6s} {'rec_inf':>8s} {'det':>7s} {'cls':>6s} {'acc':>5s}  flags")
    base_ips = None
    for label, j in rows:
        if not j:
            print(f"{label:18s} {'ERR':>7s}"); continue
        ips = j["img_per_s"]
        if base_ips is None:
            base_ips = ips
        d = (ips / base_ips - 1) * 100
        s = j["stage_ms_per_img"]
        acc = "OK" if (j["text_sig"], j["boxes_total"]) == ref else "FAIL"
        flags = " ".join(f"{k}={v}" for k, v in j["extra_env"].items())
        print(f"{label:18s} {ips:7.2f} {d:+6.1f} {s.get('rec_infer',0):8.0f} {s.get('det',0):7.0f} {s.get('cls',0):6.0f} {acc:>5s}  {flags}")
        if acc == "FAIL":
            print(f"    ^ sig {j['text_sig']}/{j['boxes_total']}boxes vs ref {ref[0]}/{ref[1]}")


if __name__ == "__main__":
    main()
