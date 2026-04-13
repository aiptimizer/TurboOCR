#!/usr/bin/env python3
"""Generate clean linear-axis benchmark plots + proper pandas tables."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ── Load results ──
with open("benchmark_results.json") as f:
    all_results = json.load(f)
all_results.sort(
    key=lambda x: np.mean([a["f1"] for a in x["accuracy"]]) if x["accuracy"] else 0,
    reverse=True,
)

# ── Model setup descriptions ──
MODEL_CONFIG = {
    "Qwen3-VL-2B": {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "engine": "vLLM serve",
        "invocation": "bf16 · enforce-eager · 8k ctx · OpenAI chat API · prompt: 'Extract all text from this image'",
    },
    "Turbo-OCR (C++/TRT)": {
        "model": "PP-OCRv5 mobile latin",
        "engine": "Custom C++/TensorRT server",
        "invocation": "FP16 · pool=5 · HTTP /ocr/raw · concurrent=8",
    },
    "PaddleOCR mobile latin (Python)": {
        "model": "PP-OCRv5_mobile_det + latin_PP-OCRv5_mobile_rec",
        "engine": "paddlepaddle-gpu 3.4 cu129",
        "invocation": "predict() Python API · device=gpu · same models as Turbo-OCR",
    },
    "EasyOCR (Python/GPU)": {
        "model": "CRAFT det + CRNN rec",
        "engine": "easyocr Python",
        "invocation": "langs=['en'] · GPU default config",
    },
    "GLM-OCR (0.9B)*": {
        "model": "zai-org/GLM-OCR",
        "engine": "vLLM serve",
        "invocation": "bf16 · max-num-batched=32768 · * repetition loop bug on SM120",
    },
    "PaddleOCR-VL (pipeline)": {
        "model": "PP-DocLayoutV3 + PaddleOCR-VL-1.5-0.9B",
        "engine": "paddleocr PaddleOCRVL (layout) + vLLM serve (VLM)",
        "invocation": "device=gpu · bf16 · max-num-seqs=64 · vl_rec_max_concurrency=16 · 2-stage pipeline",
    },
}

# ── Light modern theme ──
BG = "#ffffff"
TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"

COLORS = {
    "Qwen3-VL-2B": "#8b5cf6",
    "Turbo-OCR (C++/TRT)": "#06b6d4",
    "PaddleOCR mobile latin (Python)": "#f59e0b",
    "PaddleOCR-VL (pipeline)": "#10b981",
    "EasyOCR (Python/GPU)": "#ef4444",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "grid.color": GRID,
    "grid.alpha": 0.7,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

names = [r["name"] for r in all_results]
def _short(n):
    return (n.replace(" (Python/GPU)", "\n(Python)")
             .replace("PaddleOCR mobile latin (Python)", "PaddleOCR\nmobile latin")
             .replace(" (C++/TRT)", "\n(C++/TRT)")
             .replace(" (0.9B)*", "\n(0.9B)*")
             .replace(" (pipeline)", "\n(pipeline)"))
short_names = [_short(n) for n in names]
colors_ordered = [COLORS.get(n, "#6366f1") for n in names]


# ── Precompute metrics ──
rows = []
for r in all_results:
    f1 = np.mean([a["f1"] for a in r["accuracy"]]) * 100
    recall = np.mean([a["recall"] for a in r["accuracy"]]) * 100
    prec = np.mean([a["precision"] for a in r["accuracy"]]) * 100
    lat = np.array(r["latencies_ms"])
    rows.append({
        "Model / Engine": r["name"],
        "F1 (%)": f1,
        "Recall (%)": recall,
        "Precision (%)": prec,
        "p50 (ms)": float(np.percentile(lat, 50)),
        "p95 (ms)": float(np.percentile(lat, 95)),
        "Mean (ms)": float(np.mean(lat)),
        "Throughput (img/s)": r["throughput_img_per_sec"],
    })
df = pd.DataFrame(rows)

f1s = df["F1 (%)"].tolist()
recalls = df["Recall (%)"].tolist()
precisions = df["Precision (%)"].tolist()
p50s = df["p50 (ms)"].tolist()
p95s = df["p95 (ms)"].tolist()
throughputs = df["Throughput (img/s)"].tolist()
x = np.arange(len(all_results))


# ═══════════════════════════════════════════════════════════════════════
# PLOT 0: Hero — side-by-side F1 and throughput bars (for README top)
# ═══════════════════════════════════════════════════════════════════════
print("Generating hero plot...")
hero_order = sorted(zip(names, f1s, throughputs, colors_ordered),
                    key=lambda t: (-t[1], -t[2]))  # sort by F1 desc, then throughput
h_names = [_short(t[0]) for t in hero_order]
h_f1 = [t[1] for t in hero_order]
h_tp = [t[2] for t in hero_order]
h_col = [t[3] for t in hero_order]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"wspace": 0.35})
y = np.arange(len(h_names))

# Left: accuracy
barsL = axL.barh(y, h_f1, 0.62, color=h_col, edgecolor="white", linewidth=1.5)
for bar, v in zip(barsL, h_f1):
    axL.text(v + max(h_f1) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", ha="left", va="center", fontsize=12, fontweight="bold", color=TEXT)
axL.set_yticks(y)
axL.set_yticklabels(h_names, fontsize=11, fontweight="bold")
axL.set_xlabel("Word-level F1 (%)", fontsize=12, fontweight="bold")
axL.set_xlim(0, max(h_f1) * 1.15)
axL.set_title("Accuracy", fontsize=16, fontweight="bold", pad=12)
axL.invert_yaxis()
axL.grid(axis="x", alpha=0.7)
axL.grid(axis="y", alpha=0)

# Right: throughput
barsR = axR.barh(y, h_tp, 0.62, color=h_col, edgecolor="white", linewidth=1.5)
for bar, v in zip(barsR, h_tp):
    axR.text(v + max(h_tp) * 0.01, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f} img/s", ha="left", va="center", fontsize=12, fontweight="bold", color=TEXT)
axR.set_yticks(y)
axR.set_yticklabels([])
axR.set_xlabel("Throughput (images / second)", fontsize=12, fontweight="bold")
axR.set_xlim(0, max(h_tp) * 1.18)
axR.set_title("Throughput", fontsize=16, fontweight="bold", pad=12)
axR.invert_yaxis()
axR.grid(axis="x", alpha=0.7)
axR.grid(axis="y", alpha=0)

fig.suptitle("Turbo-OCR vs PaddleOCR · EasyOCR · VLMs  —  FUNSD (50 pages, RTX 5090)",
             fontsize=17, fontweight="bold", y=1.02)
plt.savefig("plot_hero.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved plot_hero.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Accuracy grouped bars
# ═══════════════════════════════════════════════════════════════════════
print("Generating accuracy plot...")
fig, ax = plt.subplots(figsize=(14, 7))
width = 0.26

bars1 = ax.bar(x - width, f1s, width, label="F1 Score", color="#6366f1", edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x, recalls, width, label="Recall", color="#06b6d4", edgecolor="white", linewidth=1.5)
bars3 = ax.bar(x + width, precisions, width, label="Precision", color="#f59e0b", edgecolor="white", linewidth=1.5)

for bars, vals in [(bars1, f1s), (bars2, recalls), (bars3, precisions)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11, fontweight="bold")
ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_title("OCR Accuracy — FUNSD · 50 images · ~174 words/img",
             fontsize=18, fontweight="bold", pad=18)
ax.text(0.5, 1.01, "RTX 5090 · CUDA 13.2 · Word-level F1 (alphanumeric tokens, case-insensitive)",
        ha="center", transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")
ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
plt.tight_layout()
plt.savefig("plot_accuracy.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved plot_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Latency bars (LINEAR)
# ═══════════════════════════════════════════════════════════════════════
print("Generating latency plot...")
fig, ax = plt.subplots(figsize=(14, 7))

bars_p50 = ax.bar(x - 0.2, p50s, 0.38, label="p50 (median)", color="#06b6d4", edgecolor="white", linewidth=1.5)
bars_p95 = ax.bar(x + 0.2, p95s, 0.38, label="p95", color="#ef4444", edgecolor="white", linewidth=1.5, alpha=0.9)

max_h = max(p95s)
for bar, v in zip(bars_p50, p50s):
    label = f"{v:.0f} ms"
    ax.text(bar.get_x() + bar.get_width() / 2, v + max_h * 0.01,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT)
for bar, v in zip(bars_p95, p95s):
    label = f"{v:.0f} ms"
    ax.text(bar.get_x() + bar.get_width() / 2, v + max_h * 0.01,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=11, fontweight="bold")
ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max_h * 1.13)
ax.set_title("OCR Latency — FUNSD Dataset · Lower is Better",
             fontsize=18, fontweight="bold", pad=18)
ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
plt.tight_layout()
plt.savefig("plot_latency.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved plot_latency.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Throughput bars (LINEAR)
# ═══════════════════════════════════════════════════════════════════════
print("Generating throughput plot...")
fig, ax = plt.subplots(figsize=(14, 7))

# Sort so fastest is first (horizontal)
sorted_pairs = sorted(zip(names, throughputs, colors_ordered), key=lambda t: t[1], reverse=True)
sn = [p[0] for p in sorted_pairs]
st = [p[1] for p in sorted_pairs]
sc = [p[2] for p in sorted_pairs]
short_sn = [_short(n) for n in sn]

bars = ax.barh(np.arange(len(sn)), st, 0.6, color=sc, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, st):
    ax.text(v + max(st) * 0.008, bar.get_y() + bar.get_height() / 2,
            f"{v:.1f} img/s", ha="left", va="center", fontsize=12, fontweight="bold", color=TEXT)

ax.set_yticks(np.arange(len(sn)))
ax.set_yticklabels(short_sn, fontsize=11, fontweight="bold")
ax.set_xlabel("Throughput (images / second)", fontsize=12, fontweight="bold")
ax.set_xlim(0, max(st) * 1.13)
ax.set_title("OCR Throughput — FUNSD Dataset · Higher is Better",
             fontsize=18, fontweight="bold", pad=18)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.7)
ax.grid(axis="y", alpha=0)
plt.tight_layout()
plt.savefig("plot_throughput.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved plot_throughput.png")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Accuracy vs Throughput scatter
# ═══════════════════════════════════════════════════════════════════════
print("Generating scatter plot...")
fig, ax = plt.subplots(figsize=(13, 8))

for i, r in enumerate(all_results):
    f1 = np.mean([a["f1"] for a in r["accuracy"]]) * 100
    tp = r["throughput_img_per_sec"]
    color = colors_ordered[i]
    ax.scatter(tp, f1, s=500, color=color, edgecolor="white", linewidth=2.5, zorder=3, alpha=0.9)
    ax.annotate(r["name"], xy=(tp, f1), xytext=(tp + 5, f1 + 1.5),
                fontsize=11, fontweight="bold", color=TEXT)

ax.set_xlabel("Throughput (images/sec)", fontsize=12, fontweight="bold")
ax.set_ylabel("F1 Score (%)", fontsize=12, fontweight="bold")
ax.set_title("Accuracy vs Throughput",
             fontsize=18, fontweight="bold", pad=18)
ax.text(0.5, 1.01, "Top-right = best · FUNSD · 50 imgs · RTX 5090",
        ha="center", transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")
ax.set_ylim(20, 95)
ax.set_xlim(-5, max(throughputs) * 1.15)
ax.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("plot_scatter.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved plot_scatter.png")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 1: Summary (pandas Styler → PNG via dataframe_image)
# ═══════════════════════════════════════════════════════════════════════
print("Generating summary table...")
summary_df = df.copy()
summary_df["F1 (%)"] = summary_df["F1 (%)"].map("{:.1f}".format)
summary_df["Recall (%)"] = summary_df["Recall (%)"].map("{:.1f}".format)
summary_df["Precision (%)"] = summary_df["Precision (%)"].map("{:.1f}".format)
summary_df["p50 (ms)"] = summary_df["p50 (ms)"].map("{:.0f}".format)
summary_df["p95 (ms)"] = summary_df["p95 (ms)"].map("{:.0f}".format)
summary_df["Mean (ms)"] = summary_df["Mean (ms)"].map("{:.0f}".format)
summary_df["Throughput (img/s)"] = summary_df["Throughput (img/s)"].map("{:.1f}".format)

styled = (
    summary_df.style
    .hide(axis="index")
    .set_caption("OCR Benchmark Summary — FUNSD · 50 images · RTX 5090 · CUDA 13.2")
    .set_table_styles([
        {"selector": "caption", "props": [
            ("font-size", "18px"), ("font-weight", "bold"),
            ("color", TEXT), ("padding", "14px"), ("caption-side", "top"),
        ]},
        {"selector": "th", "props": [
            ("background-color", "#6366f1"), ("color", "white"),
            ("font-weight", "bold"), ("font-size", "13px"),
            ("padding", "12px 16px"), ("text-align", "center"),
            ("border", "1px solid #4f46e5"),
        ]},
        {"selector": "td", "props": [
            ("padding", "10px 16px"), ("text-align", "center"),
            ("font-size", "13px"), ("border", "1px solid #e2e8f0"),
            ("color", TEXT),
        ]},
        {"selector": "td:nth-child(1)", "props": [
            ("text-align", "left"), ("font-weight", "bold"),
        ]},
        {"selector": "tr:nth-child(even) td", "props": [
            ("background-color", "#f8fafc"),
        ]},
        {"selector": "tr:nth-child(odd) td", "props": [
            ("background-color", "#ffffff"),
        ]},
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("font-family", "DejaVu Sans, sans-serif"),
            ("box-shadow", "0 4px 12px rgba(0,0,0,0.08)"),
            ("border-radius", "8px"),
            ("overflow", "hidden"),
        ]},
    ])
)
dfi.export(styled, "plot_summary.png", table_conversion="matplotlib", dpi=200)
print("  Saved plot_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# TABLE 2: Config table
# ═══════════════════════════════════════════════════════════════════════
print("Generating config table...")
config_df = pd.DataFrame([
    {
        "Model": MODEL_CONFIG.get(n, {}).get("model", n),
        "Engine": MODEL_CONFIG.get(n, {}).get("engine", "-"),
        "Invocation": MODEL_CONFIG.get(n, {}).get("invocation", "-"),
    }
    for n in names
])

styled_config = (
    config_df.style
    .hide(axis="index")
    .set_caption("How the Models Were Invoked")
    .set_table_styles([
        {"selector": "caption", "props": [
            ("font-size", "18px"), ("font-weight", "bold"),
            ("color", TEXT), ("padding", "14px"), ("caption-side", "top"),
        ]},
        {"selector": "th", "props": [
            ("background-color", "#6366f1"), ("color", "white"),
            ("font-weight", "bold"), ("font-size", "13px"),
            ("padding", "12px 16px"), ("text-align", "left"),
            ("border", "1px solid #4f46e5"),
        ]},
        {"selector": "td", "props": [
            ("padding", "12px 16px"), ("text-align", "left"),
            ("font-size", "12px"), ("border", "1px solid #e2e8f0"),
            ("color", TEXT),
        ]},
        {"selector": "td:nth-child(1)", "props": [
            ("font-weight", "bold"), ("background-color", "#f1f5f9"),
            ("width", "240px"), ("white-space", "nowrap"),
        ]},
        {"selector": "td:nth-child(2)", "props": [
            ("width", "280px"),
        ]},
        {"selector": "tr:nth-child(even) td:not(:nth-child(1))", "props": [
            ("background-color", "#f8fafc"),
        ]},
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("font-family", "DejaVu Sans, sans-serif"),
            ("box-shadow", "0 4px 12px rgba(0,0,0,0.08)"),
            ("border-radius", "8px"),
            ("overflow", "hidden"),
        ]},
    ])
)
dfi.export(styled_config, "plot_config.png", table_conversion="matplotlib", dpi=200)
print("  Saved plot_config.png")


print("\nAll plots saved!")
