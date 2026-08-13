#!/usr/bin/env python3
"""Rebuild the shipped fast-formula bundle (fused + FAST split) from a raw
paddle2onnx export of the PP-FormulaNet-S ARCHITECTURE family, by weight-
swapping the shipped split graphs. This reconstructs the previously
unversioned bundle-build recipe.

PROVENANCE FINDING (2026-08-05): the bundle shipped as
models/formula/ppformulanet_s is PP-FormulaNet_plus-S. Run against paddle's
official PP-FormulaNet_plus-S download, this script reproduces ALL FIVE shipped
files BYTE-IDENTICALLY (inference_trt.onnx, tokenizer.json, fast/{encoder,
prep,step_batched}.onnx). True PP-FormulaNet-S differs in 417/418 weights;
paddle's plus-S adds only 3 DEAD tensors (conv2d_80, linear_0 — declared,
never consumed) on top of an otherwise weight-identical program, so -S and
plus-S share one architecture: 2-layer 384-d MBart decoder with 3-token MTP,
PP-HGNetV2-B4 encoder, 50000 vocab, 1029 posemb, byte-identical
tokenizer.json.

Method: both models' fused exports use the SAME paddle parameter indices
(lm=linear_191, layer l = linear_{192+10l} in order
[sk,sv,sq,so,ck,cv,cq,co,f1,f2] with cross-k/v 196/197 & 206/207 in prep, LNs
layer_norm_{61+3l..}, final 67, embed LN create_parameter_14/15, memory
projector linear_212 — derived by exact value-matching the shipped split back
to its fused graph). So instead of re-deriving the torch fastdec export
(export_plusm_step.py style), copy the template split graphs and replace their
weight tensors 1:1 — the graph contract the C++ decode_chunk host loop binds
against (tokens[B,3] / pos[1] / kb,vb[2,B,16,1056,24] / ck,cv[2,B,16,144,24]
-> logits[B,3,50000]) is preserved by construction. The encoder is not
swapped: it is EXTRACTED from the fused graph at the same boundary tensor
(x -> p2o.pd_op.transpose.0.0, the [B,144,2048] pre-projection memory).

Usage:
  export_ppformulanet_plus_s_fast.py --fused-raw fused_raw.onnx \
      [--template models/formula/ppformulanet_s] \
      [--out models/formula/ppformulanet_plus_s]

fused_raw.onnx comes from (paddle2onnx 2.1.0):
  paddle2onnx --model_dir ~/.paddlex/official_models/PP-FormulaNet_plus-S \
      --model_filename inference.json --params_filename inference.pdiparams \
      --save_file fused_raw.onnx --opset_version 17

Outputs into --out:
  inference_trt.onnx   patched fused graph (CPU OrtFormulaRecognizer path)
  tokenizer.json       copied from the template (byte-identical family-wide)
  fast/encoder.onnx    extracted encoder
  fast/prep.onnx       template prep graph, cross-KV weights swapped
  fast/step_batched.onnx  template step graph, all 52 initializers swapped

For any FUTURE weight drop of this architecture: run this, then gate with a
fused-vs-fast greedy bit-exactness check before shipping (for the current
weights the byte-identity to the shipped bundle IS the gate).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_ppformulanet_s_onnx import patch_bitwise, patch_loop_carries  # noqa: E402
import onnx_graphsurgeon as gs  # noqa: E402

# fast/step_batched.onnx initializer name -> paddle parameter name.
# Derived by exact value-matching the shipped -S split against its fused graph;
# layer l linears are 192+10*l + offset [sk,sv,sq,so,(ck,cv in prep),cq,co,f1,f2].
STEP_MAP = {"emb": "embedding_3.w_0",
            "posemb": "m_bart_learned_positional_embedding_3.w_0",
            "lm": "linear_191.w_0",
            "lmb": None,  # lm has no bias; the template carries an all-zero vector
            "emb_s": "create_parameter_14.w_0", "emb_b": "create_parameter_15.w_0",
            "dec_final_s": "layer_norm_67.w_0", "dec_final_b": "layer_norm_67.b_0"}
for _l in range(2):
    _b = 192 + 10 * _l
    for _off, _nm in ((0, "sk"), (1, "sv"), (2, "sq"), (3, "so"),
                      (6, "cq"), (7, "co"), (8, "f1"), (9, "f2")):
        STEP_MAP[f"{_nm}{_l}_w"] = f"linear_{_b + _off}.w_0"
        STEP_MAP[f"{_nm}{_l}_b"] = f"linear_{_b + _off}.b_0"
    for _ln, _nm in ((61 + 3 * _l, "self"), (62 + 3 * _l, "cross"), (63 + 3 * _l, "final")):
        STEP_MAP[f"{_nm}{_l}_s"] = f"layer_norm_{_ln}.w_0"
        STEP_MAP[f"{_nm}{_l}_b"] = f"layer_norm_{_ln}.b_0"

# fast/prep.onnx Constant OUTPUT name -> paddle parameter name (same derivation).
PREP_MAP = {"/Constant_output_0": "linear_212.w_0", "/Constant_1_output_0": "linear_212.b_0",
            "/Constant_2_output_0": "linear_196.w_0", "/Constant_3_output_0": "linear_196.b_0",
            "/Constant_8_output_0": "linear_206.w_0", "/Constant_9_output_0": "linear_206.b_0",
            "/Constant_16_output_0": "linear_197.w_0", "/Constant_17_output_0": "linear_197.b_0",
            "/Constant_22_output_0": "linear_207.w_0", "/Constant_23_output_0": "linear_207.b_0"}

ENC_IN, ENC_OUT = "x", "p2o.pd_op.transpose.0.0"


def collect_weights(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """All named tensors: initializers + Constant node outputs, recursively."""
    w: dict[str, np.ndarray] = {}
    for t in graph.initializer:
        w[t.name] = nh.to_array(t)
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    w[n.output[0]] = nh.to_array(a.t)
        for a in n.attribute:
            if a.g.node:
                w.update(collect_weights(a.g))
            for sg in a.graphs:
                w.update(collect_weights(sg))
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--fused-raw", required=True)
    ap.add_argument("--template", default=str(repo / "models/formula/ppformulanet_s"))
    ap.add_argument("--out", default=str(repo / "models/formula/ppformulanet_plus_s"))
    args = ap.parse_args()
    tpl, out = Path(args.template), Path(args.out)
    (out / "fast").mkdir(parents=True, exist_ok=True)

    # 1) Patch the fused export (BitwiseAnd/Not + Loop scalar-carry rank bugs —
    #    plus-S has the same paddle2onnx defects as -S) -> the CPU-path model.
    print(f"[plus-s] loading {args.fused_raw}")
    fused = onnx.load(args.fused_raw)
    fg = gs.import_onnx(fused)
    n_bw = patch_bitwise(fg)
    # Same export defects as -S, same node names (If.3/Identity.668, the MTP
    # arange carry): reuse the -S Loop-carry patch verbatim.
    n_lc = patch_loop_carries(fg)
    print(f"[plus-s] patch: bitwise rewritten {n_bw}, loop carries patched {n_lc}")
    fg.cleanup().toposort()
    fused_patched = gs.export_onnx(fg)
    onnx.save(fused_patched, str(out / "inference_trt.onnx"))
    print(f"[plus-s] wrote {out / 'inference_trt.onnx'}")

    # 2) plus-S weights (the decoder params are top-level Constants in the raw
    #    export; collect from the PATCHED graph so names survived the rewrite).
    w = collect_weights(fused_patched.graph)

    # 3) step_batched: swap every initializer of the -S template by name.
    step = onnx.load(str(tpl / "fast/step_batched.onnx"))
    for t in step.graph.initializer:
        src = STEP_MAP[t.name]  # KeyError => template drifted; fail loudly
        if src is None:
            arr = nh.to_array(t)
            if arr.any():
                raise SystemExit(f"[plus-s] template {t.name} expected all-zero")
            continue
        if src not in w:
            raise SystemExit(f"[plus-s] fused graph is missing {src}")
        if tuple(w[src].shape) != tuple(t.dims):
            raise SystemExit(f"[plus-s] shape mismatch {t.name}: template {tuple(t.dims)} "
                             f"vs plus-S {w[src].shape}")
        t.CopyFrom(nh.from_array(w[src].astype(np.float32), name=t.name))
    onnx.save(step, str(out / "fast/step_batched.onnx"))
    print(f"[plus-s] wrote fast/step_batched.onnx ({len(step.graph.initializer)} weights swapped)")

    # 4) prep: swap the 10 weight-carrying Constant nodes by output name.
    prep = onnx.load(str(tpl / "fast/prep.onnx"))
    swapped = 0
    for n in prep.graph.node:
        if n.op_type == "Constant" and n.output[0] in PREP_MAP:
            src = PREP_MAP[n.output[0]]
            if src not in w:
                raise SystemExit(f"[plus-s] fused graph is missing {src}")
            old = nh.to_array(n.attribute[0].t)
            if old.shape != w[src].shape:
                raise SystemExit(f"[plus-s] prep shape mismatch {n.output[0]}: "
                                 f"{old.shape} vs {w[src].shape}")
            n.attribute[0].t.CopyFrom(nh.from_array(w[src].astype(np.float32)))
            swapped += 1
    if swapped != len(PREP_MAP):
        raise SystemExit(f"[plus-s] prep swapped {swapped}/{len(PREP_MAP)} — template drifted")
    onnx.save(prep, str(out / "fast/prep.onnx"))
    print(f"[plus-s] wrote fast/prep.onnx ({swapped} weights swapped)")

    # 5) encoder: extract the plus-S encoder subgraph at the -S boundary.
    onnx.utils.extract_model(str(out / "inference_trt.onnx"),
                             str(out / "fast/encoder.onnx"), [ENC_IN], [ENC_OUT])
    print("[plus-s] wrote fast/encoder.onnx (extracted)")

    # 6) tokenizer: byte-identical across the PP-FormulaNet family (verified
    #    against the paddle configs of -S, plus-S and plus-M).
    shutil.copyfile(tpl / "tokenizer.json", out / "tokenizer.json")
    print("[plus-s] wrote tokenizer.json (family-shared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
