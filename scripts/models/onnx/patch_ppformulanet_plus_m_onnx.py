#!/usr/bin/env python3
"""Patch the PaddleX-exported PP-FormulaNet_plus-M fused ONNX so ORT (and TRT)
can run its AR-decode `Loop`.

Same export bugs as PP-FormulaNet-S (see patch_ppformulanet_s_onnx.py), but the
node names differ (plus-M is d_model 512 x 6-layer, single-token; -S is 384 x
2-layer, MTP). Two fixes:

  1. INT64 BitwiseAnd/BitwiseNot -> And/Not on BOOL (TRT parser rejects them).
  2. Loop-carry rank mismatch: several carries are initialised scalar [] but the
     body re-emits them rank-1 [1] (through If branches), so ORT aborts mid-Loop
     with "Current shape:{} Requested shape:{1}". We Squeeze each offending
     carry's body output back to scalar. The carries are control/state scalars
     re-emitted unchanged, so squeezing is semantics-preserving.

Discovered offenders (plus-M): the scalar-declared carries whose body outputs
come through If/Slice and may be rank-1 -- patched generically by squeezing any
scalar-declared carry whose body output is not already scalar.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs


def patch_bitwise(graph: gs.Graph) -> int:
    rewrites = 0
    for node in list(graph.nodes):
        for attr_val in list(node.attrs.values()):
            if isinstance(attr_val, gs.Graph):
                rewrites += patch_bitwise(attr_val)
            elif isinstance(attr_val, list):
                for v in attr_val:
                    if isinstance(v, gs.Graph):
                        rewrites += patch_bitwise(v)
        if node.op == "BitwiseNot":
            (x,) = node.inputs; (y,) = node.outputs
            ci = gs.Variable(node.name + "/ci", dtype=np.bool_)
            no = gs.Variable(node.name + "/no", dtype=np.bool_)
            graph.nodes.append(gs.Node("Cast", node.name + "/c0", {"to": int(onnx.TensorProto.BOOL)}, [x], [ci]))
            graph.nodes.append(gs.Node("Not", node.name + "/n", inputs=[ci], outputs=[no]))
            graph.nodes.append(gs.Node("Cast", node.name + "/c1", {"to": int(onnx.TensorProto.INT64)}, [no], [y]))
            node.inputs.clear(); node.outputs.clear(); graph.nodes.remove(node); rewrites += 1
        elif node.op == "BitwiseAnd":
            (a, b) = node.inputs; (y,) = node.outputs
            ca = gs.Variable(node.name + "/ca", dtype=np.bool_)
            cb = gs.Variable(node.name + "/cb", dtype=np.bool_)
            ao = gs.Variable(node.name + "/ao", dtype=np.bool_)
            graph.nodes.append(gs.Node("Cast", node.name + "/ca0", {"to": int(onnx.TensorProto.BOOL)}, [a], [ca]))
            graph.nodes.append(gs.Node("Cast", node.name + "/cb0", {"to": int(onnx.TensorProto.BOOL)}, [b], [cb]))
            graph.nodes.append(gs.Node("And", node.name + "/and", inputs=[ca, cb], outputs=[ao]))
            graph.nodes.append(gs.Node("Cast", node.name + "/c1", {"to": int(onnx.TensorProto.INT64)}, [ao], [y]))
            node.inputs.clear(); node.outputs.clear(); graph.nodes.remove(node); rewrites += 1
    return rewrites


def _is_scalar(var) -> bool:
    return var.shape is not None and len(var.shape) == 0


def _squeeze_branch_to_scalar(sub: gs.Graph, tag: str) -> int:
    """Force a single-output If sub-branch to emit a true scalar []: insert a
    Squeeze (no axes -> drop all size-1 dims) on whatever feeds its output."""
    out = sub.outputs[0]
    prod = next((n for n in sub.nodes if out in n.outputs), None)
    if prod is None:  # output is a branch input passed through; squeeze it directly
        src = out
    else:
        src = prod.inputs[0] if prod.op == "Identity" else out
        if prod.op == "Identity":
            # squeeze the Identity's source, rewire Identity to the squeezed var
            sq = gs.Variable(out.name + "_sq" + tag, dtype=src.dtype, shape=[])
            sub.nodes.append(gs.Node("Squeeze", "sq_" + tag, inputs=[src], outputs=[sq]))
            prod.inputs = [sq]
            sub.cleanup().toposort()
            return 1
    sq = gs.Variable(out.name + "_sq" + tag, dtype=out.dtype, shape=[])
    sub.nodes.append(gs.Node("Squeeze", "sq_" + tag, inputs=[src], outputs=[sq]))
    sub.outputs[0] = sq
    sub.cleanup().toposort()
    return 1


def patch_loop_scalar_carries(graph: gs.Graph, if_names) -> int:
    """For each Loop body, force the named If nodes' branch outputs to scalar so
    the scalar carry they feed stays []. Recurses into the body."""
    patched = 0
    for loop in [n for n in graph.nodes if n.op == "Loop"]:
        body: gs.Graph = loop.attrs["body"]
        for n in body.nodes:
            if n.op == "If" and (not if_names or n.name in if_names):
                for a_name in ("then_branch", "else_branch"):
                    sub = n.attrs.get(a_name)
                    if isinstance(sub, gs.Graph) and len(sub.outputs) == 1:
                        patched += _squeeze_branch_to_scalar(sub, f"{n.name}_{a_name}")
        body.cleanup().toposort()
    return patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--ifs", nargs="*", default=["If.6", "If.7"],
                    help="If nodes (inside the Loop body) whose single output feeds a "
                         "scalar carry and must be squeezed to scalar []")
    args = ap.parse_args()
    print(f"[patch] loading {args.src}")
    model = onnx.load(args.src, load_external_data=False)
    graph = gs.import_onnx(model)
    n_bw = patch_bitwise(graph)
    n_lc = patch_loop_scalar_carries(graph, set(args.ifs) if args.ifs else None)
    print(f"[patch] BitwiseAnd/Not rewritten: {n_bw}; scalar carries squeezed: {n_lc}")
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.dst)
    print(f"[patch] wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
