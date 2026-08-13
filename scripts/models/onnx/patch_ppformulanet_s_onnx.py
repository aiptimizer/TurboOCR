#!/usr/bin/env python3
"""Patch the PaddleX-exported PP-FormulaNet-S inference.onnx so TRT 10.x can
build it.

Two upstream-export bugs block native TRT loading:

  1. INT64 `BitwiseAnd`/`BitwiseNot` are used to AND two boolean-semantics
     masks stored as INT64; TRT's parser rejects BitwiseAnd on those
     tensors. Replace with `And` on BOOL, casting in/out of INT64.

  2. Two Loop-carry pairs have a runtime-shape mismatch that TRT's
     IRecurrenceLayer rejects:
       - carry "arange.0.0": initialised as Range(0,3,1) → shape [3] but
         the body emits `slice(arange, [-1:])` → shape [1].
       - carry "full.17.0":  initialised as scalar shape [] but the body
         emits `if.3.0` → shape [1].
     These carries are NEVER consumed inside the body apart from their own
     re-emission, so we can re-shape the body output to match the carry
     shape without changing semantics — for arange we `Expand` the [1]
     scalar back to [3]; for full.17 we `Squeeze` the [1] back to [].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import numpy as np


def patch_bitwise(graph: gs.Graph) -> int:
    """Replace BitwiseAnd / BitwiseNot inside a graph with And / Not, casting
    integer operands to BOOL on the way in and back to INT64 on the way out.
    Returns the number of nodes rewritten across this graph and its
    sub-graphs (recursive into Loop / If body sub-graphs).
    """
    rewrites = 0
    for node in list(graph.nodes):
        # Recurse into sub-graphs first.
        for attr_name, attr_val in list(node.attrs.items()):
            if isinstance(attr_val, gs.Graph):
                rewrites += patch_bitwise(attr_val)
            elif isinstance(attr_val, list):
                for v in attr_val:
                    if isinstance(v, gs.Graph):
                        rewrites += patch_bitwise(v)

        if node.op == "BitwiseNot":
            (x,) = node.inputs
            (y,) = node.outputs
            cast_in = gs.Variable(node.name + "/cast_in", dtype=np.bool_)
            not_out = gs.Variable(node.name + "/not_out", dtype=np.bool_)
            graph.nodes.append(gs.Node(
                op="Cast", name=node.name + "/cast_to_bool",
                attrs={"to": int(onnx.TensorProto.BOOL)},
                inputs=[x], outputs=[cast_in]))
            graph.nodes.append(gs.Node(
                op="Not", name=node.name + "/not",
                inputs=[cast_in], outputs=[not_out]))
            graph.nodes.append(gs.Node(
                op="Cast", name=node.name + "/cast_to_int64",
                attrs={"to": int(onnx.TensorProto.INT64)},
                inputs=[not_out], outputs=[y]))
            node.inputs.clear()
            node.outputs.clear()
            graph.nodes.remove(node)
            rewrites += 1
        elif node.op == "BitwiseAnd":
            (a, b) = node.inputs
            (y,) = node.outputs
            ca = gs.Variable(node.name + "/cast_a", dtype=np.bool_)
            cb = gs.Variable(node.name + "/cast_b", dtype=np.bool_)
            and_out = gs.Variable(node.name + "/and_out", dtype=np.bool_)
            graph.nodes.append(gs.Node(
                op="Cast", name=node.name + "/cast_a",
                attrs={"to": int(onnx.TensorProto.BOOL)},
                inputs=[a], outputs=[ca]))
            graph.nodes.append(gs.Node(
                op="Cast", name=node.name + "/cast_b",
                attrs={"to": int(onnx.TensorProto.BOOL)},
                inputs=[b], outputs=[cb]))
            graph.nodes.append(gs.Node(
                op="And", name=node.name + "/and",
                inputs=[ca, cb], outputs=[and_out]))
            graph.nodes.append(gs.Node(
                op="Cast", name=node.name + "/cast_to_int64",
                attrs={"to": int(onnx.TensorProto.INT64)},
                inputs=[and_out], outputs=[y]))
            node.inputs.clear()
            node.outputs.clear()
            graph.nodes.remove(node)
            rewrites += 1
    return rewrites


def patch_if3_branch_shape(body: gs.Graph) -> int:
    """The If.3 op inside the Loop body has a then-branch whose Mul/Add
    produces a [1]-shape tensor, but the if-output is declared scalar
    [] (the body merges with the else-branch which is genuinely scalar).
    Insert a Squeeze inside the then-branch so both branches yield
    matching shapes — ORT crashes mid-run without this even though TRT
    rejects the graph at parse time.
    """
    if3 = next((n for n in body.nodes if n.name == "If.3"), None)
    if if3 is None:
        return 0
    then_g: gs.Graph = if3.attrs["then_branch"]
    # The then_branch's only output is `p2o.sub_block.pd_op.scale.16.0` —
    # produced by Identity Identity.668 from Add.121. We squeeze on the
    # Add.121 source so the Identity passes a true scalar through.
    add121 = next((n for n in then_g.nodes if n.name == "Add.120"), None)
    ident668 = next((n for n in then_g.nodes if n.name == "Identity.668"), None)
    if add121 is None or ident668 is None:
        return 0
    add_out = add121.outputs[0]  # Add.121
    axes = gs.Constant(name="If.3/_then_sq_axes",
                       values=np.array([0], dtype=np.int64))
    squeezed = gs.Variable(name="Add.121_squeezed", dtype=add_out.dtype, shape=[])
    sq = gs.Node(op="Squeeze", name="If.3/_then_squeeze",
                 inputs=[add_out, axes], outputs=[squeezed])
    then_g.nodes.append(sq)
    # Rewire Identity.668 to consume squeezed scalar.
    ident668.inputs = [squeezed]
    then_g.cleanup().toposort()
    return 1


def patch_loop_carries(graph: gs.Graph) -> int:
    """Fix the two Paddle-export Loop-carry shape mismatches at the top of
    the main graph (PP-FormulaNet-S has exactly one Loop). Returns the
    number of carries patched (expect 2)."""
    loops = [n for n in graph.nodes if n.op == "Loop"]
    if not loops:
        return 0
    patched = 0
    for loop in loops:
        body: gs.Graph = loop.attrs["body"]
        patched += patch_if3_branch_shape(body)

        # Map body-output index -> new replacement variable.
        # body.outputs[0] is the loop-continue cond. body.outputs[k] for k>=1
        # is the k-th carry; it must match body.inputs[k+1].
        for k in range(1, len(body.outputs)):
            in_idx = k + 1
            if in_idx >= len(body.inputs):
                break
            bi = body.inputs[in_idx]
            bo = body.outputs[k]

            # Carry "arange.0.0": initial [3] vs body output [1] (slice last)
            # The Paddle exporter dropped the K-element semantics: the
            # body computes `slice(arange,[-1]) * 1 + 1` (shape [1]) and
            # tries to feed that as the next carry. The intended next
            # carry is `arange + K` element-wise (shape [3]), so the next
            # iter's slice([-1]) gives the right last-position value AND
            # the carry shape stays [3]. We re-derive that here: ignore
            # the original [1] body output entirely and compute
            # `Add(arange, ConstantOfShape([3], K=3))` as the new carry.
            if bi.name == "p2o.pd_op.arange.0.0":
                # The body input (= current iter's carry value) is bi itself.
                k_const = gs.Constant(
                    name=loop.name + "/_arange_k",
                    values=np.array([3, 3, 3], dtype=np.int64))
                new_out = gs.Variable(
                    name=bo.name + "_addk", dtype=bi.dtype, shape=[3])
                add = gs.Node(
                    op="Add", name=loop.name + "/_arange_addk",
                    inputs=[bi, k_const], outputs=[new_out])
                body.nodes.append(add)
                body.outputs[k] = new_out
                patched += 1

            # Carry "full.17.0": initial [] (scalar) vs body output [1]
            # Already fixed inside If.3's then-branch via patch_if3_branch_shape.

        body.cleanup().toposort()
    return patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default="models/formula/ppformulanet_s/inference.onnx")
    ap.add_argument("--out", dest="dst",
                    default="models/formula/ppformulanet_s/inference_trt.onnx")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"[patch] up-to-date: {dst}")
        return 0

    print(f"[patch] loading {src}")
    model = onnx.load(str(src), load_external_data=False)
    graph = gs.import_onnx(model)
    n_bw = patch_bitwise(graph)
    print(f"[patch] rewrote {n_bw} BitwiseAnd/BitwiseNot nodes")
    n_lc = patch_loop_carries(graph)
    print(f"[patch] patched {n_lc} Loop carry shape mismatches")
    graph.cleanup().toposort()
    out = gs.export_onnx(graph)
    onnx.save(out, str(dst))
    print(f"[patch] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
