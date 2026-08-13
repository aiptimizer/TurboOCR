#!/usr/bin/env python3
"""Export rec_tiny.onnx into a flat (graph.json + weights.bin) the MPSGraph .mm
consumer can rebuild without parsing protobuf, plus a fixed input + ORT golden
output for a bit-correctness check.

Python owns the ONNX heavy-lifting (topo order, weights, attributes); the .mm
just walks nodes and maps each op_type -> an MPSGraph op, wiring by tensor name.
"""
import json, struct, sys, os
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
FEEDS = None                                          # extra-input values, see below
for a in sys.argv[1:]:
    if a.startswith("--feeds="):
        FEEDS = json.load(open(a.split("=", 1)[1]))

SRC = argv[0] if len(argv) > 0 else "models/rec_tiny.onnx"
OUT = argv[1] if len(argv) > 1 else "rec_export"
Hh = int(argv[2]) if len(argv) > 2 else 48            # fixed input height (static)
W = int(argv[3]) if len(argv) > 3 else 320            # fixed input width (static)
os.makedirs(OUT, exist_ok=True)

m = onnx.load(SRC)
g = m.graph

# ---- weights blob ----------------------------------------------------------
blob = bytearray()
inits = []
init_names = set()
for init in g.initializer:
    arr = numpy_helper.to_array(init).astype(np.float32)
    off = len(blob)
    blob += arr.tobytes()
    inits.append({"name": init.name, "shape": list(arr.shape), "offset": off,
                  "nbytes": arr.nbytes})
    init_names.add(init.name)
with open(f"{OUT}/weights.bin", "wb") as f:
    f.write(blob)

def attrs_of(node):
    d = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT: d[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT: d[a.name] = a.f
        elif a.type == onnx.AttributeProto.INTS: d[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS: d[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.STRING: d[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.TENSOR:
            d[a.name] = numpy_helper.to_array(a.t).astype(np.int64).tolist()
    return d

nodes = []
for n in g.node:
    nodes.append({"op": n.op_type, "in": list(n.input), "out": list(n.output),
                  "attr": attrs_of(n)})

# ---- IO description --------------------------------------------------------
# THE PRIMARY INPUT IS THE RANK-4 ONE, NOT g.input[0]. Every det/rec/cls model
# has exactly one input so the two coincide, but PP-DocLayoutV3 declares
# (im_shape[N,2], image[N,3,800,800], scale_factor[N,2]) — image is input #1.
# Taking input[0] would put im_shape's name and a [1,3,H,W] shape into the
# legacy keys, i.e. a confidently wrong description of the wrong tensor.
def _dims(v):
    return v.type.tensor_type.shape.dim


def _is_static(d):
    return d.HasField("dim_value") and d.dim_value > 0


prim_idx = next((i for i, v in enumerate(g.input) if len(_dims(v)) == 4), 0)


def resolve_shape(v, primary):
    """Full static shape. Batch resolves to 1; the primary input's dynamic H/W
    take the CLI values; anything else still unknown is -1 (so a consumer can
    tell 'unknown' from 'one')."""
    out = []
    for i, d in enumerate(_dims(v)):
        if _is_static(d):
            out.append(d.dim_value)
        elif i == 0:
            out.append(1)
        elif primary and i == 2:
            out.append(Hh)
        elif primary and i == 3:
            out.append(W)
        else:
            out.append(-1)
    return out


in_specs = [{"name": v.name, "shape": resolve_shape(v, i == prim_idx)}
            for i, v in enumerate(g.input)]
prim_name = g.input[prim_idx].name
prim_shape = in_specs[prim_idx]["shape"]

graph = {
    # Legacy single-IO keys — every existing export and the current builder
    # read these; they describe the PRIMARY input and the FIRST output.
    "input": prim_name,
    "output": g.output[0].name,
    "input_shape": prim_shape,
    # Multi-IO description, graph order preserved.
    "inputs": in_specs,
    "outputs": [v.name for v in g.output],
    "initializers": inits,
    "init_names": sorted(init_names),
    "nodes": nodes,
}
with open(f"{OUT}/graph.json", "w") as f:
    json.dump(graph, f)

# ---- fixed input + ORT golden ---------------------------------------------
# A golden is only worth writing if every input is fed a MEANINGFUL value. The
# primary gets a fixed random tensor; any other input (im_shape, scale_factor)
# carries semantics this script cannot invent — feeding zeros would produce a
# golden that runs and is silently wrong, which is worse than no golden at all.
# Supply them with --feeds=<json> mapping name -> nested list.
rng = np.random.default_rng(1234)
x = rng.standard_normal(prim_shape).astype(np.float32)
x.tofile(f"{OUT}/input.bin")

feeds = {prim_name: x}
missing = []
for i, v in enumerate(g.input):
    if i == prim_idx:
        continue
    if FEEDS and v.name in FEEDS:
        want = onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type)
        arr = np.asarray(FEEDS[v.name], dtype=want)
        arr.tofile(f"{OUT}/input_{v.name}.bin")
        feeds[v.name] = arr
    else:
        missing.append(v.name)

if missing:
    print(f"  NO GOLDEN: inputs {missing} have no value. Pass "
          f"--feeds=<json> mapping each name to its value "
          f"(e.g. {{\"im_shape\": [[800,800]], \"scale_factor\": [[1.0,1.0]]}}). "
          f"graph.json + weights.bin were still written.")
    y = None
else:
    sess = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    y = outs[0].astype(np.float32)
    y.tofile(f"{OUT}/golden.bin")           # first output keeps the legacy name
    for name, o in zip(graph["outputs"], outs):
        np.asarray(o).astype(np.float32).tofile(f"{OUT}/golden_{name}.bin")

op_hist = {}
for n in nodes: op_hist[n["op"]] = op_hist.get(n["op"], 0) + 1
print(f"exported -> {OUT}")
print(f"  nodes={len(nodes)}  initializers={len(inits)}  weights={len(blob)/1e6:.1f} MB")
print(f"  inputs:  {[(s['name'], s['shape']) for s in in_specs]}  (primary: {prim_name})")
print(f"  outputs: {graph['outputs']}")
print(f"  op set: {op_hist}")
if y is not None:
    print(f"  input {x.shape} -> golden {y.shape} ({y.dtype})")
    print(f"  golden stats: mean={y.mean():.4f} std={y.std():.4f} min={y.min():.3f} max={y.max():.3f}")
