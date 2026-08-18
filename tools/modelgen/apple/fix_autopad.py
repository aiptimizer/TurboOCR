import onnx, math
from onnx import helper, shape_inference

# onnx2torch doesn't implement auto_pad=SAME_UPPER/SAME_LOWER. Freeze the input to a
# static [1,3,48,W] shape, run shape inference, and rewrite every Conv/MaxPool/AveragePool
# auto_pad into explicit pads computed from the (now known) input spatial dims.

def _spatial(vi_map, name):
    d = vi_map.get(name)
    if d is None: return None
    dims = [x.dim_value for x in d.type.tensor_type.shape.dim]
    if len(dims) != 4 or any(v <= 0 for v in dims[2:]): return None
    return dims[2], dims[3]  # H, W

def fix(onnx_path, W, H=48, B=1):
    m = onnx.load(onnx_path)
    g = m.graph
    # freeze input dims
    inp = g.input[0]
    dims = inp.type.tensor_type.shape.dim
    for d, v in zip(dims, [B, 3, H, W]):
        d.ClearField("dim_param"); d.dim_value = v
    m = shape_inference.infer_shapes(m)
    g = m.graph
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
    for n in g.node:
        if n.op_type not in ("Conv", "MaxPool", "AveragePool"): continue
        ap = None
        for a in n.attribute:
            if a.name == "auto_pad": ap = a.s.decode()
        if ap in (None, "NOTSET", ""): continue
        sp = _spatial(vi, n.input[0])
        if sp is None:
            raise RuntimeError(f"no static shape for {n.name} input {n.input[0]}")
        ih, iw = sp
        def geta(name, default):
            for a in n.attribute:
                if a.name == name: return list(a.ints)
            return default
        k = geta("kernel_shape", None)
        s = geta("strides", [1, 1])
        dil = geta("dilations", [1, 1])
        if k is None:  # Conv: kernel from weight
            w = next((i for i in g.initializer if i.name == n.input[1]), None)
            k = list(w.dims[2:])
        def pad_for(insz, ks, st, dl):
            out = math.ceil(insz / st)
            tot = max((out - 1) * st + ((ks - 1) * dl + 1) - insz, 0)
            return tot
        pt_h = pad_for(ih, k[0], s[0], dil[0]); pt_w = pad_for(iw, k[1], s[1], dil[1])
        if ap == "SAME_LOWER":
            bh, eh = pt_h - pt_h // 2, pt_h // 2; bw, ew = pt_w - pt_w // 2, pt_w // 2
        else:  # SAME_UPPER
            bh, eh = pt_h // 2, pt_h - pt_h // 2; bw, ew = pt_w // 2, pt_w - pt_w // 2
        # strip auto_pad, set explicit pads
        newattr = [a for a in n.attribute if a.name != "auto_pad"]
        del n.attribute[:]
        n.attribute.extend(newattr)
        n.attribute.append(helper.make_attribute("pads", [bh, bw, eh, ew]))
    return m
