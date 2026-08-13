"""Static INT8 quantization of the CPU OCR models for AVX-VNNI (i5-13600K).

Produces QOperator-format models (pre-baked QLinearConv / QLinearMatMul) so the
INT8 path is taken regardless of the C++ session's graph-optimization level --
unlike the existing models/det.int8.onnx, which is QDQ (Q/DQ around float Conv)
and only goes fast if the runtime fuses Q->Conv->DQ at extended optimization.

VNNI fast path = uint8 activations x int8 weights -> VPDPBUSD. So default is
activation=QUInt8, weight=QInt8, per-channel weights, reduce_range=False.

Usage:
  quantize_ocr.py rec [--out models/rec.int8.static.onnx] [--calib-pages 20]
                      [--format QOperator|QDQ] [--reduce-range]
                      [--act uint8|int8] [--calib minmax|percentile|entropy]
  quantize_ocr.py det [--out models/det.int8.static.onnx] ...
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_common as Q


class RecReader(CalibrationDataReader):
    def __init__(self, crops, input_name, fixed_width=None):
        # Histogram calibration (entropy/percentile) stacks activations, so all
        # samples must share a width -> fixed_width resizes every crop.
        self.data = []
        for c in crops:
            nchw, _ = Q.rec_preprocess(c, target_w=fixed_width)
            self.data.append({input_name: nchw})
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


class DetReader(CalibrationDataReader):
    def __init__(self, pages, input_name, fixed_size=None):
        self.data = []
        import cv2
        for p in pages:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if fixed_size:  # histogram calib needs uniform shape
                img = cv2.resize(img, (fixed_size, fixed_size))
            inp, _, _ = Q.det_preprocess(img)
            self.data.append({input_name: inp})
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


def prep_fp32(src, opset, workdir):
    """version-convert to a quantizable opset, then quant_pre_process."""
    m = onnx.load(src)
    cur = max((i.version for i in m.opset_import if (i.domain or "ai.onnx") == "ai.onnx"),
              default=0)
    upgraded = os.path.join(workdir, "upgraded.onnx")
    if cur < opset:
        m = version_converter.convert_version(m, opset)
        onnx.save(m, upgraded)
        src_for_pre = upgraded
    else:
        src_for_pre = src
    pre = os.path.join(workdir, "pre.onnx")
    quant_pre_process(src_for_pre, pre, skip_symbolic_shape=False, auto_merge=True)
    return pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["rec", "det"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib-pages", type=int, default=20)
    ap.add_argument("--max-crops", type=int, default=400)
    ap.add_argument("--format", choices=["QOperator", "QDQ"], default="QOperator")
    ap.add_argument("--reduce-range", action="store_true")
    ap.add_argument("--act", choices=["uint8", "int8"], default="uint8")
    ap.add_argument("--weight", choices=["uint8", "int8"], default="int8")
    ap.add_argument("--per-channel", dest="per_channel", action="store_true", default=True)
    ap.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    ap.add_argument("--calib", choices=["minmax", "percentile", "entropy"],
                    default="minmax")
    ap.add_argument("--op-types", default="Conv,MatMul",
                    help="comma list of op types to quantize; empty=all")
    ap.add_argument("--exclude", default="",
                    help="comma list of node names to keep in fp32")
    ap.add_argument("--exclude-last-matmul", action="store_true",
                    help="keep the final classifier MatMul in fp32")
    ap.add_argument("--exclude-depthwise", action="store_true",
                    help="keep depthwise convs in fp32 (cheap, quant-sensitive)")
    ap.add_argument("--exclude-first-conv", action="store_true",
                    help="keep the first conv (raw input) in fp32")
    ap.add_argument("--calib-width", type=int, default=640,
                    help="fixed width for histogram calibration (entropy/percentile)")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    src = f"{Q.REPO}/models/{args.target}.onnx"
    out = args.out or f"{Q.REPO}/models/{args.target}.int8.static.onnx"

    qtype = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}
    qformat = QuantFormat.QOperator if args.format == "QOperator" else QuantFormat.QDQ
    cmethod = {"minmax": CalibrationMethod.MinMax,
               "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}[args.calib]

    with tempfile.TemporaryDirectory() as workdir:
        print(f"[prep] {src} -> opset>={args.opset} + shape inference")
        pre = prep_fp32(src, args.opset, workdir)

        in_name = onnx.load(pre).graph.input[0].name
        print(f"[calib] input='{in_name}', building reader ...")
        if args.target == "rec":
            det = Q.make_cpu_session(f"{Q.REPO}/models/det.onnx")
            pages = Q.list_pages(args.calib_pages)
            crops = Q.extract_line_crops(det, pages, max_crops=args.max_crops)
            print(f"[calib] {len(crops)} real line crops from {len(pages)} pages")
            fw = args.calib_width if args.calib != "minmax" else None
            reader = RecReader(crops, in_name, fixed_width=fw)
        else:
            pages = Q.list_pages(args.calib_pages)
            print(f"[calib] {len(pages)} det pages")
            fs = 960 if args.calib != "minmax" else None
            reader = DetReader(pages, in_name, fixed_size=fs)

        extra = {"WeightSymmetric": True, "ActivationSymmetric": False}
        if args.calib == "percentile":
            extra["CalibPercentile"] = 99.99
        op_types = [t for t in args.op_types.split(",") if t] or None
        print(f"[quant] format={args.format} act={args.act} weight={args.weight} "
              f"per_channel={args.per_channel} reduce_range={args.reduce_range} "
              f"calib={args.calib} op_types={op_types}")
        kwargs = dict(
            calibration_data_reader=reader,
            quant_format=qformat,
            activation_type=qtype[args.act],
            weight_type=qtype[args.weight],
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            calibrate_method=cmethod,
            extra_options=extra,
        )
        if op_types is not None:
            kwargs["op_types_to_quantize"] = op_types

        exclude = [n for n in args.exclude.split(",") if n]
        g = onnx.load(pre).graph
        if args.exclude_last_matmul:
            mm = [n.name for n in g.node if n.op_type == "MatMul"]
            if mm:
                exclude.append(mm[-1])
                print(f"[quant] excluding final MatMul: {mm[-1]}")
        if args.exclude_depthwise or args.exclude_first_conv:
            inits = {i.name: i for i in g.initializer}
            convs = [n for n in g.node if n.op_type == "Conv"]
            dw = []
            for n in convs:
                w = inits.get(n.input[1])
                if w is None:
                    continue
                grp = next((a.i for a in n.attribute if a.name == "group"), 1)
                if grp > 1 and grp == w.dims[0]:
                    dw.append(n.name)
            if args.exclude_depthwise:
                exclude += dw
                print(f"[quant] excluding {len(dw)} depthwise convs")
            if args.exclude_first_conv and convs:
                exclude.append(convs[0].name)
                print(f"[quant] excluding first conv: {convs[0].name}")
        if exclude:
            kwargs["nodes_to_exclude"] = exclude
        quantize_static(pre, out, **kwargs)

    sz_src = os.path.getsize(src) / 1e6
    sz_out = os.path.getsize(out) / 1e6
    print(f"[done] {out}  ({sz_src:.2f} MB fp32 -> {sz_out:.2f} MB int8)")

    # report op composition
    from collections import Counter
    ops = Counter(n.op_type for n in onnx.load(out).graph.node)
    qops = {k: v for k, v in ops.items()
            if any(t in k for t in ["QLinear", "Integer", "Quant", "Dequant"])}
    print(f"[ops] quant ops: {qops}")
    print(f"[ops] remaining float Conv: {ops.get('Conv', 0)}  MatMul: {ops.get('MatMul', 0)}")


if __name__ == "__main__":
    main()
