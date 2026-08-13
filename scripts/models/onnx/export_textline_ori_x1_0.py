#!/usr/bin/env python3
"""Export PP-LCNet_x1_0_textline_ori (text-line 0/180 flip classifier) to ONNX.

Mirrors export_doc_ori.py. Produces models/cls_x1_0.onnx — the full-width
variant of the shipped models/cls.onnx (x0_25), selectable via CLS_ONNX=x1_0 /
CLS_MODEL=x1_0. Same I/O contract as x0_25: input [N,3,80,160], output [N,2]
softmax over {0deg, 180deg}; the C++ PaddleCls/CpuPaddleCls consume it
unchanged.

Requires paddlepaddle + paddle2onnx 2.x in the environment (.venv-paddle).
"""
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
    "paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar"
)
OUT = Path(__file__).resolve().parents[3] / "models" / "cls_x1_0.onnx"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "textline_ori.tar"
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(td)
        model_dir = Path(td) / "PP-LCNet_x1_0_textline_ori_infer"
        cmd = [
            sys.executable, "-c",
            "import sys; sys.argv=['paddle2onnx','--model_dir',%r,"
            "'--model_filename','inference.json','--params_filename',"
            "'inference.pdiparams','--save_file',%r,'--opset_version','17'];"
            "from paddle2onnx.command import main; main()" % (str(model_dir), str(OUT)),
        ]
        subprocess.run(cmd, check=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
