#!/usr/bin/env python3
"""Export PP-LCNet_x1_0_doc_ori (document image orientation, 0/90/180/270) to ONNX.

Mirrors export_pp_doclayoutv3.py. Produces models/doc_ori.onnx, consumed by the
DocOrientation classifier (src/models/classification/doc_orientation.*) that powers
/ocr/pdf?autorotate=1.

Preprocess (baked into the C++ side, NOT the ONNX): resize so short side = 256,
center-crop 224, BGR->RGB, /255, ImageNet normalize (mean .485/.456/.406,
std .229/.224/.225), CHW. Output: [N,4] logits over labels [0,90,180,270]
where the label is the page's clockwise rotation from upright.

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
    "paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar"
)
OUT = Path(__file__).resolve().parents[3] / "models" / "doc_ori.onnx"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "doc_ori.tar"
        print(f"downloading {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(td)
        model_dir = Path(td) / "PP-LCNet_x1_0_doc_ori_infer"
        # paddle2onnx 2.x console entry needs paddle importable; call its main().
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
