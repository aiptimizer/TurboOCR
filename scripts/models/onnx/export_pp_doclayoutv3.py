#!/usr/bin/env python3
"""Download PP-DocLayoutV3 Paddle weights and export an ONNX file.

Requires: paddlepaddle, paddlex, paddle2onnx>=2.0, onnxsim, onnx.
Writes the final ONNX to models/layout/layout.onnx. The simplified graph
keeps a symbolic batch dimension so turbo-ocr can build a dynamic-batch
TensorRT engine (profile 1..8) at first-start.

Usage:
    python scripts/models/onnx/export_pp_doclayoutv3.py
    # or
    python scripts/models/onnx/export_pp_doclayoutv3.py --out models/layout/layout.onnx
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "models" / "layout" / "layout.onnx"


def fetch_paddle_weights() -> Path:
    """Use paddlex to download the official PP-DocLayoutV3 inference files."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddlex import create_model
    except ImportError as e:
        sys.exit(
            f"paddlex is required to download PP-DocLayoutV3 weights: {e}\n"
            f"Install with: pip install 'paddlepaddle>=3.4' 'paddlex>=3.4'"
        )

    # Triggers the download if not already cached
    create_model("PP-DocLayoutV3")

    cache = Path.home() / ".paddlex" / "official_models" / "PP-DocLayoutV3"
    if not cache.exists():
        sys.exit(f"paddlex download did not produce {cache}")
    needed = {"inference.json", "inference.pdiparams"}
    have = {p.name for p in cache.iterdir()}
    missing = needed - have
    if missing:
        sys.exit(f"paddlex cache is missing files: {sorted(missing)}")
    return cache


def run_paddle2onnx(paddle_dir: Path, onnx_path: Path) -> None:
    try:
        import paddle2onnx  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"paddle2onnx is required: {e}\n"
            f"Install with: pip install 'paddle2onnx>=2.0'"
        )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "paddle2onnx",
        "--model_dir", str(paddle_dir),
        "--model_filename", "inference.json",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(onnx_path),
        "--opset_version", "17",
        "--enable_onnx_checker", "True",
    ]
    # paddle2onnx is importable as a module in v2.x; the CLI is also shipped
    # as `paddle2onnx` but `python -m paddle2onnx` may not exist in older
    # releases. Fall back to the console script if the module form fails.
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        fallback = shutil.which("paddle2onnx")
        if fallback is None:
            sys.stderr.write(result.stderr)
            sys.exit("paddle2onnx failed and no console script fallback found")
        cmd[0:3] = [fallback]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            sys.exit("paddle2onnx failed")
    print(result.stdout)


def simplify_preserving_batch(onnx_path: Path) -> None:
    """Run onnxsim WITHOUT overwriting input shapes — keeps batch dim symbolic."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as e:
        sys.exit(f"onnx / onnxsim required: {e}")

    model = onnx.load(str(onnx_path))
    before = len(model.graph.node)
    simplified, ok = simplify(model)
    if not ok:
        print("onnxsim: simplification reported failure, keeping raw export")
        return
    after = len(simplified.graph.node)
    onnx.save(simplified, str(onnx_path))
    print(f"onnxsim: {before} -> {after} nodes, saved {onnx_path}")

    # Verify the batch dim stayed symbolic so TRT can build dynamic profiles
    for inp in simplified.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if not dims:
            continue
        first = dims[0]
        if first.dim_value != 0:
            sys.exit(
                f"Simplified ONNX has a fixed batch dim on {inp.name} "
                f"({first.dim_value}); TRT dynamic-batch profiles will fail. "
                f"Re-export without onnxsim's overwrite_input_shapes."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="output ONNX path (default: models/layout/layout.onnx)")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        print(f"{out} already exists. Delete it first if you want to re-export.")
        return 0

    with tempfile.TemporaryDirectory(prefix="layout_export_") as tmp:
        print("Step 1: download PP-DocLayoutV3 Paddle weights via paddlex")
        paddle_dir = fetch_paddle_weights()

        print("Step 2: paddle2onnx export")
        staged = Path(tmp) / "layout.onnx"
        run_paddle2onnx(paddle_dir, staged)

        print("Step 3: onnxsim (preserve symbolic batch)")
        simplify_preserving_batch(staged)

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, out)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"Wrote {out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
