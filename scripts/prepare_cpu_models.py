#!/usr/bin/env python3
"""
Prepare ONNX models for CPU inference.

The original ONNX models from PaddleOCR work directly with ONNX Runtime.
This script is optional — it simplifies the models for slightly faster
inference by folding constant expressions and removing unused nodes.

Usage:
    python scripts/prepare_cpu_models.py [--models-dir models]
"""

import argparse
import os
import sys


def simplify_model(input_path, output_path, input_shapes):
    """Simplify ONNX model with onnxsim."""
    import onnx
    from onnxsim import simplify

    model = onnx.load(input_path)
    model_sim, check = simplify(
        model, overwrite_input_shapes=input_shapes, dynamic_input_shape=True
    )
    if not check:
        print(f"  WARNING: Simplification check failed for {input_path}")
        print(f"  Using original model instead")
        return False

    onnx.save(model_sim, output_path)
    orig_size = os.path.getsize(input_path) / 1024
    new_size = os.path.getsize(output_path) / 1024
    print(f"  {orig_size:.0f}KB -> {new_size:.0f}KB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare ONNX models for CPU inference")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing ONNX models (default: models)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir

    models = {
        "det.onnx": {"x": [1, 3, 960, 960]},
        "rec.onnx": {"x": [1, 3, 48, 320]},
        "cls.onnx": {"x": [1, 3, 80, 160]},
    }

    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("onnx and onnxsim are required for model simplification.")
        print("Install with: pip install onnx onnxsim")
        print("\nNote: The original .onnx models work fine without simplification.")
        print("Simplification only provides a marginal speed improvement.")
        sys.exit(1)

    for model_name, input_shapes in models.items():
        input_path = os.path.join(models_dir, model_name)
        if not os.path.exists(input_path):
            print(f"Skipping {model_name} (not found)")
            continue

        output_name = model_name.replace(".onnx", "_opt.onnx")
        output_path = os.path.join(models_dir, output_name)
        print(f"Simplifying {model_name} -> {output_name}...")
        simplify_model(input_path, output_path, input_shapes)

    print("\nDone! Use the *_opt.onnx models for slightly faster CPU inference.")
    print("Or use the original .onnx models directly — they work fine with ONNX Runtime.")


if __name__ == "__main__":
    main()
