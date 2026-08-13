#!/usr/bin/env python3
"""Convert ONNX models to TensorRT engines with dynamic shapes and FP16/INT8.

Supports per-bucket engine building for rec models via --width-buckets flag.
Instead of one engine with dynamic width [48-4000], builds separate engines
each optimized for a narrow width range. TRT generates better kernels when
the optimization range is narrow.

INT8 support:
  1. Pre-quantized: --int8 with a Q/DQ ONNX model (from nvidia-modelopt).
  2. Native calibration: --int8 --calibration-data <dir> with the ORIGINAL FP32
     ONNX model. TRT runs calibration images through the network to determine
     per-tensor activation ranges, then builds an INT8 engine directly.
     Uses IInt8EntropyCalibrator2 with a calibration cache for fast rebuilds.
"""

import argparse
import os
import time
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Default bucket widths matching PaddleRec's width bucketing.
# 5 buckets (reduced from 10): fewer TRT shape reconfigs, fewer engine builds.
# Merges: {160→320}, {1200,1600→1600}, {2000,2500,3200→4000}.
REC_WIDTH_BUCKETS = [160, 320, 480, 800, 1200, 1600, 2000, 2500, 3200, 4000]


class RecCalibrator(trt.IInt8EntropyCalibrator2):
    """TRT INT8 calibrator for recognition models using real images.

    Loads images from a directory, preprocesses them to match PaddleRec's
    pipeline (resize to 48xW, ImageNet normalize), and feeds batches to TRT
    for activation range calibration. Caches results for fast rebuilds.
    """

    def __init__(self, data_dir, batch_size=32, img_h=48, img_w=320,
                 max_images=500, cache_file="calibration.cache"):
        super().__init__()
        from PIL import Image
        self._Image = Image

        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.cache_file = cache_file

        # Collect calibration image paths
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(exts)
        ])[:max_images]
        print(f"Calibrator: {len(self.images)} images from {data_dir}, "
              f"batch_size={batch_size}, shape=({img_h},{img_w})")

        self.current = 0

        # Allocate device memory for one batch
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self._cuda = cuda
        nbytes = batch_size * 3 * img_h * img_w * 4  # float32
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= len(self.images):
            return None

        batch = np.zeros((self.batch_size, 3, self.img_h, self.img_w), dtype=np.float32)
        count = min(self.batch_size, len(self.images) - self.current)

        for i in range(count):
            try:
                img = self._Image.open(self.images[self.current + i]).convert('RGB')
                img = img.resize((self.img_w, self.img_h))
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                # ImageNet normalization (matches PaddleRec preprocessing)
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                batch[i] = (arr - mean) / std
            except Exception as e:
                print(f"  Warning: skipping {self.images[self.current + i]}: {e}")

        self.current += self.batch_size
        self._cuda.memcpy_htod(self.device_input, batch.tobytes())
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                data = f.read()
            print(f"Calibrator: loaded cache ({len(data)} bytes) from {self.cache_file}")
            return data
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"Calibrator: saved cache to {self.cache_file}")


class DetCalibrator(trt.IInt8EntropyCalibrator2):
    """TRT INT8 calibrator for detection models."""

    def __init__(self, data_dir, batch_size=4, img_h=640, img_w=640,
                 max_images=200, cache_file="calibration_det.cache"):
        super().__init__()
        from PIL import Image
        self._Image = Image

        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.cache_file = cache_file

        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(exts)
        ])[:max_images]
        print(f"DetCalibrator: {len(self.images)} images, batch_size={batch_size}, "
              f"shape=({img_h},{img_w})")

        self.current = 0

        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self._cuda = cuda
        nbytes = batch_size * 3 * img_h * img_w * 4
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= len(self.images):
            return None

        batch = np.zeros((self.batch_size, 3, self.img_h, self.img_w), dtype=np.float32)
        count = min(self.batch_size, len(self.images) - self.current)

        for i in range(count):
            try:
                img = self._Image.open(self.images[self.current + i]).convert('RGB')
                img = img.resize((self.img_w, self.img_h))
                arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                batch[i] = (arr - mean) / std
            except Exception as e:
                print(f"  Warning: skipping {self.images[self.current + i]}: {e}")

        self.current += self.batch_size
        self._cuda.memcpy_htod(self.device_input, batch.tobytes())
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                data = f.read()
            print(f"DetCalibrator: loaded cache ({len(data)} bytes)")
            return data
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"DetCalibrator: saved cache to {self.cache_file}")


def _setup_builder_config(builder, output_path, fp16, fp8, workspace_gb, opt_level,
                          int8=False, calibrator=None):
    """Create and configure a TRT builder config with common settings."""
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = opt_level
    print(f"Builder optimization level: {opt_level}")

    # Timing cache: speeds up repeated builds significantly
    timing_cache_path = os.path.join(os.path.dirname(output_path) or ".", "timing.cache")
    timing_cache_data = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, "rb") as f:
            timing_cache_data = f.read()
        print(f"Loaded timing cache: {len(timing_cache_data)} bytes")
    cache = config.create_timing_cache(timing_cache_data)
    config.set_timing_cache(cache, ignore_mismatch=False)

    if int8:
        # INT8 mode: enable both INT8 and FP16 flags.
        if hasattr(builder, "platform_has_fast_int8") and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 enabled")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 enabled (platform_has_fast_int8 check unavailable)")
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled (mixed precision with INT8)")
        # If a calibrator is provided, use native TRT calibration (no Q/DQ nodes needed)
        if calibrator is not None:
            config.int8_calibrator = calibrator
            print("INT8 native calibration enabled (IInt8EntropyCalibrator2)")

    if fp8:
        try:
            if hasattr(builder, "platform_has_fast_fp8") and builder.platform_has_fast_fp8:
                config.set_flag(trt.BuilderFlag.FP8)
                print("FP8 enabled")
            else:
                print("FP8 not supported on this platform, skipping")
        except Exception as e:
            print(f"FP8 flag not available in this TRT version: {e}")

    if fp16 and not int8 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # Enable sparse weights if supported (skip zero weights in mobile nets)
    if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("Sparse weights enabled")

    return config, timing_cache_path


def _save_timing_cache(config, timing_cache_path):
    """Save the timing cache after a build."""
    updated_cache = config.get_timing_cache()
    with open(timing_cache_path, "wb") as f:
        f.write(bytearray(updated_cache.serialize()))
    print(f"Saved timing cache: {timing_cache_path}")


def _parse_onnx(builder, onnx_path):
    """Parse an ONNX model. Returns network or None on failure."""
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            return None
    return network


def build_engine(onnx_path, output_path, model_type, fp16=True, fp8=False, workspace_gb=4,
                 opt_level=5, int8=False, calibrator=None):
    builder = trt.Builder(TRT_LOGGER)
    network = _parse_onnx(builder, onnx_path)
    if network is None:
        return False

    config, timing_cache_path = _setup_builder_config(
        builder, output_path, fp16, fp8, workspace_gb, opt_level, int8=int8,
        calibrator=calibrator
    )

    if model_type == "det":
        profile = builder.create_optimization_profile()
        # Detection: [batch, 3, H, W] - H,W multiples of 32, max 960, batch up to 8
        profile.set_shape(
            "x", min=(1, 3, 32, 32), opt=(4, 3, 640, 640), max=(8, 3, 960, 960)
        )
        config.add_optimization_profile(profile)
    elif model_type == "rec":
        # Recognition: [batch, 3, 48, W] - W dynamic, batch up to 32
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "x", min=(1, 3, 48, 48), opt=(32, 3, 48, 320), max=(32, 3, 48, 4000)
        )
        config.add_optimization_profile(profile)
    elif model_type == "cls":
        profile = builder.create_optimization_profile()
        # Classification: [batch, 3, 48, 192] - fixed spatial, batch up to 128
        profile.set_shape(
            "x", min=(1, 3, 48, 192), opt=(64, 3, 48, 192), max=(128, 3, 48, 192)
        )
        config.add_optimization_profile(profile)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Building {model_type} engine: {onnx_path} -> {output_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Failed to build engine")
        return False

    with open(output_path, "wb") as f:
        f.write(serialized)

    _save_timing_cache(config, timing_cache_path)

    print(f"Done: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return True


def build_bucket_engine(onnx_path, output_dir, bucket_w, fp16=True, fp8=False,
                        workspace_gb=4, opt_level=5, int8=False,
                        calibration_data=None, calib_cache_dir=None):
    """Build a TRT engine optimized for a specific width bucket.

    For bucket width W, the profile is:
        opt = (32, 3, 48, W)
        min = (1,  3, 48, max(48, W-100))
        max = (32, 3, 48, W+100)

    This narrow range lets TRT pick kernels perfectly tuned for width ~W.
    """
    output_path = os.path.join(output_dir, f"rec_w{bucket_w}.trt")

    # Create a calibrator for this bucket width if calibration data is provided
    calibrator = None
    if int8 and calibration_data:
        cache_dir = calib_cache_dir or output_dir
        cache_file = os.path.join(cache_dir, f"calibration_w{bucket_w}.cache")
        calibrator = RecCalibrator(
            calibration_data, batch_size=32, img_h=48, img_w=bucket_w,
            max_images=500, cache_file=cache_file
        )

    builder = trt.Builder(TRT_LOGGER)
    network = _parse_onnx(builder, onnx_path)
    if network is None:
        return False

    config, timing_cache_path = _setup_builder_config(
        builder, output_path, fp16, fp8, workspace_gb, opt_level, int8=int8,
        calibrator=calibrator
    )

    profile = builder.create_optimization_profile()

    min_w = max(48, bucket_w - 100)
    opt_w = bucket_w
    max_w = bucket_w + 100

    profile.set_shape(
        "x",
        min=(1, 3, 48, min_w),
        opt=(32, 3, 48, opt_w),
        max=(32, 3, 48, max_w),
    )
    config.add_optimization_profile(profile)

    print(f"\n{'='*60}")
    print(f"Building bucket engine: width={bucket_w} (range [{min_w}, {max_w}])")
    print(f"  opt=(32, 3, 48, {opt_w})")
    print(f"  min=(1, 3, 48, {min_w})")
    print(f"  max=(32, 3, 48, {max_w})")
    print(f"  output: {output_path}")
    print(f"{'='*60}")

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if serialized is None:
        print(f"Failed to build bucket engine w={bucket_w}")
        return False

    with open(output_path, "wb") as f:
        f.write(serialized)

    _save_timing_cache(config, timing_cache_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done: rec_w{bucket_w}.trt = {size_mb:.1f} MB (built in {elapsed:.1f}s)")
    return True


def build_all_bucket_engines(onnx_path, output_dir, buckets=None, fp16=True, fp8=False,
                             workspace_gb=4, opt_level=5, int8=False,
                             calibration_data=None):
    """Build per-bucket TRT engines for all specified width buckets."""
    if buckets is None:
        buckets = REC_WIDTH_BUCKETS

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nBuilding {len(buckets)} per-bucket rec engines")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Buckets: {buckets}")
    if calibration_data:
        print(f"  Calibration data: {calibration_data}")
    print()

    total_t0 = time.time()
    results = {}
    for bw in buckets:
        ok = build_bucket_engine(onnx_path, output_dir, bw, fp16, fp8, workspace_gb,
                                 opt_level, int8=int8, calibration_data=calibration_data)
        results[bw] = ok

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"All bucket engines built in {total_elapsed:.1f}s")
    for bw, ok in results.items():
        status = "OK" if ok else "FAILED"
        path = os.path.join(output_dir, f"rec_w{bw}.trt")
        size = f"{os.path.getsize(path)/1e6:.1f} MB" if ok and os.path.exists(path) else "N/A"
        print(f"  w={bw:>5}: {status} ({size})")
    print(f"{'='*60}")

    return all(results.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--output", required=True, help="Output .trt path (or output dir for --width-buckets)")
    parser.add_argument(
        "--type", required=True, choices=["det", "rec", "cls"], help="det, rec, or cls"
    )
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    parser.add_argument("--fp8", action="store_true", default=False, help="Enable FP8 (requires supported HW)")
    parser.add_argument("--int8", action="store_true", default=False,
                        help="Enable INT8 + FP16 mixed precision. Without --calibration-data, "
                             "expects a pre-quantized ONNX with Q/DQ nodes. With --calibration-data, "
                             "uses TRT native calibration on the original FP32 ONNX model.")
    parser.add_argument("--calibration-data", type=str, default=None,
                        help="Directory of images for TRT native INT8 calibration. "
                             "Use with --int8 and the ORIGINAL FP32 ONNX model (no Q/DQ nodes needed). "
                             "TRT will run these images through the network to determine activation ranges.")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace GB")
    parser.add_argument("--opt-level", type=int, default=5, choices=range(0, 6),
                        help="Builder optimization level 0-5 (default: 5, maximum)")
    parser.add_argument("--width-buckets", action="store_true", default=False,
                        help="Build per-bucket TRT engines for rec model. "
                             "Each bucket gets its own engine with a narrow width range "
                             "for better kernel selection. --output is used as the output directory. "
                             f"Default buckets: {REC_WIDTH_BUCKETS}")
    parser.add_argument("--buckets", type=str, default=None,
                        help="Comma-separated list of bucket widths (default: %(default)s). "
                             f"Only used with --width-buckets. Default: {','.join(map(str, REC_WIDTH_BUCKETS))}")
    args = parser.parse_args()

    if args.calibration_data and not args.int8:
        parser.error("--calibration-data requires --int8")

    if args.width_buckets:
        if args.type != "rec":
            parser.error("--width-buckets is only supported for --type rec")
        buckets = REC_WIDTH_BUCKETS
        if args.buckets:
            buckets = [int(x.strip()) for x in args.buckets.split(",")]
        build_all_bucket_engines(
            args.model, args.output, buckets,
            args.fp16, args.fp8, args.workspace, args.opt_level, int8=args.int8,
            calibration_data=args.calibration_data
        )
    else:
        # Create calibrator for single-engine build if calibration data provided
        calibrator = None
        if args.int8 and args.calibration_data:
            cache_file = os.path.join(
                os.path.dirname(args.output) or ".",
                f"calibration_{args.type}.cache"
            )
            if args.type == "rec":
                calibrator = RecCalibrator(
                    args.calibration_data, batch_size=32, img_h=48, img_w=320,
                    max_images=500, cache_file=cache_file
                )
            elif args.type == "det":
                calibrator = DetCalibrator(
                    args.calibration_data, batch_size=4, img_h=640, img_w=640,
                    max_images=200, cache_file=cache_file
                )
            else:
                # cls calibrator would use 48x192 -- add if needed
                print("Warning: no calibrator implemented for cls, using Q/DQ mode")

        build_engine(args.model, args.output, args.type, args.fp16, args.fp8,
                     args.workspace, args.opt_level, int8=args.int8,
                     calibrator=calibrator)


if __name__ == "__main__":
    main()
