# CPU-only quickstart

TurboOCR ships a **CPU-only server** that runs the same OCR, layout, table and
formula pipeline as the GPU build without an NVIDIA card. It is built on
ONNX Runtime, so it works on:

- Any x86_64 or ARM64 Linux machine
- Apple Silicon Macs (via Docker Desktop / OrbStack / Lima)
- Cloud VMs without GPU

The CPU image is ~500 MB vs ~10 GB for the GPU image and starts immediately
(no TensorRT engine warm-up).

## One-line start

```bash
docker compose -f docker/docker-compose.cpu.yml up -d
```

For an even smaller image (~250–350 MB) see [docs/build/cpu-slim.md](cpu-slim.md).

Then test it:

```bash
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @tests/test_data/png/receipt.png \
  -H "Content-Type: image/png"
```

Or use the convenience script:

```bash
scripts/quickstart-cpu.sh
```

## Picking an execution provider

The CPU server can route ONNX inference through several execution providers.
The right one depends on your hardware:

| Provider | Env var | Best for | Notes |
|---|---|---|---|
| MLAS (default CPU) | `ORT_EP=cpu` | Any CPU | Safe baseline; good single-core throughput. |
| XNNPACK | `ORT_EP=xnnpack` | x86_64 / ARM64 | Often the fastest CPU EP for conv-heavy PP-OCRv6 models; uses its own threadpool. |
| oneDNN | `ORT_EP=dnnl` | Intel/AMD | Good on AVX-512 machines; requires oneDNN-enabled ORT build. |
| OpenVINO | `ORT_EP=openvino` | Intel CPUs | Best throughput on modern Intel cores; requires OpenVINO-enabled ORT build. |
| OpenVINO iGPU | `ORT_EP=openvino_gpu` | Intel integrated GPUs | Offload conv-heavy models to Intel iGPU where available. |
| CoreML | (auto on macOS) | Apple Silicon | Enabled automatically inside the container on Apple hosts when using an ARM64 image. |

Example with XNNPACK:

```bash
ORT_EP=xnnpack docker compose -f docker/docker-compose.cpu.yml up -d
```

## Model tiers

The same `OCR_MODEL` values work as on GPU:

| Tier | Speed | Accuracy | RAM |
|---|---|---|---|
| `tiny` | fastest | good | ~1 GB |
| `small` | ~2–4× slower than tiny | better | ~2 GB |
| `medium` | slowest | best | ~3 GB |

```bash
OCR_MODEL=small docker compose -f docker/docker-compose.cpu.yml up -d
```

## Optional stages

Layout, tables and formulas are opt-in, exactly like the GPU build:

```bash
# layout + reading order (default enabled; disable with DISABLE_LAYOUT=1)
docker compose -f docker/docker-compose.cpu.yml up -d

# + tables → HTML
docker compose -f docker/docker-compose.cpu.yml -e TABLE_BACKEND=slanext up -d

# + formulas → LaTeX
docker compose -f docker/docker-compose.cpu.yml \
  -e FORMULA_BACKEND=ppformulanet_s up -d
```

Per-request opt-in works identically:

```bash
curl -X POST "http://localhost:8000/ocr/raw?layout=1&tables=1&formulas=1" \
  --data-binary @paper.png -H "Content-Type: image/png"
```

## Thread tuning

Two knobs control CPU utilization:

- `ORT_NUM_THREADS` — intra-op threads per inference session (default 4).
- `PIPELINE_POOL_SIZE` — number of concurrent OCR pipelines (default 4).

On an 8-core machine, a good starting point is:

```bash
ORT_NUM_THREADS=4 PIPELINE_POOL_SIZE=2 docker compose -f docker/docker-compose.cpu.yml up -d
```

On a 4-core machine, lower both to avoid oversubscription:

```bash
ORT_NUM_THREADS=2 PIPELINE_POOL_SIZE=2 docker compose -f docker/docker-compose.cpu.yml up -d
```

Set `ORT_SHARED_POOL=1` to share one global ONNX Runtime threadpool across all
sessions instead of one pool per session; this often improves throughput under
concurrency.

### Model preparation (optional)

The original ONNX models work directly with ONNX Runtime. Two optional scripts
can further optimize them for CPU:

- `scripts/prepare_cpu_models.py` — constant-fold and simplify ONNX graphs
  (produces `*_opt.onnx`).
- `scripts/quantize_cpu_models.py` — dynamic INT8 quantization
  (produces `*_int8.onnx`).

You can combine them: simplify first, then quantize the simplified model.

```bash
pip install onnx onnxsim onnxruntime
bash scripts/fetch_release_models.sh
python3 scripts/prepare_cpu_models.py --models-dir ./models
python3 scripts/quantize_cpu_models.py --models-dir ./models
```

Then run with the optimized models:

```bash
DET_MODEL=./models/det_int8.onnx \
REC_MODEL=./models/rec_int8.onnx \
CLS_MODEL=./models/cls_int8.onnx \
./build_cpu/turboocr-cpu-server
```

Validate accuracy on your own documents before deploying.

## Building from source (Linux)

If you prefer a native binary:

```bash
cmake -B build_cpu -DUSE_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j$(nproc) --target turboocr-cpu-server

# Fetch models the first time
bash scripts/fetch_release_models.sh

./build_cpu/turboocr-cpu-server
```

Requirements: GCC 13+ or Clang 17+, CMake 3.20+, OpenCV 4.x, Drogon 1.9+,
gRPC/protobuf, libturbojpeg, libpdfium. ONNX Runtime 1.22.0 is downloaded
automatically by CMake if not already installed.

## Expected performance

Throughput is workload- and hardware-dependent. Representative single-page
numbers on common CPUs:

| Hardware | Model | Text only | + layout | + tables + formulas |
|---|---|---:|---:|---:|
| Apple M3 Max | tiny | ~8–12 pg/s | ~3–5 pg/s | ~1–2 pg/s |
| Intel i7-13700H | tiny | ~4–8 pg/s | ~2–3 pg/s | ~0.5–1 pg/s |
| AMD EPYC 7402 | tiny | ~3–6 pg/s | ~1–2 pg/s | ~0.3–0.6 pg/s |
| AWS c7g (Graviton3) | tiny | ~3–5 pg/s | ~1–2 pg/s | ~0.3–0.5 pg/s |

Run `scripts/benchmark-cpu.sh` against a running container to measure your
own hardware.

### Benchmarking an execution provider

```bash
# Start with the provider you want to test
docker run -d --name turboocr-cpu \
  -p 8000:8000 \
  -e ORT_EP=xnnpack \
  turboocr-cpu:latest

# Run 30-second benchmark
scripts/benchmark-cpu.sh tests/test_data/png/receipt.png 30

# Compare with the default CPU provider
scripts/quickstart-cpu.sh  # stop and restart with ORT_EP=cpu
scripts/benchmark-cpu.sh tests/test_data/png/receipt.png 30
```

## Troubleshooting

**Container exits immediately:** check logs with `docker logs turboocr-cpu`.
Common causes: port 8000 already bound, or `OCR_MODEL` value unsupported.

**Very high latency:** reduce `PIPELINE_POOL_SIZE` and `ORT_NUM_THREADS` so the
machine is not oversubscribed. Use `ORT_SHARED_POOL=1`.

**Layout/tables/formulas disabled in `/capabilities`:** the relevant backend
env var was not set at startup. Restart the container with
`TABLE_BACKEND=slanext` or `FORMULA_BACKEND=ppformulanet_s`.
