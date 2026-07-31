# TurboOCR: CPU vs GPU

TurboOCR is available in two builds from the same source tree:

- **GPU build** (`docker/Dockerfile.gpu`) — TensorRT + CUDA; fastest throughput on NVIDIA hardware.
- **CPU build** (`docker/Dockerfile.cpu`) — ONNX Runtime; runs anywhere, no GPU required.

Both builds expose the **same HTTP/gRPC API** and return the **same response
format**. You can switch between them without changing client code.

## Feature parity

| Capability | GPU build | CPU build |
|---|---|---|
| Text OCR (PP-OCRv6 tiny/small/medium) | ✓ | ✓ |
| PP-OCRv5 retained scripts (Arabic, Cyrillic, …) | ✓ | ✓ |
| Layout + reading order (PP-DocLayoutV3) | ✓ | ✓ |
| Tables → HTML (SLANet+) | ✓ | ✓ |
| Formulas → LaTeX (PP-FormulaNet-S) | ✓ | ✓ |
| PDF → text / markdown | ✓ | ✓ |
| Batch OCR | ✓ | ✓ |
| gRPC API | ✓ | ✓ |
| Prometheus metrics | ✓ | ✓* |

\* GPU-specific metrics (VRAM, TensorRT cache) are omitted; request latency and
throughput metrics remain.

## When to use the CPU build

Choose the CPU build when:

- You do not have an NVIDIA GPU.
- You deploy on laptops, edge devices, or CPU-only cloud VMs.
- You want a much smaller image (~500 MB vs ~10 GB).
- You need instant cold start (no TensorRT engine compilation on first run).
- You want predictable per-request cost on shared infrastructure.

Choose the GPU build when:

- You need maximum throughput (hundreds of images per second).
- You have an RTX 20-series or newer NVIDIA card available.
- You can tolerate the ~10 GB image and the first-start engine-build delay.

## Performance expectations

Throughput is highly workload-dependent. Approximate guidance for a single
instance:

| Workload | GPU (RTX 4090) | CPU (Apple M3 Max) | CPU (Intel i7-13700H) |
|---|---:|---:|---:|
| Receipts / dense text (tiny) | 200–550 img/s | 8–12 img/s | 4–8 img/s |
| Full document + layout + tables + formulas | 15–20 pg/s | 1–2 pg/s | 0.5–1 pg/s |

CPU throughput scales roughly linearly with core count and benefits from
modern SIMD (AVX-512, AMX, Apple AMX). Use `ORT_EP=xnnpack` or `ORT_EP=openvino`
on x86_64 for the best CPU throughput.

## Deployment cost

| Cost | GPU build | CPU build |
|---|---|---|
| Image size | ~10 GB | ~500 MB |
| Cold start | 90 s–1 h (TensorRT engine build) | seconds |
| Min host requirement | NVIDIA GPU, 8 GB VRAM | any CPU, ~4 GB RAM (text), ~8 GB RAM (full pipeline) |
| Container runtime | NVIDIA Container Toolkit required | Docker/Podman anywhere |

## Switching from GPU to CPU

Change only the image and remove the GPU reservations:

```yaml
services:
  ocr:
    image: ghcr.io/aiptimizer/turboocr-cpu:latest   # or local build
    # remove: deploy.resources.reservations.devices
    environment:
      - OCR_MODEL=tiny
      - PIPELINE_POOL_SIZE=4
      - ORT_EP=xnnpack
```

All client requests stay identical:

```bash
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

## Marketing angle: "No GPU required"

Because TurboOCR's CPU build runs the same pipeline as the GPU build, you can
offer users:

- **Zero GPU tax**: deploy on existing CPU servers or laptops.
- **Smaller image**: ~500 MB, fast CI/CD pulls.
- **Instant cold start**: no engine compilation, scale to zero friendly.
- **Same API**: clients written for the GPU build work unchanged.
- **Same accuracy**: identical PP-OCRv6 / layout / table / formula models.

The trade-off is throughput, not correctness.
