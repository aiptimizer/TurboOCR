# Slim CPU image

`docker/Dockerfile.cpu-slim` is a multi-stage variant of the CPU Dockerfile. It
builds the binary in a full toolchain stage, then copies only the binary,
runtime libraries and models into a smaller runtime stage.

## Size

| Image | Typical size |
|---|---|
| `docker/Dockerfile.cpu` | ~500 MB |
| `docker/Dockerfile.cpu-slim` | ~250–350 MB |

Exact size depends on the target architecture and how many layers Docker can
share with the base image.

## Build

```bash
docker build -f docker/Dockerfile.cpu-slim -t turboocr-cpu:slim .
```

For ARM64 (Apple Silicon, AWS Graviton):

```bash
docker buildx build --platform linux/arm64 \
  -f docker/Dockerfile.cpu-slim -t turboocr-cpu:slim-arm64 .
```

## Run

```bash
docker run -d --name turboocr-cpu-slim \
  -p 8000:8000 -p 50051:50051 \
  -e OCR_MODEL=tiny \
  -e ORT_EP=xnnpack \
  turboocr-cpu:slim
```

## How it works

1. **Builder stage**: installs cmake, g++, Drogon, ONNX Runtime, OpenCV, gRPC,
   PDFium and builds `turboocr-cpu-server`.
2. **Dependency collection**: runs `ldd` on the binary to discover every
   `.so` it needs, then copies them to `/opt/turboocr-libs/`.
3. **Runtime stage**: starts from a clean Ubuntu 24.04 image with only nginx,
   gosu, curl and ca-certificates, then copies in the binary, collected
   libraries and models.

Because the runtime stage has no compiler, git, wget or build headers, the
attack surface is smaller and image pulls are faster.

## Caveats

- The `ldd` collector copies all transitive runtime dependencies. If your host
  build environment injects unexpected libraries, they will be included.
- OpenVINO, oneDNN and XNNPACK execution providers require their provider
  shared libraries to be present in the copied set. They are included because
  `ldd` sees them through `libonnxruntime.so`'s provider-loading mechanism
  when the provider libraries sit next to it.
- If you build a custom ORT with additional execution providers, verify the
  provider `.so` files are copied alongside `libonnxruntime.so`.
