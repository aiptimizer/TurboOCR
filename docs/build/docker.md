# Docker

!!! abstract "TL;DR"
    Two images, both built from the same source tree. The **GPU image**
    (`docker/Dockerfile.gpu`) targets `nvcr.io/nvidia/tensorrt:26.03-py3`
    with full CUDA + TensorRT support (~10 GB). The **CPU image**
    (`docker/Dockerfile.cpu`) targets `ubuntu:24.04` with ONNX Runtime
    only (~500 MB). Both expose **8000** (HTTP) and **50051** (gRPC).

| File | Image tag | Base | Runtime |
|---|---|---|---|
| `docker/Dockerfile.gpu` | `turbo-ocr`     | `nvcr.io/nvidia/tensorrt:26.03-py3` (digest-pinned) | CUDA + TensorRT |
| `docker/Dockerfile.cpu` | `turbo-ocr-cpu` | `ubuntu:24.04` (digest-pinned)                       | ONNX Runtime    |

## GPU image

### Build + run

=== "docker"

    ```bash
    docker build -f docker/Dockerfile.gpu -t turbo-ocr .
    docker run --gpus all \
               -p 8000:8000 -p 50051:50051 \
               -v trt-cache:/home/ocr/.cache/turbo-ocr \
               turbo-ocr
    ```

=== "compose"

    ```yaml
    # docker/docker-compose.yml is the canonical example.
    services:
      turbo-ocr:
        image: turbo-ocr
        runtime: nvidia
        ports: ["8000:8000", "50051:50051"]
        volumes: ["trt-cache:/home/ocr/.cache/turbo-ocr"]
    volumes:
      trt-cache:
    ```

!!! tip "Persist the cache volume"
    The named volume on `/home/ocr/.cache/turbo-ocr` persists both the
    built TensorRT engines and the per-language recognition bundles
    fetched at image-build time. Without it, every container start
    rebuilds engines from ONNX (~90 s).

### Image layers

`docker/Dockerfile.gpu`, high level — see the file for the full
manifest:

1. apt packages: `build-essential pkg-config libopencv-dev nginx gosu
   libwebp-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
   libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libc-ares-dev git wget
   curl gettext-base`.
2. Drogon v1.9.12 from source (same flags as the native build).
3. CMake 3.31.6 from upstream tarball — the base image's 3.24 doesn't
   support CUDA C++20.
4. ONNX Runtime 1.22.0 — pinned SHA256 verified before extraction.
5. Source COPY (`CMakeLists.txt proto/ include/ src/ tools/
   third_party/ tests/`).
6. `install_pdfium.sh` then `install_fastpdf2png.sh` for the PDFium-backed PDF
   renderer, both keyed on `TARGETARCH` (on arm64 they replace the vendored
   x86-64 PDFium and rebuild the renderer; both check the ELF architecture of
   what is already installed).
7. ```bash
   cmake .. -DTENSORRT_DIR=/usr/lib/x86_64-linux-gnu -DFETCH_MODELS=OFF -DTURBO_BUILD_FASTPDF2PNG=OFF
   make -j$(nproc)
   ```
   (`TURBO_BUILD_FASTPDF2PNG=OFF` because step 6 already produced `bin/fastpdf2png`.)
8. Runtime-only layers (`docker/` configs, `scripts/entrypoint.sh`,
   model fetcher).
9. `useradd ocr`, symlink `/app/models/rec → /home/ocr/.cache/turbo-ocr/
   models/rec`, then run `fetch_release_models.sh` to seed every
   PP-OCRv5 bundle into the cache directory.

### Entrypoint

```bash
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["./build/turboocr-server"]
```

The entrypoint drops privileges to the `ocr` user, optionally starts
nginx, then execs the CMD.

### Build args

| Arg | Default | Effect |
|---|---|---|
| `TARGETARCH` | host arch | Set automatically by `docker buildx` (`amd64` / `arm64`); selects the matching ONNX Runtime + PDFium binaries. |
| `ORT_VERSION` | `1.22.0` | ONNX Runtime C++ SDK version baked in. |

All model weights (every tier + language + table/formula) are fetched from the
GitHub Release at build time, so there is no language-bundle build arg.

## CPU image

```bash
docker build -f docker/Dockerfile.cpu -t turbo-ocr-cpu .
docker run -p 8000:8000 -p 50051:50051 \
           -v "$PWD/models:/app/models" \
           turbo-ocr-cpu
```

Same package set minus the CUDA-specific pieces; build switches to
`-DUSE_CPU_ONLY=ON -DFETCH_MODELS=OFF`.

## Environment variables

These are read by the server itself and apply to both Dockerfiles:

| Var | Default | Effect |
|---|---|---|
| `TURBO_OCR_HOST` | `0.0.0.0` | HTTP bind address. |
| `TURBO_OCR_PORT` | `8080`   | HTTP backend port (nginx proxies `8000` → `8080`). |
| `TURBO_OCR_GRPC_PORT` | `50051` | gRPC bind port. |
| `DISABLE_LAYOUT` | unset | Skip loading PP-DocLayoutV3 (smaller startup, no `?layout=1`). |
| `MAX_IMAGE_DIM` | `16384` | Pre/post-decode dimension cap (clamped `[64, 65535]`). |
| `MAX_PDF_PAGES` | `2000` | Hard cap on `/ocr/pdf` page count. |
| `OCR_MODEL` | `tiny` | Recognizer tier / language: `tiny`/`small`/`medium` (Latin+Chinese+Japanese) or `arabic`/`eslav`/`korean`/`thai`/`greek`. |
| `TABLE_BACKEND` | unset | `slanext` enables table→HTML (baked encoder auto-resolves). Run per request with `?tables=1`. |
| `FORMULA_BACKEND` | unset | `ppformulanet_s` enables formula→LaTeX (baked weights auto-resolve); `ppformulanet_plus_m` for Chinese (GPU). Run per request with `?formulas=1`. |
| `MAX_BODY_MB` | `100` | nginx `client_max_body_size` (consumed by the nginx template). |
| `MODELS_RELEASE_URL` | release URL | Override the models bundle base URL at build time. |

!!! tip "Dump the resolved config"
    `./turboocr-server --print-config` writes the full effective
    configuration (including the `--grpc-response-mode` flag for gRPC
    JSON-bytes vs. structured responses).

## nginx fronting

`docker/nginx.conf.template` is `envsubst`'d at container start by the
entrypoint, then nginx daemonizes in front of the OCR server:

- Listens on `:8000` with `backlog=4096 reuseport`.
- Proxies to `127.0.0.1:8080` (the server's `TURBO_OCR_PORT`).
- Honors `MAX_BODY_MB` for `client_max_body_size`.
- Returns JSON error envelopes for 413 / 502 / 504 so clients see the
  same shape they would directly from Drogon.

!!! info "502 → 503 retry remap"
    `nginx.conf.template:36-47` remaps upstream **502 → 503** with
    `Retry-After: 15`. During first start the backend is not crashed —
    it's still building TensorRT engines — so clients should retry
    rather than treat it as a hard failure.

## Compose

`docker/docker-compose.yml` wires the GPU image with the persistent
cache volume and host-network port mapping. Useful as a starting point
for production deployments.

!!! info "See also"
    - [Build → Native](native.md) — same dependency pins, no Docker.
    - [Build → Models](models.md) — what `fetch_release_models.sh`
      bakes into the image.
    - [API → HTTP](../api/http.md) — the endpoints exposed on 8000.
