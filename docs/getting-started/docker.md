# Docker

One Dockerfile builds four images — `nvidia`, `cpu`, `intel`, `amd` — from the
same source tree:

```bash
docker build -f docker/Dockerfile --target <nvidia|cpu|intel|amd> -t turboocr:<target> .
```

Every image exposes the identical API on **8000** (HTTP, nginx-fronted) and
**50051** (gRPC). The exact `docker run` line for your hardware — device
flags included — is in the [install selector](install.md); this page covers
what the selector doesn't: scaling, what's inside the image, and the nginx
front.

| Target | Base image | Status | Run flags |
|---|---|---|---|
| `nvidia` | `nvcr.io/nvidia/tensorrt:26.03-py3` (digest-pinned) | published as `ghcr.io/aiptimizer/turboocr` | `--gpus all` |
| `cpu` | `ubuntu:24.04` (digest-pinned) | build from repo | — |
| `intel` | `openvino/ubuntu24_dev:2026.2.1` | build from repo | `--device /dev/dri` |
| `amd` | `rocm/dev-ubuntu-24.04:7.1.1` | build from repo · not yet hardware-tested | `--device /dev/kfd --device /dev/dri --group-add video` |

Two things about running it matter on day one. First start on the NVIDIA
image builds TensorRT engines — about 90 s on a 5090, up to an hour on older
cards (`TRT_OPT_LEVEL=3` cuts build time 3–5× for <5% runtime cost). And
those engines, the MIGraphX `.mxr` programs on AMD, and the per-language
recognition bundles all live under `/home/ocr/.cache/turbo-ocr` — mount a
named volume there (`-v trt-cache:/home/ocr/.cache/turbo-ocr`, as the
selector's commands already do) or every container start pays the build
again.

## Scaling with compose

`docker/compose.yaml` is a production example that plain `docker run` cannot
express: **N server instances on one GPU under NVIDIA MPS**, sharing a single
TRT-engine cache volume, with health checks and an optional gRPC profile.

```bash
sudo nvidia-cuda-mps-control -d          # MPS daemon, once per boot
docker compose -f docker/compose.yaml up -d              # 7 instances, ports 8001-8007
docker compose -f docker/compose.yaml up -d --scale ocr=5
docker compose -f docker/compose.yaml --profile grpc up -d   # + gRPC on 50051
```

Put any load balancer in front of the port range. Engines build once (the
first instance pays) and every replica reuses the shared cache volume — the
same one described in the intro; how its contents are laid out is in
[Model bundle](model-bundle.md).

## What is inside the image

High level (see `docker/Dockerfile` for the full manifest):

1. System packages, Drogon v1.9.12 from source, CMake 3.31.6 (the NVIDIA
   base image's CMake is too old for CUDA C++20).
2. ONNX Runtime `1.28.0` — SHA256-verified before extraction
   (`ORT_VERSION` build arg).
3. Source build of `turboocr-server` for the target's backend set.
4. `install_fastpdf2png.sh` for the PDFium-backed PDF renderer.
5. A non-root `ocr` user; `fetch_release_models.sh` seeds every model
   bundle into the cache directory at build time — all weights are baked
   in, nothing downloads at run time.

The entrypoint (`scripts/entrypoint.sh`) drops privileges to `ocr`,
templates and starts nginx, then execs `./build/turboocr-server`.

## Environment variables

Container-specific knobs (the full 35+ variable reference is
[Configuration](../reference/configuration.md)):

| Var | Default | Effect |
|---|---|---|
| `PORT` | `8080` | Internal backend port Drogon binds and nginx proxies to (nginx itself listens on `8000`). Rendered into the nginx upstream at start, so changing it keeps the proxy in sync. Remap the *published* port with docker `-p`. |
| `MAX_BODY_MB` | `100` | nginx `client_max_body_size`. |
| `OCR_MODEL` · `TABLE_BACKEND` · `FORMULA_BACKEND` | — | pick the model tier and opt-in stages, as everywhere ([Configuration](../reference/configuration.md)). |

!!! tip "Dump the resolved config"
    `./turboocr-server --print-config` writes the full effective
    configuration. The complete variable list is in
    [Configuration](../reference/configuration.md).

## nginx fronting

`docker/config/nginx.conf.template` is `envsubst`'d at container start:

- Listens on `:8000` with `backlog=4096 reuseport`.
- Proxies to `127.0.0.1:${PORT}` (default `8080`, the port the server binds).
- Honors `MAX_BODY_MB` for `client_max_body_size`.
- Returns JSON error envelopes for 413 / 502 / 504 so clients see the same
  shape they would directly from Drogon.

!!! warning "Upstream errors are returned as 502 — there is no 503 remap"
    The bundled nginx template preserves upstream **502** and sends no
    `Retry-After`. This matters for retry policy: during first start on the
    NVIDIA image the backend is not crashed — it is still building TensorRT
    engines — so a client that wants to ride that out must treat **502** as
    retryable, or front the service with a gateway that does.

!!! info "See also"
    - [Install](install.md) — the pick-your-hardware selector.
    - [Native build](native.md) — the same build without Docker.
    - [Model bundle](model-bundle.md) — what `fetch_release_models.sh` bakes in.
    - [HTTP API](../reference/http.md) — the endpoints exposed on 8000.
