# docker/

**One multi-stage `Dockerfile`.** Select a backend with `--target`:

```bash
docker build -f docker/Dockerfile --target cpu    -t turboocr:cpu    .
docker build -f docker/Dockerfile --target nvidia -t turboocr:nvidia .
docker build -f docker/Dockerfile --target intel  -t turboocr:intel  .
docker build -f docker/Dockerfile --target amd    -t turboocr:amd    .
```

| target | base | status |
|---|---|---|
| `cpu` | `ubuntu:24.04` (digest-pinned) | shipped, built in CI |
| `nvidia` | `nvcr.io/nvidia/tensorrt:26.03-py3` | shipped; compile-gated by `cuda-compile-gate` |
| `intel` | `openvino/ubuntu24_dev:2026.2.1` | **UNVERIFIED** — never built or run |
| `amd` | `rocm/dev-ubuntu-24.04:7.14-complete` | **builds CPU backend only** — `TURBO_BACKENDS=amd` is still a FATAL_ERROR |

**There is no `apple` target and there cannot be.** Every macOS virtualization
product builds on Apple's `Hypervisor.framework`, which provides virtual CPU
and memory but **no virtual GPU** — there is no Metal passthrough to expose, so
a Linux container on a Mac cannot reach Metal or MPSGraph. Apple runs natively;
see the *Apple Silicon* section of the top-level README.

**Why one file.** The two previous Dockerfiles were 86 lines each and 59 of
those were identical. Four vendors would have meant four copies of the same 59
lines. Shared work lives in the `base` stage; each vendor stage carries only
its own SDK and CMake flags.

`config/nginx.conf.template` is rendered by `scripts/entrypoint.sh` at
container start (`envsubst`, so `MAX_BODY_MB` reaches nginx and the server
identically).
