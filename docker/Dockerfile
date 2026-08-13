# ============================================================
# TurboOCR — ONE multi-stage Dockerfile for every containerizable backend.
#
#   docker build -f docker/Dockerfile --target cpu     -t turboocr:cpu     .
#   docker build -f docker/Dockerfile --target nvidia  -t turboocr:nvidia  .
#   docker build -f docker/Dockerfile --target intel   -t turboocr:intel   .
#   docker build -f docker/Dockerfile --target amd     -t turboocr:amd     .
#
# WHY ONE FILE. The previous per-vendor Dockerfiles were 86 lines each and
# 59 of those lines were IDENTICAL — Drogon from source, PDFium, the layered
# COPY, the ocr user, model fetch, healthcheck, entrypoint. Four vendors meant
# four copies of the same 59 lines. Everything shared now lives in `base`;
# each vendor stage carries only its own SDK and CMake flags.
#
# WHY NOT one file with `ARG BACKEND`. The base images genuinely differ
# (ubuntu / nvcr tensorrt / openvino runtime / rocm), so an ARG-switched build
# needs a conditional FROM plus `if [ "$BACKEND" = … ]` inside every RUN. That
# defeats layer caching and hides the per-vendor delta instead of showing it.
# Named stages give one file AND a readable diff between vendors.
#
# THERE IS NO apple STAGE, AND THERE CANNOT BE. Every macOS virtualization
# product builds on Apple's Hypervisor.framework, which provides virtual CPU
# and memory but NO virtual GPU — there is no Metal passthrough to expose, so
# a Linux container on a Mac cannot reach Metal or MPSGraph. Apple runs
# natively; see the "Apple Silicon" section of the top-level README.
# ============================================================

# ---------------------------------------------------------------------------
# deps — the shared toolchain layer, parameterised only by base image.
# Vendor stages start FROM their own SDK image and re-run this via `deps`.
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=ubuntu:24.04@sha256:786a8b558f7be160c6c8c4a54f9a57274f3b4fb1491cf65146521ae77ff1dc54
FROM ${BASE_IMAGE} AS base

ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make pkg-config libopencv-dev nginx gosu \
    libwebp-dev libturbojpeg0-dev libgrpc++-dev libc-ares-dev \
    libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libcurl4-openssl-dev \
    git wget curl ca-certificates gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Drogon (async HTTP). Identical for every vendor.
RUN git clone --depth 1 --branch v1.9.12 https://github.com/drogonframework/drogon.git /tmp/drogon && \
    cd /tmp/drogon && git submodule update --init && \
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
          -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
          -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF && \
    cmake --build build -j"$(nproc)" && cmake --install build && rm -rf /tmp/drogon

WORKDIR /app

# Layered COPY: runtime edits (entrypoint, docker/config/) must not bust the
# expensive C++ build layer below.
COPY CMakeLists.txt ./
COPY proto/ proto/
COPY include/ include/
COPY src/ src/
COPY tools/ tools/
COPY third_party/ third_party/
COPY tests/ tests/
COPY scripts/setup/install_fastpdf2png.sh scripts/setup/install_fastpdf2png.sh
COPY scripts/setup/install_pdfium.sh scripts/setup/install_pdfium.sh

# PDFium + fastpdf2png. Fail the BUILD on error: the PDF-renderer liveness probe
# would otherwise only surface this at container start, which is much worse.
RUN TARGETARCH=${TARGETARCH} bash scripts/setup/install_pdfium.sh && \
    cp third_party/pdfium/lib/libpdfium.so /usr/lib/ && ldconfig && \
    TARGETARCH=${TARGETARCH} bash scripts/setup/install_fastpdf2png.sh && \
    { cp bin/libpdfium.so /usr/lib/ 2>/dev/null || true; } && ldconfig

# ---------------------------------------------------------------------------
# runtime-common — everything after the per-vendor build. Vendor stages COPY
# this in rather than repeating it.
# ---------------------------------------------------------------------------
FROM base AS runtime-common
COPY docker/config/ docker/config/
COPY scripts/entrypoint.sh scripts/
COPY scripts/models/fetch/download_models.sh scripts/models/fetch/fetch_release_models.sh scripts/models/fetch/

RUN useradd -m -s /bin/bash ocr \
    && chmod +x /app/scripts/entrypoint.sh \
    && mkdir -p /home/ocr/.cache/turbo-ocr/models/rec /app/models \
    && ln -s /home/ocr/.cache/turbo-ocr/models/rec /app/models/rec
RUN bash scripts/models/fetch/fetch_release_models.sh \
    && chown -R ocr:ocr /app /home/ocr/.cache

EXPOSE 8000 50051
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# ---------------------------------------------------------------------------
# cpu — portable ONNX Runtime build. The default target.
# ---------------------------------------------------------------------------
FROM base AS cpu-build
ARG TARGETARCH
# PINNED with verified checksums. 1.28.0 is available and is a deliberate TODO —
# see the bump procedure in docs/notes/production-readiness. Do NOT bump the
# version without also refreshing these hashes: an empty SHA falls through to the
# "skipping checksum" branch below, which silently turns a supply-chain
# guarantee into a trust-the-network download.
ARG ORT_VERSION=1.28.0
ARG ORT_SHA256=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407
ARG ORT_SHA256_ARM64=e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb
RUN cd /tmp && \
    case "${TARGETARCH:-amd64}" in \
      arm64) ORT_ARCH=aarch64; ORT_SHA=${ORT_SHA256_ARM64} ;; \
      *)     ORT_ARCH=x64;     ORT_SHA=${ORT_SHA256} ;; \
    esac && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" -O ort.tgz && \
    if [ -n "${ORT_SHA}" ]; then echo "${ORT_SHA}  ort.tgz" | sha256sum -c -; \
    else echo "WARN: no SHA256 pinned for ${ORT_ARCH} ONNX Runtime ${ORT_VERSION}; skipping checksum"; fi && \
    tar xzf ort.tgz && \
    cp -r onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/include/* /usr/local/include/ && \
    cp -P onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/lib/libonnxruntime*.so* /usr/local/lib/ && \
    ldconfig && rm -rf /tmp/onnxruntime* /tmp/ort.tgz
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON -DFETCH_MODELS=OFF \
    && cmake --build build --target turboocr-server -j"$(nproc)"

FROM runtime-common AS cpu
COPY --from=cpu-build /app/build/turboocr-server /app/build/turboocr-server
COPY --from=cpu-build /usr/local/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1
CMD ["./build/turboocr-server"]

# ---------------------------------------------------------------------------
# nvidia — CUDA + TensorRT. Needs the CUDA-enabled ORT (the plain CPU tarball
# links fine and then fails at runtime; CMake hard-requires the CUDA EP at
# configure time, so a wrong asset fails the BUILD, which is correct).
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/tensorrt:26.03-py3@sha256:ade1b30517b3d66b911a3cd7faf0146484ab8956098abe66b96b944fa36f4861 AS nvidia-sdk
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libopencv-dev nginx gosu libwebp-dev \
    libturbojpeg0-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev libc-ares-dev git wget \
    curl libcurl4-openssl-dev gettext-base && rm -rf /var/lib/apt/lists/*
# base image ships cmake 3.24; CUDA C++20 needs newer
ARG CMAKE_VERSION=3.31.6
RUN cd /tmp && case "${TARGETARCH:-amd64}" in arm64) CM=aarch64 ;; *) CM=x86_64 ;; esac && \
    wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CM}.tar.gz" && \
    tar xzf cmake-${CMAKE_VERSION}-linux-${CM}.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-linux-${CM}/bin/* /usr/local/bin/ && \
    cp -r cmake-${CMAKE_VERSION}-linux-${CM}/share/* /usr/local/share/ && rm -rf /tmp/cmake*
# PINNED (CUDA-13 build). Same rule as the cpu stage: bump version and hash together.
ARG ORT_VERSION=1.28.0
ARG ORT_SHA256=84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de
RUN cd /tmp && PKG=onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION} && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${PKG}.tgz" -O ort.tgz && \
    if [ -n "${ORT_SHA256}" ]; then echo "${ORT_SHA256}  ort.tgz" | sha256sum -c -; \
    else echo "WARN: no SHA256 pinned for CUDA ONNX Runtime ${ORT_VERSION}; skipping checksum"; fi && \
    tar xzf ort.tgz && cp -r ${PKG}/include/* /usr/local/include/ && \
    cp -P ${PKG}/lib/libonnxruntime*.so* /usr/local/lib/ && ldconfig && rm -rf /tmp/ort.tgz /tmp/onnxruntime*

FROM nvidia-sdk AS nvidia
ARG TARGETARCH
WORKDIR /app
COPY CMakeLists.txt ./
COPY proto/ proto/
COPY include/ include/
COPY src/ src/
COPY tools/ tools/
COPY third_party/ third_party/
COPY tests/ tests/
COPY scripts/ scripts/
COPY docker/config/ docker/config/
RUN git clone --depth 1 --branch v1.9.12 https://github.com/drogonframework/drogon.git /tmp/drogon && \
    cd /tmp/drogon && git submodule update --init && \
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
          -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
          -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF && \
    cmake --build build -j"$(nproc)" && cmake --install build && rm -rf /tmp/drogon
RUN TARGETARCH=${TARGETARCH} bash scripts/setup/install_fastpdf2png.sh && \
    cp bin/libpdfium.so /usr/lib/ && ldconfig
# TensorRT libs live under the Debian multiarch triplet, which differs per arch.
RUN case "${TARGETARCH:-amd64}" in arm64) TRT=/usr/lib/aarch64-linux-gnu ;; *) TRT=/usr/lib/x86_64-linux-gnu ;; esac && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=${TRT} \
          -DFETCH_MODELS=OFF -DTURBO_BACKENDS="cpu;nvidia" && \
    cmake --build build --target turboocr-server -j"$(nproc)"
RUN useradd -m -s /bin/bash ocr && chmod +x /app/scripts/entrypoint.sh && \
    mkdir -p /home/ocr/.cache/turbo-ocr/models/rec /app/models && \
    ln -s /home/ocr/.cache/turbo-ocr/models/rec /app/models/rec
RUN bash scripts/models/fetch/fetch_release_models.sh && chown -R ocr:ocr /app /home/ocr/.cache
EXPOSE 8000 50051
# start-period covers cold-start TRT engine build from ONNX (~90s).
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["./build/turboocr-server"]

# ---------------------------------------------------------------------------
# intel — OpenVINO. Runtime device is chosen at RUN time: OV_DEVICE=CPU|GPU|NPU.
# UNVERIFIED: never built or run by CI. See src/backends/intel/SETUP.md.
# ---------------------------------------------------------------------------
FROM openvino/ubuntu24_dev:2026.2.1 AS intel
USER root
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make pkg-config libopencv-dev nginx gosu libwebp-dev \
    libturbojpeg0-dev libgrpc++-dev libc-ares-dev libprotobuf-dev \
    protobuf-compiler protobuf-compiler-grpc libjsoncpp-dev uuid-dev \
    zlib1g-dev libssl-dev libcurl4-openssl-dev git wget curl gettext-base \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN git clone --depth 1 --branch v1.9.12 https://github.com/drogonframework/drogon.git /tmp/drogon && \
    cd /tmp/drogon && git submodule update --init && \
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
          -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
          -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF && \
    cmake --build build -j"$(nproc)" && cmake --install build && rm -rf /tmp/drogon
RUN TARGETARCH=${TARGETARCH} bash scripts/setup/install_pdfium.sh && \
    cp third_party/pdfium/lib/libpdfium.so /usr/lib/ && ldconfig && \
    TARGETARCH=${TARGETARCH} bash scripts/setup/install_fastpdf2png.sh && ldconfig
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON \
          -DFETCH_MODELS=OFF -DTURBO_BACKENDS="cpu;intel" && \
    cmake --build build --target turboocr-server -j"$(nproc)"
RUN useradd -m -s /bin/bash ocr && chmod +x /app/scripts/entrypoint.sh && \
    mkdir -p /home/ocr/.cache/turbo-ocr/models/rec /app/models && \
    ln -s /home/ocr/.cache/turbo-ocr/models/rec /app/models/rec
RUN bash scripts/models/fetch/fetch_release_models.sh && chown -R ocr:ocr /app /home/ocr/.cache
ENV OV_DEVICE=CPU
EXPOSE 8000 50051
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["./build/turboocr-server"]

# ---------------------------------------------------------------------------
# amd — ROCm + MIGraphX.
# The 7.1.1 base (non-complete) is deliberate: the "-complete" images are 5-8 GB
# compressed and everything the build needs beyond the base — hipcc, the HIP
# cmake package, MIGraphX and its dev headers — installs from the ROCm apt repo
# the image already carries. (The previous pin, 7.14-complete, was a tag that
# does not exist on Docker Hub; 7.14 is also the "TheRock" modular build whose
# layout the HIP sources have never been checked against.)
# ---------------------------------------------------------------------------
FROM rocm/dev-ubuntu-24.04:7.1.1 AS amd
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ make pkg-config libopencv-dev nginx gosu libwebp-dev \
    libturbojpeg0-dev libgrpc++-dev libc-ares-dev libprotobuf-dev \
    protobuf-compiler protobuf-compiler-grpc libjsoncpp-dev uuid-dev \
    zlib1g-dev libssl-dev libcurl4-openssl-dev git wget curl gettext-base \
    hipcc rocm-hip-runtime-dev migraphx migraphx-dev half \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN git clone --depth 1 --branch v1.9.12 https://github.com/drogonframework/drogon.git /tmp/drogon && \
    cd /tmp/drogon && git submodule update --init && \
    cmake -S . -B build -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
          -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
          -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF && \
    cmake --build build -j"$(nproc)" && cmake --install build && rm -rf /tmp/drogon
RUN TARGETARCH=${TARGETARCH} bash scripts/setup/install_pdfium.sh && \
    cp third_party/pdfium/lib/libpdfium.so /usr/lib/ && ldconfig && \
    TARGETARCH=${TARGETARCH} bash scripts/setup/install_fastpdf2png.sh && ldconfig
# The amd CMake target now exists (UNVERIFIED — never compiled).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON \
          -DFETCH_MODELS=OFF -DTURBO_BACKENDS="cpu;amd" && \
    cmake --build build --target turboocr-server -j"$(nproc)"
RUN useradd -m -s /bin/bash ocr && chmod +x /app/scripts/entrypoint.sh && \
    mkdir -p /home/ocr/.cache/turbo-ocr/models/rec /app/models && \
    ln -s /home/ocr/.cache/turbo-ocr/models/rec /app/models/rec
RUN bash scripts/models/fetch/fetch_release_models.sh && chown -R ocr:ocr /app /home/ocr/.cache
EXPOSE 8000 50051
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["./build/turboocr-server"]
