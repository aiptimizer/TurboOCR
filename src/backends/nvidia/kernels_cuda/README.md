# `src/backends/nvidia/kernels_cuda/` — the NVIDIA CUDA kernels + their seam adapter

Hand-written CUDA kernels behind the NVIDIA-native OCR path: connected
components (`ccl_kernels.cu`), jump-flooding (`jfa.cu`), fused preprocess
(`preprocess_kernels.cu`) and reductions (`reduce_kernels.cu`), plus
`cuda_kernels.{h,cpp}` — the `backend::IKernels` adapter that translates seam
vocabulary into calls on them. Host-side signatures are in `kernels_cuda.h`;
device-only helpers in `kernels_internal.cuh`.

## Where they compile

The four `.cu` files are named **once**, in `TURBO_GPU_CU_SRCS` in the root
`CMakeLists.txt`, and compiled by `nvcc` into `turbo_ocr_gpu`.

`turbo_ocr_backend_nvidia` globs `src/backends/nvidia/*.cu` as a landing pad for
a genuinely new vendor kernel, and **subtracts `TURBO_GPU_CU_SRCS`**. It has to:
it PUBLIC-links `turbo_ocr_gpu`, so letting the glob take these four as well puts
identical objects in two archives, and any target linking both fails with
multiple definitions. `turbo_ocr_tests` links both; `turboocr-server` does not,
which is why that failure showed up only in the test binary.

Removing them from `turbo_ocr_gpu` and relying on the glob is not the
alternative: `TURBO_BACKENDS` defaults to `""` on the GPU configure and
`docker/Dockerfile`'s `nvidia` target passes none, so `turbo_ocr_backend_nvidia`
is often
never created at all and the shipping server would lose every kernel.

## Why this directory is now the right home

It was not always. These kernels used to live outside the vendor arm, in a
top-level `src/cuda/`, and a long argument was recorded here for keeping them
there: their header had **11 consumers outside `src/backends/`**, nine of them in
the analysis layer that vendor arms sit *above*. That made them a shared runtime
primitive rather than a leaf, and moving them in would have inverted the
layering — the analysis layer, and a public header, depending on a vendor arm.

That premise no longer holds. Routing those consumers through
`backend::IKernels` removed every one of them: **no file outside
`src/backends/` includes a CUDA kernel header today** except two tests
(`tests/cpp/pipeline/test_gpu_safety.cpp`,
`tests/cpp/backends/test_db_postprocess_parity.cpp`), which reach in
deliberately to check the device path against the host path. NVIDIA is now a
leaf like every other vendor, and the `backends/<vendor>/kernels_<toolchain>/`
convention that `docs/contributing/adding-a-backend.md` teaches and
`tools/new_backend.py` scaffolds applies to it unchanged.

The argument is kept rather than deleted because the *reasoning* is what stops
someone moving the kernels back out on the grounds that they once lived
elsewhere: the fan-out is what decided it, and the fan-out is gone.
