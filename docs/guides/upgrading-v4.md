# What changed in v4 (alpha)

v4 is an architectural release: the CUDA-specific pipeline and the portable
CPU pipeline were replaced by **one** orchestration
(`UnifiedOcrPipeline`) running against a device-agnostic backend seam, with
one backend library per vendor. The rebuild was gated on the NVIDIA server's
output staying **byte-identical**, so for existing NVIDIA deployments v4 is
not a breaking upgrade.

**Alpha status:** the NVIDIA backend is shipped and validated. Apple and
Intel are functional and benchmarked but not yet hardened; AMD is not yet
hardware-tested; the Python library's API may still change and is not yet
on PyPI.

## For existing v3 NVIDIA users

Nothing breaks:

- Same HTTP and gRPC API, same request options, same error codes.
- Same environment variables and CLI flags.
- Same Docker image name; TensorRT engine caches remain valid.

If you are still on v2.x, read
[Upgrading to v3 — breaking changes](upgrading-v3.md) first; those renames
and default changes all still apply.

## New in v4

- **Backend selection.** One `turboocr-server` binary serves whichever
  backends were compiled in; `--backend nvidia|apple|intel|amd|cpu|auto` picks at
  startup (`auto` probes the hardware). `GET /capabilities/backend` reports
  what is loaded and what the device can do.
- **Per-vendor builds.** `-DTURBO_BACKENDS="cpu;apple"` (and friends) choose
  which backend libraries compile in; the default is the host platform's
  natural set. See [Install — pick your hardware](../getting-started/install.md).
- **Apple backend (testing).** Detection and warp on Metal + MPSGraph;
  recognition split between the GPU and the Apple Neural Engine (CoreML) —
  native only, no container.
- **Intel backend (testing).** One backend for Intel CPUs, iGPUs, Arc and
  NPUs through OpenVINO. `--backend intel` is required — without it the
  server runs the ONNX Runtime CPU path.
- **Python library (testing).** The C++ pipeline behind a native wheel with
  a built-in replica pool — `OCR(replicas=3)` reaches server-class
  throughput from one object.
- **Engine modes.** `native`/`ultra` (vendor graph engine) vs `onnx`/`fast`
  (ONNX Runtime with the vendor's execution provider), selectable per build
  and per backend.
- **Real shutdown bound.** `SHUTDOWN_GRACE_SECONDS` is now an upper bound:
  when it expires, queued-but-unstarted work is shed (counted on
  `/metrics`) while in-flight requests finish.
