# `src/pipeline/`

**ONE OCR orchestration, plus the policy and the job lifecycle around it.**

| path | what |
|---|---|
| [`unified/`](unified) | the orchestration. Device-agnostic, written entirely against the `include/turbo_ocr/backend/` seam, so it runs every backend (cpu, apple, intel, amd, nvidia) unchanged. `UnifiedOcrPipeline` is ONE class across four TUs — single-image path, dispatch, batch, stage bootstrap — beside the replica pool, the detection batcher and the staging ring. Built into `turbo_ocr_pipeline`, in **both** configures. |
| [`job/`](job) | the multi-page **PDF job**: render → per-page OCR → assemble, including the streaming hooks `/ocr/stream` and gRPC `RecognizeStream` fire from. Named for the RULE (a job lifecycle), not the medium — `src/pdf/` already owns the medium, and two directories called `pdf` would have said nothing about which was which. |
| [`router/`](router) | CUA routing: which recognizer a region goes to. Device-neutral. |
| `ocr_pipeline_detail.cpp` | **shared, device-free** result policy: text-degraded / dropped-crop accounting, the combine step, table-region adjust. Header at `include/turbo_ocr/pipeline/ocr_pipeline_detail.h`, compiled into `turbo_ocr_common`. |
| `finalize_deferred.cpp` | **shared, device-free** finalization of the deferred VLM crop futures. Declared in `include/turbo_ocr/pipeline/pipeline_result.h`. |

Header-only peer: `include/turbo_ocr/pipeline/pool_sizing.h` — the pipeline-pool
auto-sizing policy. It is arithmetic over two device-memory numbers, which is why
it is testable on the CPU build, and it is consumed by a backend answering
`BackendCaps::recommended_pool_size`. It used to live under
`service/server/bootstrap/`, which made it the only edge from a vendor arm up
into the transport layer.

## There used to be two

A second, CUDA-native orchestration lived at `src/pipeline/cuda/`: `cudaStream_t`
and `cudaEvent_t` woven through the control flow, nvJPEG, and the concrete
TensorRT-backed model classes, with no seam. It is **gone** — and with it the
reason this README used to open by explaining how to tell the two apart.

Worth stating rather than quietly deleting, because the duplication it caused is
the failure this directory is now arranged to prevent: every stage existed twice,
and a fix applied to one orchestration silently did not reach the other. The
NVIDIA path now runs through the same code as every other backend.

**Generic policy is shared, never per backend.** If you find yourself copying a
helper out of `unified/` to specialise it for one device, it belongs in
`ocr_pipeline_detail.{h,cpp}` behind a parameter instead. Every copy this tree
has grown was found later, having drifted.

## Device-free means device-free

Nothing under `src/pipeline/` may name `cudaStream_t`, `MTLBuffer`,
`hipStream_t` or a Level-Zero handle. The vocabulary is `ImageView`,
`DeviceQueue` and the stage interfaces; anything a device has to do reaches it
through `include/turbo_ocr/backend/`. `tools/checks/architecture.sh` holds the
half of that rule a compiler cannot state.
