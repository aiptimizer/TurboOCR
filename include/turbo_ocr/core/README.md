# `include/turbo_ocr/core/` — domain vocabulary

**The rule: types every layer may include, which include nothing back.**

Header-only. No transport, no device, no orchestration. If a header here ever
needs Drogon, gRPC, protobuf, CUDA, Metal, OpenVINO or OpenCV, it does not
belong here.

## `core/` vs `base/`

Both are leaf layers, and the split is deliberate:

| | holds | example |
|---|---|---|
| `base/` | **language and platform** utilities — nothing OCR-specific | `logger.h`, `env_utils.h`, `geometry/box.h`, `uuid.h` |
| `core/` | **OCR domain vocabulary** — the nouns this product is about | `layout_types.h`, `router_types.h`, `capability.h` |

Same shape as LLVM's `Support/` vs `IR/`. If you cannot say which one a new
header belongs in, it is probably not a leaf type at all.

## Why this directory exists

`include/turbo_ocr/backend/` is the device seam — the architectural floor that
all five vendor arms implement. It was reaching **upward** into three higher
layers:

```
backend/backend.h:32,33   -> service/capability/, service/server/   (the ceiling)
backend/kernels.h:61      -> models/detection/db_post_config.h
backend/stages.h:36       -> models/layout/layout_types.h
backend/table_recognizer.h:32 -> pipeline/router/router_types.h
```

So the "foundation" depended on the serving layer, the model layer and the
orchestration layer at once — and every vendor arm inherited those edges.

None of it was a real dependency. `service_fns.h` is 31 lines of `std::function`
typedefs with zero transport includes; `capability.h` is STL plus one `.def`.
They lived under `service/` because their **C++ namespace** is
`turbo_ocr::server` — the tree was carved by namespace, not by dependency.

Moving the transport-free vocabulary here removed 13 of the 14 upward edges into
`service/`. The one that remains is a genuine violation, not misfiling:
`src/backends/nvidia/backend/cuda_backend.cpp` -> `service/server/cuda/stages_gpu.h`,
which is CUDA-only and therefore deferred (nothing can compile that arm to
verify a fix — see `docs/notes/production-readiness-2026-07.md` §2.4).

## What deliberately did NOT move

`service/capability/proto_capability_bridge.h` stays put. It includes
`<google/protobuf/descriptor.h>` and `<google/protobuf/message.h>` — genuinely
transport-coupled, and correctly a `service/` concern. Only `capability.h` and
`capability_table.def` moved.

## Adding a header here

Ask: *would a vendor backend, the model layer, the orchestration and the serving
layer all be entitled to include this?* If yes, and it pulls no heavy dependency,
it belongs here. If it is only needed by one layer, put it in that layer.
