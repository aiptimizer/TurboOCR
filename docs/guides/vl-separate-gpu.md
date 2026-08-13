# Deploying PaddleOCR-VL on a separate GPU (external `kind:openai` backend)

The C++ OCR server can route any modality (table / formula) to PaddleOCR-VL running
as a **separate process on its own GPU**, over the OpenAI-compatible HTTP protocol.
This is the throughput lever for formula/table-heavy workloads: it removes VL from the
C++ server's GPU (freeing ~18 GB so `PIPELINE_POOL_SIZE` can grow past the single-GPU
VRAM ceiling), and — combined with the async `/ocr/raw` decouple — the C++ GPU pipeline
never blocks on the VL network call, so both GPUs saturate independently.

No code change is needed: it is a routing-config entry.

## 1. Launch PaddleOCR-VL, pinned to its own GPU

`vllm` (the `.venv-vllm` environment) supports the `PaddleOCRVL` architecture natively —
do **not** use `paddlex_genai_server` (it requires the PaddleX GenAI engine plugin, which
is not installed; vllm logs a *caught* "genai engine plugins not available" traceback and
then falls back to native support). Plain `vllm serve` is the correct path.

```bash
# On the VL host / the GPU you want VL to own (here GPU 1):
CUDA_VISIBLE_DEVICES=1 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PATH="/path/to/vllm-venv/bin:$PATH" \
vllm serve /path/to/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/<snap> \
  --port 8077 \
  --trust-remote-code \
  --served-model-name PaddleOCR-VL \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Notes:
- `--gpu-memory-utilization 0.90` is safe when VL owns the GPU (raise from the ~0.55 used
  when co-located with the C++ server).
- Keep the `.venv-vllm/bin` on `PATH` so the `ninja` JIT used by `torch.compile` resolves
  in the spawned EngineCore subprocess. (`ninja` is already installed in `.venv-vllm/bin`;
  the failure mode when it is missing is a JIT `FileNotFoundError: 'ninja'` during compile.)
- Full CUDA-graph capture is on (we never use `--enforce-eager`), so first-boot compile
  takes a few minutes; the engine is ready when `GET /v1/models` returns 200.

Health check:
```bash
curl -sf http://<vl-host>:8077/v1/models
```

## 2. Point the C++ server at it (routing config)

Table and formula need different prompts/parsers, so define two `kind:openai` backends at
the same `base_url`. Save as `routing.json`:

```json
{
  "backends": {
    "vl_table":   { "kind": "openai", "base_url": "http://<vl-host>:8077",
                    "model": "PaddleOCR-VL", "prompt": "Table Recognition:",
                    "parser": "otsl",  "max_tokens": 4096, "timeout_s": 60 },
    "vl_formula": { "kind": "openai", "base_url": "http://<vl-host>:8077",
                    "model": "PaddleOCR-VL", "prompt": "Formula Recognition:",
                    "parser": "latex", "max_tokens": 1024, "timeout_s": 60 },
    "slanext_local": { "kind": "local", "engine": "slanext" }
  },
  "routes": { "table": "vl_table", "formula": "vl_formula", "text": "default" }
}
```

- `api_key` is optional and **env-indirection only** (`"api_key": "env:VL_API_KEY"`); never a
  literal. Auth/TLS belong in the fronting gateway, not here.
- To keep the fast local table path and only send formulas to VL: `"routes": { "table":
  "slanext_local", "formula": "vl_formula" }`.

Launch the C++ server with the config:
```bash
TURBO_ROUTING_CONFIG=/path/to/routing.json \
LD_LIBRARY_PATH=/path/to/TensorRT/lib \
./build/turboocr-server   # + your usual server flags
```

Verify the wiring (no secrets are emitted — names + kinds only):
```bash
curl -s http://127.0.0.1:8080/capabilities | jq .routing
# { "routes": {"table":"vl_table","formula":"vl_formula","text":"default"},
#   "backends": {"vl_table":{"kind":"openai"}, "vl_formula":{"kind":"openai"},
#                "slanext_local":{"kind":"local"}} }
```

A `/ocr/raw` request on a document with tables/formulas then returns VL-recognized
`tables[].html` and `formulas[].latex`, routed through the external endpoint.

## 3. Why this is the throughput lever

On a single shared 32 GB GPU, VL (~18 GB) + the C++ engines (~12 GB) leave no room to grow
`PIPELINE_POOL_SIZE` (it OOMs at pool≥3 with medium text, or pool≥8 otherwise), and the
GPU is contended between the two workloads. Giving VL its own GPU:

1. **Frees ~18 GB** on the C++ GPU → raise `PIPELINE_POOL_SIZE` to feed more pages
   concurrently (the VRAM-OOM ceiling is removed).
2. **Decouples the two GPUs** — the async `/ocr/raw` path submits VL crops without blocking
   a C++ GPU pipeline worker (it returns immediately; the futures finalize off the GPU
   worker). The C++ GPU stays busy on det/rec/layout/table while the VL GPU runs the VLM.

Measured decouple effect (mock endpoint, 3 s latency, pool=2): 2.4 pages/s vs the 0.67
pages/s a blocked worker would cap at — **3.6×**, bounded by the crop pool rather than the
GPU workers.

## Single-GPU caveat

With only one GPU, you can still run VL as a separate **process** (split
`--gpu-memory-utilization` between VL and the C++ server) and route via `kind:openai`. You
get the **async-decouple** benefit (the GPU pipeline no longer blocks on the VL HTTP wait),
but **not** the VRAM win — both still share the 32 GB, so `PIPELINE_POOL_SIZE` stays
constrained. The full benefit needs a second GPU (or a second host).
