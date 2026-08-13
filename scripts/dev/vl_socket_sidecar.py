#!/usr/bin/env python3
"""In-process PaddleOCR-VL-1.6 sidecar — NO HTTP server.

vLLM's AsyncLLM engine runs inside this process (continuous batching preserved);
the frontend is a raw Unix-domain socket instead of the OpenAI HTTP server.
turboocr sends raw BGR crop bytes (no PNG / no base64 / no JSON), we hand the
pixels to vLLM as a numpy array (vLLM accepts np.ndarray/torch.Tensor images
directly — HfImageItem), and stream back the recognized text.

Only the table + formula tasks are served.

Wire protocol (length-prefixed, little-endian), one request per framed message:
  req : u32 req_id | u8 task(0=formula,1=table) | u32 H | u32 W | u8[H*W*3] BGR
  rep : u32 req_id | u32 status(0=ok) | u32 len | utf8[len]

Concurrency: each request is its own asyncio task → AsyncLLM coalesces them
into continuous batches exactly like the HTTP server did.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import asyncio, struct, sys, itertools
import numpy as np
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser("~/.cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/04f4914a147b84d7dabe3490dcf67152d2e097a9")
SOCK = os.environ.get("VL_SOCK", "/tmp/vl_sidecar.sock")

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

# Chat-template-equivalent wrapping (see chat_template.jinja). vLLM's
# PaddleOCRVL processor expands <|IMAGE_PLACEHOLDER|> to the right token count.
PROMPTS = {0: "Formula Recognition:", 1: "Table Recognition:"}
MAXTOK  = {0: 1024, 1: 4096}
def wrap(task):
    return ("<|begin_of_sentence|>User: "
            "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
            f"{PROMPTS[task]}\nAssistant:\n")

engine = None
_ids = itertools.count(1)

async def recvn(reader, n):
    buf = await reader.readexactly(n)
    return buf

async def run_one(task, img_rgb):
    prompt = {
        "prompt": wrap(task),
        "multi_modal_data": {"image": img_rgb},
        "mm_processor_kwargs": {"min_pixels": 112896, "max_pixels": 1003520},
    }
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK[task])
    rid = f"r{next(_ids)}"
    out_text = ""
    async for out in engine.generate(prompt=prompt, sampling_params=sp, request_id=rid):
        out_text = out.outputs[0].text
    return out_text

async def handle(reader, writer):
    try:
        while True:
            hdr = await recvn(reader, 13)  # u32 req_id, u8 task, u32 H, u32 W
            req_id, task, H, W = struct.unpack("<IBII", hdr)
            raw = await recvn(reader, H * W * 3)
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
            rgb = bgr[:, :, ::-1]  # BGR->RGB
            pil = Image.fromarray(np.ascontiguousarray(rgb))  # match HTTP/PNG path
            try:
                text = await run_one(task, pil)
                payload = text.encode("utf-8")
                writer.write(struct.pack("<III", req_id, 0, len(payload)) + payload)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}".encode()
                writer.write(struct.pack("<III", req_id, 1, len(msg)) + msg)
            await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    finally:
        writer.close()

async def main():
    global engine
    args = AsyncEngineArgs(
        model=MODEL, served_model_name="PaddleOCR-VL-1.6",
        gpu_memory_utilization=0.66, max_model_len=8192, max_num_seqs=512,
        dtype="bfloat16", trust_remote_code=True,
    )
    engine = AsyncLLM.from_engine_args(args)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    server = await asyncio.start_unix_server(handle, path=SOCK)
    print(f"[vl-sidecar] READY on {SOCK}", flush=True)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
