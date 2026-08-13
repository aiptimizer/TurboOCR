"""recognize_batch: OCR multiple images in a single HTTP call."""

from pathlib import Path

from turboocr import Client

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

client = Client()
batch = client.recognize_batch([IMAGE, IMAGE, IMAGE], include_blocks=True)

for i, (resp, err) in enumerate(zip(batch.batch_results, batch.errors, strict=True)):
    if err is not None:
        print(f"[{i}] error: {err}")
    else:
        print(f"[{i}] ok: {len(resp.results)} items, {len(resp.blocks)} blocks")

# Output:
# [0] ok: 71 items, 12 blocks
# [1] ok: 71 items, 12 blocks
# [2] ok: 71 items, 12 blocks
