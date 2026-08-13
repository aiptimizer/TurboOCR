"""AsyncClient: OCR a list of images concurrently with asyncio.gather."""

import asyncio
from pathlib import Path

from turboocr import AsyncClient

SAMPLE = Path(__file__).parent / "sample"
IMAGES = [SAMPLE / "acme_invoice.png"] * 3


async def main() -> None:
    async with AsyncClient() as client:
        responses = await asyncio.gather(*(client.recognize_image(img) for img in IMAGES))

    for img, resp in zip(IMAGES, responses, strict=True):
        print(f"{img.name}: {len(resp.results)} items")


asyncio.run(main())

# Output:
# acme_invoice.png: 71 items
# acme_invoice.png: 71 items
# acme_invoice.png: 71 items
