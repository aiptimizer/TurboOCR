"""Process a folder of PDFs concurrently, bounded by an asyncio.Semaphore."""

import asyncio
from pathlib import Path

from turboocr import AsyncClient, render_to_markdown

FOLDER = Path(__file__).parent / "sample"
OUT_DIR = Path("/tmp/turboocr-folder-pipeline")


async def process(client: AsyncClient, sem: asyncio.Semaphore, pdf: Path) -> tuple[Path, int]:
    async with sem:
        response = await client.recognize_pdf(pdf, include_blocks=True)
    markdown = render_to_markdown(response).markdown
    out = OUT_DIR / pdf.with_suffix(".md").name
    out.write_text(markdown)
    return out, len(markdown)


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(FOLDER.glob("*.pdf"))
    sem = asyncio.Semaphore(4)
    async with AsyncClient() as client:
        results = await asyncio.gather(*(process(client, sem, p) for p in pdfs))
    for out, chars in results:
        print(f"{out}: {chars:,} chars")


asyncio.run(main())

# Output:
# /tmp/turboocr-folder-pipeline/acme_invoice.md: 1,379 chars
