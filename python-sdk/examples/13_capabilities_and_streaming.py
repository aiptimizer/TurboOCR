"""Feature discovery + NDJSON streaming (server v3.2+).

`capabilities()` reports what the running server actually loaded — check it
before opting into tables/formulas/autorotate. `stream()` yields per-page
results as each page completes (out of order by design), so a RAG ingester
can start chunking page 1 while page N is still being OCR'd.
"""

from pathlib import Path

from turboocr import Client

PDF = Path(__file__).parent / "sample" / "acme_invoice.pdf"

client = Client(timeout=120.0)

caps = client.capabilities()
print(f"build={caps.build} features={caps.features.model_dump()}")
print(f"pdf modes={caps.pdf.modes} max_pages={caps.pdf.max_pages}")

for event in client.stream(PDF, layout=True):
    if event.event == "meta":
        print(f"streaming {event.pages} pages (mode={event.mode})")
    elif event.event == "page":
        page = event.page
        assert page is not None
        print(f"  page {page.page_index}: {len(page.results)} lines")
    elif event.event == "end":
        print(f"done, failed={event.failed}")

# Output:
# build=gpu features={'layout': True, 'tables': True, 'formulas': True, 'autorotate': True}
# pdf modes=['ocr', 'geometric', 'auto', 'auto_verified'] max_pages=2000
# streaming 2 pages (mode=ocr)
#   page 0: 33 lines
#   page 1: 7 lines
# done, failed=0
