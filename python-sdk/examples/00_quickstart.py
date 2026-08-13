"""30-second tour: sync image OCR."""

from pathlib import Path

from turboocr import Client

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

client = Client()  # reads TURBO_OCR_BASE_URL or defaults to http://localhost:8000

response = client.recognize_image(IMAGE)
print(f"recognized {len(response.results)} text items")
if response.results:
    first = response.results[0]
    print(f"first item: {first.text!r} (confidence={first.confidence:.2f})")

# Output:
# recognized 71 text items
# first item: 'ACME Corporation' (confidence=1.00)
