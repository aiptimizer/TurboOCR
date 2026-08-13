"""Image OCR with layout + reading order. Bounding boxes for every text item and block."""

from pathlib import Path

from turboocr import Client

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

client = Client()
response = client.recognize_image(
    IMAGE, layout=True, reading_order=True, include_blocks=True
)

print(f"{len(response.results)} text items, {len(response.blocks)} blocks")

print("\n--- text items (first 3) — per-word OCR with bbox ---")
for item in response.results[:3]:
    x0, y0, x1, y1 = item.bounding_box.aabb
    print(f"  '{item.text}' (conf={item.confidence:.2f}) bbox=({x0},{y0})-({x1},{y1})")

print("\n--- blocks (first 3) — reading-order-grouped paragraphs with bbox ---")
for b in response.blocks[:3]:
    x0, y0, x1, y1 = b.bounding_box.aabb
    print(f"  [{b.id}] {b.class_name} bbox=({x0},{y0})-({x1},{y1})")
    print(f"      {b.content[:80]!r}")

# Output:
# 71 text items, 12 blocks
#
# --- text items (first 3) — per-word OCR with bbox ---
#   'ACME Corporation' (conf=1.00) bbox=(121,114)-(452,155)
#   '123 Roadrunner Way, Tumbleweed, AZ 86001' (conf=0.98) bbox=(123,167)-(499,191)
#   'invoices@acme.example · +1 (555) 010-0182' (conf=0.97) bbox=(121,203)-(494,228)
#
# --- blocks (first 3) — reading-order-grouped paragraphs with bbox ---
#   [0] paragraph_title bbox=(121,116)-(451,156)
#       'ACME Corporation'
#   [1] text bbox=(122,168)-(502,231)
#       '123 Roadrunner Way, Tumbleweed, AZ 86001 invoices@acme.example · +1 (555) 010-01'
#   [2] text bbox=(259,355)-(632,381)
#       'Invoice # INV-2026-00482'
