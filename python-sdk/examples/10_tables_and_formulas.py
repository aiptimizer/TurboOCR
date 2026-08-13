"""Real table → HTML and formula → LaTeX recognition (server v3.1+).

Tables and formulas are strict per-request opt-ins: start the server with the
backends loaded, then pass `tables=True` / `formulas=True`:

    docker run --gpus all -p 8000:8000 \
      -e TABLE_BACKEND=slanext -e FORMULA_BACKEND=ppformulanet_s \
      -v trt-cache:/home/ocr/.cache/turbo-ocr ghcr.io/aiptimizer/turboocr:latest

Asking a server without the backend raises `BackendDisabled` (never a silent
empty result) — check `client.capabilities()` first. Without the opt-in, the
SDK still synthesizes region-level `tables` / `formulas` entries from layout
blocks (`text` only, `html`/`latex` stay None).
"""

from pathlib import Path

from turboocr import BackendDisabled, Client

PDF = Path(__file__).parent / "sample" / "acme_invoice.pdf"

client = Client(timeout=120.0)

caps = client.capabilities()
print(f"server: tables={caps.features.tables} formulas={caps.features.formulas}")

try:
    response = client.recognize_pdf(
        PDF, tables=caps.features.tables or None, formulas=caps.features.formulas or None
    )
except BackendDisabled as exc:
    raise SystemExit(f"backend missing on this server: {exc}") from exc

print(
    f"pages={len(response.pages)} "
    f"tables={len(response.tables)} formulas={len(response.formulas)}"
)

for t in response.tables:
    print(f"table aabb={t.bounding_box.aabb} confidence={t.confidence}")
    if t.html:
        print(f"  html: {t.html[:120]}...")
for f in response.formulas:
    print(f"formula aabb={f.bounding_box.aabb} latex={f.latex!r}")

# Output (server with SLANet-Plus loaded):
# server: tables=True formulas=True
# pages=2 tables=1 formulas=0
# table aabb=(128, 890, 1140, 1422) confidence=0.99...
#   html: <html><body><table><tr><td>#</td><td>Description</td><td>Qty</td>...
