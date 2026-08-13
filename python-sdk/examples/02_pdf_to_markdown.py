"""PDF -> Markdown, two ways.

1. Server-side (recommended, v3.3+): one call to `/ocr/pdf?markdown=1` — the
   server's parallel page pipeline renders faithful Markdown (tables → HTML,
   formulas → LaTeX when loaded, figures embedded as data URIs).
2. Client-side: `recognize_pdf` + `render_to_markdown` — when you want to
   customize the rendering with a `MarkdownStyle`.
"""

from pathlib import Path

from turboocr import Client, render_to_markdown

PDF = Path(__file__).parent / "sample" / "acme_invoice.pdf"

client = Client(timeout=120.0)

# --- 1. server-side, whole document in one call ---
md = client.pdf_markdown(PDF, dpi=150)
print(f"server-side: {len(md)} chars")
print(md[:300])

# per-page shape for chunked / RAG consumers:
pages = client.pdf_markdown(PDF, dpi=150, as_pages=True)
print(f"pages={len(pages.pages)} first_page_chars={len(pages.pages[0].markdown)}")

# --- 2. client-side rendering (customizable via MarkdownStyle) ---
response = client.recognize_pdf(
    PDF, dpi=150, layout=True, reading_order=True, include_blocks=True
)
doc = render_to_markdown(response)
print(f"client-side: pages={len(response.pages)} chars={len(doc.markdown)}")

# Output:
# server-side: 1710 chars
# ## ACME Corporation
# ...
# pages=2 first_page_chars=1391
# client-side: pages=2 chars=1379
