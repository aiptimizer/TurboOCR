"""Customize markdown rendering with MarkdownStyle.

`MarkdownStyle` maps each server layout label (e.g. `paragraph_title`,
`table`) to a `NodeKind` and an optional `level`. The default maps
`paragraph_title` → `NodeKind.heading` at level 2 (rendered as `##`). We
override it to level 1 (`#`) and swap the figure renderer.
"""

from pathlib import Path

from turboocr import Client, MarkdownStyle, NodeKind, render_to_markdown

IMAGE = Path(__file__).parent / "sample" / "acme_invoice.png"

style = MarkdownStyle()
style.register("paragraph_title", NodeKind.heading, level=1)  # default was level=2
style.register_renderer(NodeKind.figure, lambda n: f"<!-- figure: {n.text} -->")

client = Client()
response = client.recognize_image(
    IMAGE, layout=True, reading_order=True, include_blocks=True
)

doc = render_to_markdown(response, style=style)
print(doc.markdown[:500])

# Output:
# # ACME Corporation
#
# 123 Roadrunner Way, Tumbleweed, AZ 86001 invoices@acme.example · +1 (555) 010-0182
#
# Invoice # INV-2026-00482
#
# Issue date 2026-04-15
#
# Due date 2026-05-15
#
# Customer Coyote Logistics Ltd.
#
# Customer # C-1049
#
# # Invoice
#
# # Bill to
#
# Coyote Logistics Ltd.
# Attn: Accounts Payable 742 Mesa Drive, Suite 4B Flagstaff, AZ 86004
#
# # Line items
#
# ```
# # Description Unit price Amount Qty
# 1 Industrial-grade Rocket Skates (model RS-9) $249.00 $2,988.00 12
# 4 $89.50 $358.00 2 Anvil, 100 Ib, painted
