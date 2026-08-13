"""Generate the example fixtures used by all docs/*.py scripts.

Produces:
  docs/sample/acme_invoice.pdf  — 2-page fictional invoice + terms
  docs/sample/acme_invoice.png  — page 1 rendered to PNG at 144 DPI

Re-run after editing the content. Requires `reportlab` (ships with the core
install) and `pypdfium2` (part of the `[dev]` extra; install with
`pip install -e ".[dev]"` from a clone).
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
PDF_PATH = HERE / "acme_invoice.pdf"
PNG_PATH = HERE / "acme_invoice.png"


def build_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="ACME Corp Invoice INV-2026-00482",
        author="ACME Corporation",
    )
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6, fontSize=10)
    small = ParagraphStyle("small", parent=body, fontSize=9, textColor=colors.grey)

    story = []

    # ─── Page 1: header + bill-to + line items ──────────────────────────
    story.append(Paragraph("ACME Corporation", h1))
    story.append(Paragraph("123 Roadrunner Way, Tumbleweed, AZ 86001", small))
    story.append(Paragraph("invoices@acme.example · +1 (555) 010-0182", small))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Invoice", h2))
    meta_data = [
        ["Invoice #", "INV-2026-00482"],
        ["Issue date", "2026-04-15"],
        ["Due date", "2026-05-15"],
        ["Customer", "Coyote Logistics Ltd."],
        ["Customer #", "C-1049"],
    ]
    meta_table = Table(meta_data, colWidths=[1.5 * inch, 3.5 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Bill to", h2))
    story.append(
        Paragraph(
            "Coyote Logistics Ltd.<br/>"
            "Attn: Accounts Payable<br/>"
            "742 Mesa Drive, Suite 4B<br/>"
            "Flagstaff, AZ 86004",
            body,
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Line items", h2))
    items = [
        ["#", "Description", "Qty", "Unit price", "Amount"],
        ["1", "Industrial-grade Rocket Skates (model RS-9)", "12", "$249.00", "$2,988.00"],
        ["2", "Anvil, 100 lb, painted black", "4", "$89.50", "$358.00"],
        ["3", "Portable Hole, vinyl, 36-inch", "2", "$425.00", "$850.00"],
        ["4", "Earthquake Pills, blister pack of 10", "6", "$32.75", "$196.50"],
        ["5", "Triple-Strength Fortified Leg Muscle Vitamins", "10", "$18.40", "$184.00"],
        ["6", "Dehydrated Boulders, 50-lb sack", "3", "$54.95", "$164.85"],
        ["7", "Giant Magnet, horseshoe, 24-inch", "1", "$1,150.00", "$1,150.00"],
        ["8", "Roadrunner Trap Schematic (revision F)", "1", "$75.00", "$75.00"],
    ]
    line_table = Table(
        items,
        colWidths=[0.4 * inch, 3.6 * inch, 0.6 * inch, 1.0 * inch, 1.1 * inch],
    )
    line_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f8")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#222244")),
            ]
        )
    )
    story.append(line_table)
    story.append(Spacer(1, 0.2 * inch))

    totals_data = [
        ["Subtotal", "$5,966.35"],
        ["Sales tax (8.6%)", "$513.11"],
        ["Total due", "$6,479.46"],
    ]
    totals_table = Table(totals_data, colWidths=[2.0 * inch, 1.1 * inch])
    totals_table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -2), "Helvetica", 10),
                ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 11),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black),
                ("TOPPADDING", (0, -1), (-1, -1), 6),
            ]
        )
    )
    # Right-align the totals block by wrapping it in an outer table
    outer = Table([[None, totals_table]], colWidths=[3.6 * inch, 3.1 * inch])
    outer.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "RIGHT")]))
    story.append(outer)

    # ─── Page 2: terms + signature ──────────────────────────────────────
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Payment terms", h2))
    story.append(
        Paragraph(
            "Net 30 from invoice date. Pay by ACH transfer to "
            "<b>Cliffside Bank</b>, routing 081234567, account 0294-882-118. "
            "Reference the invoice number on the wire.",
            body,
        )
    )
    story.append(Paragraph("Returns and warranty", h2))
    story.append(
        Paragraph(
            "All ACME products carry a 90-day defects-in-materials warranty. "
            "Returns require an RMA number, obtainable by emailing "
            "support@acme.example with the invoice number and item SKU. "
            "Detonatable items are final sale.",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Authorized signature", h2))
    story.append(Spacer(1, 0.6 * inch))
    sig = Table(
        [["________________________", "____________________"]],
        colWidths=[3.0 * inch, 2.0 * inch],
    )
    sig.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 10)]))
    story.append(sig)
    story.append(Paragraph("W. E. Coyote, Sales Director", small))

    doc.build(story)


def render_first_page_to_png(pdf_path: Path, png_path: Path, dpi: int = 144) -> None:
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    bitmap = page.render(scale=dpi / 72.0)
    img = bitmap.to_pil()
    img.save(png_path, format="PNG", optimize=True)
    pdf.close()


def main() -> None:
    build_pdf(PDF_PATH)
    render_first_page_to_png(PDF_PATH, PNG_PATH)
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
    print(f"wrote {PNG_PATH} ({PNG_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
