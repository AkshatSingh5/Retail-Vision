from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.config import STORE_ADDRESS, STORE_NAME
from backend.app.utils.money import money_json


def _rupee(value) -> str:
    amount = money_json(value)
    if amount is None:
        amount = 0
    return f"Rs {amount}"


def build_invoice_pdf(transaction, items: list, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = transaction.created_at
    if created.tzinfo is not None:
        local = created.astimezone()
    else:
        local = created
    date_text = local.strftime("%d %b %Y")
    time_text = local.strftime("%H:%M:%S")

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "InvoiceTitle",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1b2430"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "InvoiceSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#5c6570"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    meta = ParagraphStyle(
        "InvoiceMeta",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1b2430"),
        leading=14,
    )
    right = ParagraphStyle("InvoiceRight", parent=meta, alignment=TA_RIGHT)
    footer = ParagraphStyle(
        "InvoiceFooter",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#5c6570"),
        alignment=TA_CENTER,
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Invoice {transaction.invoice_number}",
        author=STORE_NAME,
    )

    story = [
        Paragraph(STORE_NAME, title),
        Paragraph(STORE_ADDRESS, subtitle),
        Table(
            [
                [
                    Paragraph(f"<b>Invoice</b> {transaction.invoice_number}", meta),
                    Paragraph(f"<b>Date</b> {date_text}<br/><b>Time</b> {time_text}", right),
                ]
            ],
            colWidths=[100 * mm, 70 * mm],
        ),
        Spacer(1, 8 * mm),
    ]

    header = ["#", "Product", "SKU", "Weight", "Qty", "Unit Price", "Tax", "Total"]
    rows: list[list] = [header]
    for index, item in enumerate(items, start=1):
        weight_val = getattr(item, "weight", None)
        weight_str = str(weight_val) if weight_val else "-"
        rows.append(
            [
                str(index),
                str(item.name),
                str(item.sku),
                weight_str,
                str(item.quantity),
                _rupee(item.unit_price),
                _rupee(item.tax),
                _rupee(item.total),
            ]
        )

    table = Table(rows, colWidths=[10 * mm, 48 * mm, 24 * mm, 20 * mm, 14 * mm, 24 * mm, 17 * mm, 17 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2430")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (1, 0), (2, -1), "LEFT"),
                ("ALIGN", (3, 0), (3, -1), "CENTER"),
                ("ALIGN", (4, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbe3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8 * mm))

    totals = [
        ["Subtotal", _rupee(transaction.subtotal)],
        [f"Tax / GST", _rupee(transaction.tax)],
        [f"Discount ({money_json(transaction.discount_percent)}%)", _rupee(transaction.discount)],
        ["Grand Total", _rupee(transaction.grand_total)],
    ]
    totals_table = Table(totals, colWidths=[40 * mm, 30 * mm], hAlign="RIGHT")
    totals_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8c547")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(totals_table)
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph("Thank you for shopping with Retail Vision.", footer))
    story.append(Paragraph("This invoice was generated automatically from the live cart.", footer))
    document.build(story)
    return output_path
