"""Build the plain-English ChoiceForge explainer PDF from its Markdown source."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "choiceforge-plain-english-explainer.md"
OUTPUT = ROOT / "output" / "pdf" / "choiceforge-plain-english-explainer.pdf"

NAVY = colors.HexColor("#14283D")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#32A6A1")
PALE = colors.HexColor("#EAF5F4")
SKY = colors.HexColor("#EDF5FA")
GOLD = colors.HexColor("#F2B84B")
INK = colors.HexColor("#1C2732")
MUTED = colors.HexColor("#586875")
GRID = colors.HexColor("#C8D5DD")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str]:
    """Use a readable bundled Windows font, with safe built-in fallbacks."""
    candidates = [
        (
            Path("C:/Windows/Fonts/aptos.ttf"),
            Path("C:/Windows/Fonts/aptosb.ttf"),
            Path("C:/Windows/Fonts/aptosi.ttf"),
            Path("C:/Windows/Fonts/aptosbi.ttf"),
            "Aptos",
        ),
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/ariali.ttf"),
            Path("C:/Windows/Fonts/arialbi.ttf"),
            "Arial",
        ),
    ]
    for regular, bold, italic, bold_italic, family in candidates:
        if all(path.exists() for path in (regular, bold, italic, bold_italic)):
            pdfmetrics.registerFont(TTFont(family, str(regular)))
            pdfmetrics.registerFont(TTFont(f"{family}-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont(f"{family}-Italic", str(italic)))
            pdfmetrics.registerFont(TTFont(f"{family}-BoldItalic", str(bold_italic)))
            pdfmetrics.registerFontFamily(
                family,
                normal=family,
                bold=f"{family}-Bold",
                italic=f"{family}-Italic",
                boldItalic=f"{family}-BoldItalic",
            )
            return family, f"{family}-Bold", f"{family}-Italic", f"{family}-BoldItalic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


FONT, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC = register_fonts()


def inline_markup(text: str) -> str:
    """Convert the small inline Markdown subset used by the explainer."""
    token_pattern = re.compile(r"(`[^`]+`|\[([^\]]+)\]\(([^)]+)\)|\*\*[^*]+\*\*)")
    parts: list[str] = []
    cursor = 0
    for match in token_pattern.finditer(text):
        parts.append(html.escape(text[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            parts.append(f'<font name="Courier">{html.escape(token[1:-1])}</font>')
        elif token.startswith("["):
            label = html.escape((match.group(2) or "").replace("`", ""))
            href = html.escape(match.group(3) or "", quote=True)
            parts.append(f'<link href="{href}" color="#176B87"><u>{label}</u></link>')
        else:
            parts.append(f"<b>{html.escape(token[2:-2])}</b>")
        cursor = match.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def make_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=FONT,
        fontSize=9.4,
        leading=13.4,
        textColor=INK,
        spaceAfter=7,
        allowWidows=0,
        allowOrphans=0,
    )
    styles["lead"] = ParagraphStyle(
        "Lead",
        parent=styles["body"],
        fontSize=11.2,
        leading=16.2,
        textColor=NAVY,
        spaceAfter=10,
    )
    styles["h2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=17,
        leading=20,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=7,
        keepWithNext=True,
    )
    styles["h3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName=FONT_BOLD,
        fontSize=11.4,
        leading=14.2,
        textColor=BLUE,
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["bullet"] = ParagraphStyle(
        "BulletBody",
        parent=styles["body"],
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=2,
    )
    styles["code"] = ParagraphStyle(
        "Code",
        parent=styles["body"],
        fontName="Courier",
        fontSize=8.2,
        leading=11.2,
        leftIndent=10,
        rightIndent=8,
        borderColor=GRID,
        borderWidth=0.7,
        borderPadding=8,
        backColor=colors.HexColor("#F5F8FA"),
        textColor=NAVY,
        spaceBefore=4,
        spaceAfter=9,
    )
    styles["table_head"] = ParagraphStyle(
        "TableHead",
        parent=styles["body"],
        fontName=FONT_BOLD,
        fontSize=7.8,
        leading=9.4,
        textColor=WHITE,
        alignment=TA_LEFT,
    )
    styles["table_cell"] = ParagraphStyle(
        "TableCell",
        parent=styles["body"],
        fontSize=7.8,
        leading=9.8,
        spaceAfter=0,
    )
    styles["cover_kicker"] = ParagraphStyle(
        "CoverKicker",
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        textColor=TEAL,
        tracking=1.1,
        spaceAfter=13,
    )
    styles["cover_title"] = ParagraphStyle(
        "CoverTitle",
        fontName=FONT_BOLD,
        fontSize=28,
        leading=33,
        textColor=NAVY,
        spaceAfter=16,
    )
    styles["cover_sub"] = ParagraphStyle(
        "CoverSub",
        fontName=FONT,
        fontSize=13,
        leading=19,
        textColor=MUTED,
        spaceAfter=18,
    )
    styles["metric_num"] = ParagraphStyle(
        "MetricNum",
        fontName=FONT_BOLD,
        fontSize=17,
        leading=19,
        textColor=BLUE,
        alignment=TA_CENTER,
    )
    styles["metric_label"] = ParagraphStyle(
        "MetricLabel",
        fontName=FONT,
        fontSize=7.7,
        leading=10,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
    styles["small"] = ParagraphStyle(
        "Small",
        parent=styles["body"],
        fontSize=7.8,
        leading=10.5,
        textColor=MUTED,
    )
    return styles


STYLES = make_styles()


class ExplainerDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.72 * inch,
            rightMargin=0.72 * inch,
            topMargin=0.72 * inch,
            bottomMargin=0.65 * inch,
            title="ChoiceForge: A Plain-English Guide",
            author="ChoiceForge project",
            subject="Travel demand modeling and GPU kernels for beginners",
        )
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [PageTemplate(id="AllPages", frames=[body_frame], onPage=draw_page)]
        )


def draw_page(canvas, doc):
    if doc.page == 1:
        draw_cover(canvas, doc)
    else:
        draw_body(canvas, doc)


def draw_cover(canvas, doc):
    width, height = letter
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.22 * inch, width, 0.22 * inch, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, width, 0.18 * inch, fill=1, stroke=0)
    canvas.setFillColor(PALE)
    canvas.circle(width - 0.55 * inch, height - 1.0 * inch, 0.95 * inch, fill=1, stroke=0)
    canvas.setFillColor(SKY)
    canvas.circle(width - 0.1 * inch, height - 2.0 * inch, 0.75 * inch, fill=1, stroke=0)
    canvas.restoreState()


def draw_body(canvas, doc):
    width, height = letter
    canvas.saveState()
    # Use a built-in PDF font for stable page-decoration rendering.
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    page_text = str(doc.page)
    canvas.drawRightString(width - doc.rightMargin, 0.36 * inch, page_text)
    canvas.restoreState()


def cover_story():
    metrics = Table(
        [
            [
                Paragraph("1.348x", STYLES["metric_num"]),
                Paragraph("69.5 s", STYLES["metric_num"]),
                Paragraph("0", STYLES["metric_num"]),
            ],
            [
                Paragraph("median whole-model<br/>speedup", STYLES["metric_label"]),
                Paragraph("median whole-model<br/>seconds saved", STYLES["metric_label"]),
                Paragraph("changed modeled<br/>decision cells", STYLES["metric_label"]),
            ],
        ],
        colWidths=[2.08 * inch] * 3,
        rowHeights=[0.4 * inch, 0.52 * inch],
    )
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.8, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return [
        Spacer(1, 0.75 * inch),
        Paragraph("BEGINNER EXPLAINER", STYLES["cover_kicker"]),
        Paragraph("ChoiceForge", STYLES["cover_title"]),
        Paragraph(
            "How a GPU kernel can accelerate a real travel demand modeling component - from the first idea through a public ActivitySim proof.",
            STYLES["cover_sub"],
        ),
        HRFlowable(width="100%", thickness=3, color=GOLD, spaceBefore=4, spaceAfter=20),
        Paragraph(
            "Written for readers with no background in transportation modeling, probability, or GPU computing.",
            STYLES["lead"],
        ),
        Spacer(1, 0.18 * inch),
        metrics,
        Spacer(1, 0.25 * inch),
        Paragraph(
            "Read left to right: Phase 34 cuts the median complete 34-step public model from 269.1 to 199.6 seconds and changes zero modeled decision cells in three matched pairs. Phases 35 and 36 then extend the trip runtime with exact outputs. Phase 36 replaces a 1.79 GB dense CPU input factory with a 351.8 MB compact packet and device-side generation, while withholding a new speed claim because concurrent GPU load prevents a clean stopwatch comparison. Boundaries, variance, and limits are explained inside.",
            STYLES["small"],
        ),
        Spacer(1, 0.9 * inch),
        Paragraph("ChoiceForge project | Evidence updated August 28, 2026", STYLES["small"]),
        PageBreak(),
    ]


def parse_table(lines: list[str]) -> Table:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    rows = [rows[0]] + rows[2:]
    ncols = len(rows[0])
    usable = 7.06 * inch
    first = usable * (0.28 if ncols <= 3 else 0.24)
    remaining = (usable - first) / max(1, ncols - 1)
    widths = [first] + [remaining] * (ncols - 1)
    formatted = []
    for r_index, row in enumerate(rows):
        style = STYLES["table_head"] if r_index == 0 else STYLES["table_cell"]
        formatted.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BACKGROUND", (0, 1), (-1, -1), WHITE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SKY]),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def markdown_story(text: str):
    lines = text.splitlines()
    story = []
    paragraph: list[str] = []
    i = 1  # The cover replaces the Markdown H1.
    first_body_paragraph = True

    def flush_paragraph():
        nonlocal paragraph, first_body_paragraph
        if paragraph:
            joined = " ".join(item.strip() for item in paragraph)
            style = STYLES["lead"] if first_body_paragraph else STYLES["body"]
            story.append(Paragraph(inline_markup(joined), style))
            paragraph = []
            first_body_paragraph = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(html.escape(lines[i]).replace(" ", "&nbsp;"))
                i += 1
            story.append(Paragraph("<br/>".join(code_lines), STYLES["code"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            if (
                stripped.startswith("## 8. A negative result")
                or stripped.startswith("## 22. Three different meanings")
                or stripped.startswith("## 26. What must be done next")
                or stripped.startswith("## 27. Phase 16")
                or stripped.startswith("## 32. The practical path")
                or stripped.startswith("## 76. Phase 28")
                or stripped.startswith("## 81. Phase 29")
                or stripped.startswith("## 86. Phase 30")
                or stripped.startswith("## 92. Phase 31")
            ):
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(stripped[3:]), STYLES["h2"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            if (
                stripped.startswith("### Phase 5:")
                or stripped.startswith("### Phase 7:")
                or stripped.startswith("### Phase 8:")
            ):
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(stripped[4:]), STYLES["h3"]))
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1].strip().lstrip("|")):
            flush_paragraph()
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table = parse_table(table_lines)
            if len(table_lines) <= 8 and story and isinstance(story[-1], Paragraph):
                intro = story.pop()
                story.append(KeepTogether([intro, table]))
            else:
                story.append(table)
            story.append(Spacer(1, 8))
            continue
        if re.match(r"^-\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^-\s+", lines[i].strip()):
                item_text = re.sub(r"^-\s+", "", lines[i].strip())
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^[-\d]+\.?(?:\s+)", lines[i].strip())
                    and lines[i][:1].isspace()
                ):
                    item_text += " " + lines[i].strip()
                    i += 1
                items.append(ListItem(Paragraph(inline_markup(item_text), STYLES["bullet"]), leftIndent=11))
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=17,
                    bulletFontName=FONT_BOLD,
                    bulletFontSize=6,
                    bulletColor=TEAL,
                    spaceAfter=7,
                )
            )
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^\d+\.\s+", lines[i].strip())
                    and lines[i][:1].isspace()
                ):
                    item_text += " " + lines[i].strip()
                    i += 1
                items.append(ListItem(Paragraph(inline_markup(item_text), STYLES["bullet"]), leftIndent=16))
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    leftIndent=22,
                    bulletFontName=FONT_BOLD,
                    bulletFontSize=8,
                    bulletColor=BLUE,
                    spaceAfter=7,
                )
            )
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return story


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    markdown = SOURCE.read_text(encoding="utf-8")
    if "\u2011" in markdown or "\u2013" in markdown or "\u2014" in markdown:
        raise ValueError("Use ASCII hyphens in PDF source text.")
    doc = ExplainerDocTemplate(str(OUTPUT))
    story = cover_story() + markdown_story(markdown)
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
