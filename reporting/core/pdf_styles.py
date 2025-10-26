#!/usr/bin/env python3
"""
PDF styles and page templates.

Defines:
- Color palette
- Font styles (title, heading, body, KPI labels)
- ParagraphStyle definitions
- Page template with header/footer
"""
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import PageTemplate, Frame, Paragraph
from reportlab.pdfgen import canvas

# ============================================================================
# PAGE DIMENSIONS
# ============================================================================
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 1.5 * cm
MARGIN_RIGHT = 1.5 * cm
MARGIN_TOP = 2.5 * cm
MARGIN_BOTTOM = 2.0 * cm

CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
CONTENT_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# ============================================================================
# COLOR PALETTE
# ============================================================================
COLORS = {
    # Threshold colors (from HTML)
    'excellent': HexColor('#2ecc71'),
    'good': HexColor('#3498db'),
    'marginal': HexColor('#f39c12'),
    'poor': HexColor('#e74c3c'),

    # UI colors
    'primary': HexColor('#3498db'),
    'secondary': HexColor('#7f8c8d'),
    'background_light': HexColor('#ecf0f1'),
    'border': HexColor('#bdc3c7'),
    'text_dark': HexColor('#2c3e50'),
    'text_light': HexColor('#95a5a6'),
}

# ============================================================================
# FONT DEFINITIONS
# ============================================================================
FONT_TITLE = 'Helvetica-Bold'
FONT_HEADING = 'Helvetica-Bold'
FONT_BODY = 'Helvetica'
FONT_MONO = 'Courier'

# ============================================================================
# PARAGRAPH STYLES
# ============================================================================
base_styles = getSampleStyleSheet()

# Title (Report main title)
style_title = ParagraphStyle(
    'ReportTitle',
    parent=base_styles['Title'],
    fontName=FONT_TITLE,
    fontSize=24,
    textColor=COLORS['text_dark'],
    spaceAfter=0.3 * cm,
    alignment=TA_LEFT,
)

# Subtitle (Target/Run ID)
style_subtitle = ParagraphStyle(
    'ReportSubtitle',
    parent=base_styles['Normal'],
    fontName=FONT_HEADING,
    fontSize=14,
    textColor=COLORS['secondary'],
    spaceAfter=0.5 * cm,
    alignment=TA_LEFT,
)

# Heading 1 (Section titles)
style_h1 = ParagraphStyle(
    'Heading1',
    parent=base_styles['Heading1'],
    fontName=FONT_HEADING,
    fontSize=18,
    textColor=COLORS['primary'],
    spaceBefore=0.8 * cm,
    spaceAfter=0.4 * cm,
    alignment=TA_LEFT,
)

# Heading 2 (Subsection titles)
style_h2 = ParagraphStyle(
    'Heading2',
    parent=base_styles['Heading2'],
    fontName=FONT_HEADING,
    fontSize=14,
    textColor=COLORS['text_dark'],
    spaceBefore=0.5 * cm,
    spaceAfter=0.3 * cm,
    alignment=TA_LEFT,
)

# Body text
style_body = ParagraphStyle(
    'Body',
    parent=base_styles['Normal'],
    fontName=FONT_BODY,
    fontSize=10,
    textColor=COLORS['text_dark'],
    leading=14,
    spaceAfter=0.3 * cm,
)

# KPI label (small caps style)
style_kpi_label = ParagraphStyle(
    'KPILabel',
    parent=base_styles['Normal'],
    fontName=FONT_BODY,
    fontSize=8,
    textColor=COLORS['text_light'],
    spaceBefore=0,
    spaceAfter=0.1 * cm,
    alignment=TA_CENTER,
)

# KPI value (large numbers)
style_kpi_value = ParagraphStyle(
    'KPIValue',
    parent=base_styles['Normal'],
    fontName=FONT_HEADING,
    fontSize=28,
    textColor=COLORS['text_dark'],
    spaceBefore=0.1 * cm,
    spaceAfter=0.1 * cm,
    alignment=TA_CENTER,
)

# KPI metric (small descriptor)
style_kpi_metric = ParagraphStyle(
    'KPIMetric',
    parent=base_styles['Normal'],
    fontName=FONT_BODY,
    fontSize=9,
    textColor=COLORS['text_light'],
    spaceBefore=0,
    spaceAfter=0,
    alignment=TA_CENTER,
)

# Caption (figure captions)
style_caption = ParagraphStyle(
    'Caption',
    parent=base_styles['Normal'],
    fontName=FONT_BODY,
    fontSize=9,
    textColor=COLORS['secondary'],
    spaceBefore=0.2 * cm,
    spaceAfter=0.5 * cm,
    alignment=TA_CENTER,
)


# ============================================================================
# PAGE TEMPLATE
# ============================================================================
def create_header_footer(canvas_obj: canvas.Canvas, doc, run_id: str, target_id: str):
    """
    Draw header and footer on each page.

    Args:
        canvas_obj: ReportLab canvas
        doc: Document object
        run_id: Run identifier
        target_id: Target identifier
    """
    canvas_obj.saveState()

    # Header line
    canvas_obj.setStrokeColor(COLORS['border'])
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(
        MARGIN_LEFT,
        PAGE_HEIGHT - MARGIN_TOP + 0.5 * cm,
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - MARGIN_TOP + 0.5 * cm
    )

    # Header text
    canvas_obj.setFont(FONT_BODY, 9)
    canvas_obj.setFillColor(COLORS['secondary'])
    canvas_obj.drawString(
        MARGIN_LEFT,
        PAGE_HEIGHT - MARGIN_TOP + 0.7 * cm,
        f"Architecture Performance Report — {target_id}"
    )
    canvas_obj.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - MARGIN_TOP + 0.7 * cm,
        f"Run: {run_id}"
    )

    # Footer line
    canvas_obj.line(
        MARGIN_LEFT,
        MARGIN_BOTTOM - 0.5 * cm,
        PAGE_WIDTH - MARGIN_RIGHT,
        MARGIN_BOTTOM - 0.5 * cm
    )

    # Footer text (page number)
    canvas_obj.setFont(FONT_BODY, 8)
    canvas_obj.drawCentredString(
        PAGE_WIDTH / 2,
        MARGIN_BOTTOM - 0.8 * cm,
        f"Page {doc.page}"
    )

    canvas_obj.restoreState()


def create_page_template_function(run_id: str, target_id: str):
    """
    Create page template factory function.

    Args:
        run_id: Run identifier
        target_id: Target identifier

    Returns:
        Callable for onFirstPage/onLaterPages
    """

    def _template(canvas_obj, doc):
        create_header_footer(canvas_obj, doc, run_id, target_id)

    return _template


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_color_for_threshold(value: float, threshold_level_color: str) -> HexColor:
    """
    Get color for a threshold level.

    Args:
        value: KPI value
        threshold_level_color: Color hex string

    Returns:
        HexColor object
    """
    return HexColor(threshold_level_color)