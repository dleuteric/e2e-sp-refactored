#!/usr/bin/env python3
"""
Header section for PDF report.

Renders:
- Main title
- Subtitle (Run ID / Target ID)
- Architecture info box
"""
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib import colors

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import (
    style_title, style_subtitle, style_body,
    COLORS, CONTENT_WIDTH
)


class HeaderSection(Section):
    """Header section with title and architecture info."""

    def render(self, context):
        """
        Render header content.

        Args:
            context: RenderContext with run_id, target_id

        Returns:
            List of Flowables
        """
        flowables = []

        # Main title
        flowables.append(Paragraph(
            "Architecture Performance Report",
            style_title
        ))

        # Subtitle
        subtitle_text = f"<b>Target:</b> {context.target_id} — <b>Run:</b> {context.run_id}"
        flowables.append(Paragraph(subtitle_text, style_subtitle))

        flowables.append(Spacer(1, 0.5 * cm))

        # Architecture info box (2-column table)
        info_data = [
            ['Run ID', context.run_id],
            ['Target ID', context.target_id],
            ['Generated', 'Test Mode'],
        ]

        info_table = Table(
            info_data,
            colWidths=[4 * cm, CONTENT_WIDTH - 4 * cm],
        )

        info_table.setStyle(TableStyle([
            # Background
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['background_light']),

            # Borders
            ('BOX', (0, 0), (-1, -1), 1.5, COLORS['primary']),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, COLORS['border']),

            # Text formatting
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), COLORS['text_dark']),
            ('TEXTCOLOR', (1, 0), (1, -1), COLORS['secondary']),

            # Padding
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),

            # Alignment
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        flowables.append(info_table)
        flowables.append(Spacer(1, 1 * cm))

        return flowables

    def get_bookmark_title(self):
        return "Header"


# Auto-register this section
SECTION_REGISTRY.register(
    HeaderSection(
        SectionConfig(
            name="header",
            title="Report Header",
            order=0,  # First section
            enabled=True,
            page_break_before=False,
        )
    )
)