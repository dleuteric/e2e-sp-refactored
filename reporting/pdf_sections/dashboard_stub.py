#!/usr/bin/env python3
"""
Executive Dashboard stub for testing.

Shows:
- Section heading
- Dummy KPI cards (colored boxes)
- Styled paragraphs
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
    style_h1, style_body, style_kpi_label, style_kpi_value, style_kpi_metric,
    COLORS, CONTENT_WIDTH
)


class DashboardStubSection(Section):
    """Stub executive dashboard for testing."""

    def render(self, context):
        """
        Render dashboard stub.

        Args:
            context: RenderContext

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Executive Dashboard", style_h1))

        flowables.append(Paragraph(
            "This is a test section demonstrating the registry pattern. "
            "In the full report, this would contain KPI cards with real metrics.",
            style_body
        ))

        flowables.append(Spacer(1, 0.5 * cm))

        # Dummy KPI cards (3 columns)
        kpi_data = self._create_dummy_kpi_cards()

        kpi_table = Table(
            kpi_data,
            colWidths=[CONTENT_WIDTH / 3] * 3,
        )

        kpi_table.setStyle(TableStyle([
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, COLORS['border']),

            # Padding
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),

            # Alignment
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),

            # Border-left simulation (colored strip)
            ('LINEABOVE', (0, 0), (0, 0), 4, COLORS['excellent']),
            ('LINEABOVE', (1, 0), (1, 0), 4, COLORS['good']),
            ('LINEABOVE', (2, 0), (2, 0), 4, COLORS['marginal']),
        ]))

        flowables.append(kpi_table)
        flowables.append(Spacer(1, 1 * cm))

        return flowables

    def _create_dummy_kpi_cards(self):
        """Create 3 dummy KPI cards with different colors."""
        card1 = [
            Paragraph("TRACKING ACCURACY", style_kpi_label),
            Paragraph("0.15 km", style_kpi_value),
            Paragraph("CEP50 · Excellent", style_kpi_metric),
        ]

        card2 = [
            Paragraph("COVERAGE", style_kpi_label),
            Paragraph("92%", style_kpi_value),
            Paragraph("≥2 sats · Good", style_kpi_metric),
        ]

        card3 = [
            Paragraph("DATA AGE", style_kpi_label),
            Paragraph("850 ms", style_kpi_value),
            Paragraph("p50 latency · Marginal", style_kpi_metric),
        ]

        return [[card1, card2, card3]]

    def get_bookmark_title(self):
        return "Executive Dashboard"


# Auto-register this section
SECTION_REGISTRY.register(
    DashboardStubSection(
        SectionConfig(
            name="dashboard_stub",
            title="Executive Dashboard (Stub)",
            order=10,  # After header
            enabled=True,
            page_break_before=True,  # Start on new page
        )
    )
)