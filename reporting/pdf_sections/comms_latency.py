#!/usr/bin/env python3
"""
Communications Latency Analysis section.

Renders:
- Latency metrics table (ground/onboard processing times)
- Latency distribution plot
"""
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import cm

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import style_h1, style_body
from reporting.rendering import create_metrics_table, create_figure_flowable


class CommsLatencySection(Section):
    """Communications latency analysis with metrics and plots."""

    def render(self, context):
        """
        Render communications latency section.

        Args:
            context: RenderContext with data.plugin_data['comms_latency']

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Communications Latency Analysis", style_h1))
        flowables.append(Spacer(1, 0.3 * cm))

        flowables.append(Paragraph(
            "This section analyzes the latency between data generation and processing, "
            "comparing ground and onboard processing times.",
            style_body
        ))
        flowables.append(Spacer(1, 0.5 * cm))

        # Get plugin data
        plugin_df = context.data.plugin_data['comms_latency']

        # Render metrics table
        flowables.extend(
            create_metrics_table(
                plugin_df,
                title="Latency Statistics",
                round_decimals=2,
            )
        )
        flowables.append(Spacer(1, 0.5 * cm))

        # Render latency plot
        latency_plot = context.chart_dir / "latency_plot.png"
        flowables.extend(
            create_figure_flowable(
                latency_plot,
                caption=(
                    "Figure: Distribution of ground and onboard processing latency. "
                    "Lower latency indicates faster data processing."
                )
            )
        )

        return flowables

    def get_bookmark_title(self):
        return "Communications Latency Analysis"


# ============================================================================
# AUTO-REGISTER
# ============================================================================
SECTION_REGISTRY.register(
    CommsLatencySection(
        SectionConfig(
            name="comms_latency",
            title="Communications Latency Analysis",
            order=17,
            enabled=True,
            page_break_before=True,
        )
    )
)