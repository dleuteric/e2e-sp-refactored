#!/usr/bin/env python3
"""
Metrics Summary section.

Renders all computed metrics as formatted tables:
- Filter performance (KF errors)
- Triangulation performance (GPM errors)
- Geometry statistics
- Covariance statistics
- Data age statistics
"""
from reportlab.platypus import Paragraph, Spacer, PageBreak

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import style_h1, style_h2
from reporting.rendering import create_metrics_table

from reportlab.lib.units import cm


class MetricsSummarySection(Section):
    """Metrics summary tables."""

    def render(self, context):
        """
        Render all metrics tables.

        Args:
            context: RenderContext with metrics bundle

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Summary Metrics", style_h1))
        flowables.append(Spacer(1, 0.5 * cm))

        metrics = context.metrics

        # ====================================================================
        # 1. FILTER PERFORMANCE
        # ====================================================================
        if not metrics.kf_errors.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.kf_errors,
                    title="Filter Performance",
                    round_decimals=4,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 2. TRIANGULATION PERFORMANCE
        # ====================================================================
        if not metrics.gpm_errors.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.gpm_errors,
                    title="Triangulation Performance",
                    round_decimals=4,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 3. GEOMETRY STATISTICS
        # ====================================================================
        if not metrics.geometry.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.geometry,
                    title="Geometry & Access Statistics",
                    round_decimals=3,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 4. COVARIANCE STATISTICS
        # ====================================================================
        if not metrics.covariance.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.covariance,
                    title="Covariance Statistics",
                    round_decimals=6,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 5. NIS/NEES DIAGNOSTICS
        # ====================================================================
        if not metrics.nis_nees.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.nis_nees,
                    title="Filter Consistency (NIS/NEES)",
                    round_decimals=4,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        # ====================================================================
        # 6. DATA AGE
        # ====================================================================
        if not metrics.data_age.empty:
            flowables.extend(
                create_metrics_table(
                    metrics.data_age,
                    title="Data Age Statistics",
                    round_decimals=1,
                )
            )
            flowables.append(Spacer(1, 0.8 * cm))

        return flowables

    def get_bookmark_title(self):
        return "Summary Metrics"


# ============================================================================
# AUTO-REGISTER
# ============================================================================
SECTION_REGISTRY.register(
    MetricsSummarySection(
        SectionConfig(
            name="metrics_summary",
            title="Summary Metrics",
            order=20,  # After dashboard
            enabled=True,
            page_break_before=True,
        )
    )
)