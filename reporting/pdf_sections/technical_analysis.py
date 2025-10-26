#!/usr/bin/env python3
"""
Technical Analysis section.

Renders technical plots:
- Timeseries errors (KF vs truth)
- 3D trajectory overlay
- Contact analysis
- Geometry timeline
"""
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import cm

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_section_registry import Section, SectionConfig, SECTION_REGISTRY
from reporting.core.pdf_styles import style_h1, style_h2, style_body
from reporting.rendering import create_figure_flowable


class TechnicalAnalysisSection(Section):
    """Technical analysis plots."""

    def render(self, context):
        """
        Render all technical plots.

        Args:
            context: RenderContext with chart_dir

        Returns:
            List of Flowables
        """
        flowables = []

        # Section heading
        flowables.append(Paragraph("Technical Analysis", style_h1))
        flowables.append(Spacer(1, 0.3 * cm))

        flowables.append(Paragraph(
            "This section presents detailed technical analysis of tracking performance, "
            "including time-series error evolution, trajectory overlays, and geometry diagnostics.",
            style_body
        ))
        flowables.append(Spacer(1, 0.5 * cm))

        chart_dir = context.chart_dir

        if not chart_dir or not chart_dir.exists():
            flowables.append(Paragraph(
                "<i>No charts available for this report.</i>",
                style_body
            ))
            return flowables

        # ====================================================================
        # 1. TIMESERIES ERRORS
        # ====================================================================
        timeseries_png = chart_dir / "errors_timeseries.png"
        if timeseries_png.exists():
            flowables.append(Paragraph("Filter Performance vs Truth", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    timeseries_png,
                    caption=(
                        "Figure: Time evolution of tracking errors (X, Y, Z, norm) "
                        "and filter consistency diagnostics (NIS/NEES). "
                        "Triangulation errors shown with dashed lines for comparison."
                    )
                )
            )

        # ====================================================================
        # 2. 3D TRAJECTORY OVERLAY
        # ====================================================================
        overlay_png = chart_dir / "overlay_3d.png"
        if overlay_png.exists():
            flowables.append(Paragraph("Trajectory Overlay (3D)", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    overlay_png,
                    caption=(
                        "Figure: 3D visualization of truth trajectory (gray), "
                        "Kalman filter track (blue line), and triangulation estimates (orange points). "
                        "Truth points colored by tracking error magnitude. "
                        "Earth sphere shown for reference (R = 6371 km)."
                    )
                )
            )

        # ====================================================================
        # 3. CONTACT ANALYSIS
        # ====================================================================
        contact_png = chart_dir / "contact_analysis.png"
        if contact_png.exists():
            flowables.append(Paragraph("Satellite Access Windows", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    contact_png,
                    caption=(
                        "Figure: Timeline of satellite-to-target visibility windows. "
                        "Each row represents one satellite, with colored bars indicating "
                        "periods of line-of-sight access. Gaps between bars represent "
                        "occultation or below minimum elevation angle."
                    )
                )
            )

        # ====================================================================
        # 4. GEOMETRY TIMELINE
        # ====================================================================
        geometry_png = chart_dir / "geometry_timeline.png"
        if geometry_png.exists():
            flowables.append(Paragraph("Geometry Quality Metrics", style_h2))
            flowables.append(Spacer(1, 0.2 * cm))

            flowables.extend(
                create_figure_flowable(
                    geometry_png,
                    caption=(
                        "Figure: Evolution of geometry quality over time. "
                        "Top panel shows GDOP (Geometric Dilution of Precision). "
                        "Bottom panel shows number of satellites with line-of-sight. "
                        "Lower GDOP values indicate better geometry."
                    )
                )
            )

        # Fallback if no plots
        if not any([
            timeseries_png.exists(),
            overlay_png.exists(),
            contact_png.exists(),
            geometry_png.exists()
        ]):
            flowables.append(Paragraph(
                "<i>No technical plots generated for this run.</i>",
                style_body
            ))

        return flowables

    def get_bookmark_title(self):
        return "Technical Analysis"


# ============================================================================
# AUTO-REGISTER
# ============================================================================
SECTION_REGISTRY.register(
    TechnicalAnalysisSection(
        SectionConfig(
            name="technical_analysis",
            title="Technical Analysis",
            order=15,  # Between dashboard (10) and metrics (20)
            enabled=True,
            page_break_before=True,
        )
    )
)