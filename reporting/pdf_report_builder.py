#!/usr/bin/env python3
"""
PDF Report Builder - Main Orchestrator

Coordinates the entire report generation pipeline:
1. Load data (DataBundle)
2. Compute metrics (MetricsBundle)
3. Extract KPIs
4. Export charts
5. Render sections
6. Build PDF

Usage:
    builder = PDFReportBuilder(run_id, target_id, output_path, cfg)
    pdf_path = builder.build()
"""
from __future__ import annotations
import numpy as np
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Optional

from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.lib.pagesizes import A4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reporting.core.pdf_styles import (
    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM,
    create_page_template_function,
)
from reporting.core.pdf_section_registry import SECTION_REGISTRY, RenderContext
from reporting.core.pdf_kpi_config import extract_all_kpis
from reporting.pipeline.data_loader import load_all_data
from reporting.pipeline.metrics_computer import compute_all_metrics
from reporting.rendering import export_all_report_figures
from reporting.docx_companion import generate_docx_companion
from reporting.pipeline.chart_exporter import ChartExporter
from reporting.pipeline.section_renderer import SectionRenderer

# Import sections (triggers auto-registration)
import reporting.pdf_sections.header
import reporting.pdf_sections.executive_dashboard
import reporting.pdf_sections.metrics_summary

LOGGER = logging.getLogger(__name__)


# ============================================================================
# REPORT BUILDER
# ============================================================================
class PDFReportBuilder:
    """
    Main orchestrator for PDF report generation.

    Coordinates data loading, metrics computation, chart export,
    and section rendering into a single PDF document.

    Attributes:
        run_id: Run identifier
        target_id: Target identifier
        output_path: Output PDF path
        cfg: Configuration dict

    Example:
        >>> builder = PDFReportBuilder(
        ...     run_id="20251020T154731Z",
        ...     target_id="HGV_330",
        ...     output_path=Path("report.pdf"),
        ...     cfg=get_config()
        ... )
        >>> pdf_path = builder.build()
    """

    def __init__(self,
                 run_id: str,
                 target_id: str,
                 output_path: Path,
                 cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize report builder.

        Args:
            run_id: Run identifier
            target_id: Target identifier
            output_path: Output PDF file path
            cfg: Configuration dictionary (optional)
        """
        self.run_id = str(run_id)
        self.target_id = str(target_id)
        self.output_path = Path(output_path)
        self.cfg = cfg or {}

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize document
        self.doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        # Story (list of flowables)
        self.story = []

        # Data containers (populated during build)
        self.data = None
        self.metrics = None
        self.kpis = None
        self.chart_dir = None
        self.figures = {}
        self.docx_path = None


        LOGGER.info("PDFReportBuilder initialized:")
        LOGGER.info("  Run ID:    %s", self.run_id)
        LOGGER.info("  Target ID: %s", self.target_id)
        LOGGER.info("  Output:    %s", self.output_path)

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    def build(self) -> Path:
        """
        Execute complete report generation pipeline.

        Pipeline stages:
        1. Load data
        2. Compute metrics
        3. Extract KPIs
        4. Export charts
        5. Render sections
        6. Build PDF

        Returns:
            Path to generated PDF file

        Raises:
            RuntimeError: If any stage fails
        """
        LOGGER.info("=" * 70)
        LOGGER.info("PDF REPORT GENERATION")
        LOGGER.info("=" * 70)

        try:
            # Stage 1: Data loading
            LOGGER.info("\n[1/6] Loading data...")
            self._load_data()

            # Stage 2: Metrics computation
            LOGGER.info("\n[2/6] Computing metrics...")
            self._compute_metrics()

            # Stage 3: KPI extraction
            LOGGER.info("\n[3/6] Extracting KPIs...")
            self._extract_kpis()

            # Stage 4: Chart export
            LOGGER.info("\n[4/6] Exporting charts...")
            self._export_charts()

            # Stage 5: Section rendering
            LOGGER.info("\n[5/6] Rendering sections...")
            self._render_sections()

            # Stage 6: PDF assembly
            LOGGER.info("\n[6/6] Building PDF...")
            self._build_pdf()

            # Stage 7: DOCX companion (NEW!)
            LOGGER.info("\n[7/7] Generating DOCX companion...")
            self._generate_docx()

            # Success
            size_kb = self.output_path.stat().st_size / 1024
            LOGGER.info("\n" + "=" * 70)
            LOGGER.info("✓ REPORT GENERATION COMPLETE")
            LOGGER.info("=" * 70)
            LOGGER.info("  Output: %s", self.output_path)
            LOGGER.info("  Size:   %.1f KB", size_kb)
            LOGGER.info("=" * 70)

            return self.output_path

        except Exception as e:
            LOGGER.error("Report generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Report generation failed: {e}") from e

    # ========================================================================
    # PRIVATE METHODS (Pipeline Stages)
    # ========================================================================
    def _load_data(self):
        """Load all data (truth, KF, GPM, MGM, comms)."""
        self.data = load_all_data(
            self.run_id,
            self.target_id,
            include_comms=True
        )

        LOGGER.info("  Truth:         %6d samples", len(self.data.truth))
        LOGGER.info("  KF Track:      %6d samples", len(self.data.kf))
        LOGGER.info("  GPM:           %6d samples", len(self.data.gpm))
        LOGGER.info("  MGM:           %6d samples", len(self.data.mgm))
        LOGGER.info("  Comms Ground:  %6d samples", len(self.data.comms_ground))
        LOGGER.info("  Comms Onboard: %6d samples", len(self.data.comms_onboard))

    def _compute_metrics(self):
        """Compute all metrics from loaded data."""
        self.metrics = compute_all_metrics(
            self.data,
            self.target_id,
            measurement_dof=3,
            state_dim=9,
        )

        LOGGER.info("  Alignment:     %6d KF samples",
                    len(self.metrics.alignment.kf_vs_truth))
        LOGGER.info("  KF errors:     %6d metrics", len(self.metrics.kf_errors))
        LOGGER.info("  GPM errors:    %6d metrics", len(self.metrics.gpm_errors))
        LOGGER.info("  Geometry:      %6d metrics", len(self.metrics.geometry))
        LOGGER.info("  Covariance:    %6d metrics", len(self.metrics.covariance))
        LOGGER.info("  NIS/NEES:      %6d metrics", len(self.metrics.nis_nees))
        LOGGER.info("  Data age:      %6d modes", len(self.metrics.data_age))

    def _extract_kpis(self):
        """Extract KPIs from metrics."""
        # Create context for KPI extraction
        kpi_context = SimpleNamespace(
            metrics=self.metrics,
            data=self.data,
        )

        self.kpis = extract_all_kpis(kpi_context)

        LOGGER.info("  Extracted %d KPIs:", len(self.kpis))
        for kpi_name, kpi_value in self.kpis.items():
            if np.isfinite(kpi_value):
                LOGGER.info("    • %s: %.3f", kpi_name, kpi_value)
            else:
                LOGGER.info("    • %s: N/A", kpi_name)


    def _export_charts(self):
        """Export Plotly figures to PNG."""
        # Create chart directory
        self.chart_dir = self.output_path.parent / f"{self.target_id}_charts"
        self.chart_dir.mkdir(exist_ok=True)

        # Use ChartExporter
        exporter = ChartExporter(
            run_id=self.run_id,
            target_id=self.target_id,
            chart_dir=self.chart_dir
        )

        self.figures = exporter.export_all(
            data=self.data,
            metrics=self.metrics,
            cfg=self.cfg
        )

    def _render_sections(self):
        """Render all enabled sections."""
        from reporting.core.pdf_section_registry import RenderContext

        # Create render context
        context = RenderContext(
            run_id=self.run_id,
            target_id=self.target_id,
            data=self.data,
            metrics=self.metrics,
            kpis=self.kpis,
            chart_dir=self.chart_dir,
            config=self.cfg,
        )

        # Use SectionRenderer
        renderer = SectionRenderer(
            run_id=self.run_id,
            target_id=self.target_id
        )

        self.story = renderer.render_all(context)

    def _build_pdf(self):
        """Build final PDF document."""
        if not self.story:
            raise RuntimeError("Story is empty - no content to render!")

        # Create page template function
        page_template = create_page_template_function(
            self.run_id,
            self.target_id
        )

        # Build PDF
        self.doc.build(
            self.story,
            onFirstPage=page_template,
            onLaterPages=page_template
        )

        LOGGER.info("  PDF written: %s", self.output_path)


    def _generate_docx(self):
        """Generate companion DOCX document."""
        try:
            # Determine DOCX output path (same dir as PDF)
            docx_filename = self.output_path.stem + "_notes.docx"
            self.docx_path = self.output_path.parent / docx_filename

            # Get PDF filename for reference
            pdf_filename = self.output_path.name

            LOGGER.info("  Generating DOCX companion...")

            # Generate DOCX
            generate_docx_companion(
                run_id=self.run_id,
                target_id=self.target_id,
                output_path=self.docx_path,
                pdf_filename=pdf_filename,
                kpis=self.kpis
            )

            LOGGER.info("  DOCX companion written: %s", self.docx_path)

        except ImportError:
            LOGGER.warning("  python-docx not available, skipping DOCX companion")
            self.docx_path = None
        except Exception as e:
            LOGGER.error("  Failed to generate DOCX: %s", e)
            self.docx_path = None

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def generate_pdf_report(run_id: str,
                        target_id: str,
                        output_dir: Optional[Path] = None,
                        cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Generate PDF report for a specific run and target.

    Convenience wrapper around PDFReportBuilder.

    Args:
        run_id: Run identifier
        target_id: Target identifier
        output_dir: Output directory (default: valid/reports/{run_id})
        cfg: Configuration dictionary

    Returns:
        Path to generated PDF

    Example:
        >>> pdf_path = generate_pdf_report(
        ...     run_id="20251025T211158Z",
        ...     target_id="HGV_330"
        ... )
    """
    # Default output directory
    if output_dir is None:
        output_dir = REPO / "valid" / "reports" / run_id

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    output_path = output_dir / f"{target_id}_report.pdf"

    # Build report
    builder = PDFReportBuilder(run_id, target_id, output_path, cfg)
    return builder.build()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PDF performance report"
    )
    parser.add_argument(
        "run_id",
        help="Run identifier (e.g., 20251020T154731Z)"
    )
    parser.add_argument(
        "target_id",
        help="Target identifier (e.g., HGV_330)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output PDF path (default: auto)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose logging",
        action="store_true"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Generate report
    try:
        if args.output:
            builder = PDFReportBuilder(
                args.run_id,
                args.target_id,
                args.output,
                cfg={}
            )
            pdf_path = builder.build()
        else:
            pdf_path = generate_pdf_report(
                args.run_id,
                args.target_id,
                cfg={}
            )

        print(f"\n✓ Report generated: {pdf_path}")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":


    sys.exit(main())